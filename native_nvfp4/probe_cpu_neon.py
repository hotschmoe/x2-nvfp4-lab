#!/usr/bin/env python3
"""Validate and benchmark the ARM64 NEON native-NVFP4 fallback."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from probe_native_nvfp4 import cpu_gemm
from safetensors import safe_open


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors",
    )
    parser.add_argument(
        "--tensor", default="model.language_model.layers.0.mlp.gate_proj"
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=5120)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 0])
    args = parser.parse_args()
    if (
        args.rows <= 0
        or args.cols <= 0
        or args.cols % 16
        or args.iterations <= 0
        or any(thread < 0 for thread in args.threads)
    ):
        parser.error("invalid matrix dimensions, iterations, or thread count")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        here / "runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        here / "kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(here.parent / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    with safe_open(args.model, framework="pt", device="cpu") as checkpoint:
        packed_tensor = checkpoint.get_slice(args.tensor + ".weight_packed")[
            : args.rows, : args.cols // 2
        ]
        scale_tensor = checkpoint.get_slice(args.tensor + ".weight_scale")[
            : args.rows, : args.cols // 16
        ]
        global_scale = float(
            checkpoint.get_tensor(args.tensor + ".weight_global_scale").item()
        )
    packed = np.ascontiguousarray(
        packed_tensor.numpy().astype(np.uint8, copy=False)
    )
    scales = np.ascontiguousarray(scale_tensor.view(torch.uint8).numpy())
    x = np.ascontiguousarray(
        np.random.default_rng(20260822)
        .standard_normal((1, args.cols))
        .astype(np.float32)
    )
    reference = cpu_gemm(packed, scales, x, global_scale)

    runtime = Runtime(*runtime_paths())
    try:
        for threads in args.threads:
            result = runtime.linear_cpu(packed, scales, global_scale, x, threads)
            max_abs = float(np.max(np.abs(reference - result)))
            if not np.allclose(reference, result, rtol=3e-5, atol=3e-5):
                raise SystemExit(
                    f"NEON mismatch for threads={threads}: max_abs={max_abs}"
                )
            started = time.perf_counter()
            for _ in range(args.iterations):
                result = runtime.linear_cpu(
                    packed, scales, global_scale, x, threads
                )
            elapsed = (time.perf_counter() - started) / args.iterations
            gflops = 2.0 * args.rows * args.cols / elapsed / 1e9
            label = "auto" if threads == 0 else str(threads)
            print(
                f"threads={label} latency_ms={elapsed * 1e3:.3f} "
                f"gflops={gflops:.3f} max_abs_err={max_abs:.8g}"
            )
        print(
            f"rows={args.rows} cols={args.cols} "
            "PASS: ARM64 NEON consumes checkpoint-native NVFP4"
        )
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
