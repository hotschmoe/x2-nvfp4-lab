#!/usr/bin/env python3
"""Validate native scalar-F32-scaled FP8 on a real Ornith attention slice."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from probe_native_fp8 import decode_e4m3
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/Ornith-1.5-35B-A3B-NVFP4",
    )
    parser.add_argument(
        "--tensor", default="model.language_model.layers.3.self_attn.q_proj"
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--vectors", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.rows <= 0 or args.vectors <= 0 or args.iterations <= 0:
        parser.error("rows, vectors, and iterations must be positive")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(ROOT / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    index = json.loads(
        (args.model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    weight_key = args.tensor + ".weight"
    scale_key = args.tensor + ".weight_scale"
    with safe_open(
        args.model / index[weight_key], framework="pt", device="cpu"
    ) as shard:
        tensor = shard.get_slice(weight_key)
        weights = np.ascontiguousarray(
            tensor[: args.rows].view(torch.uint8).numpy()
        )
        weight_scale = float(shard.get_tensor(scale_key).item())

    x = np.ascontiguousarray(
        np.random.default_rng(20260822)
        .standard_normal((args.vectors, weights.shape[1]))
        .astype(np.float32)
    )
    reference = x @ (decode_e4m3(weights) * np.float32(weight_scale)).T
    runtime = Runtime(*runtime_paths())
    matrix = runtime.upload_fp8_tensor_scaled(weights, weight_scale)
    kernel_kind = 2 if args.vectors > 1 else 3
    try:
        runtime.linear_fp8(matrix, x, kernel_kind=kernel_kind)
        started = time.perf_counter()
        for _ in range(args.iterations):
            actual = runtime.linear_fp8(matrix, x, kernel_kind=kernel_kind)
        elapsed = (time.perf_counter() - started) / args.iterations
        maximum_error = float(np.max(np.abs(reference - actual)))
        if not np.allclose(reference, actual, rtol=3e-5, atol=5e-4):
            raise SystemExit(f"tensor-scaled FP8 mismatch: {maximum_error:.9g}")
        operations = 2 * weights.shape[0] * weights.shape[1] * args.vectors
        print(
            f"device={runtime.device_name} tensor={args.tensor} "
            f"shape={weights.shape} vectors={args.vectors} scale={weight_scale:.9g}"
        )
        print(
            f"call_us={elapsed * 1e6:.3f} "
            f"effective_gflops={operations / elapsed / 1e9:.3f} "
            f"max_abs={maximum_error:.9g}"
        )
        print("TENSOR_SCALED_FP8_PASS")
        return 0
    finally:
        matrix.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
