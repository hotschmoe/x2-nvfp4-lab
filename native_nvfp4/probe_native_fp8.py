#!/usr/bin/env python3
"""Validate direct row-scaled FP8 linear using the persistent OpenCL runtime."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from probe_runtime import Runtime
from safetensors import safe_open


def decode_e4m3(raw: np.ndarray) -> np.ndarray:
    magnitude = raw & np.uint8(0x7F)
    exp = (magnitude >> np.uint8(3)).astype(np.int32)
    man = (magnitude & np.uint8(7)).astype(np.float32)
    value = np.where(exp == 0, man / 512.0, (1.0 + man / 8.0) * np.exp2(exp - 7))
    value[magnitude == 0x7F] = 0.0
    value = np.where((raw & np.uint8(0x80)) != 0, -value, value)
    return value.astype(np.float32)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=Path, default=here / "runtime/build/nvfp4_runtime.dll")
    parser.add_argument("--model", type=Path, default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors")
    parser.add_argument("--tensor", default="model.language_model.layers.0.linear_attn.in_proj_qkv")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=5120)
    parser.add_argument("--vectors", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--kernel",
        choices=("scalar", "subgroup", "tiled", "rows_tiled"),
        default="rows_tiled",
    )
    args = parser.parse_args()
    if args.rows < 1 or args.cols < 1 or args.vectors < 1 or args.iterations < 1:
        parser.error("dimensions and iterations must be positive")
    if (args.kernel == "tiled") != (args.vectors > 1):
        parser.error(
            "use tiled for multi-vector inputs and another kernel for one vector"
        )

    with safe_open(args.model, framework="pt", device="cpu") as model:
        weight_t = model.get_slice(args.tensor + ".weight")[:args.rows, :args.cols]
        scale_t = model.get_slice(args.tensor + ".weight_scale")[:args.rows, :]
    weights = np.ascontiguousarray(weight_t.view(torch.uint8).numpy())
    scales_bf16 = np.ascontiguousarray(scale_t.view(torch.uint16).numpy())
    scales_f32 = scale_t.float().numpy().reshape(-1)
    x = np.random.default_rng(20260822).standard_normal((args.vectors, args.cols)).astype(np.float32)

    cpu_started = time.perf_counter()
    decoded = decode_e4m3(weights) * scales_f32[:, None]
    reference = x @ decoded.T
    cpu_elapsed = time.perf_counter() - cpu_started

    runtime = Runtime(args.dll.resolve(), here / "kernels/nvfp4_gemv.cl")
    matrix = runtime.upload_fp8(weights, scales_bf16)
    try:
        kind = {"scalar": 0, "subgroup": 1, "tiled": 2, "rows_tiled": 3}[
            args.kernel
        ]
        runtime.linear_fp8(matrix, x, args.rows, kind)
        started = time.perf_counter()
        for _ in range(args.iterations):
            result = runtime.linear_fp8(matrix, x, args.rows, kind)
        elapsed = time.perf_counter() - started
        max_abs = float(np.max(np.abs(reference - result)))
        max_rel = float(np.max(np.abs(reference - result) / np.maximum(np.abs(reference), 1e-6)))
        operations = 2.0 * args.rows * args.cols * args.vectors
        print(f"tensor={args.tensor} device={runtime.device_name} rows={args.rows} "
              f"cols={args.cols} vectors={args.vectors} kernel={args.kernel}")
        print(f"native_weight_bytes={weights.nbytes + scales_bf16.nbytes} "
              f"cpu_seconds={cpu_elapsed:.6f} call_us={elapsed/args.iterations*1e6:.3f} "
              f"effective_gflops={operations/(elapsed/args.iterations)/1e9:.3f}")
        print(f"max_abs_err={max_abs:.8g} max_rel_err={max_rel:.8g}")
        if not np.allclose(reference, result, rtol=3e-5, atol=5e-4):
            raise SystemExit("direct FP8 runtime does not match CPU reference")
        print("PASS: native row-scaled FP8 linear matches CPU reference")
    finally:
        runtime.destroy_fp8_matrix(matrix)
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
