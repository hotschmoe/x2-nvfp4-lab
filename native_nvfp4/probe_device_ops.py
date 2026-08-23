#!/usr/bin/env python3
"""Validate queueable float32 graph primitives on reusable OpenCL buffers."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def assert_close(label: str, expected: np.ndarray, actual: np.ndarray) -> float:
    error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=3e-5, atol=3e-5):
        raise SystemExit(f"{label} mismatch: max_abs={error}")
    return error


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=5120)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.iterations <= 0:
        parser.error("dimensions and iteration count must be positive")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        here / "runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        here / "kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(here.parent / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    rng = np.random.default_rng(20260822)
    shape = (args.rows, args.cols)
    x = np.ascontiguousarray(rng.standard_normal(shape).astype(np.float32))
    other = np.ascontiguousarray(rng.standard_normal(shape).astype(np.float32))
    weight = np.ascontiguousarray(rng.standard_normal(args.cols).astype(np.float32))
    epsilon = 1e-6
    add_reference = x + other
    silu_reference = x / (1.0 + np.exp(-x)) * other
    inverse_rms = 1.0 / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + epsilon)
    norm_reference = x * inverse_rms * weight

    runtime = Runtime(*runtime_paths())
    x_buffer = runtime.upload_buffer(x)
    other_buffer = runtime.upload_buffer(other)
    weight_buffer = runtime.upload_buffer(weight)
    output_buffer = runtime.create_buffer(x.nbytes)
    try:
        runtime.add_device(x_buffer, other_buffer, x.size, output_buffer)
        add_profile = runtime.synchronize()
        add_error = assert_close("add", add_reference, output_buffer.download(shape))

        runtime.silu_mul_device(x_buffer, other_buffer, x.size, output_buffer)
        silu_profile = runtime.synchronize()
        silu_error = assert_close(
            "silu_mul", silu_reference, output_buffer.download(shape)
        )

        runtime.rmsnorm_device(
            x_buffer,
            weight_buffer,
            args.rows,
            args.cols,
            epsilon,
            output_buffer,
        )
        norm_profile = runtime.synchronize()
        norm_error = assert_close(
            "rmsnorm", norm_reference, output_buffer.download(shape)
        )

        started = time.perf_counter()
        for _ in range(args.iterations):
            runtime.rmsnorm_device(
                x_buffer,
                weight_buffer,
                args.rows,
                args.cols,
                epsilon,
                output_buffer,
            )
        queued_profile = runtime.synchronize()
        elapsed = (time.perf_counter() - started) / args.iterations

        print(
            f"device={runtime.lib.nvfp4_runtime_device_name(runtime.handle).decode()} "
            f"rows={args.rows} cols={args.cols}"
        )
        print(
            f"add_us={add_profile.kernel_ns / 1e3:.3f} max_abs_err={add_error:.8g}"
        )
        print(
            f"silu_mul_us={silu_profile.kernel_ns / 1e3:.3f} "
            f"max_abs_err={silu_error:.8g}"
        )
        print(
            f"rmsnorm_us={norm_profile.kernel_ns / 1e3:.3f} "
            f"max_abs_err={norm_error:.8g}"
        )
        print(
            f"queued_rmsnorm_kernel_us="
            f"{queued_profile.kernel_ns / args.iterations / 1e3:.3f} "
            f"queued_rmsnorm_wall_us={elapsed * 1e6:.3f}"
        )
        print("PASS: queued device graph primitives match NumPy references")
    finally:
        output_buffer.close()
        weight_buffer.close()
        other_buffer.close()
        x_buffer.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
