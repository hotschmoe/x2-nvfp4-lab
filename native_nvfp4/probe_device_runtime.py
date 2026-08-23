#!/usr/bin/env python3
"""Validate profiled device-resident NVFP4 linear execution."""

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
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--shared-svm", action="store_true")
    parser.add_argument("--default-allocation", action="store_true")
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.cols % 16 or args.iterations <= 0:
        parser.error("invalid matrix dimensions or iteration count")

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
    allocation = None if args.default_allocation else args.shared_svm
    matrix = runtime.upload(packed, scales, global_scale, shared_svm=allocation)
    input_buffer = runtime.upload_buffer(x)
    output_buffer = runtime.create_buffer(args.rows * np.dtype(np.float32).itemsize)
    try:
        subgroup = runtime.linear(matrix, x, kernel_kind=1)
        subgroup_profile = runtime.last_profile()
        row_tiled = runtime.linear(matrix, x, kernel_kind=3)
        row_tiled_profile = runtime.last_profile()
        runtime.linear_device(
            matrix,
            input_buffer,
            1,
            out=output_buffer,
            kernel_kind=3,
        )
        device_result = output_buffer.download((1, args.rows))

        for label, result in (
            ("subgroup", subgroup),
            ("row_tiled", row_tiled),
            ("device_row_tiled", device_result),
        ):
            if not np.allclose(reference, result, rtol=2e-5, atol=2e-5):
                error = float(np.max(np.abs(reference - result)))
                raise SystemExit(f"{label} mismatch: max_abs={error}")

        kernel_times = []
        started = time.perf_counter()
        for _ in range(args.iterations):
            runtime.linear_device(
                matrix,
                input_buffer,
                1,
                out=output_buffer,
                kernel_kind=3,
            )
            kernel_times.append(runtime.last_profile().kernel_ns)
        wall_seconds = (time.perf_counter() - started) / args.iterations
        device_result = output_buffer.download((1, args.rows))
        max_abs = float(np.max(np.abs(reference - device_result)))

        started = time.perf_counter()
        for _ in range(args.iterations):
            runtime.linear_device(
                matrix,
                input_buffer,
                1,
                out=output_buffer,
                kernel_kind=3,
                enqueue=True,
            )
        queued_profile = runtime.synchronize()
        queued_wall_seconds = (time.perf_counter() - started) / args.iterations
        queued_result = output_buffer.download((1, args.rows))
        queued_max_abs = float(np.max(np.abs(reference - queued_result)))
        if not np.allclose(reference, queued_result, rtol=2e-5, atol=2e-5):
            raise SystemExit(
                f"queued row-tiled mismatch: max_abs={queued_max_abs}"
            )

        print(
            f"device={runtime.lib.nvfp4_runtime_device_name(runtime.handle).decode()} "
            f"rows={args.rows} cols={args.cols} shared_svm={matrix.shared_svm}"
        )
        print(
            "subgroup_profile_us="
            f"{subgroup_profile.upload_ns / 1e3:.3f}/"
            f"{subgroup_profile.kernel_ns / 1e3:.3f}/"
            f"{subgroup_profile.download_ns / 1e3:.3f} upload/kernel/download"
        )
        print(
            "row_tiled_profile_us="
            f"{row_tiled_profile.upload_ns / 1e3:.3f}/"
            f"{row_tiled_profile.kernel_ns / 1e3:.3f}/"
            f"{row_tiled_profile.download_ns / 1e3:.3f} upload/kernel/download"
        )
        print(
            f"resident_kernel_us={np.mean(kernel_times) / 1e3:.3f} "
            f"resident_wall_us={wall_seconds * 1e6:.3f} max_abs_err={max_abs:.8g}"
        )
        print(
            f"queued_kernel_us={queued_profile.kernel_ns / args.iterations / 1e3:.3f} "
            f"queued_wall_us={queued_wall_seconds * 1e6:.3f} "
            f"max_abs_err={queued_max_abs:.8g}"
        )
        print("PASS: device-resident row-tiled NVFP4 matches CPU reference")
    finally:
        output_buffer.close()
        input_buffer.close()
        matrix.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
