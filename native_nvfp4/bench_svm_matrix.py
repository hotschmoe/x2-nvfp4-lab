"""Interleaved copied-buffer versus shared-SVM benchmark on a real NVFP4 matrix."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from bench_islands import memory_status, percentile, power_status, system_model
from probe_native_nvfp4 import cpu_gemm
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "campaign_results/bandwidth-first"


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors",
    )
    parser.add_argument(
        "--tensor", default="model.language_model.layers.0.mlp.gate_proj"
    )
    parser.add_argument("--rows", type=int, default=17408)
    parser.add_argument("--cols", type=int, default=5120)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if (
        args.rows <= 0
        or args.cols <= 0
        or args.cols % 16
        or args.warmups < 0
        or args.samples <= 0
        or args.cpu_threads < 0
    ):
        parser.error("invalid dimensions or sample counts")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(ROOT / "vllm_nvfp4_opencl/src"))
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
    payload = packed.nbytes + scales.nbytes

    runtime = Runtime(*runtime_paths())
    memory_before = memory_status().available_physical
    copied = runtime.upload(packed, scales, global_scale, shared_svm=False)
    memory_after_copied = memory_status().available_physical
    shared = runtime.upload(packed, scales, global_scale, shared_svm=True)
    memory_after_shared = memory_status().available_physical
    del packed, scales, packed_tensor, scale_tensor
    gc.collect()
    memory_after_sources_released = memory_status().available_physical
    input_buffer = runtime.upload_buffer(x)
    output_buffer = runtime.create_buffer(args.rows * np.dtype(np.float32).itemsize)

    kernel_ms: dict[str, list[float]] = {"copied": [], "shared_svm": []}
    wall_ms: dict[str, list[float]] = {"copied": [], "shared_svm": []}

    def execute(label: str) -> None:
        matrix = copied if label == "copied" else shared
        started = time.perf_counter_ns()
        runtime.linear_device(
            matrix, input_buffer, 1, out=output_buffer, kernel_kind=3
        )
        wall_ms[label].append((time.perf_counter_ns() - started) / 1e6)
        kernel_ms[label].append(runtime.last_profile().kernel_ns / 1e6)

    try:
        for iteration in range(args.warmups):
            for label in (
                ("copied", "shared_svm")
                if iteration % 2 == 0
                else ("shared_svm", "copied")
            ):
                execute(label)
        for values in (kernel_ms, wall_ms):
            for samples in values.values():
                samples.clear()

        for iteration in range(args.samples):
            for label in (
                ("copied", "shared_svm")
                if iteration % 2 == 0
                else ("shared_svm", "copied")
            ):
                execute(label)

        results: dict[str, np.ndarray] = {}
        for label, matrix in (("copied", copied), ("shared_svm", shared)):
            runtime.linear_device(
                matrix, input_buffer, 1, out=output_buffer, kernel_kind=3
            )
            results[label] = output_buffer.download((1, args.rows))
        cpu_shared = runtime.linear_shared_cpu(
            shared, x, threads=args.cpu_threads
        )
        errors = {
            "copied_gpu_max_abs": float(np.max(np.abs(reference - results["copied"]))),
            "shared_gpu_max_abs": float(np.max(np.abs(reference - results["shared_svm"]))),
            "shared_cpu_max_abs": float(np.max(np.abs(reference - cpu_shared))),
            "gpu_exact_match": bool(np.array_equal(results["copied"], results["shared_svm"])),
        }
        if not all(
            np.allclose(reference, result, rtol=2e-5, atol=2e-5)
            for result in (results["copied"], results["shared_svm"], cpu_shared)
        ):
            raise SystemExit(f"correctness failure: {errors}")

        copied_kernel = summary(kernel_ms["copied"])
        shared_kernel = summary(kernel_ms["shared_svm"])
        copied_wall = summary(wall_ms["copied"])
        shared_wall = summary(wall_ms["shared_svm"])
        record = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hardware": {
                "system_model": system_model(),
                "gpu": runtime.lib.nvfp4_runtime_device_name(runtime.handle).decode(),
                "physical_memory_bytes": memory_status().total_physical,
            },
            "software": {"command": sys.argv},
            "environment": {
                **power_status(),
                "thermal_regime": "warm-interleaved",
                "available_physical_bytes": {
                    "before_upload": memory_before,
                    "after_copied_upload": memory_after_copied,
                    "after_shared_upload": memory_after_shared,
                    "after_source_arrays_released": memory_after_sources_released,
                },
            },
            "workload": {
                "operation": "nvfp4_exact_matrix_svm_comparison",
                "format": "nvfp4",
                "tensor": args.tensor,
                "rows": args.rows,
                "cols": args.cols,
                "vectors": 1,
                "logical_payload_bytes": payload,
            },
            "timing": {
                "warmups": args.warmups,
                "samples": args.samples,
                "kernel_ms": {
                    "copied": copied_kernel,
                    "shared_svm": shared_kernel,
                },
                "wall_ms": {"copied": copied_wall, "shared_svm": shared_wall},
                "kernel_speedup": copied_kernel["median"] / shared_kernel["median"],
                "wall_speedup": copied_wall["median"] / shared_wall["median"],
            },
            "bandwidth": {
                "copied_kernel_logical_gbs": payload / (copied_kernel["median"] * 1e6),
                "shared_kernel_logical_gbs": payload / (shared_kernel["median"] * 1e6),
                "matched_gpu_raw_ceiling_gbs": 129.0,
                "copied_island_utilization": payload
                / (copied_kernel["median"] * 1e6)
                / 129.0,
                "shared_island_utilization": payload
                / (shared_kernel["median"] * 1e6)
                / 129.0,
                "physical_gbs": None,
            },
            "correctness": {
                "passed": True,
                **errors,
                "explicit_completion_marker": True,
            },
            "samples": {"kernel_ms": kernel_ms, "wall_ms": wall_ms},
        }
        args.results.mkdir(parents=True, exist_ok=True)
        slug = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = args.results / f"{slug}-nvfp4-svm-comparison.json"
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"copied_kernel_ms={copied_kernel['median']:.4f} "
            f"shared_kernel_ms={shared_kernel['median']:.4f} "
            f"speedup={record['timing']['kernel_speedup']:.3f}x"
        )
        print(
            f"copied_logical_gbs={record['bandwidth']['copied_kernel_logical_gbs']:.2f} "
            f"shared_logical_gbs={record['bandwidth']['shared_kernel_logical_gbs']:.2f} "
            "shared_raw_ceiling_utilization="
            f"{record['bandwidth']['shared_island_utilization']:.1%}"
        )
        print(f"correctness={errors} result={output}")
        print("SVM_NVFP4_INTERLEAVED_PASS")
    finally:
        output_buffer.close()
        input_buffer.close()
        shared.close()
        copied.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
