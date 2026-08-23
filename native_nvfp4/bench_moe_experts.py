"""Validate and benchmark one routed top-k NVFP4 MoE expert set."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from bench_islands import percentile, power_status, system_model
from probe_native_nvfp4 import cpu_gemm
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/Ornith-1.5-35B-A3B-NVFP4"
RESULTS = ROOT / "campaign_results/bandwidth-first"


def describe(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def load_experts(
    model_dir: Path,
    layer: int,
    expert_ids: list[int],
) -> list[list[tuple[np.ndarray, np.ndarray, float]]]:
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    requested: dict[str, tuple[int, int, str]] = {}
    projection_names = ("gate_proj", "up_proj", "down_proj")
    for expert_index, expert_id in enumerate(expert_ids):
        prefix = (
            f"model.language_model.layers.{layer}.mlp.experts."
            f"{expert_id}"
        )
        for projection_index, projection in enumerate(projection_names):
            base = prefix + "." + projection
            for suffix in ("weight", "weight_scale", "weight_scale_2"):
                key = base + "." + suffix
                if key not in index:
                    raise KeyError(f"checkpoint index is missing {key}")
                requested[key] = (expert_index, projection_index, suffix)

    by_shard: dict[str, list[str]] = {}
    for key in requested:
        by_shard.setdefault(index[key], []).append(key)
    loaded: list[list[dict[str, Any]]] = [
        [dict() for _ in projection_names] for _ in expert_ids
    ]
    for shard_name, keys in by_shard.items():
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as shard:
            for key in keys:
                expert_index, projection_index, suffix = requested[key]
                tensor = shard.get_tensor(key)
                if suffix == "weight":
                    value: Any = np.ascontiguousarray(
                        tensor.numpy().astype(np.uint8, copy=False)
                    )
                elif suffix == "weight_scale":
                    value = np.ascontiguousarray(tensor.view(torch.uint8).numpy())
                else:
                    # compressed-tensors scale_2 multiplies the block scale;
                    # the native ABI accepts the equivalent global divisor.
                    multiplier = float(tensor.item())
                    if not np.isfinite(multiplier) or multiplier <= 0:
                        raise ValueError(f"invalid {key}: {multiplier}")
                    value = 1.0 / multiplier
                loaded[expert_index][projection_index][suffix] = value

    result: list[list[tuple[np.ndarray, np.ndarray, float]]] = []
    for expert in loaded:
        result.append(
            [
                (projection["weight"], projection["weight_scale"], projection["weight_scale_2"])
                for projection in expert
            ]
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    expert_ids = [int(value) for value in args.experts.split(",")]
    if (
        args.layer < 0
        or not expert_ids
        or len(set(expert_ids)) != len(expert_ids)
        or any(expert < 0 or expert >= 256 for expert in expert_ids)
        or args.warmups < 0
        or args.samples <= 0
    ):
        parser.error("invalid layer, expert set, or sample counts")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(ROOT / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    hosts = load_experts(args.model, args.layer, expert_ids)
    for expert in hosts:
        gate, up, down = expert
        if gate[0].shape != (512, 1024) or up[0].shape != (512, 1024):
            raise SystemExit("expert gate/up must be packed [512, 1024]")
        if down[0].shape != (2048, 256):
            raise SystemExit("expert down must be packed [2048, 256]")
    payload = sum(
        packed.nbytes + scales.nbytes
        for expert in hosts
        for packed, scales, _global_divisor in expert
    )

    x = np.ascontiguousarray(
        np.random.default_rng(20260822)
        .standard_normal((1, 2048))
        .astype(np.float32)
        * np.float32(0.2)
    )
    gate_host, up_host, down_host = hosts[0]
    gate_reference = cpu_gemm(*gate_host[:2], x, gate_host[2])
    up_reference = cpu_gemm(*up_host[:2], x, up_host[2])
    activation_reference = np.ascontiguousarray(
        gate_reference / (1.0 + np.exp(-gate_reference)) * up_reference
    )
    reference = cpu_gemm(
        *down_host[:2], activation_reference, down_host[2]
    )

    runtime = Runtime(*runtime_paths())
    matrices = [
        tuple(runtime.upload(*projection) for projection in expert)
        for expert in hosts
    ]
    input_buffer = runtime.upload_buffer(x)
    gate_buffer = runtime.create_buffer(512 * np.dtype(np.float32).itemsize)
    up_buffer = runtime.create_buffer(512 * np.dtype(np.float32).itemsize)
    activation_buffer = runtime.create_buffer(
        512 * np.dtype(np.float32).itemsize
    )
    output_buffer = runtime.create_buffer(2048 * np.dtype(np.float32).itemsize)

    def enqueue_expert(expert_matrices: tuple[Any, Any, Any]) -> None:
        gate, up, down = expert_matrices
        runtime.linear_device(
            gate, input_buffer, 1, out=gate_buffer, kernel_kind=3, enqueue=True
        )
        runtime.linear_device(
            up, input_buffer, 1, out=up_buffer, kernel_kind=3, enqueue=True
        )
        runtime.silu_mul_device(
            gate_buffer, up_buffer, 512, activation_buffer
        )
        runtime.linear_device(
            down,
            activation_buffer,
            1,
            out=output_buffer,
            kernel_kind=3,
            enqueue=True,
        )

    try:
        enqueue_expert(matrices[0])
        runtime.synchronize()
        result = output_buffer.download((1, 2048))
        max_abs = float(np.max(np.abs(reference - result)))
        if not np.allclose(reference, result, rtol=5e-5, atol=5e-5):
            raise SystemExit(f"expert oracle mismatch: max_abs={max_abs}")

        def execute_set() -> tuple[float, float]:
            started = time.perf_counter_ns()
            for expert_matrices in matrices:
                enqueue_expert(expert_matrices)
            profile = runtime.synchronize()
            return profile.kernel_ns / 1e6, (time.perf_counter_ns() - started) / 1e6

        for _ in range(args.warmups):
            execute_set()
        kernel_ms: list[float] = []
        wall_ms: list[float] = []
        for _ in range(args.samples):
            kernel, wall = execute_set()
            kernel_ms.append(kernel)
            wall_ms.append(wall)
        if not np.isfinite(output_buffer.download((1, 2048))).all():
            raise SystemExit("repeated expert output is non-finite")

        kernel_stats = describe(kernel_ms)
        wall_stats = describe(wall_ms)
        logical_gbs = payload / (kernel_stats["median"] * 1e6)
        record = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hardware": {
                "system_model": system_model(),
                "gpu": runtime.device_name,
            },
            "software": {"command": sys.argv},
            "environment": {**power_status(), "thermal_regime": "warm-burst"},
            "workload": {
                "operation": "qwen35_moe_routed_experts",
                "format": "nvfp4",
                "layer": args.layer,
                "expert_ids": expert_ids,
                "rows": 0,
                "cols": 0,
                "vectors": 1,
                "logical_payload_bytes": payload,
            },
            "timing": {
                "warmups": args.warmups,
                "samples": args.samples,
                "kernel_ms": kernel_stats,
                "wall_ms": wall_stats,
                "per_expert_kernel_ms": kernel_stats["median"] / len(expert_ids),
            },
            "bandwidth": {
                "logical_gbs": logical_gbs,
                "matched_island_ceiling_gbs": 129.0,
                "island_utilization": logical_gbs / 129.0,
                "physical_gbs": None,
            },
            "correctness": {
                "passed": True,
                "first_expert_max_abs_error": max_abs,
                "finite_outputs": True,
                "explicit_completion_marker": True,
            },
            "limitations": [
                "router top-k selection is supplied by expert_ids",
                "router weights and expert-output reduction are not included",
                "expert kernels execute serially on one in-order queue",
            ],
            "samples": {"kernel_ms": kernel_ms, "wall_ms": wall_ms},
        }
        args.results.mkdir(parents=True, exist_ok=True)
        slug = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = args.results / f"{slug}-moe-experts.json"
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"device={runtime.device_name} layer={args.layer} "
            f"experts={expert_ids} native_payload_bytes={payload}"
        )
        print(
            f"kernel_ms={kernel_stats['median']:.4f} "
            f"wall_ms={wall_stats['median']:.4f} "
            f"per_expert_kernel_ms={kernel_stats['median'] / len(expert_ids):.4f} "
            f"logical_gbs={logical_gbs:.2f}"
        )
        print(f"first_expert_max_abs_err={max_abs:.8g} result={output}")
        print("MOE_NVFP4_EXPERTS_PASS")
    finally:
        output_buffer.close()
        activation_buffer.close()
        up_buffer.close()
        gate_buffer.close()
        input_buffer.close()
        for expert in reversed(matrices):
            for matrix in reversed(expert):
                matrix.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
