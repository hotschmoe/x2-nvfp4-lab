"""Benchmark a fully device-routed contiguous-SVM Qwen3.5 MoE expert bank."""

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
from bench_islands import memory_status, power_status, system_model
from bench_moe_experts import load_experts
from bench_moe_routed_layer import (
    MODEL,
    RESULTS,
    describe,
    expert_reference,
    load_layer_tensors,
    route,
)
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]


def stream_experts_into_bank(bank: object, model_dir: Path, layer: int) -> int:
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    projection_names = ("gate_proj", "up_proj", "down_proj")
    by_shard: dict[str, list[int]] = {}
    for expert in range(256):
        prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
        shards = {
            index[prefix + f".{projection}.{suffix}"]
            for projection in projection_names
            for suffix in ("weight", "weight_scale", "weight_scale_2")
        }
        if len(shards) != 1:
            raise RuntimeError(f"expert {expert} spans checkpoint shards")
        by_shard.setdefault(shards.pop(), []).append(expert)

    payload_bytes = 0
    for shard_name, expert_ids in by_shard.items():
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as shard:
            for expert in expert_ids:
                prefix = (
                    f"model.language_model.layers.{layer}.mlp.experts.{expert}"
                )
                projections: list[tuple[np.ndarray, np.ndarray, float]] = []
                for projection in projection_names:
                    base = prefix + f".{projection}"
                    packed = np.ascontiguousarray(
                        shard.get_tensor(base + ".weight")
                        .numpy()
                        .astype(np.uint8, copy=False)
                    )
                    scales = np.ascontiguousarray(
                        shard.get_tensor(base + ".weight_scale")
                        .view(torch.uint8)
                        .numpy()
                    )
                    multiplier = float(
                        shard.get_tensor(base + ".weight_scale_2").item()
                    )
                    projections.append((packed, scales, 1.0 / multiplier))
                    payload_bytes += packed.nbytes + scales.nbytes
                bank.upload_expert(expert, projections)
    return payload_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.layer < 0 or args.warmups < 0 or args.samples <= 0:
        parser.error("invalid layer or sample counts")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(ROOT / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    router_bf16, router_f32, shared_gate_bf16, shared_host = load_layer_tensors(
        args.model, args.layer
    )
    x = np.ascontiguousarray(
        np.random.default_rng(20260822).standard_normal((1, 2048)).astype(np.float32)
        * np.float32(0.2)
    )
    expected_ids, expected_weights = route(router_f32 @ x[0], 8)
    shared_gate_f32 = (
        np.left_shift(shared_gate_bf16.astype(np.uint32), 16)
        .view(np.float32)
        .reshape(-1)
    )
    shared_logit = float((shared_gate_f32 @ x[0]).item())
    shared_weight = float(1.0 / (1.0 + np.exp(-shared_logit)))

    selected_hosts = load_experts(args.model, args.layer, expected_ids)
    reference = np.zeros((1, 2048), dtype=np.float32)
    for weight, expert in zip(expected_weights, selected_hosts, strict=True):
        reference += np.float32(weight) * expert_reference(expert, x)
    reference += np.float32(shared_weight) * expert_reference(shared_host, x)

    available_before = memory_status().available_physical
    runtime = Runtime(*runtime_paths())
    bank_started = time.perf_counter()
    bank = runtime.create_moe_bank(router_bf16, shared_gate_bf16, 512)
    available_after_allocation = memory_status().available_physical
    payload_bytes = stream_experts_into_bank(bank, args.model, args.layer)
    bank.upload_expert(256, shared_host)
    payload_bytes += sum(
        packed.nbytes + scales.nbytes
        for packed, scales, _divisor in shared_host
    )
    bank_load_seconds = time.perf_counter() - bank_started
    available_with_sources = memory_status().available_physical
    del selected_hosts
    gc.collect()
    available_after_source_release = memory_status().available_physical

    input_buffer = runtime.upload_buffer(x)
    output_buffer = runtime.create_buffer(2048 * 4)

    def execute() -> tuple[float, float]:
        started = time.perf_counter_ns()
        bank.decode_device(input_buffer, output_buffer)
        profile = runtime.synchronize()
        return profile.kernel_ns / 1e6, (time.perf_counter_ns() - started) / 1e6

    try:
        execute()
        result = output_buffer.download((1, 2048))
        max_abs = float(np.max(np.abs(reference - result)))
        if not np.allclose(reference, result, rtol=1e-4, atol=1e-4):
            raise SystemExit(f"device-bank oracle mismatch: max_abs={max_abs}")
        for _ in range(args.warmups):
            execute()
        samples = [execute() for _ in range(args.samples)]
        kernel_ms = [sample[0] for sample in samples]
        wall_ms = [sample[1] for sample in samples]
        kernel_stats = describe(kernel_ms)
        wall_stats = describe(wall_ms)
        active_payload_bytes = payload_bytes // 257 * 9 + router_bf16.nbytes + shared_gate_bf16.nbytes
        record = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hardware": {"system_model": system_model(), "gpu": runtime.device_name},
            "software": {"command": sys.argv},
            "environment": {**power_status(), "thermal_regime": "warm-burst"},
            "workload": {
                "operation": "qwen35_moe_device_routed_svm_bank",
                "format": "nvfp4_expert_bank_bf16_router",
                "layer": args.layer,
                "experts_resident": 256,
                "top_k": 8,
                "selected_experts": expected_ids,
                "routing_weights": expected_weights.tolist(),
                "shared_expert_gate": shared_weight,
                "resident_bank_payload_bytes": payload_bytes,
                "active_logical_payload_bytes": active_payload_bytes,
            },
            "loading": {
                "bank_upload_seconds": bank_load_seconds,
                "available_memory_before_bytes": available_before,
                "available_after_allocation_bytes": available_after_allocation,
                "available_with_sources_bytes": available_with_sources,
                "available_after_source_release_bytes": available_after_source_release,
            },
            "timing": {
                "warmups": args.warmups,
                "samples": args.samples,
                "kernel_ms": kernel_stats,
                "wall_ms": wall_stats,
            },
            "bandwidth": {
                "logical_gbs": active_payload_bytes / (kernel_stats["median"] * 1e6),
                "matched_island_ceiling_gbs": 129.0,
                "island_utilization": active_payload_bytes / (kernel_stats["median"] * 1e6) / 129.0,
                "physical_gbs": None,
            },
            "correctness": {
                "passed": True,
                "max_abs_error": max_abs,
                "expected_router_ids": expected_ids,
                "finite_outputs": bool(np.isfinite(result).all()),
                "explicit_completion_marker": True,
            },
            "limitations": [
                "single-token one-layer decode micrograph",
                "all 256 routed experts plus the shared expert are resident for one layer only",
                "attention, norms, residuals, sampling, and serving overhead are excluded",
                "physical DRAM traffic and per-kernel component timings are unavailable",
            ],
            "samples": {"kernel_ms": kernel_ms, "wall_ms": wall_ms},
        }
        args.results.mkdir(parents=True, exist_ok=True)
        result_path = args.results / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-moe-device-bank.json"
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"device={runtime.device_name} layer={args.layer} "
            f"experts={expected_ids} resident_bytes={payload_bytes}"
        )
        print(
            f"bank_stream_load_s={bank_load_seconds:.3f} "
            f"source_release_recovered_bytes={available_after_source_release - available_with_sources}"
        )
        print(
            f"kernel_ms={kernel_stats['median']:.4f} wall_ms={wall_stats['median']:.4f} "
            f"logical_gbs={record['bandwidth']['logical_gbs']:.2f}"
        )
        print(f"max_abs_err={max_abs:.8g} result={result_path}")
        print("MOE_NVFP4_DEVICE_BANK_PASS")
    finally:
        output_buffer.close()
        input_buffer.close()
        bank.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
