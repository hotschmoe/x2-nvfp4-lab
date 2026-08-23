#!/usr/bin/env python3
"""Compose an exact Ornith attention block and device-routed MoE bank."""

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
from bench_moe_device_bank import stream_experts_into_bank
from bench_moe_experts import load_experts
from bench_moe_full_attention import load_layer as load_attention_layer
from bench_moe_routed_layer import (
    MODEL,
    RESULTS,
    expert_reference,
    load_layer_tensors,
    route,
)
from bench_resident_full_attention import rmsnorm, rope
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]


def describe(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def load_post_norm(model_dir: Path, layer: int) -> np.ndarray:
    key = f"model.language_model.layers.{layer}.post_attention_layernorm.weight"
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    with safe_open(model_dir / index[key], framework="pt", device="cpu") as shard:
        return np.ascontiguousarray(shard.get_tensor(key).float().numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--kv-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.layer < 0 or args.warmups < 0 or args.samples <= 0:
        parser.error("invalid layer, warmup, or sample count")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(ROOT / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.graph import ResidentQwen35FullAttention
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    matrix_hosts, input_norm, q_norm, k_norm = load_attention_layer(
        args.model, args.layer
    )
    post_norm = load_post_norm(args.model, args.layer)
    router_bf16, router_f32, shared_gate_bf16, shared_host = load_layer_tensors(
        args.model, args.layer
    )
    runtime = Runtime(*runtime_paths())
    matrices = [runtime.upload_fp8_tensor_scaled(*host) for host in matrix_hosts]
    pool = runtime.create_paged_attention_pool(
        1, kv_dtype=args.kv_dtype, query_heads=16, kv_heads=2
    )
    attention = ResidentQwen35FullAttention(
        runtime,
        *matrices,
        input_norm_weight=input_norm,
        q_norm_weight=q_norm,
        k_norm_weight=k_norm,
        max_tokens=1,
        attention_pool=pool,
        hidden=2048,
        query_heads=16,
        kv_heads=2,
    )
    post_norm_buffer = runtime.upload_buffer(
        np.ascontiguousarray(post_norm + np.float32(1.0))
    )
    available_before_bank = memory_status().available_physical
    bank_started = time.perf_counter()
    bank = runtime.create_moe_bank(router_bf16, shared_gate_bf16, 512)
    payload_bytes = stream_experts_into_bank(bank, args.model, args.layer)
    bank.upload_expert(256, shared_host)
    payload_bytes += sum(
        packed.nbytes + scales.nbytes
        for packed, scales, _divisor in shared_host
    )
    bank_load_seconds = time.perf_counter() - bank_started
    available_after_bank = memory_status().available_physical

    x = np.ascontiguousarray(
        np.random.default_rng(20260822).standard_normal((1, 2048)).astype(np.float32)
        * np.float32(0.2)
    )
    cos, sin = rope(0)
    input_buffer = runtime.upload_buffer(x)
    cos_buffer = runtime.upload_buffer(cos)
    sin_buffer = runtime.upload_buffer(sin)
    attention_output = runtime.create_buffer(x.nbytes)
    normalized = runtime.create_buffer(x.nbytes)
    moe_output = runtime.create_buffer(x.nbytes)
    final_output = runtime.create_buffer(x.nbytes)

    try:
        attention.enqueue(input_buffer, cos_buffer, sin_buffer, attention_output)
        runtime.synchronize()
        attention_host = attention_output.download(x.shape)
        normalized_host = np.ascontiguousarray(rmsnorm(attention_host, post_norm))
        expected_ids, expected_weights = route(router_f32 @ normalized_host[0], 8)
        shared_gate_f32 = (
            np.left_shift(shared_gate_bf16.astype(np.uint32), 16)
            .view(np.float32)
            .reshape(-1)
        )
        shared_weight = float(
            1.0
            / (
                1.0
                + np.exp(-float((shared_gate_f32 @ normalized_host[0]).item()))
            )
        )
        selected_hosts = load_experts(args.model, args.layer, expected_ids)
        moe_reference = np.zeros_like(x)
        for weight, expert in zip(expected_weights, selected_hosts, strict=True):
            moe_reference += np.float32(weight) * expert_reference(
                expert, normalized_host
            )
        moe_reference += np.float32(shared_weight) * expert_reference(
            shared_host, normalized_host
        )
        reference = attention_host + moe_reference
        del selected_hosts
        gc.collect()

        def execute() -> tuple[float, float]:
            attention.reset()
            started = time.perf_counter_ns()
            attention.enqueue(input_buffer, cos_buffer, sin_buffer, attention_output)
            runtime.rmsnorm_device(
                attention_output,
                post_norm_buffer,
                1,
                2048,
                1e-6,
                normalized,
            )
            bank.decode_device(normalized, moe_output)
            runtime.add_device(attention_output, moe_output, 2048, final_output)
            profile = runtime.synchronize()
            return profile.kernel_ns / 1e6, (time.perf_counter_ns() - started) / 1e6

        execute()
        actual = final_output.download(x.shape)
        maximum_error = float(np.max(np.abs(reference - actual)))
        if not np.allclose(reference, actual, rtol=2e-4, atol=1e-4):
            raise SystemExit(f"full MoE layer mismatch: max_abs={maximum_error:.9g}")
        for _ in range(args.warmups):
            execute()
        samples = [execute() for _ in range(args.samples)]
        kernel_ms = [sample[0] for sample in samples]
        wall_ms = [sample[1] for sample in samples]
        record = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hardware": {"system_model": system_model(), "gpu": runtime.device_name},
            "environment": {**power_status(), "thermal_regime": "warm-burst"},
            "workload": {
                "operation": "ornith_35b_complete_full_attention_moe_layer",
                "layer": args.layer,
                "kv_dtype": args.kv_dtype,
                "resident_bank_payload_bytes": payload_bytes,
                "kv_pool_storage_bytes": pool.storage_bytes,
                "selected_experts": expected_ids,
                "shared_expert_gate": shared_weight,
            },
            "loading": {
                "bank_upload_seconds": bank_load_seconds,
                "available_before_bank_bytes": available_before_bank,
                "available_after_bank_bytes": available_after_bank,
            },
            "timing": {
                "warmups": args.warmups,
                "samples": args.samples,
                "kernel_ms": describe(kernel_ms),
                "wall_ms": describe(wall_ms),
            },
            "correctness": {
                "passed": True,
                "maximum_absolute_error": maximum_error,
                "expected_router_ids": expected_ids,
                "finite_outputs": bool(np.isfinite(actual).all()),
                "explicit_completion_marker": True,
            },
            "limitations": [
                "single-token one-layer decode with one resident expert bank",
                "linear-attention layers and the complete 40-layer graph are excluded",
                "full-model residency remains gated beyond 19 expert banks",
            ],
            "samples": {"kernel_ms": kernel_ms, "wall_ms": wall_ms},
        }
        args.results.mkdir(parents=True, exist_ok=True)
        result_path = args.results / (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-"
            f"moe-full-layer-{args.kv_dtype}.json"
        )
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"device={runtime.device_name} layer={args.layer} "
            f"kv_dtype={args.kv_dtype} experts={expected_ids}"
        )
        print(
            f"resident_bank_bytes={payload_bytes} kv_bytes={pool.storage_bytes} "
            f"bank_load_s={bank_load_seconds:.3f}"
        )
        print(
            f"kernel_ms={record['timing']['kernel_ms']['median']:.6f} "
            f"wall_ms={record['timing']['wall_ms']['median']:.6f} "
            f"max_abs={maximum_error:.9g} result={result_path}"
        )
        print("MOE_FULL_LAYER_PASS")
        return 0
    finally:
        final_output.close()
        moe_output.close()
        normalized.close()
        attention_output.close()
        sin_buffer.close()
        cos_buffer.close()
        input_buffer.close()
        bank.close()
        post_norm_buffer.close()
        attention.close()
        pool.close()
        for matrix in reversed(matrices):
            matrix.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
