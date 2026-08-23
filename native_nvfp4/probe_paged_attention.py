#!/usr/bin/env python3
"""Validate interleaved Qwen3.5 paged attention and page reclamation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from bench_islands import power_status, system_model

from probe_full_attention import apply_rope, rmsnorm


def round_to_bf16(array: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return ((rounded >> 16) << 16).view(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--profile", choices=("dense", "moe"), default="dense")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "campaign_results/bandwidth-first",
    )
    args = parser.parse_args()
    query_heads, kv_heads = (24, 4) if args.profile == "dense" else (16, 2)
    here = Path(__file__).resolve().parent
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
    runtime = Runtime(*runtime_paths())
    pool = runtime.create_paged_attention_pool(
        4,
        kv_dtype=args.kv_dtype,
        query_heads=query_heads,
        kv_heads=kv_heads,
    )
    element_bytes = 2 if args.kv_dtype == "bf16" else 4
    expected_storage = 4 * 16 * kv_heads * 256 * element_bytes * 2
    if pool.storage_bytes != expected_storage:
        raise SystemExit(
            f"pool storage mismatch: {pool.storage_bytes} != {expected_storage}"
        )
    states = [runtime.create_paged_full_attention_state(pool, 32) for _ in range(2)]
    q_weight = np.ascontiguousarray(rng.normal(0, 0.1, 256).astype(np.float32))
    k_weight = np.ascontiguousarray(rng.normal(0, 0.1, 256).astype(np.float32))
    buffers = []

    def upload(array: np.ndarray):
        result = runtime.upload_buffer(array)
        buffers.append(result)
        return result

    def create(elements: int):
        result = runtime.create_buffer(elements * 4)
        buffers.append(result)
        return result

    q_weight_buffer = upload(q_weight)
    k_weight_buffer = upload(k_weight)
    q_buffer = create(query_heads * 512)
    k_buffer = create(kv_heads * 256)
    v_buffer = create(kv_heads * 256)
    cos_buffer = create(64)
    sin_buffer = create(64)
    output_buffer = create(query_heads * 256)
    references = [([], []), ([], [])]
    full_references = [([], []), ([], [])]
    maximum_error = 0.0
    maximum_fp32_delta = 0.0
    squared_fp32_delta = 0.0
    squared_fp32_reference = 0.0
    kernel_ms: list[float] = []
    wall_ms: list[float] = []

    try:
        for position in range(18):
            for request in range(2):
                q_projected = np.ascontiguousarray(
                    rng.normal(0, 0.2, (query_heads, 512)).astype(np.float32)
                )
                k_projected = np.ascontiguousarray(
                    rng.normal(0, 0.2, (kv_heads, 256)).astype(np.float32)
                )
                v_projected = np.ascontiguousarray(
                    rng.normal(0, 0.2, (kv_heads, 256)).astype(np.float32)
                )
                angles = np.arange(64, dtype=np.float32) * np.float32(0.013)
                angles += np.float32(position * 0.071)
                cos = np.ascontiguousarray(np.cos(angles).astype(np.float32))
                sin = np.ascontiguousarray(np.sin(angles).astype(np.float32))
                q_buffer.upload(q_projected)
                k_buffer.upload(k_projected)
                v_buffer.upload(v_projected)
                cos_buffer.upload(cos)
                sin_buffer.upload(sin)

                q = apply_rope(
                    rmsnorm(q_projected[:, :256], q_weight), cos, sin
                )
                gate = q_projected[:, 256:]
                key = apply_rope(rmsnorm(k_projected, k_weight), cos, sin)
                full_references[request][0].append(key)
                full_references[request][1].append(v_projected)
                if args.kv_dtype == "bf16":
                    key = round_to_bf16(key)
                    v_projected = round_to_bf16(v_projected)
                references[request][0].append(key)
                references[request][1].append(v_projected)
                cached_k = np.stack(references[request][0])
                cached_v = np.stack(references[request][1])
                expected = np.empty((query_heads, 256), dtype=np.float32)
                for head in range(query_heads):
                    kv_head = head // (query_heads // kv_heads)
                    logits = cached_k[:, kv_head] @ q[head] * np.float32(0.0625)
                    probability = np.exp(logits - np.max(logits))
                    probability /= np.sum(probability)
                    expected[head] = (probability @ cached_v[:, kv_head]) / (
                        1.0 + np.exp(-gate[head])
                    )

                full_k = np.stack(full_references[request][0])
                full_v = np.stack(full_references[request][1])
                expected_fp32 = np.empty((query_heads, 256), dtype=np.float32)
                for head in range(query_heads):
                    kv_head = head // (query_heads // kv_heads)
                    logits = full_k[:, kv_head] @ q[head] * np.float32(0.0625)
                    probability = np.exp(logits - np.max(logits))
                    probability /= np.sum(probability)
                    expected_fp32[head] = (
                        probability @ full_v[:, kv_head]
                    ) / (1.0 + np.exp(-gate[head]))

                call_started = time.perf_counter()
                runtime.paged_full_attention_decode_device(
                    states[request], q_buffer, k_buffer, v_buffer,
                    q_weight_buffer, k_weight_buffer, cos_buffer, sin_buffer,
                    1e-6, output_buffer,
                )
                runtime_profile = runtime.synchronize()
                actual = output_buffer.download((query_heads, 256))
                wall_ms.append((time.perf_counter() - call_started) * 1e3)
                kernel_ms.append(runtime_profile.kernel_ns / 1e6)
                error = float(np.max(np.abs(expected - actual)))
                maximum_error = max(maximum_error, error)
                if not np.allclose(expected, actual, rtol=2e-4, atol=8e-5):
                    raise SystemExit(
                        f"request={request} token={position} mismatch: {error:.9g}"
                    )
                fp32_delta = actual.astype(np.float64) - expected_fp32.astype(
                    np.float64
                )
                fp32_reference = expected_fp32.astype(np.float64)
                maximum_fp32_delta = max(
                    maximum_fp32_delta, float(np.max(np.abs(fp32_delta)))
                )
                squared_fp32_delta += float(np.sum(fp32_delta * fp32_delta))
                squared_fp32_reference += float(
                    np.sum(fp32_reference * fp32_reference)
                )

        if [state.pages for state in states] != [2, 2] or pool.free_pages != 0:
            raise SystemExit("unexpected page allocation after crossing block boundary")
        runtime.reset_paged_full_attention_state(states[0])
        if states[0].tokens != 0 or states[0].pages != 0 or pool.free_pages != 2:
            raise SystemExit("reset did not return request pages to the pool")
        runtime.reset_paged_full_attention_state(states[1])
        if pool.free_pages != 4:
            raise SystemExit("page pool did not fully recover")
        relative_rmse = (squared_fp32_delta / squared_fp32_reference) ** 0.5
        full_attention_layers = 16 if args.profile == "dense" else 10
        record = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hardware": {"system_model": system_model(), "gpu": runtime.device_name},
            "environment": {**power_status(), "thermal_regime": "warm-burst"},
            "workload": {
                "operation": "qwen35_paged_attention_shape_gate",
                "profile": args.profile,
                "query_heads": query_heads,
                "kv_heads": kv_heads,
                "head_dim": 256,
                "kv_dtype": args.kv_dtype,
                "requests": 2,
                "tokens_per_request": 18,
                "pages": 4,
                "pool_storage_bytes": pool.storage_bytes,
                "projected_pool_bytes_all_full_attention_layers": (
                    pool.storage_bytes * full_attention_layers
                ),
            },
            "timing": {
                "samples": len(kernel_ms),
                "kernel_ms_median": statistics.median(kernel_ms),
                "kernel_ms_min": min(kernel_ms),
                "kernel_ms_max": max(kernel_ms),
                "wall_ms_median": statistics.median(wall_ms),
            },
            "correctness": {
                "passed": True,
                "maximum_absolute_error_vs_storage_oracle": maximum_error,
                "maximum_absolute_delta_vs_fp32_cache": maximum_fp32_delta,
                "relative_rmse_vs_fp32_cache": relative_rmse,
                "finite_outputs": True,
                "explicit_completion_marker": True,
            },
            "limitations": [
                "synthetic 18-token decode crosses one page boundary",
                "timing samples span token positions 1 through 18 rather than a fixed context",
                "this is attention state/softmax only, excluding checkpoint projections",
            ],
        }
        args.results.mkdir(parents=True, exist_ok=True)
        result_path = args.results / (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-paged-attention-"
            f"{args.profile}-{args.kv_dtype}.json"
        )
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"device={runtime.device_name} profile={args.profile} "
            f"heads={query_heads}/{kv_heads} kv_dtype={args.kv_dtype} "
            f"storage_bytes={pool.storage_bytes} requests=2 tokens_each=18 "
            f"pages=4 page_tokens=16 max_abs={maximum_error:.9g}"
        )
        print(
            f"kernel_ms_median={statistics.median(kernel_ms):.6f} "
            f"wall_ms_median={statistics.median(wall_ms):.6f} "
            f"fp32_delta={maximum_fp32_delta:.9g} "
            f"relative_rmse={relative_rmse:.9g} result={result_path}"
        )
        print("PASS: interleaved block tables, KV dtype, and reclamation are exact")
        return 0
    finally:
        for state in reversed(states):
            state.close()
        for buffer in reversed(buffers):
            buffer.close()
        pool.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
