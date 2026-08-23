#!/usr/bin/env python3
"""Validate and benchmark two-request paged resident cadence scheduling."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from bench_islands import power_status, system_model


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors",
    )
    parser.add_argument("--tokens", type=int, default=18)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--kv-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--results",
        type=Path,
        default=here.parent / "campaign_results/bandwidth-first",
    )
    args = parser.parse_args()
    if args.tokens < 17 or args.tokens > 32:
        parser.error("tokens must be between 17 and 32 to exercise two pages")
    if args.requests < 1 or args.requests > 4:
        parser.error("requests must be between 1 and 4")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        here / "runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        here / "kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(here.parent / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths
    from vllm_nvfp4_opencl.serving import Qwen35CadenceWeights

    runtime = Runtime(*runtime_paths())
    weights = Qwen35CadenceWeights.load(runtime, args.model)
    request_ids = tuple(chr(ord("a") + index) for index in range(args.requests))
    max_pages = args.requests * 2
    oracle = {
        request_id: weights.create_session(args.tokens)
        for request_id in request_ids
    }
    scheduler = weights.create_paged_scheduler(
        max_pages=max_pages,
        max_batch_size=args.requests,
        kv_dtype=args.kv_dtype,
    )
    for request_id in request_ids:
        scheduler.add_request(request_id, args.tokens)
    rng = np.random.default_rng(20260822)
    hidden = {
        request_id: [
            np.ascontiguousarray(
                rng.standard_normal((1, 5120)).astype(np.float32)
                * np.float32(0.2)
            )
            for _ in range(args.tokens)
        ]
        for request_id in request_ids
    }
    maximum_error = 0.0
    maximum_reference = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    absolute_error = 0.0
    compared_elements = 0
    kernel_ns = 0
    started = time.perf_counter()
    try:
        for position in range(args.tokens):
            expected = {
                request_id: oracle[request_id].step(hidden[request_id][position])[0]
                for request_id in request_ids
            }
            actual, profile = scheduler.decode_batch(
                {
                    request_id: hidden[request_id][position]
                    for request_id in request_ids
                }
            )
            kernel_ns += profile.kernel_ns
            for request_id in request_ids:
                error = float(np.max(np.abs(expected[request_id] - actual[request_id])))
                maximum_error = max(maximum_error, error)
                delta = actual[request_id].astype(np.float64) - expected[
                    request_id
                ].astype(np.float64)
                reference64 = expected[request_id].astype(np.float64)
                maximum_reference = max(
                    maximum_reference, float(np.max(np.abs(reference64)))
                )
                squared_error += float(np.sum(delta * delta))
                squared_reference += float(np.sum(reference64 * reference64))
                absolute_error += float(np.sum(np.abs(delta)))
                compared_elements += delta.size
                rtol, atol = (
                    (2e-4, 1e-4) if args.kv_dtype == "fp32" else (5e-3, 2e-3)
                )
                if not np.allclose(expected[request_id], actual[request_id], rtol=rtol, atol=atol):
                    raise SystemExit(
                        f"request={request_id} token={position} mismatch: {error:.9g}"
                    )
        wall_ms = (time.perf_counter() - started) * 1e3 / args.tokens
        if scheduler.free_pages != 0:
            raise SystemExit(
                f"expected full four-page pool, found {scheduler.free_pages} free"
            )
        for request_id in request_ids:
            scheduler.reset_request(request_id)
        scheduled_kernel_ns = 0
        scheduled_started = time.perf_counter()
        for position in range(args.tokens):
            _outputs, profile = scheduler.decode_batch(
                {
                    request_id: hidden[request_id][position]
                    for request_id in request_ids
                }
            )
            scheduled_kernel_ns += profile.kernel_ns
        scheduled_wall_ms = (
            (time.perf_counter() - scheduled_started) * 1e3 / args.tokens
        )
        scheduler.remove_request(request_ids[0])
        if scheduler.free_pages != 2:
            raise SystemExit("request removal did not reclaim two pages")
        replacement = "replacement"
        scheduler.add_request(replacement, args.tokens)
        output, _ = scheduler.decode_batch(
            {replacement: hidden[request_ids[0]][0]}
        )
        if not np.isfinite(output[replacement]).all() or scheduler.free_pages != 1:
            raise SystemExit("reclaimed page was not reusable by a new request")
        for request_id in request_ids[1:]:
            scheduler.remove_request(request_id)
        scheduler.remove_request(replacement)
        if scheduler.free_pages != max_pages:
            raise SystemExit("pool did not fully recover after request removal")

        relative_rmse = (squared_error / squared_reference) ** 0.5
        mean_absolute_error = absolute_error / compared_elements
        record = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hardware": {"system_model": system_model(), "gpu": runtime.device_name},
            "environment": {**power_status(), "thermal_regime": "warm-burst"},
            "workload": {
                "operation": "qwen35_four_layer_paged_decode",
                "kv_dtype": args.kv_dtype,
                "requests": args.requests,
                "tokens_per_request": args.tokens,
                "page_tokens": 16,
                "pages": max_pages,
                "pool_storage_bytes_one_full_attention_layer": scheduler.pool.storage_bytes,
                "projected_pool_bytes_16_full_attention_layers": scheduler.pool.storage_bytes * 16,
            },
            "timing": {
                "scheduler_only_kernel_ms_per_step": scheduled_kernel_ns
                / args.tokens
                / 1e6,
                "scheduler_only_wall_ms_per_step": scheduled_wall_ms,
                "wall_with_fp32_oracle_ms_per_step": wall_ms,
            },
            "correctness": {
                "passed": True,
                "oracle_kv_dtype": "fp32",
                "maximum_absolute_error": maximum_error,
                "maximum_reference_magnitude": maximum_reference,
                "mean_absolute_error": mean_absolute_error,
                "relative_rmse": relative_rmse,
                "finite_outputs": True,
                "explicit_completion_marker": True,
            },
            "limitations": [
                "four-layer cadence contains one of the dense model's sixteen full-attention layers",
                "18 synthetic decode tokens are a numerical and page-boundary gate, not a long-context quality evaluation",
                "full-model pool bytes are an exact shape projection, not a successful full-model allocation",
            ],
        }
        args.results.mkdir(parents=True, exist_ok=True)
        result_path = args.results / (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-"
            f"paged-{args.kv_dtype}-batch{args.requests}.json"
        )
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"device={runtime.device_name} kv_dtype={args.kv_dtype} "
            f"pool_storage_bytes={scheduler.pool.storage_bytes} requests={args.requests} "
            f"tokens_each={args.tokens} pages={max_pages} "
            f"max_abs={maximum_error:.9g}"
        )
        print(
            f"batch_kernel_ms={kernel_ns / args.tokens / 1e6:.3f} "
            f"batch_wall_with_oracle_ms={wall_ms:.3f}"
        )
        print(
            f"scheduler_only_kernel_ms="
            f"{scheduled_kernel_ns / args.tokens / 1e6:.3f} "
            f"scheduler_only_wall_ms={scheduled_wall_ms:.3f}"
        )
        print(
            f"mean_abs={mean_absolute_error:.9g} relative_rmse={relative_rmse:.9g} "
            f"result={result_path}"
        )
        print("PASS: paged cadence scheduler matched contiguous request oracles")
        return 0
    finally:
        scheduler.close()
        for session in oracle.values():
            session.close()
        weights.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
