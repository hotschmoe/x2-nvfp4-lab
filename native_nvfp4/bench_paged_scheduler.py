#!/usr/bin/env python3
"""Validate and benchmark two-request paged resident cadence scheduling."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


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
        max_pages=max_pages, max_batch_size=args.requests
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
                if not np.allclose(
                    expected[request_id], actual[request_id], rtol=2e-4, atol=1e-4
                ):
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

        print(
            f"device={runtime.device_name} requests={args.requests} "
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
