#!/usr/bin/env python3
"""Validate Ornith final RMSNorm and full checkpoint-native NVFP4 LM head."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from bench_islands import memory_status, percentile, power_status, system_model
from bench_moe_routed_layer import MODEL, RESULTS
from bench_resident_full_attention import rmsnorm
from probe_native_nvfp4 import cpu_gemv
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


def load_head(
    model: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    index = json.loads(
        (model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    names = (
        "model.language_model.norm.weight",
        "lm_head.weight",
        "lm_head.weight_scale",
        "lm_head.weight_scale_2",
    )
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(index[name], []).append(name)
    tensors: dict[str, torch.Tensor] = {}
    for shard_name, shard_names in by_shard.items():
        with safe_open(model / shard_name, framework="pt", device="cpu") as shard:
            for name in shard_names:
                tensors[name] = shard.get_tensor(name)
    norm = np.ascontiguousarray(
        tensors["model.language_model.norm.weight"].float().numpy()
    )
    packed = np.ascontiguousarray(
        tensors["lm_head.weight"].numpy().astype(np.uint8, copy=False)
    )
    scales = np.ascontiguousarray(
        tensors["lm_head.weight_scale"].view(torch.uint8).numpy()
    )
    multiplier = float(tensors["lm_head.weight_scale_2"].item())
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(f"invalid lm_head.weight_scale_2: {multiplier}")
    return norm, packed, scales, 1.0 / multiplier


def reference_logits(
    packed: np.ndarray,
    scales: np.ndarray,
    hidden: np.ndarray,
    global_divisor: float,
    chunk_rows: int,
) -> np.ndarray:
    logits = np.empty(packed.shape[0], dtype=np.float32)
    for start in range(0, packed.shape[0], chunk_rows):
        stop = min(start + chunk_rows, packed.shape[0])
        logits[start:stop] = cpu_gemv(
            packed[start:stop],
            scales[start:stop],
            hidden,
            global_divisor,
        )
    return logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--oracle-chunk-rows", type=int, default=2048)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.warmups < 0 or args.samples <= 0 or args.oracle_chunk_rows <= 0:
        parser.error("warmups, samples, and oracle chunk rows must be valid")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(ROOT / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.graph import ResidentNvFp4LmHead
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    norm, packed, scales, divisor = load_head(args.model)
    hidden = np.ascontiguousarray(
        np.random.default_rng(20260822)
        .standard_normal((1, norm.size))
        .astype(np.float32)
        * np.float32(0.2)
    )
    normalized = np.ascontiguousarray(rmsnorm(hidden, norm)[0])
    oracle_started = time.perf_counter()
    reference = reference_logits(
        packed, scales, normalized, divisor, args.oracle_chunk_rows
    )
    oracle_seconds = time.perf_counter() - oracle_started

    available_before = memory_status().available_physical
    runtime = Runtime(*runtime_paths())
    matrix = runtime.upload(packed, scales, divisor)
    head = ResidentNvFp4LmHead(runtime, norm, matrix)

    def execute() -> tuple[np.ndarray, float, float]:
        started = time.perf_counter_ns()
        logits, profile = head.execute(hidden)
        return (
            logits[0],
            profile.kernel_ns / 1e6,
            (time.perf_counter_ns() - started) / 1e6,
        )

    try:
        output, _kernel, _wall = execute()
        maximum_absolute_error = float(np.max(np.abs(output - reference)))
        if not np.allclose(output, reference, rtol=1e-4, atol=1e-4):
            raise SystemExit(
                f"LM-head oracle mismatch: max_abs={maximum_absolute_error}"
            )
        expected_token = int(np.argmax(reference))
        actual_token = int(np.argmax(output))
        if actual_token != expected_token:
            raise SystemExit(
                f"LM-head argmax mismatch: expected={expected_token} actual={actual_token}"
            )
        for _ in range(args.warmups):
            execute()
        samples = [execute() for _ in range(args.samples)]
        kernel = describe([sample[1] for sample in samples])
        wall = describe([sample[2] for sample in samples])
        available_after = memory_status().available_physical
        record = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hardware": {
                "system_model": system_model(),
                "gpu": runtime.device_name,
            },
            "software": {"command": sys.argv},
            "environment": {**power_status(), "thermal_regime": "warm-burst"},
            "workload": {
                "operation": "ornith_final_norm_nvfp4_lm_head",
                "hidden_size": norm.size,
                "vocabulary_size": packed.shape[0],
                "packed_shape": list(packed.shape),
                "scale_shape": list(scales.shape),
                "native_payload_bytes": packed.nbytes + scales.nbytes,
                "checkpoint_global_divisor": divisor,
            },
            "timing": {
                "warmups": args.warmups,
                "samples": args.samples,
                "kernel_ms": kernel,
                "wall_ms": wall,
                "independent_cpu_oracle_seconds": oracle_seconds,
            },
            "memory": {
                "available_before_bytes": available_before,
                "available_after_bytes": available_after,
            },
            "correctness": {
                "passed": True,
                "maximum_absolute_error": maximum_absolute_error,
                "expected_argmax_token": expected_token,
                "actual_argmax_token": actual_token,
                "finite_outputs": bool(np.isfinite(output).all()),
                "explicit_completion_marker": True,
            },
            "limitations": [
                "synthetic hidden state rather than a complete 40-layer output",
                "greedy argmax is performed on the downloaded logits",
            ],
        }
        args.results.mkdir(parents=True, exist_ok=True)
        path = args.results / (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}-"
            "moe-lm-head.json"
        )
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"vocab={packed.shape[0]} payload_bytes={packed.nbytes + scales.nbytes} "
            f"kernel_ms={kernel['median']:.6f} wall_ms={wall['median']:.6f} "
            f"max_abs={maximum_absolute_error:.8g} argmax={actual_token} result={path}"
        )
        print("MOE_LM_HEAD_PASS")
    finally:
        head.close()
        matrix.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
