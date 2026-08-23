#!/usr/bin/env python3
"""Validate one exact Ornith 35B MoE full-attention layer with paged KV."""

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
import torch
from bench_islands import percentile, power_status, system_model
from bench_resident_full_attention import apply_rope, rmsnorm, rope
from probe_paged_attention import round_to_bf16
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


def load_layer(
    model_dir: Path, layer: int
) -> tuple[list[tuple[np.ndarray, float]], np.ndarray, np.ndarray, np.ndarray]:
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    prefix = f"model.language_model.layers.{layer}"
    attention = prefix + ".self_attn"
    matrix_bases = [
        attention + "." + name
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    keys = [
        key
        for base in matrix_bases
        for key in (base + ".weight", base + ".weight_scale")
    ] + [
        prefix + ".input_layernorm.weight",
        attention + ".q_norm.weight",
        attention + ".k_norm.weight",
    ]
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index[key], []).append(key)
    tensors: dict[str, torch.Tensor] = {}
    for shard_name, shard_keys in by_shard.items():
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as shard:
            for key in shard_keys:
                tensors[key] = shard.get_tensor(key)

    matrices = []
    for base in matrix_bases:
        matrices.append(
            (
                np.ascontiguousarray(
                    tensors[base + ".weight"].view(torch.uint8).numpy()
                ),
                float(tensors[base + ".weight_scale"].item()),
            )
        )

    def f32(key: str) -> np.ndarray:
        return np.ascontiguousarray(tensors[key].float().numpy())

    return (
        matrices,
        f32(prefix + ".input_layernorm.weight"),
        f32(attention + ".q_norm.weight"),
        f32(attention + ".k_norm.weight"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=18)
    parser.add_argument("--kv-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.layer < 0 or args.tokens < 17:
        parser.error("use a nonnegative full-attention layer and at least 17 tokens")

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

    matrix_hosts, input_norm, q_norm, k_norm = load_layer(args.model, args.layer)
    runtime = Runtime(*runtime_paths())
    matrices = [runtime.upload_fp8_tensor_scaled(*host) for host in matrix_hosts]
    pool = runtime.create_paged_attention_pool(
        (args.tokens + 15) // 16,
        kv_dtype=args.kv_dtype,
        query_heads=16,
        kv_heads=2,
    )
    graph = ResidentQwen35FullAttention(
        runtime,
        *matrices,
        input_norm_weight=input_norm,
        q_norm_weight=q_norm,
        k_norm_weight=k_norm,
        max_tokens=args.tokens,
        attention_pool=pool,
        hidden=2048,
        query_heads=16,
        kv_heads=2,
    )
    rng = np.random.default_rng(20260822)
    inputs = [
        np.ascontiguousarray(
            rng.standard_normal((1, 2048)).astype(np.float32) * np.float32(0.2)
        )
        for _ in range(args.tokens)
    ]
    input_buffers = [runtime.upload_buffer(value) for value in inputs]
    output = runtime.create_buffer(inputs[0].nbytes)
    rope_buffers = [
        tuple(runtime.upload_buffer(value) for value in rope(position))
        for position in range(args.tokens)
    ]
    k_cache: list[np.ndarray] = []
    v_cache: list[np.ndarray] = []
    full_k_cache: list[np.ndarray] = []
    full_v_cache: list[np.ndarray] = []
    kernel_ms: list[float] = []
    wall_ms: list[float] = []
    maximum_storage_error = 0.0
    maximum_fp32_delta = 0.0
    squared_fp32_delta = 0.0
    squared_fp32_reference = 0.0

    try:
        for position, x in enumerate(inputs):
            normalized = np.ascontiguousarray(rmsnorm(x, input_norm))
            q_projected = runtime.linear_fp8(matrices[0], normalized).reshape(16, 512)
            k_projected = runtime.linear_fp8(matrices[1], normalized).reshape(2, 256)
            v_projected = runtime.linear_fp8(matrices[2], normalized).reshape(2, 256)
            cos, sin = rope(position)
            q = apply_rope(rmsnorm(q_projected[:, :256], q_norm), cos, sin)
            gate = q_projected[:, 256:]
            key = apply_rope(rmsnorm(k_projected, k_norm), cos, sin)
            full_k_cache.append(key)
            full_v_cache.append(v_projected)
            if args.kv_dtype == "bf16":
                key = round_to_bf16(key)
                v_projected = round_to_bf16(v_projected)
            k_cache.append(key)
            v_cache.append(v_projected)

            def attend(keys: list[np.ndarray], values: list[np.ndarray]) -> np.ndarray:
                cached_k = np.stack(keys)
                cached_v = np.stack(values)
                attended = np.empty((16, 256), dtype=np.float32)
                for head in range(16):
                    kv_head = head // 8
                    logits = cached_k[:, kv_head] @ q[head] * np.float32(0.0625)
                    probability = np.exp(logits - np.max(logits))
                    probability /= np.sum(probability)
                    attended[head] = (probability @ cached_v[:, kv_head]) / (
                        1.0 + np.exp(-gate[head])
                    )
                return attended

            expected = x + runtime.linear_fp8(
                matrices[3], np.ascontiguousarray(attend(k_cache, v_cache).reshape(1, -1))
            )
            fp32_reference = x + runtime.linear_fp8(
                matrices[3],
                np.ascontiguousarray(attend(full_k_cache, full_v_cache).reshape(1, -1)),
            )

            started = time.perf_counter()
            graph.enqueue(input_buffers[position], *rope_buffers[position], output)
            profile = runtime.synchronize()
            actual = output.download(x.shape)
            wall_ms.append((time.perf_counter() - started) * 1e3)
            kernel_ms.append(profile.kernel_ns / 1e6)
            storage_error = float(np.max(np.abs(expected - actual)))
            maximum_storage_error = max(maximum_storage_error, storage_error)
            if not np.allclose(expected, actual, rtol=2e-4, atol=1e-4):
                raise SystemExit(
                    f"token {position} storage-oracle mismatch: {storage_error:.9g}"
                )
            delta = actual.astype(np.float64) - fp32_reference.astype(np.float64)
            reference64 = fp32_reference.astype(np.float64)
            maximum_fp32_delta = max(
                maximum_fp32_delta, float(np.max(np.abs(delta)))
            )
            squared_fp32_delta += float(np.sum(delta * delta))
            squared_fp32_reference += float(np.sum(reference64 * reference64))

        relative_rmse = (squared_fp32_delta / squared_fp32_reference) ** 0.5
        record = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hardware": {"system_model": system_model(), "gpu": runtime.device_name},
            "environment": {**power_status(), "thermal_regime": "warm-burst"},
            "workload": {
                "operation": "ornith_35b_exact_full_attention_layer",
                "layer": args.layer,
                "kv_dtype": args.kv_dtype,
                "tokens": args.tokens,
                "hidden": 2048,
                "query_heads": 16,
                "kv_heads": 2,
                "pool_storage_bytes": pool.storage_bytes,
            },
            "timing": {
                "kernel_ms": describe(kernel_ms),
                "wall_ms": describe(wall_ms),
            },
            "correctness": {
                "passed": True,
                "maximum_absolute_error_vs_storage_oracle": maximum_storage_error,
                "maximum_absolute_delta_vs_fp32_cache": maximum_fp32_delta,
                "relative_rmse_vs_fp32_cache": relative_rmse,
                "finite_outputs": True,
                "explicit_completion_marker": True,
            },
            "limitations": [
                "one exact full-attention layer, excluding MoE and layer-output norm",
                "synthetic 18-token decode crosses one page boundary",
                "timing spans context positions 1 through 18",
            ],
        }
        args.results.mkdir(parents=True, exist_ok=True)
        result_path = args.results / (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-"
            f"moe-full-attention-{args.kv_dtype}.json"
        )
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"device={runtime.device_name} layer={args.layer} kv_dtype={args.kv_dtype} "
            f"storage_bytes={pool.storage_bytes} tokens={args.tokens}"
        )
        print(
            f"kernel_ms_median={record['timing']['kernel_ms']['median']:.6f} "
            f"wall_ms_median={record['timing']['wall_ms']['median']:.6f} "
            f"storage_max_abs={maximum_storage_error:.9g} "
            f"fp32_delta={maximum_fp32_delta:.9g} relative_rmse={relative_rmse:.9g}"
        )
        print(f"result={result_path}")
        print("MOE_FULL_ATTENTION_PASS")
        return 0
    finally:
        for cos_buffer, sin_buffer in reversed(rope_buffers):
            sin_buffer.close()
            cos_buffer.close()
        output.close()
        for buffer in reversed(input_buffers):
            buffer.close()
        graph.close()
        pool.close()
        for matrix in reversed(matrices):
            matrix.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
