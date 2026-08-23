#!/usr/bin/env python3
"""Validate an exact Ornith linear-attention layer plus resident MoE bank."""

from __future__ import annotations

import argparse
import gc
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
from bench_moe_device_bank import stream_experts_into_bank
from bench_moe_experts import load_experts
from bench_moe_routed_layer import (
    MODEL,
    RESULTS,
    expert_reference,
    load_layer_tensors,
    route,
)
from probe_causal_conv import cpu_causal_conv
from probe_gated_delta import cpu_gated_delta
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


def load_linear_layer(model_dir: Path, layer: int) -> dict[str, object]:
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    prefix = f"model.language_model.layers.{layer}"
    linear = prefix + ".linear_attn"
    matrix_bases = [
        linear + "." + name for name in ("in_proj_qkv", "in_proj_z", "out_proj")
    ]
    keys = [
        key
        for base in matrix_bases
        for key in (base + ".weight", base + ".weight_scale")
    ] + [
        prefix + ".input_layernorm.weight",
        prefix + ".post_attention_layernorm.weight",
        linear + ".in_proj_a.weight",
        linear + ".in_proj_b.weight",
        linear + ".A_log",
        linear + ".dt_bias",
        linear + ".conv1d.weight",
        linear + ".norm.weight",
    ]
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index[key], []).append(key)
    tensors: dict[str, torch.Tensor] = {}
    for shard_name, shard_keys in by_shard.items():
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as shard:
            for key in shard_keys:
                tensors[key] = shard.get_tensor(key)

    def f32(key: str) -> np.ndarray:
        return np.ascontiguousarray(tensors[key].float().numpy())

    matrices = [
        (
            np.ascontiguousarray(
                tensors[base + ".weight"].view(torch.uint8).numpy()
            ),
            float(tensors[base + ".weight_scale"].item()),
        )
        for base in matrix_bases
    ]
    return {
        "matrices": matrices,
        "input_norm": f32(prefix + ".input_layernorm.weight"),
        "post_norm": f32(prefix + ".post_attention_layernorm.weight"),
        "a_weight": f32(linear + ".in_proj_a.weight"),
        "b_weight": f32(linear + ".in_proj_b.weight"),
        "a_log": f32(linear + ".A_log"),
        "dt_bias": f32(linear + ".dt_bias"),
        "conv_weight": np.ascontiguousarray(
            f32(linear + ".conv1d.weight").reshape(8192, 4)
        ),
        "gated_norm": f32(linear + ".norm.weight"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--layer", type=int, default=0)
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
    from vllm_nvfp4_opencl.graph import ResidentQwen35LinearAttention
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    layer = load_linear_layer(args.model, args.layer)
    router_bf16, router_f32, shared_gate_bf16, shared_host = load_layer_tensors(
        args.model, args.layer
    )
    rng = np.random.default_rng(20260822)
    x = np.ascontiguousarray(
        rng.standard_normal((1, 2048)).astype(np.float32) * np.float32(0.2)
    )
    initial_recurrent = np.ascontiguousarray(
        rng.standard_normal((32, 128, 128)).astype(np.float32) * np.float32(0.01)
    )
    initial_conv = np.ascontiguousarray(
        rng.standard_normal((8192, 4)).astype(np.float32) * np.float32(0.05)
    )
    epsilon = 1e-6

    runtime = Runtime(*runtime_paths())
    matrices = [
        runtime.upload_fp8_tensor_scaled(*host) for host in layer["matrices"]
    ]
    qkv_matrix, z_matrix, out_matrix = matrices
    attention = ResidentQwen35LinearAttention(
        runtime,
        qkv_matrix,
        z_matrix,
        out_matrix,
        input_norm_weight=layer["input_norm"],
        a_weight=layer["a_weight"],
        b_weight=layer["b_weight"],
        a_log=layer["a_log"],
        dt_bias=layer["dt_bias"],
        conv_weight=layer["conv_weight"],
        gated_norm_weight=layer["gated_norm"],
        recurrent_state=initial_recurrent,
        conv_state=initial_conv,
        epsilon=epsilon,
        hidden=2048,
        key_heads=16,
        value_heads=32,
    )
    post_norm_buffer = runtime.upload_buffer(
        np.ascontiguousarray(layer["post_norm"] + np.float32(1.0))
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

    input_buffer = runtime.upload_buffer(x)
    attention_output = runtime.create_buffer(x.nbytes)
    normalized_buffer = runtime.create_buffer(x.nbytes)
    moe_output = runtime.create_buffer(x.nbytes)
    final_output = runtime.create_buffer(x.nbytes)

    try:
        inverse_rms = 1.0 / np.sqrt(
            np.mean(x * x, axis=-1, keepdims=True) + epsilon
        )
        normalized = np.ascontiguousarray(
            x * inverse_rms * (1.0 + layer["input_norm"])
        )
        mixed = runtime.linear_fp8(qkv_matrix, normalized)
        z = runtime.linear_fp8(z_matrix, normalized).reshape(32, 128)
        a = layer["a_weight"] @ normalized[0]
        b = layer["b_weight"] @ normalized[0]
        convolved = cpu_causal_conv(
            initial_conv, layer["conv_weight"], mixed
        )[0]
        q = np.repeat(convolved[:2048].reshape(16, 128), 2, axis=0)
        k = np.repeat(convolved[2048:4096].reshape(16, 128), 2, axis=0)
        q = np.ascontiguousarray(
            q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + epsilon)
        )
        k = np.ascontiguousarray(
            k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + epsilon)
        )
        v = np.ascontiguousarray(convolved[4096:].reshape(1, 32, 128))
        g = np.ascontiguousarray(
            (
                -np.exp(layer["a_log"])
                * np.logaddexp(0.0, a + layer["dt_bias"])
            ).reshape(1, 32)
        )
        beta = np.ascontiguousarray(
            (1.0 / (1.0 + np.exp(-b))).reshape(1, 32)
        )
        recurrent = cpu_gated_delta(
            initial_recurrent, q[None], k[None], v, g, beta
        )[0]
        recurrent_inverse_rms = 1.0 / np.sqrt(
            np.mean(recurrent * recurrent, axis=-1, keepdims=True) + epsilon
        )
        gated = np.ascontiguousarray(
            recurrent
            * recurrent_inverse_rms
            * layer["gated_norm"]
            * (z / (1.0 + np.exp(-z)))
        )
        attention_reference = x + runtime.linear_fp8(
            out_matrix, gated.reshape(1, -1)
        )
        post_inverse_rms = 1.0 / np.sqrt(
            np.mean(
                attention_reference * attention_reference,
                axis=-1,
                keepdims=True,
            )
            + epsilon
        )
        post_normalized = np.ascontiguousarray(
            attention_reference
            * post_inverse_rms
            * (1.0 + layer["post_norm"])
        )
        expected_ids, expected_weights = route(router_f32 @ post_normalized[0], 8)
        shared_gate_f32 = (
            np.left_shift(shared_gate_bf16.astype(np.uint32), 16)
            .view(np.float32)
            .reshape(-1)
        )
        shared_weight = float(
            1.0 / (1.0 + np.exp(-float((shared_gate_f32 @ post_normalized[0]).item())))
        )
        selected_hosts = load_experts(args.model, args.layer, expected_ids)
        moe_reference = np.zeros_like(x)
        for weight, expert in zip(expected_weights, selected_hosts, strict=True):
            moe_reference += np.float32(weight) * expert_reference(
                expert, post_normalized
            )
        moe_reference += np.float32(shared_weight) * expert_reference(
            shared_host, post_normalized
        )
        reference = attention_reference + moe_reference
        del selected_hosts
        gc.collect()

        def execute() -> tuple[float, float]:
            attention.reset(initial_recurrent, initial_conv)
            started = time.perf_counter_ns()
            attention.enqueue(input_buffer, attention_output)
            runtime.rmsnorm_device(
                attention_output,
                post_norm_buffer,
                1,
                2048,
                epsilon,
                normalized_buffer,
            )
            bank.decode_device(normalized_buffer, moe_output)
            runtime.add_device(attention_output, moe_output, 2048, final_output)
            profile = runtime.synchronize()
            return profile.kernel_ns / 1e6, (time.perf_counter_ns() - started) / 1e6

        execute()
        actual = final_output.download(x.shape)
        maximum_error = float(np.max(np.abs(reference - actual)))
        if not np.allclose(reference, actual, rtol=2e-4, atol=1e-4):
            raise SystemExit(
                f"linear-attention MoE layer mismatch: max_abs={maximum_error:.9g}"
            )
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
                "operation": "ornith_35b_complete_linear_attention_moe_layer",
                "layer": args.layer,
                "hidden": 2048,
                "key_heads": 16,
                "value_heads": 32,
                "resident_bank_payload_bytes": payload_bytes,
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
                "initial recurrent and convolution states are synthetic",
                "complete 40-layer cadence and residency remain excluded",
            ],
            "samples": {"kernel_ms": kernel_ms, "wall_ms": wall_ms},
        }
        args.results.mkdir(parents=True, exist_ok=True)
        result_path = args.results / (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}-"
            "moe-linear-full-layer.json"
        )
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"device={runtime.device_name} layer={args.layer} "
            f"heads=16/32 experts={expected_ids}"
        )
        print(
            f"resident_bank_bytes={payload_bytes} bank_load_s={bank_load_seconds:.3f} "
            f"kernel_ms={record['timing']['kernel_ms']['median']:.6f} "
            f"wall_ms={record['timing']['wall_ms']['median']:.6f}"
        )
        print(f"max_abs={maximum_error:.9g} result={result_path}")
        print("MOE_LINEAR_FULL_LAYER_PASS")
        return 0
    finally:
        final_output.close()
        moe_output.close()
        normalized_buffer.close()
        attention_output.close()
        input_buffer.close()
        bank.close()
        post_norm_buffer.close()
        attention.close()
        for matrix in reversed(matrices):
            matrix.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
