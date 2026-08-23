"""Validate and benchmark one checkpoint-routed Qwen3.5 NVFP4 MoE layer."""

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
from bench_moe_experts import load_experts
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


def load_layer_tensors(
    model_dir: Path, layer: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray, float]]]:
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp"
    router_key = prefix + ".gate.weight"
    shared_gate_key = prefix + ".shared_expert_gate.weight"
    projection_keys = [
        prefix + f".shared_expert.{name}"
        for name in ("gate_proj", "up_proj", "down_proj")
    ]
    keys = [router_key, shared_gate_key]
    for base in projection_keys:
        keys.extend(base + suffix for suffix in (".weight", ".weight_scale", ".weight_scale_2"))
    missing = [key for key in keys if key not in index]
    if missing:
        raise KeyError(f"checkpoint index is missing {missing[0]}")

    tensors: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index[key], []).append(key)
    for shard_name, shard_keys in by_shard.items():
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as shard:
            for key in shard_keys:
                tensors[key] = shard.get_tensor(key)

    router_tensor = tensors[router_key]
    shared_gate_tensor = tensors[shared_gate_key]
    router_bf16 = np.ascontiguousarray(router_tensor.view(torch.uint16).numpy())
    router_f32 = np.ascontiguousarray(router_tensor.float().numpy())
    shared_gate_bf16 = np.ascontiguousarray(
        shared_gate_tensor.view(torch.uint16).numpy()
    )
    shared: list[tuple[np.ndarray, np.ndarray, float]] = []
    for base in projection_keys:
        packed = np.ascontiguousarray(
            tensors[base + ".weight"].numpy().astype(np.uint8, copy=False)
        )
        scales = np.ascontiguousarray(
            tensors[base + ".weight_scale"].view(torch.uint8).numpy()
        )
        multiplier = float(tensors[base + ".weight_scale_2"].item())
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(f"invalid {base}.weight_scale_2: {multiplier}")
        shared.append((packed, scales, 1.0 / multiplier))
    return router_bf16, router_f32, shared_gate_bf16, shared


def route(logits: np.ndarray, top_k: int) -> tuple[list[int], np.ndarray]:
    values = torch.from_numpy(np.ascontiguousarray(logits)).float()
    probabilities = torch.softmax(values, dim=-1)
    top_values, top_indices = torch.topk(probabilities, top_k, dim=-1)
    top_values /= top_values.sum(dim=-1, keepdim=True)
    return top_indices.tolist(), top_values.numpy().astype(np.float32, copy=False)


def expert_reference(
    projections: list[tuple[np.ndarray, np.ndarray, float]], x: np.ndarray
) -> np.ndarray:
    gate = cpu_gemm(*projections[0][:2], x, projections[0][2])
    up = cpu_gemm(*projections[1][:2], x, projections[1][2])
    activation = np.ascontiguousarray(gate / (1.0 + np.exp(-gate)) * up)
    return cpu_gemm(*projections[2][:2], activation, projections[2][2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.layer < 0 or not 0 < args.top_k <= 256 or args.warmups < 0 or args.samples <= 0:
        parser.error("invalid layer, top-k, or sample counts")

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
    reference_logits = np.ascontiguousarray(router_f32 @ x[0])
    expected_ids, expected_weights = route(reference_logits, args.top_k)
    experts_host = load_experts(args.model, args.layer, expected_ids)
    shared_gate_f32 = (
        torch.from_numpy(shared_gate_bf16.view(np.uint16).copy())
        .view(torch.bfloat16)
        .float()
        .numpy()
    )
    shared_gate_reference = float((shared_gate_f32 @ x[0]).item())
    shared_scale = float(1.0 / (1.0 + np.exp(-shared_gate_reference)))
    reference = np.zeros((1, 2048), dtype=np.float32)
    for weight, projections in zip(expected_weights, experts_host, strict=True):
        reference += np.float32(weight) * expert_reference(projections, x)
    reference += np.float32(shared_scale) * expert_reference(shared_host, x)

    runtime = Runtime(*runtime_paths())
    router = runtime.upload_buffer(router_bf16)
    shared_gate = runtime.upload_buffer(shared_gate_bf16)
    experts = [tuple(runtime.upload(*p) for p in item) for item in experts_host]
    shared = tuple(runtime.upload(*p) for p in shared_host)
    input_buffer = runtime.upload_buffer(x)
    router_output = runtime.create_buffer(256 * 4)
    shared_gate_output = runtime.create_buffer(4)
    expert_gate = runtime.create_buffer(512 * 4)
    expert_up = runtime.create_buffer(512 * 4)
    expert_activation = runtime.create_buffer(512 * 4)
    expert_output = runtime.create_buffer(2048 * 4)
    shared_gate_proj = runtime.create_buffer(512 * 4)
    shared_up_proj = runtime.create_buffer(512 * 4)
    shared_activation = runtime.create_buffer(512 * 4)
    shared_output = runtime.create_buffer(2048 * 4)
    output = runtime.create_buffer(2048 * 4)

    def enqueue_mlp(matrices: tuple[Any, Any, Any], buffers: tuple[Any, Any, Any, Any]) -> None:
        gate_matrix, up_matrix, down_matrix = matrices
        gate_buffer, up_buffer, activation_buffer, output_buffer = buffers
        runtime.linear_device(gate_matrix, input_buffer, 1, out=gate_buffer, kernel_kind=3, enqueue=True)
        runtime.linear_device(up_matrix, input_buffer, 1, out=up_buffer, kernel_kind=3, enqueue=True)
        runtime.silu_mul_device(gate_buffer, up_buffer, 512, activation_buffer)
        runtime.linear_device(down_matrix, activation_buffer, 1, out=output_buffer, kernel_kind=3, enqueue=True)

    def execute() -> tuple[float, float, float, float, list[int], np.ndarray, float]:
        started = time.perf_counter_ns()
        runtime.bf16_gemv_device(router, input_buffer, 256, 2048, router_output)
        runtime.bf16_gemv_device(shared_gate, input_buffer, 1, 2048, shared_gate_output)
        enqueue_mlp(
            shared,
            (shared_gate_proj, shared_up_proj, shared_activation, shared_output),
        )
        prefix = runtime.synchronize()
        logits = router_output.download((256,))
        gate_value = float(shared_gate_output.download((1,))[0])
        routing_started = time.perf_counter_ns()
        selected_ids, weights = route(logits, args.top_k)
        host_routing_ms = (time.perf_counter_ns() - routing_started) / 1e6
        if selected_ids != expected_ids:
            raise RuntimeError(f"router selected {selected_ids}, expected {expected_ids}")
        for index, (weight, matrices) in enumerate(zip(weights, experts, strict=True)):
            enqueue_mlp(
                matrices,
                (expert_gate, expert_up, expert_activation, expert_output),
            )
            runtime.weighted_accumulate_device(
                expert_output, float(weight), output, 2048, reset=index == 0
            )
        gate_scale = float(1.0 / (1.0 + np.exp(-gate_value)))
        runtime.weighted_accumulate_device(shared_output, gate_scale, output, 2048)
        suffix = runtime.synchronize()
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        return (
            (prefix.kernel_ns + suffix.kernel_ns) / 1e6,
            prefix.kernel_ns / 1e6,
            suffix.kernel_ns / 1e6,
            wall_ms,
            selected_ids,
            weights,
            host_routing_ms,
        )

    buffers = [
        router,
        shared_gate,
        input_buffer,
        router_output,
        shared_gate_output,
        expert_gate,
        expert_up,
        expert_activation,
        expert_output,
        shared_gate_proj,
        shared_up_proj,
        shared_activation,
        shared_output,
        output,
    ]
    try:
        first = execute()
        result = output.download((1, 2048))
        max_abs = float(np.max(np.abs(reference - result)))
        if not np.allclose(reference, result, rtol=8e-5, atol=8e-5):
            raise SystemExit(f"routed MoE oracle mismatch: max_abs={max_abs}")
        for _ in range(args.warmups):
            execute()
        samples = [execute() for _ in range(args.samples)]
        kernel_ms = [sample[0] for sample in samples]
        dispatch_kernel_ms = [sample[1] for sample in samples]
        experts_kernel_ms = [sample[2] for sample in samples]
        wall_ms = [sample[3] for sample in samples]
        host_routing_ms = [sample[6] for sample in samples]
        native_payload = (
            router_bf16.nbytes
            + shared_gate_bf16.nbytes
            + sum(p[0].nbytes + p[1].nbytes for p in shared_host)
            + sum(p[0].nbytes + p[1].nbytes for expert in experts_host for p in expert)
        )
        record = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hardware": {"system_model": system_model(), "gpu": runtime.device_name},
            "software": {"command": sys.argv},
            "environment": {**power_status(), "thermal_regime": "warm-burst"},
            "workload": {
                "operation": "qwen35_moe_checkpoint_routed_layer",
                "format": "nvfp4_experts_bf16_router",
                "layer": args.layer,
                "top_k": args.top_k,
                "selected_experts": expected_ids,
                "routing_weights": expected_weights.tolist(),
                "shared_expert_gate": shared_scale,
                "logical_payload_bytes": native_payload,
            },
            "timing": {
                "warmups": args.warmups,
                "samples": args.samples,
                "kernel_ms": describe(kernel_ms),
                "dispatch_and_shared_kernel_ms": describe(dispatch_kernel_ms),
                "selected_experts_and_reduction_kernel_ms": describe(experts_kernel_ms),
                "host_topk_ms": describe(host_routing_ms),
                "wall_ms": describe(wall_ms),
            },
            "bandwidth": {
                "logical_gbs": native_payload / (statistics.median(kernel_ms) * 1e6),
                "matched_island_ceiling_gbs": 129.0,
                "island_utilization": native_payload / (statistics.median(kernel_ms) * 1e6) / 129.0,
                "physical_gbs": None,
            },
            "correctness": {
                "passed": True,
                "max_abs_error": max_abs,
                "router_ids_exact": first[4] == expected_ids,
                "router_weight_max_abs_error": float(np.max(np.abs(first[5] - expected_weights))),
                "explicit_completion_marker": True,
            },
            "limitations": [
                "single-token decode micrograph with a fixed deterministic activation",
                "router logits and shared-gate scalar cross to the host at one dispatch boundary",
                "attention, layer norms, residuals, sampling, and serving overhead are excluded",
                "expert kernels execute serially on one in-order queue",
            ],
            "samples": {
                "kernel_ms": kernel_ms,
                "dispatch_and_shared_kernel_ms": dispatch_kernel_ms,
                "selected_experts_and_reduction_kernel_ms": experts_kernel_ms,
                "host_topk_ms": host_routing_ms,
                "wall_ms": wall_ms,
            },
        }
        args.results.mkdir(parents=True, exist_ok=True)
        result_path = args.results / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-moe-routed-layer.json"
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"device={runtime.device_name} layer={args.layer} experts={expected_ids}")
        print(f"weights={expected_weights.tolist()} shared_gate={shared_scale:.8f}")
        print(
            f"kernel_ms={statistics.median(kernel_ms):.4f} "
            f"dispatch_shared_ms={statistics.median(dispatch_kernel_ms):.4f} "
            f"experts_reduce_ms={statistics.median(experts_kernel_ms):.4f} "
            f"host_topk_ms={statistics.median(host_routing_ms):.4f} "
            f"wall_ms={statistics.median(wall_ms):.4f}"
        )
        print(f"max_abs_err={max_abs:.8g} result={result_path}")
        print("MOE_NVFP4_ROUTED_LAYER_PASS")
    finally:
        for buffer in reversed(buffers):
            buffer.close()
        for item in reversed(experts):
            for matrix in reversed(item):
                matrix.close()
        for matrix in reversed(shared):
            matrix.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
