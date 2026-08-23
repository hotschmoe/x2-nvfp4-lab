#!/usr/bin/env python3
"""Benchmark the exact Ornith 3-linear + 1-full sparse decoder cadence."""

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
from typing import Any

import numpy as np
from bench_islands import memory_status, percentile, power_status, system_model
from bench_moe_device_bank import stream_experts_into_bank
from bench_moe_full_attention import load_layer as load_full_attention
from bench_moe_full_layer import load_post_norm
from bench_moe_linear_layer import load_linear_layer
from bench_moe_routed_layer import MODEL, RESULTS, load_layer_tensors
from bench_resident_full_attention import rope

ROOT = Path(__file__).resolve().parents[1]


def describe(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


class CompleteSparseLayer:
    def __init__(
        self,
        runtime: Any,
        attention: Any,
        post_norm: np.ndarray,
        bank: Any,
        matrices: list[Any],
        payload_bytes: int,
        cos: Any | None = None,
        sin: Any | None = None,
    ):
        self.runtime = runtime
        self.attention = attention
        self.bank = bank
        self.matrices = matrices
        self.payload_bytes = payload_bytes
        self.cos = cos
        self.sin = sin
        self.post_norm = runtime.upload_buffer(
            np.ascontiguousarray(post_norm + np.float32(1.0))
        )
        self.attention_output = runtime.create_buffer(2048 * 4)
        self.normalized = runtime.create_buffer(2048 * 4)
        self.moe_output = runtime.create_buffer(2048 * 4)
        self.output = runtime.create_buffer(2048 * 4)

    def reset(self) -> None:
        self.attention.reset()

    def enqueue(self, source: Any) -> Any:
        if self.cos is None:
            self.attention.enqueue(source, self.attention_output)
        else:
            self.attention.enqueue(
                source, self.cos, self.sin, self.attention_output
            )
        self.runtime.rmsnorm_device(
            self.attention_output,
            self.post_norm,
            1,
            2048,
            1e-6,
            self.normalized,
        )
        self.bank.decode_device(self.normalized, self.moe_output)
        self.runtime.add_device(
            self.attention_output, self.moe_output, 2048, self.output
        )
        return self.output

    def close(self) -> None:
        self.output.close()
        self.moe_output.close()
        self.normalized.close()
        self.attention_output.close()
        self.post_norm.close()
        if self.sin is not None:
            self.sin.close()
        if self.cos is not None:
            self.cos.close()
        self.bank.close()
        self.attention.close()
        for matrix in reversed(self.matrices):
            matrix.close()


def build_bank(runtime: Any, model: Path, layer: int) -> tuple[Any, int]:
    router, _router_f32, shared_gate, shared = load_layer_tensors(model, layer)
    bank = runtime.create_moe_bank(router, shared_gate, 512)
    payload = stream_experts_into_bank(bank, model, layer)
    bank.upload_expert(256, shared)
    payload += sum(
        packed.nbytes + scales.nbytes for packed, scales, _divisor in shared
    )
    return bank, payload


def build_linear(runtime: Any, model: Path, layer: int) -> CompleteSparseLayer:
    from vllm_nvfp4_opencl.graph import ResidentQwen35LinearAttention

    values = load_linear_layer(model, layer)
    matrices = [
        runtime.upload_fp8_tensor_scaled(*host) for host in values["matrices"]
    ]
    attention = ResidentQwen35LinearAttention(
        runtime,
        *matrices,
        input_norm_weight=values["input_norm"],
        a_weight=values["a_weight"],
        b_weight=values["b_weight"],
        a_log=values["a_log"],
        dt_bias=values["dt_bias"],
        conv_weight=values["conv_weight"],
        gated_norm_weight=values["gated_norm"],
        hidden=2048,
        key_heads=16,
        value_heads=32,
    )
    bank, payload = build_bank(runtime, model, layer)
    return CompleteSparseLayer(
        runtime, attention, values["post_norm"], bank, matrices, payload
    )


def build_full(runtime: Any, model: Path, layer: int) -> CompleteSparseLayer:
    from vllm_nvfp4_opencl.graph import ResidentQwen35FullAttention

    hosts, input_norm, q_norm, k_norm = load_full_attention(model, layer)
    matrices = [runtime.upload_fp8_tensor_scaled(*host) for host in hosts]
    pool = runtime.create_paged_attention_pool(
        1, kv_dtype="bf16", query_heads=16, kv_heads=2
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
    bank, payload = build_bank(runtime, model, layer)
    cos, sin = (runtime.upload_buffer(value) for value in rope(0))
    result = CompleteSparseLayer(
        runtime,
        attention,
        load_post_norm(model, layer),
        bank,
        matrices,
        payload,
        cos,
        sin,
    )
    # The attention state owns a shared reference to the pool; release the
    # public pool handle after the layer object has been constructed.
    pool.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.first_layer < 0 or args.warmups < 0 or args.samples <= 0:
        parser.error("invalid layer, warmup, or sample count")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(ROOT / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    runtime = Runtime(*runtime_paths())
    available_before = memory_status().available_physical
    load_started = time.perf_counter()
    layers: list[CompleteSparseLayer] = []
    try:
        for offset in range(4):
            layer = args.first_layer + offset
            layers.append(
                build_full(runtime, args.model, layer)
                if offset == 3
                else build_linear(runtime, args.model, layer)
            )
        load_seconds = time.perf_counter() - load_started
        available_after = memory_status().available_physical
        gc.collect()
        x = np.ascontiguousarray(
            np.random.default_rng(20260822)
            .standard_normal((1, 2048))
            .astype(np.float32)
            * np.float32(0.2)
        )
        input_buffer = runtime.upload_buffer(x)

        def reset() -> None:
            for layer in layers:
                layer.reset()

        def enqueue_all(sync_each_layer: bool) -> tuple[Any, float]:
            source = input_buffer
            kernel_ms = 0.0
            for layer in layers:
                source = layer.enqueue(source)
                if sync_each_layer:
                    kernel_ms += runtime.synchronize().kernel_ns / 1e6
            return source, kernel_ms

        reset()
        oracle_buffer, oracle_kernel_ms = enqueue_all(True)
        oracle = oracle_buffer.download((1, 2048))
        reset()
        queued_buffer, _ = enqueue_all(False)
        queued_profile = runtime.synchronize()
        queued = queued_buffer.download((1, 2048))
        maximum_error = float(np.max(np.abs(oracle - queued)))
        if not np.allclose(oracle, queued, rtol=1e-5, atol=1e-5):
            raise SystemExit(f"queued sparse cadence mismatch: {maximum_error:.9g}")
        if not np.isfinite(queued).all():
            raise SystemExit("queued sparse cadence produced non-finite output")

        def execute() -> tuple[float, float]:
            reset()
            started = time.perf_counter_ns()
            _output, _ = enqueue_all(False)
            profile = runtime.synchronize()
            return profile.kernel_ns / 1e6, (time.perf_counter_ns() - started) / 1e6

        for _ in range(args.warmups):
            execute()
        samples = [execute() for _ in range(args.samples)]
        kernel_ms = [sample[0] for sample in samples]
        wall_ms = [sample[1] for sample in samples]
        payload_bytes = sum(layer.payload_bytes for layer in layers)
        record = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hardware": {"system_model": system_model(), "gpu": runtime.device_name},
            "environment": {**power_status(), "thermal_regime": "warm-burst"},
            "workload": {
                "operation": "ornith_35b_exact_four_layer_sparse_cadence",
                "layers": list(range(args.first_layer, args.first_layer + 4)),
                "layer_types": ["linear", "linear", "linear", "full"],
                "kv_dtype": "bf16",
                "resident_expert_bank_payload_bytes": payload_bytes,
            },
            "loading": {
                "load_seconds": load_seconds,
                "available_before_bytes": available_before,
                "available_after_bytes": available_after,
            },
            "timing": {
                "warmups": args.warmups,
                "samples": args.samples,
                "kernel_ms": describe(kernel_ms),
                "wall_ms": describe(wall_ms),
                "layer_synchronized_oracle_kernel_ms": oracle_kernel_ms,
                "first_queued_kernel_ms": queued_profile.kernel_ns / 1e6,
            },
            "correctness": {
                "passed": True,
                "maximum_absolute_error_vs_layer_synchronized_device_oracle": maximum_error,
                "finite_outputs": True,
                "explicit_completion_marker": True,
            },
            "limitations": [
                "one-token four-layer cadence with zero recurrent/conv initial states",
                "device oracle synchronizes identical proven layer kernels after each layer",
                "complete 40-layer residency, final head, sampling, and serving are excluded",
            ],
            "samples": {"kernel_ms": kernel_ms, "wall_ms": wall_ms},
        }
        args.results.mkdir(parents=True, exist_ok=True)
        result_path = args.results / (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}-moe-cadence.json"
        )
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"device={runtime.device_name} layers={record['workload']['layers']} "
            f"resident_bank_bytes={payload_bytes} load_s={load_seconds:.3f}"
        )
        print(
            f"kernel_ms={record['timing']['kernel_ms']['median']:.6f} "
            f"wall_ms={record['timing']['wall_ms']['median']:.6f} "
            f"oracle_max_abs={maximum_error:.9g} result={result_path}"
        )
        print("MOE_CADENCE_PASS")
        input_buffer.close()
        return 0
    finally:
        for layer in reversed(layers):
            layer.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
