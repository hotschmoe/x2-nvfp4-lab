#!/usr/bin/env python3
"""Load the complete Ornith text model and execute one checkpoint token."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from bench_islands import memory_status, percentile, power_status, system_model
from bench_moe_device_bank import stream_experts_into_bank
from bench_moe_experts import load_experts
from bench_moe_full_attention import load_layer as load_full_attention
from bench_moe_full_layer import load_post_norm
from bench_moe_linear_layer import load_linear_layer
from bench_moe_routed_layer import (
    MODEL,
    RESULTS,
    expert_reference,
    load_layer_tensors,
    route,
)
from bench_resident_full_attention import rope
from inventory_checkpoint_memory import DTYPE_BYTES, tensor_category
from probe_moe_lm_head import load_head
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
LAYERS = 40
FULL_ATTENTION_INTERVAL = 4
HIDDEN = 2048


def describe(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def checkpoint_payloads(model: Path) -> tuple[int, list[int], int]:
    """Return upfront, per-layer MoE, and complete text-compute bytes."""
    upfront = 0
    moe_by_layer = [0] * LAYERS
    complete = 0
    for path in sorted(model.glob("*.safetensors")):
        with safe_open(path, framework="np") as shard:
            # safetensors.safe_open exposes keys() but is not itself iterable.
            for name in shard.keys():  # noqa: SIM118
                category = tensor_category(name)
                if not category.startswith("text_") or category == "text_embedding":
                    continue
                tensor = shard.get_slice(name)
                size = math.prod(tensor.get_shape()) * DTYPE_BYTES[tensor.get_dtype()]
                complete += size
                if category.startswith("text_moe_"):
                    marker = ".layers."
                    layer = int(name.split(marker, 1)[1].split(".", 1)[0])
                    moe_by_layer[layer] += size
                else:
                    upfront += size
    if upfront + sum(moe_by_layer) != complete:
        raise RuntimeError("checkpoint payload classification is inconsistent")
    return upfront, moe_by_layer, complete


def load_embedding_row(model: Path, token_id: int) -> np.ndarray:
    name = "model.language_model.embed_tokens.weight"
    index = json.loads(
        (model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    with safe_open(model / index[name], framework="pt", device="cpu") as shard:
        shape = shard.get_slice(name).get_shape()
        if token_id < 0 or token_id >= shape[0]:
            raise ValueError(f"token ID {token_id} is outside vocabulary {shape[0]}")
        row = shard.get_slice(name)[token_id : token_id + 1]
        return np.ascontiguousarray(row.float().numpy())


@dataclass
class LayerSlot:
    attention: Any
    matrices: list[Any]
    post_norm: Any
    full_attention: bool
    bank: Any | None = None

    def close(self) -> None:
        if self.bank is not None:
            self.bank.close()
            self.bank = None
        self.post_norm.close()
        self.attention.close()
        for matrix in reversed(self.matrices):
            matrix.close()
        self.matrices.clear()


class OrnithModelRegistry:
    """One-copy model weights plus one-request state and reusable activations."""

    def __init__(
        self,
        runtime: Any,
        model: Path,
        max_tokens: int,
        kv_dtype: str,
    ):
        from vllm_nvfp4_opencl.graph import ResidentNvFp4LmHead

        self.runtime = runtime
        self.model = model
        self.max_tokens = max_tokens
        self.kv_dtype = kv_dtype
        self.layers: list[LayerSlot] = []
        self.head_matrix: Any | None = None
        self.head: ResidentNvFp4LmHead | None = None
        self.pool: Any | None = None
        self._buffers: list[Any] = []
        self._closed = False

        norm, packed, scales, divisor = load_head(model)
        try:
            self.head_matrix = runtime.upload(packed, scales, divisor)
            self.head = ResidentNvFp4LmHead(runtime, norm, self.head_matrix)
        finally:
            del norm, packed, scales
            gc.collect()

        full_layers = LAYERS // FULL_ATTENTION_INTERVAL
        max_pages = full_layers * ((max_tokens + 15) // 16)
        self.pool = runtime.create_paged_attention_pool(
            max_pages,
            kv_dtype=kv_dtype,
            query_heads=16,
            kv_heads=2,
        )
        try:
            for layer in range(LAYERS):
                slot = (
                    self._load_full(layer)
                    if layer % FULL_ATTENTION_INTERVAL == 3
                    else self._load_linear(layer)
                )
                self.layers.append(slot)
                gc.collect()
                print(
                    f"attention_layer_loaded={layer} "
                    f"available_bytes={memory_status().available_physical}",
                    flush=True,
                )
            for elements in (HIDDEN, HIDDEN, HIDDEN, HIDDEN, HIDDEN):
                self._buffers.append(runtime.create_buffer(elements * 4))
            self.input, self.attention_residual, self.normalized = self._buffers[:3]
            self.moe_output, self.scratch0 = self._buffers[3:]
            self.scratch1 = runtime.create_buffer(HIDDEN * 4)
            self._buffers.append(self.scratch1)
            cos, sin = rope(0)
            self.cos = runtime.upload_buffer(cos)
            self.sin = runtime.upload_buffer(sin)
            self._buffers.extend((self.cos, self.sin))
        except Exception:
            self.close()
            raise

    def _load_linear(self, layer: int) -> LayerSlot:
        from vllm_nvfp4_opencl.graph import ResidentQwen35LinearAttention

        values = load_linear_layer(self.model, layer)
        matrices: list[Any] = []
        attention = None
        post_norm = None
        try:
            matrices = [
                self.runtime.upload_fp8_tensor_scaled(*host)
                for host in values["matrices"]
            ]
            attention = ResidentQwen35LinearAttention(
                self.runtime,
                *matrices,
                input_norm_weight=values["input_norm"],
                a_weight=values["a_weight"],
                b_weight=values["b_weight"],
                a_log=values["a_log"],
                dt_bias=values["dt_bias"],
                conv_weight=values["conv_weight"],
                gated_norm_weight=values["gated_norm"],
                hidden=HIDDEN,
                key_heads=16,
                value_heads=32,
            )
            post_norm = self.runtime.upload_buffer(
                np.ascontiguousarray(values["post_norm"] + np.float32(1.0))
            )
            return LayerSlot(attention, matrices, post_norm, False)
        except Exception:
            if post_norm is not None:
                post_norm.close()
            if attention is not None:
                attention.close()
            for matrix in reversed(matrices):
                matrix.close()
            raise
        finally:
            del values

    def _load_full(self, layer: int) -> LayerSlot:
        from vllm_nvfp4_opencl.graph import ResidentQwen35FullAttention

        values, input_norm, q_norm, k_norm = load_full_attention(self.model, layer)
        matrices: list[Any] = []
        attention = None
        post_norm = None
        try:
            matrices = [
                self.runtime.upload_fp8_tensor_scaled(*host) for host in values
            ]
            if self.pool is None:
                raise RuntimeError("paged pool is unavailable")
            attention = ResidentQwen35FullAttention(
                self.runtime,
                *matrices,
                input_norm_weight=input_norm,
                q_norm_weight=q_norm,
                k_norm_weight=k_norm,
                max_tokens=self.max_tokens,
                attention_pool=self.pool,
                hidden=HIDDEN,
                query_heads=16,
                kv_heads=2,
            )
            post_norm = self.runtime.upload_buffer(
                np.ascontiguousarray(load_post_norm(self.model, layer) + np.float32(1.0))
            )
            return LayerSlot(attention, matrices, post_norm, True)
        except Exception:
            if post_norm is not None:
                post_norm.close()
            if attention is not None:
                attention.close()
            for matrix in reversed(matrices):
                matrix.close()
            raise
        finally:
            del values, input_norm, q_norm, k_norm

    def load_bank(
        self,
        layer: int,
        validation_input: np.ndarray | None = None,
    ) -> tuple[int, float | None, list[int] | None]:
        slot = self.layers[layer]
        if slot.bank is not None:
            raise ValueError(f"layer {layer} bank is already loaded")
        router, router_f32, shared_gate, shared = load_layer_tensors(
            self.model, layer
        )
        bank = self.runtime.create_moe_bank(router, shared_gate, 512)
        try:
            payload = stream_experts_into_bank(bank, self.model, layer)
            bank.upload_expert(256, shared)
            payload += sum(
                packed.nbytes + scales.nbytes
                for packed, scales, _divisor in shared
            )
            maximum_error = None
            selected_ids = None
            if validation_input is not None:
                selected_ids, selected_weights = route(
                    router_f32 @ validation_input[0], 8
                )
                selected_hosts = load_experts(self.model, layer, selected_ids)
                reference = np.zeros((1, HIDDEN), dtype=np.float32)
                for weight, expert in zip(
                    selected_weights, selected_hosts, strict=True
                ):
                    reference += np.float32(weight) * expert_reference(
                        expert, validation_input
                    )
                shared_f32 = (
                    np.left_shift(shared_gate.astype(np.uint32), 16)
                    .view(np.float32)
                    .reshape(-1)
                )
                shared_weight = float(
                    1.0
                    / (
                        1.0
                        + np.exp(
                            -float((shared_f32 @ validation_input[0]).item())
                        )
                    )
                )
                reference += np.float32(shared_weight) * expert_reference(
                    shared, validation_input
                )
                source = self.runtime.upload_buffer(validation_input)
                output = self.runtime.create_buffer(HIDDEN * 4)
                try:
                    bank.decode_device(source, output)
                    self.runtime.synchronize()
                    actual = output.download((1, HIDDEN))
                finally:
                    output.close()
                    source.close()
                maximum_error = float(np.max(np.abs(reference - actual)))
                if not np.allclose(reference, actual, rtol=1e-4, atol=1e-4):
                    raise RuntimeError(
                        f"layer {layer} bank mismatch: max_abs={maximum_error}"
                    )
                del selected_hosts, reference, actual
            slot.bank = bank
            return payload, maximum_error, selected_ids
        except Exception:
            bank.close()
            raise
        finally:
            del router, router_f32, shared_gate, shared
            gc.collect()

    def reset(self) -> None:
        for slot in self.layers:
            slot.attention.reset()

    def enqueue_layer(self, layer: int, source: Any, destination: Any) -> Any:
        slot = self.layers[layer]
        if slot.bank is None:
            raise RuntimeError(f"layer {layer} bank is not loaded")
        if slot.full_attention:
            slot.attention.enqueue(
                source, self.cos, self.sin, self.attention_residual
            )
        else:
            slot.attention.enqueue(source, self.attention_residual)
        self.runtime.rmsnorm_device(
            self.attention_residual,
            slot.post_norm,
            1,
            HIDDEN,
            1e-6,
            self.normalized,
        )
        slot.bank.decode_device(self.normalized, self.moe_output)
        return self.runtime.add_device(
            self.attention_residual,
            self.moe_output,
            HIDDEN,
            destination,
        )

    def execute(
        self,
        hidden: np.ndarray,
        *,
        sync_each_layer: bool,
    ) -> tuple[np.ndarray, float, float]:
        if len(self.layers) != LAYERS or any(
            slot.bank is None for slot in self.layers
        ):
            raise RuntimeError("complete model execution requires all 40 banks")
        if self.head is None:
            raise RuntimeError("LM head is unavailable")
        self.reset()
        self.input.upload(hidden)
        # Finish state resets and input upload outside the measured token step.
        self.runtime.synchronize()
        started = time.perf_counter_ns()
        current = self.input
        kernel_ns = 0
        for layer in range(LAYERS):
            destination = self.scratch0 if layer % 2 == 0 else self.scratch1
            current = self.enqueue_layer(layer, current, destination)
            if sync_each_layer:
                kernel_ns += self.runtime.synchronize().kernel_ns
        self.head.enqueue(current)
        profile = self.runtime.synchronize()
        kernel_ns += profile.kernel_ns
        logits = self.head.logits.download((1, self.head.vocab_size))
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        return logits, kernel_ns / 1e6, wall_ms

    def close(self) -> None:
        if self._closed:
            return
        for buffer in reversed(self._buffers):
            buffer.close()
        self._buffers.clear()
        for slot in reversed(self.layers):
            slot.close()
        self.layers.clear()
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        if self.head is not None:
            self.head.close()
            self.head = None
        if self.head_matrix is not None:
            self.head_matrix.close()
            self.head_matrix = None
        self._closed = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--gates", default="24,30,35,40")
    parser.add_argument("--max-tokens", type=int, default=32_768)
    parser.add_argument("--kv-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--token-id", type=int, default=248044)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    gates = [int(value) for value in args.gates.split(",")]
    if (
        not gates
        or gates != sorted(set(gates))
        or gates[0] <= 0
        or gates[-1] > LAYERS
        or args.max_tokens <= 0
        or args.warmups < 0
        or args.samples <= 0
    ):
        parser.error("invalid gates, context, warmups, or sample count")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(ROOT / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    upfront_payload, moe_payloads, complete_payload = checkpoint_payloads(
        args.model
    )
    hidden = load_embedding_row(args.model, args.token_id)
    validation_input = np.ascontiguousarray(
        np.random.default_rng(20260822)
        .standard_normal((1, HIDDEN))
        .astype(np.float32)
        * np.float32(0.2)
    )
    baseline_available = memory_status().available_physical
    runtime = Runtime(*runtime_paths())
    registry: OrnithModelRegistry | None = None
    gate_records: list[dict[str, Any]] = []
    bank_payload = 0
    checkpoint_payload = upfront_payload
    load_started = time.perf_counter()
    full_result: dict[str, Any] | None = None
    before_release = baseline_available
    after_release = baseline_available
    try:
        registry = OrnithModelRegistry(
            runtime, args.model, args.max_tokens, args.kv_dtype
        )
        nonexpert_load_seconds = time.perf_counter() - load_started
        print(
            f"nonexpert_loaded checkpoint_payload_bytes={upfront_payload} "
            f"available_bytes={memory_status().available_physical}",
            flush=True,
        )
        for layer in range(gates[-1]):
            layer_started = time.perf_counter()
            gate = layer + 1 in gates
            native_bank_payload, max_abs, selected_ids = registry.load_bank(
                layer, validation_input if gate else None
            )
            bank_payload += native_bank_payload
            checkpoint_payload += moe_payloads[layer]
            available = memory_status().available_physical
            print(
                f"bank_loaded={layer + 1} layer={layer} "
                f"bank_native_bytes={bank_payload} available_bytes={available}",
                flush=True,
            )
            if gate:
                gate_record = {
                    "banks_resident": layer + 1,
                    "last_layer": layer,
                    "checkpoint_native_payload_bytes": checkpoint_payload,
                    "expert_packed_scale_payload_bytes": bank_payload,
                    "available_physical_bytes": available,
                    "available_delta_bytes": available - baseline_available,
                    "last_bank_load_and_validate_seconds": (
                        time.perf_counter() - layer_started
                    ),
                    "last_bank_maximum_absolute_error": max_abs,
                    "last_bank_selected_experts": selected_ids,
                }
                gate_records.append(gate_record)
                print(
                    f"gate_banks={layer + 1} "
                    f"checkpoint_bytes={checkpoint_payload} "
                    f"max_abs={max_abs:.8g}",
                    flush=True,
                )

        load_and_validate_seconds = time.perf_counter() - load_started

        if gates[-1] == LAYERS:
            if checkpoint_payload != complete_payload:
                raise RuntimeError(
                    "full registry payload does not equal text-compute inventory: "
                    f"{checkpoint_payload} != {complete_payload}"
                )
            queued_logits, queued_kernel, queued_wall = registry.execute(
                hidden, sync_each_layer=False
            )
            oracle_logits, oracle_kernel, oracle_wall = registry.execute(
                hidden, sync_each_layer=True
            )
            maximum_error = float(
                np.max(np.abs(queued_logits - oracle_logits))
            )
            if not np.array_equal(
                queued_logits, oracle_logits
            ) and not np.allclose(
                queued_logits, oracle_logits, rtol=1e-6, atol=1e-6
            ):
                raise RuntimeError(
                    f"queued full-model mismatch: max_abs={maximum_error}"
                )
            queued_token = int(np.argmax(queued_logits))
            oracle_token = int(np.argmax(oracle_logits))
            if queued_token != oracle_token:
                raise RuntimeError(
                    f"full-model argmax mismatch: {queued_token} != {oracle_token}"
                )
            for _ in range(args.warmups):
                registry.execute(hidden, sync_each_layer=False)
            samples = [
                registry.execute(hidden, sync_each_layer=False)
                for _ in range(args.samples)
            ]
            kernel = describe([sample[1] for sample in samples])
            wall = describe([sample[2] for sample in samples])
            full_result = {
                "input_token_id": args.token_id,
                "greedy_output_token_id": queued_token,
                "queued_kernel_ms": kernel,
                "queued_wall_ms": wall,
                "first_queued_kernel_ms": queued_kernel,
                "first_queued_wall_ms": queued_wall,
                "layer_synchronized_kernel_ms": oracle_kernel,
                "layer_synchronized_wall_ms": oracle_wall,
                "maximum_absolute_error_vs_layer_synchronized_oracle": (
                    maximum_error
                ),
                "finite_logits": bool(np.isfinite(queued_logits).all()),
            }
            print(
                f"full_token input={args.token_id} output={queued_token} "
                f"kernel_ms={kernel['median']:.6f} "
                f"wall_ms={wall['median']:.6f} max_abs={maximum_error:.8g}",
                flush=True,
            )

        before_release = memory_status().available_physical
        registry.close()
        registry = None
        gc.collect()
        after_release = memory_status().available_physical
        total_seconds = time.perf_counter() - load_started
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
                "operation": "ornith_full_text_registry_and_token",
                "layers": LAYERS,
                "loaded_banks": gates[-1],
                "gate_bank_counts": gates,
                "max_tokens": args.max_tokens,
                "kv_dtype": args.kv_dtype,
                "lazy_embedding_rows_touched": 1,
                "optional_vision_loaded": False,
                "optional_mtp_loaded": False,
            },
            "loading": {
                "nonexpert_load_seconds": nonexpert_load_seconds,
                "total_load_and_validate_seconds": load_and_validate_seconds,
                "total_load_benchmark_and_teardown_seconds": total_seconds,
                "upfront_checkpoint_native_payload_bytes": upfront_payload,
                "resident_checkpoint_native_payload_bytes": checkpoint_payload,
                "complete_text_compute_payload_bytes": complete_payload,
                "expert_packed_scale_payload_bytes": bank_payload,
            },
            "memory": {
                "baseline_available_physical_bytes": baseline_available,
                "before_release_available_physical_bytes": before_release,
                "after_release_available_physical_bytes": after_release,
                "recovered_on_release_bytes": after_release - before_release,
                "reported_opencl_global_budget_bytes": 24379 * 1024 * 1024,
            },
            "gates": gate_records,
            "full_token": full_result,
            "correctness": {
                "passed": True,
                "gates_validated": len(gate_records),
                "maximum_gate_bank_error": max(
                    float(gate["last_bank_maximum_absolute_error"])
                    for gate in gate_records
                ),
                "complete_text_payload_accounted": (
                    gates[-1] < LAYERS
                    or checkpoint_payload == complete_payload
                ),
                "full_token_executed": full_result is not None,
                "explicit_completion_marker": True,
            },
            "limitations": [
                "one request and one decoded token; sustained generation is not measured",
                "greedy argmax is performed after downloading full logits",
                "available physical memory is system-wide, not an OpenCL allocator counter",
                "vision and MTP tensors are intentionally excluded from the coding endpoint",
            ],
        }
        args.results.mkdir(parents=True, exist_ok=True)
        suffix = "full-model-token" if full_result is not None else "full-model-residency"
        path = args.results / (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}-"
            f"moe-{suffix}.json"
        )
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"loaded_banks={gates[-1]} checkpoint_bytes={checkpoint_payload} "
            f"released_bytes={after_release - before_release} result={path}",
            flush=True,
        )
        print("MOE_FULL_MODEL_REGISTRY_PASS", flush=True)
    finally:
        if registry is not None:
            registry.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
