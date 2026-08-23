#!/usr/bin/env python3
"""Load the complete dense Qwen3.5 text model and execute one real token."""

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
import torch
from bench_islands import memory_status, percentile, power_status, system_model
from bench_resident_full_attention import rope
from inventory_checkpoint_memory import DTYPE_BYTES, tensor_category
from safetensors import safe_open
from trace_utils import summarize_trace, summarize_trace_samples

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/Qwen3.8-27B-NVFP4-Unsloth"
RESULTS = ROOT / "campaign_results/bandwidth-first"
LAYERS = 64
FULL_ATTENTION_INTERVAL = 4
HIDDEN = 5120


def describe(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def checkpoint_payloads(model: Path) -> tuple[int, list[int], int]:
    upfront = 0
    by_layer = [0] * LAYERS
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
                if ".layers." in name:
                    layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
                    by_layer[layer] += size
                else:
                    upfront += size
    if upfront + sum(by_layer) != complete:
        raise RuntimeError("dense checkpoint payload classification is inconsistent")
    return upfront, by_layer, complete


def load_embedding_row(checkpoint: Any, token_id: int) -> np.ndarray:
    name = "model.language_model.embed_tokens.weight"
    tensor = checkpoint.get_slice(name)
    shape = tensor.get_shape()
    if token_id < 0 or token_id >= shape[0]:
        raise ValueError(f"token ID {token_id} is outside vocabulary {shape[0]}")
    return np.ascontiguousarray(
        tensor[token_id : token_id + 1].float().numpy()
    )


class DenseLazyEmbeddingRows:
    def __init__(self, checkpoint_path: Path):
        self._context = safe_open(
            checkpoint_path, framework="pt", device="cpu"
        )
        self._checkpoint = self._context.__enter__()
        self._slice = self._checkpoint.get_slice(
            "model.language_model.embed_tokens.weight"
        )
        self.shape = self._slice.get_shape()
        self.touched: set[int] = set()
        self._closed = False

    def row(self, token_id: int) -> np.ndarray:
        if self._closed:
            raise RuntimeError("dense embedding table is closed")
        if token_id < 0 or token_id >= self.shape[0]:
            raise ValueError(
                f"token ID {token_id} is outside vocabulary {self.shape[0]}"
            )
        self.touched.add(token_id)
        return np.ascontiguousarray(
            self._slice[token_id : token_id + 1].float().numpy()
        )

    def close(self) -> None:
        if self._closed:
            return
        self._context.__exit__(None, None, None)
        self._closed = True


@dataclass
class DenseLayerSlot:
    attention: Any
    mlp: Any
    matrices: list[Any]
    full_attention: bool

    def close(self) -> None:
        self.mlp.close()
        self.attention.close()
        for matrix in reversed(self.matrices):
            matrix.close()
        self.matrices.clear()


class DenseModelRegistry:
    """One-copy dense weights plus one-request state and shared hidden scratch."""

    def __init__(
        self,
        runtime: Any,
        checkpoint_path: Path,
        max_tokens: int,
        kv_dtype: str,
    ):
        from vllm_nvfp4_opencl.graph import ResidentFp8LmHead

        self.runtime = runtime
        self.max_tokens = max_tokens
        self.layers: list[DenseLayerSlot] = []
        self.head_matrix: Any | None = None
        self.head: ResidentFp8LmHead | None = None
        self.pool: Any | None = None
        self._buffers: list[Any] = []
        self.last_trace: dict[str, Any] | None = None
        self._closed = False
        try:
            with safe_open(
                checkpoint_path, framework="pt", device="cpu"
            ) as checkpoint:
                norm = self._f32(
                    checkpoint, "model.language_model.norm.weight"
                )
                head_host = self._fp8(checkpoint, "lm_head")
            self.head_matrix = runtime.upload_fp8(*head_host)
            self.head = ResidentFp8LmHead(runtime, norm, self.head_matrix)
            del norm, head_host
            gc.collect()

            full_layers = LAYERS // FULL_ATTENTION_INTERVAL
            max_pages = full_layers * ((max_tokens + 15) // 16)
            self.pool = runtime.create_paged_attention_pool(
                max_pages,
                kv_dtype=kv_dtype,
                query_heads=24,
                kv_heads=4,
            )
            self.input = runtime.create_buffer(HIDDEN * 4)
            self.attention_residual = runtime.create_buffer(HIDDEN * 4)
            self.scratch0 = runtime.create_buffer(HIDDEN * 4)
            self.scratch1 = runtime.create_buffer(HIDDEN * 4)
            self._buffers.extend(
                (
                    self.input,
                    self.attention_residual,
                    self.scratch0,
                    self.scratch1,
                )
            )
            cos, sin = rope(0)
            self.cos = runtime.upload_buffer(cos)
            self.sin = runtime.upload_buffer(sin)
            self._buffers.extend((self.cos, self.sin))
        except Exception:
            self.close()
            raise

    def load_next_layer(self, checkpoint_path: Path) -> None:
        if self._closed:
            raise RuntimeError("dense registry is closed")
        layer = len(self.layers)
        if layer >= LAYERS:
            raise RuntimeError("all dense layers are already loaded")
        with safe_open(
            checkpoint_path, framework="pt", device="cpu"
        ) as checkpoint:
            slot = self._load_layer(checkpoint, layer)
        self.layers.append(slot)
        gc.collect()
        print(
            f"dense_layer_loaded={layer} "
            f"mlp_format={'fp8' if layer >= 56 else 'nvfp4'} "
            f"available_bytes={memory_status().available_physical}",
            flush=True,
        )

    @staticmethod
    def _f32(checkpoint: Any, name: str) -> np.ndarray:
        return np.ascontiguousarray(checkpoint.get_tensor(name).float().numpy())

    @staticmethod
    def _fp8(checkpoint: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.ascontiguousarray(
                checkpoint.get_tensor(name + ".weight").view(torch.uint8).numpy()
            ),
            np.ascontiguousarray(
                checkpoint.get_tensor(name + ".weight_scale")
                .view(torch.uint16)
                .numpy()
            ),
        )

    @staticmethod
    def _nvfp4(
        checkpoint: Any, name: str
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return (
            np.ascontiguousarray(checkpoint.get_tensor(name + ".weight_packed").numpy()),
            np.ascontiguousarray(
                checkpoint.get_tensor(name + ".weight_scale")
                .view(torch.uint8)
                .numpy()
            ),
            float(checkpoint.get_tensor(name + ".weight_global_scale").item()),
        )

    def _load_layer(self, checkpoint: Any, layer: int) -> DenseLayerSlot:
        from vllm_nvfp4_opencl.graph import (
            ResidentFp8Mlp,
            ResidentNvFp4Mlp,
            ResidentQwen35FullAttention,
            ResidentQwen35LinearAttention,
        )

        prefix = f"model.language_model.layers.{layer}"
        input_norm = self._f32(checkpoint, prefix + ".input_layernorm.weight")
        post_norm = self._f32(
            checkpoint, prefix + ".post_attention_layernorm.weight"
        )
        matrices: list[Any] = []
        attention = None
        mlp = None
        try:
            if layer % FULL_ATTENTION_INTERVAL == 3:
                base = prefix + ".self_attn"
                attention_matrices = [
                    self.runtime.upload_fp8(
                        *self._fp8(checkpoint, base + "." + name)
                    )
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                ]
                matrices.extend(attention_matrices)
                if self.pool is None:
                    raise RuntimeError("dense paged pool is unavailable")
                attention = ResidentQwen35FullAttention(
                    self.runtime,
                    *attention_matrices,
                    input_norm_weight=input_norm,
                    q_norm_weight=self._f32(checkpoint, base + ".q_norm.weight"),
                    k_norm_weight=self._f32(checkpoint, base + ".k_norm.weight"),
                    max_tokens=self.max_tokens,
                    attention_pool=self.pool,
                    hidden=HIDDEN,
                    query_heads=24,
                    kv_heads=4,
                )
                full_attention = True
            else:
                base = prefix + ".linear_attn"
                attention_matrices = [
                    self.runtime.upload_fp8(
                        *self._fp8(checkpoint, base + "." + name)
                    )
                    for name in ("in_proj_qkv", "in_proj_z", "out_proj")
                ]
                matrices.extend(attention_matrices)
                attention = ResidentQwen35LinearAttention(
                    self.runtime,
                    *attention_matrices,
                    input_norm_weight=input_norm,
                    a_weight=self._f32(checkpoint, base + ".in_proj_a.weight"),
                    b_weight=self._f32(checkpoint, base + ".in_proj_b.weight"),
                    a_log=self._f32(checkpoint, base + ".A_log"),
                    dt_bias=self._f32(checkpoint, base + ".dt_bias"),
                    conv_weight=np.ascontiguousarray(
                        self._f32(checkpoint, base + ".conv1d.weight").reshape(
                            10240, 4
                        )
                    ),
                    gated_norm_weight=self._f32(
                        checkpoint, base + ".norm.weight"
                    ),
                    hidden=HIDDEN,
                    key_heads=16,
                    value_heads=48,
                )
                full_attention = False

            mlp_base = prefix + ".mlp."
            if layer >= 56:
                mlp_matrices = [
                    self.runtime.upload_fp8(
                        *self._fp8(checkpoint, mlp_base + name)
                    )
                    for name in ("gate_proj", "up_proj", "down_proj")
                ]
                mlp = ResidentFp8Mlp(
                    self.runtime, post_norm, *mlp_matrices
                )
            else:
                mlp_matrices = [
                    self.runtime.upload(*self._nvfp4(checkpoint, mlp_base + name))
                    for name in ("gate_proj", "up_proj", "down_proj")
                ]
                mlp = ResidentNvFp4Mlp(
                    self.runtime, post_norm, *mlp_matrices
                )
            matrices.extend(mlp_matrices)
            return DenseLayerSlot(attention, mlp, matrices, full_attention)
        except Exception:
            if mlp is not None:
                mlp.close()
            if attention is not None:
                attention.close()
            for matrix in reversed(matrices):
                matrix.close()
            raise

    def reset(self) -> None:
        for slot in self.layers:
            slot.attention.reset()

    def begin_sequence(self) -> None:
        self.reset()
        self.runtime.synchronize()

    def enqueue_layer(
        self,
        layer: int,
        source: Any,
        destination: Any,
        *,
        trace: bool = False,
    ) -> Any:
        slot = self.layers[layer]
        if trace:
            attention_kind = (
                "full_attention" if slot.full_attention else "linear_attention"
            )
            self.runtime.set_trace_scope(
                f"dense.layer.{layer:02d}.{attention_kind}"
            )
        if slot.full_attention:
            slot.attention.enqueue(
                source, self.cos, self.sin, self.attention_residual
            )
        else:
            slot.attention.enqueue(source, self.attention_residual)
        if trace:
            mlp_kind = "mlp_fp8" if layer >= 56 else "mlp_nvfp4"
            self.runtime.set_trace_scope(
                f"dense.layer.{layer:02d}.{mlp_kind}"
            )
        return slot.mlp.enqueue(self.attention_residual, destination)

    def validate_last_layer(self, hidden: np.ndarray) -> float:
        layer = len(self.layers) - 1
        slot = self.layers[layer]
        self.reset()
        self.input.upload(hidden)
        self.runtime.synchronize()
        queued = self.enqueue_layer(layer, self.input, self.scratch0)
        self.runtime.synchronize()
        queued_host = queued.download((1, HIDDEN))
        slot.attention.reset()
        self.runtime.synchronize()
        if slot.full_attention:
            slot.attention.enqueue(
                self.input, self.cos, self.sin, self.attention_residual
            )
        else:
            slot.attention.enqueue(self.input, self.attention_residual)
        self.runtime.synchronize()
        slot.mlp.enqueue(self.attention_residual, self.scratch1)
        self.runtime.synchronize()
        synchronized_host = self.scratch1.download((1, HIDDEN))
        maximum_error = float(np.max(np.abs(queued_host - synchronized_host)))
        if not np.array_equal(queued_host, synchronized_host):
            raise RuntimeError(
                f"dense layer {layer} composition mismatch: max_abs={maximum_error}"
            )
        return maximum_error

    def step(
        self,
        hidden: np.ndarray,
        position: int,
        *,
        sync_each_layer: bool,
        project_logits: bool = True,
        trace: bool = False,
    ) -> tuple[np.ndarray | None, float, float]:
        if len(self.layers) != LAYERS or self.head is None:
            raise RuntimeError("complete dense execution requires all 64 layers")
        if position < 0 or position >= self.max_tokens:
            raise ValueError("position is outside the dense context capacity")
        if trace and sync_each_layer:
            raise ValueError("trace collection requires the queued execution path")
        cos, sin = rope(position)
        self.cos.upload(cos)
        self.sin.upload(sin)
        self.input.upload(hidden)
        self.runtime.synchronize()
        self.last_trace = None
        if trace:
            self.runtime.set_trace_enabled(True)
        started = time.perf_counter_ns()
        try:
            current = self.input
            kernel_ns = 0
            for layer in range(LAYERS):
                destination = self.scratch0 if layer % 2 == 0 else self.scratch1
                current = self.enqueue_layer(
                    layer, current, destination, trace=trace
                )
                if sync_each_layer:
                    kernel_ns += self.runtime.synchronize().kernel_ns
            if project_logits:
                if trace:
                    self.runtime.set_trace_scope("dense.head")
                self.head.enqueue(current)
            profile = self.runtime.synchronize()
            kernel_ns += profile.kernel_ns
            if trace:
                self.last_trace = summarize_trace(self.runtime.trace_events())
        finally:
            if trace:
                self.runtime.set_trace_enabled(False)
        logits = (
            self.head.logits.download((1, self.head.vocab_size))
            if project_logits
            else None
        )
        return (
            logits,
            kernel_ns / 1e6,
            (time.perf_counter_ns() - started) / 1e6,
        )

    def execute(
        self,
        hidden: np.ndarray,
        *,
        sync_each_layer: bool,
        trace: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        self.begin_sequence()
        logits, kernel_ms, wall_ms = self.step(
            hidden, 0, sync_each_layer=sync_each_layer, trace=trace
        )
        assert logits is not None
        return logits, kernel_ms, wall_ms

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
    parser.add_argument("--gates", default="16,32,48,56,64")
    parser.add_argument("--max-tokens", type=int, default=32_768)
    parser.add_argument("--kv-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--token-id", type=int, default=248044)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--generate-tokens", type=int, default=0)
    parser.add_argument("--trace-token", action="store_true")
    parser.add_argument("--trace-samples", type=int, default=3)
    parser.add_argument("--prompt")
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
        or args.generate_tokens < 0
        or args.generate_tokens > args.max_tokens
        or args.trace_samples <= 0
    ):
        parser.error("invalid gates, context, warmups, or samples")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(ROOT / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    upfront_payload, layer_payloads, complete_payload = checkpoint_payloads(
        args.model
    )
    checkpoint_path = args.model / "model.safetensors"
    baseline_available = memory_status().available_physical
    runtime = Runtime(*runtime_paths())
    registry: DenseModelRegistry | None = None
    gate_records: list[dict[str, Any]] = []
    checkpoint_payload = upfront_payload
    load_started = time.perf_counter()
    full_result: dict[str, Any] | None = None
    generation_result: dict[str, Any] | None = None
    embedding_rows: DenseLazyEmbeddingRows | None = None
    before_release = baseline_available
    after_release = baseline_available
    try:
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            hidden = load_embedding_row(checkpoint, args.token_id)
        registry = DenseModelRegistry(
            runtime,
            checkpoint_path,
            args.max_tokens,
            args.kv_dtype,
        )
        for layer in range(gates[-1]):
            registry.load_next_layer(checkpoint_path)
            checkpoint_payload += layer_payloads[layer]
            if layer + 1 in gates:
                maximum_error = registry.validate_last_layer(hidden)
                available = memory_status().available_physical
                gate_records.append(
                    {
                        "layers_resident": layer + 1,
                        "last_layer": layer,
                        "checkpoint_native_payload_bytes": checkpoint_payload,
                        "available_physical_bytes": available,
                        "available_delta_bytes": available - baseline_available,
                        "last_layer_composition_maximum_absolute_error": (
                            maximum_error
                        ),
                        "last_layer_mlp_format": (
                            "fp8" if layer >= 56 else "nvfp4"
                        ),
                    }
                )
                print(
                    f"gate_layers={layer + 1} "
                    f"checkpoint_bytes={checkpoint_payload} "
                    f"available_bytes={available} "
                    f"max_abs={maximum_error:.8g}",
                    flush=True,
                )
        load_and_validate_seconds = time.perf_counter() - load_started

        if gates[-1] == LAYERS:
            if checkpoint_payload != complete_payload:
                raise RuntimeError(
                    "dense registry payload does not equal text inventory: "
                    f"{checkpoint_payload} != {complete_payload}"
                )
            queued_logits, first_kernel, first_wall = registry.execute(
                hidden, sync_each_layer=False
            )
            oracle_logits, oracle_kernel, oracle_wall = registry.execute(
                hidden, sync_each_layer=True
            )
            maximum_error = float(
                np.max(np.abs(queued_logits - oracle_logits))
            )
            if not np.array_equal(queued_logits, oracle_logits):
                raise RuntimeError(
                    f"dense full-token composition mismatch: max_abs={maximum_error}"
                )
            queued_token = int(np.argmax(queued_logits))
            oracle_token = int(np.argmax(oracle_logits))
            if queued_token != oracle_token:
                raise RuntimeError("dense full-token argmax mismatch")
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
                "wall_tokens_per_second": 1000.0 / wall["median"],
                "first_queued_kernel_ms": first_kernel,
                "first_queued_wall_ms": first_wall,
                "layer_synchronized_kernel_ms": oracle_kernel,
                "layer_synchronized_wall_ms": oracle_wall,
                "maximum_absolute_error_vs_layer_synchronized_oracle": (
                    maximum_error
                ),
                "finite_logits": bool(np.isfinite(queued_logits).all()),
            }
            if args.trace_token:
                trace_samples = []
                for _ in range(args.trace_samples):
                    traced_logits, traced_kernel, traced_wall = registry.execute(
                        hidden, sync_each_layer=False, trace=True
                    )
                    if not np.array_equal(queued_logits, traced_logits):
                        raise RuntimeError("dense trace replay changed full logits")
                    if registry.last_trace is None:
                        raise RuntimeError("dense trace replay produced no events")
                    registry.last_trace.update(
                        replay_reported_kernel_ms=traced_kernel,
                        replay_wall_ms=traced_wall,
                        maximum_absolute_error_vs_untraced_logits=float(
                            np.max(np.abs(queued_logits - traced_logits))
                        ),
                    )
                    trace_samples.append(registry.last_trace)
                full_result["trace"] = summarize_trace_samples(trace_samples)
                trace_kernel_median = full_result["trace"]["sampling"][
                    "kernel_sum_ms"
                ]["median"]
                print(
                    "dense_trace "
                    f"events={full_result['trace']['event_count']} "
                    f"kernel_median_ms={trace_kernel_median:.6f}",
                    flush=True,
                )
            print(
                f"dense_full_token input={args.token_id} output={queued_token} "
                f"kernel_ms={kernel['median']:.6f} "
                f"wall_ms={wall['median']:.6f} max_abs={maximum_error:.8g}",
                flush=True,
            )

            if args.generate_tokens:
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(
                    str(args.model / "tokenizer.json")
                )
                prompt = args.prompt or (
                    "Write a Python function add(a, b) with type hints. "
                    "Return only code."
                )
                rendered_prompt = (
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n"
                )
                seed_token_ids = tokenizer.encode(
                    rendered_prompt, add_special_tokens=False
                ).ids
                maximum_positions = (
                    len(seed_token_ids) + args.generate_tokens - 1
                )
                if maximum_positions > args.max_tokens:
                    raise ValueError(
                        "dense prompt plus generation exceed context capacity"
                    )
                stop_token_ids = {248044, 248046}
                embedding_rows = DenseLazyEmbeddingRows(checkpoint_path)

                def generate(
                    sync_each_layer: bool,
                ) -> tuple[dict[str, Any], list[np.ndarray]]:
                    registry.begin_sequence()
                    prefill_kernel: list[float] = []
                    prefill_end_to_end: list[float] = []
                    started = time.perf_counter_ns()
                    last_logits = None
                    for position, token in enumerate(seed_token_ids):
                        host_started = time.perf_counter_ns()
                        logits, kernel, _device_wall = registry.step(
                            embedding_rows.row(token),
                            position,
                            sync_each_layer=sync_each_layer,
                            project_logits=position == len(seed_token_ids) - 1,
                        )
                        prefill_kernel.append(kernel)
                        prefill_end_to_end.append(
                            (time.perf_counter_ns() - host_started) / 1e6
                        )
                        if logits is not None:
                            last_logits = logits
                    assert last_logits is not None
                    generated = [int(np.argmax(last_logits))]
                    generated_logits = [last_logits]
                    time_to_first_ms = (
                        time.perf_counter_ns() - started
                    ) / 1e6
                    decode_kernel: list[float] = []
                    decode_end_to_end: list[float] = []
                    for generated_index in range(1, args.generate_tokens):
                        if generated[-1] in stop_token_ids:
                            break
                        position = len(seed_token_ids) + generated_index - 1
                        host_started = time.perf_counter_ns()
                        logits, kernel, _device_wall = registry.step(
                            embedding_rows.row(generated[-1]),
                            position,
                            sync_each_layer=sync_each_layer,
                        )
                        assert logits is not None
                        generated.append(int(np.argmax(logits)))
                        generated_logits.append(logits)
                        decode_kernel.append(kernel)
                        decode_end_to_end.append(
                            (time.perf_counter_ns() - host_started) / 1e6
                        )
                    processed_positions = (
                        len(seed_token_ids) + len(generated) - 1
                    )
                    result: dict[str, Any] = {
                        "prompt": prompt,
                        "rendered_prompt": rendered_prompt,
                        "prompt_token_ids": seed_token_ids,
                        "prompt_tokens": len(seed_token_ids),
                        "maximum_tokens_requested": args.generate_tokens,
                        "generated_token_ids": generated,
                        "generated_text": tokenizer.decode(
                            generated, skip_special_tokens=False
                        ),
                        "tokens_generated": len(generated),
                        "positions_processed": processed_positions,
                        "stop_token_ids": sorted(stop_token_ids),
                        "finish_reason": (
                            "stop"
                            if generated[-1] in stop_token_ids
                            else "length"
                        ),
                        "finish_token_id": (
                            generated[-1]
                            if generated[-1] in stop_token_ids
                            else None
                        ),
                        "prefill_kernel_ms_per_token": describe(prefill_kernel),
                        "prefill_end_to_end_ms": sum(prefill_end_to_end),
                        "prefill_tokens_per_second": (
                            1000.0
                            * len(seed_token_ids)
                            / sum(prefill_end_to_end)
                        ),
                        "time_to_first_token_ms": time_to_first_ms,
                    }
                    if decode_kernel:
                        result.update(
                            decode_steps=len(decode_kernel),
                            decode_kernel_ms_per_token=describe(decode_kernel),
                            decode_end_to_end_wall_ms_per_token=describe(
                                decode_end_to_end
                            ),
                            decode_end_to_end_tokens_per_second=(
                                1000.0 / statistics.mean(decode_end_to_end)
                            ),
                        )
                    return result, generated_logits

                generation_result, generated_logits = generate(False)
                oracle_generation, oracle_generation_logits = generate(True)
                maximum_generation_error = max(
                    float(np.max(np.abs(actual - expected)))
                    for actual, expected in zip(
                        generated_logits,
                        oracle_generation_logits,
                        strict=True,
                    )
                )
                if (
                    generation_result["generated_token_ids"]
                    != oracle_generation["generated_token_ids"]
                    or maximum_generation_error != 0
                ):
                    raise RuntimeError(
                        "dense stateful generation differs from synchronized replay"
                    )
                generation_result.update(
                    layer_synchronized_prefill_kernel_ms_per_token=(
                        oracle_generation["prefill_kernel_ms_per_token"]
                    ),
                    layer_synchronized_decode_kernel_ms_per_token=(
                        oracle_generation.get("decode_kernel_ms_per_token")
                    ),
                    maximum_absolute_error_vs_layer_synchronized_oracle=(
                        maximum_generation_error
                    ),
                    token_sequence_matches_oracle=True,
                )
                decode_tps = generation_result.get(
                    "decode_end_to_end_tokens_per_second", 0
                )
                print(
                    f"dense_prompt_tokens={len(seed_token_ids)} "
                    f"generated_tokens={generation_result['tokens_generated']} "
                    f"finish_reason={generation_result['finish_reason']} "
                    f"prefill_tps={generation_result['prefill_tokens_per_second']:.6f} "
                    f"decode_tps={decode_tps:.6f} "
                    f"max_abs={maximum_generation_error:.8g}",
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
            "environment": {
                **power_status(),
                "thermal_regime": "warm-burst",
                "shape_tuning_enabled": (
                    os.environ.get("VLLM_NVFP4_OPENCL_SHAPE_TUNING", "1") != "0"
                ),
            },
            "workload": {
                "operation": "dense_qwen_full_text_registry_and_token",
                "layers_loaded": gates[-1],
                "gate_layer_counts": gates,
                "max_tokens": args.max_tokens,
                "kv_dtype": args.kv_dtype,
                "lazy_embedding_rows_touched": (
                    len(embedding_rows.touched)
                    if embedding_rows is not None
                    else 1
                ),
                "optional_vision_loaded": False,
                "optional_mtp_loaded": False,
                "nvfp4_mlp_layers": min(gates[-1], 56),
                "fp8_mlp_layers": max(gates[-1] - 56, 0),
            },
            "loading": {
                "load_and_validate_seconds": load_and_validate_seconds,
                "total_load_benchmark_and_teardown_seconds": total_seconds,
                "resident_checkpoint_native_payload_bytes": checkpoint_payload,
                "complete_text_compute_payload_bytes": complete_payload,
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
            "generation": generation_result,
            "correctness": {
                "passed": True,
                "gates_validated": len(gate_records),
                "maximum_gate_layer_error": max(
                    float(gate["last_layer_composition_maximum_absolute_error"])
                    for gate in gate_records
                ),
                "complete_text_payload_accounted": (
                    gates[-1] < LAYERS
                    or checkpoint_payload == complete_payload
                ),
                "full_token_executed": full_result is not None,
                "stateful_generation_executed": generation_result is not None,
                "explicit_completion_marker": True,
            },
            "limitations": [
                *(
                    [
                        "one request and one decoded token; sustained "
                        "generation is not measured"
                    ]
                    if generation_result is None
                    else [
                        "one request; multi-request scheduling and sustained "
                        "thermal behavior are not measured"
                    ]
                ),
                "prefill is sequential batch-one execution rather than "
                "shared-weight GEMM",
                "greedy argmax is performed after downloading full logits",
                "available physical memory is system-wide, not an OpenCL "
                "allocator counter",
                "vision and MTP tensors are intentionally excluded from the "
                "coding endpoint",
            ],
        }
        args.results.mkdir(parents=True, exist_ok=True)
        suffix = (
            "generation"
            if generation_result is not None
            else (
                "trace"
                if args.trace_token and full_result is not None
                else ("token" if full_result is not None else "residency")
            )
        )
        path = args.results / (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}-"
            f"dense-full-model-{suffix}.json"
        )
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"dense_layers={gates[-1]} checkpoint_bytes={checkpoint_payload} "
            f"released_bytes={after_release - before_release} result={path}",
            flush=True,
        )
        print("DENSE_FULL_MODEL_REGISTRY_PASS", flush=True)
    finally:
        if registry is not None:
            registry.close()
        if embedding_rows is not None:
            embedding_rows.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
