"""Serving-oriented Qwen3.5 cadence loader and per-request decode sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
import torch
from safetensors import safe_open

from .graph import (
    ResidentBatchedNvFp4Mlp,
    ResidentNvFp4Mlp,
    ResidentQwen35DecodeCadence,
    ResidentQwen35FullAttention,
    ResidentQwen35LinearAttention,
)
from .runtime import (
    DeviceBuffer,
    Fp8Matrix,
    NativeMatrix,
    PagedAttentionPool,
    Profile,
    Runtime,
)


def _rope(position: int) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.float32(position) / np.power(
        np.float32(10_000_000.0), np.arange(0, 64, 2, dtype=np.float32) / 64.0
    )
    angles = np.concatenate((frequencies, frequencies))
    return (
        np.ascontiguousarray(np.cos(angles).astype(np.float32)),
        np.ascontiguousarray(np.sin(angles).astype(np.float32)),
    )


class Qwen35CadenceWeights:
    """Shared immutable matrices and small parameters for four decoder layers."""

    def __init__(
        self,
        runtime: Runtime,
        layer_parameters: list[dict[str, object]],
        matrices: list[NativeMatrix | Fp8Matrix],
        first_layer: int,
    ):
        self.runtime = runtime
        self.layer_parameters = layer_parameters
        self.matrices = matrices
        self.first_layer = first_layer
        self._closed = False

    @classmethod
    def load(
        cls,
        runtime: Runtime,
        checkpoint_path: str | Path,
        first_layer: int = 0,
    ) -> Qwen35CadenceWeights:
        if first_layer < 0 or first_layer % 4:
            raise ValueError("first_layer must be a nonnegative cadence boundary")
        hosts: list[dict[str, object]] = []
        checkpoint_path = Path(checkpoint_path)
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:

            def f32(name: str) -> np.ndarray:
                return np.ascontiguousarray(
                    checkpoint.get_tensor(name).float().numpy()
                )

            def fp8(name: str) -> tuple[np.ndarray, np.ndarray]:
                return (
                    np.ascontiguousarray(
                        checkpoint.get_tensor(name + ".weight")
                        .view(torch.uint8)
                        .numpy()
                    ),
                    np.ascontiguousarray(
                        checkpoint.get_tensor(name + ".weight_scale")
                        .view(torch.uint16)
                        .numpy()
                    ),
                )

            def nvfp4(name: str) -> tuple[np.ndarray, np.ndarray, float]:
                return (
                    np.ascontiguousarray(
                        checkpoint.get_tensor(name + ".weight_packed").numpy()
                    ),
                    np.ascontiguousarray(
                        checkpoint.get_tensor(name + ".weight_scale")
                        .view(torch.uint8)
                        .numpy()
                    ),
                    float(
                        checkpoint.get_tensor(
                            name + ".weight_global_scale"
                        ).item()
                    ),
                )

            for layer_index in range(first_layer, first_layer + 4):
                prefix = f"model.language_model.layers.{layer_index}"
                host: dict[str, object] = {
                    "input_norm": f32(prefix + ".input_layernorm.weight"),
                    "post_norm": f32(
                        prefix + ".post_attention_layernorm.weight"
                    ),
                    "mlp_hosts": [
                        nvfp4(prefix + ".mlp." + name)
                        for name in ("gate_proj", "up_proj", "down_proj")
                    ],
                }
                if layer_index % 4 == 3:
                    attention = prefix + ".self_attn"
                    host.update(
                        type="full",
                        attention_hosts=[
                            fp8(attention + "." + name)
                            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                        ],
                        q_norm=f32(attention + ".q_norm.weight"),
                        k_norm=f32(attention + ".k_norm.weight"),
                    )
                else:
                    attention = prefix + ".linear_attn"
                    host.update(
                        type="linear",
                        attention_hosts=[
                            fp8(attention + "." + name)
                            for name in ("in_proj_qkv", "in_proj_z", "out_proj")
                        ],
                        a_weight=f32(attention + ".in_proj_a.weight"),
                        b_weight=f32(attention + ".in_proj_b.weight"),
                        a_log=f32(attention + ".A_log"),
                        dt_bias=f32(attention + ".dt_bias"),
                        conv_weight=np.ascontiguousarray(
                            f32(attention + ".conv1d.weight").reshape(10240, 4)
                        ),
                        gated_norm=f32(attention + ".norm.weight"),
                    )
                hosts.append(host)

        matrices: list[NativeMatrix | Fp8Matrix] = []
        try:
            for host in hosts:
                attention_matrices = [
                    runtime.upload_fp8(*item)
                    for item in host.pop("attention_hosts")  # type: ignore[union-attr]
                ]
                mlp_matrices = [
                    runtime.upload(*item)
                    for item in host.pop("mlp_hosts")  # type: ignore[union-attr]
                ]
                host["attention_matrices"] = attention_matrices
                host["mlp_matrices"] = mlp_matrices
                matrices.extend(attention_matrices)
                matrices.extend(mlp_matrices)
        except Exception:
            for matrix in reversed(matrices):
                matrix.close()
            raise
        return cls(runtime, hosts, matrices, first_layer)

    def create_session(self, max_tokens: int) -> Qwen35CadenceSession:
        if self._closed:
            raise RuntimeError("cadence weights are closed")
        return Qwen35CadenceSession(self, max_tokens)

    def create_paged_scheduler(
        self,
        max_pages: int,
        max_batch_size: int = 4,
        kv_dtype: str = "fp32",
    ) -> Qwen35PagedScheduler:
        if self._closed:
            raise RuntimeError("cadence weights are closed")
        return Qwen35PagedScheduler(self, max_pages, max_batch_size, kv_dtype)

    def close(self) -> None:
        if self._closed:
            return
        for matrix in reversed(self.matrices):
            matrix.close()
        self.matrices.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


class Qwen35CadenceSession:
    """Per-request state and activation buffers over shared cadence weights."""

    def __init__(
        self,
        weights: Qwen35CadenceWeights,
        max_tokens: int,
        attention_pool: PagedAttentionPool | None = None,
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.weights = weights
        self.runtime = weights.runtime
        self.max_tokens = max_tokens
        self.position = 0
        self._closed = False
        self.cadence: ResidentQwen35DecodeCadence | None = None
        self.input: DeviceBuffer | None = None
        self.output: DeviceBuffer | None = None
        self.cos: DeviceBuffer | None = None
        self.sin: DeviceBuffer | None = None
        layers = []
        try:
            for parameters in weights.layer_parameters:
                attention_matrices = parameters["attention_matrices"]
                if parameters["type"] == "full":
                    attention = ResidentQwen35FullAttention(
                        self.runtime,
                        *attention_matrices,  # type: ignore[arg-type]
                        input_norm_weight=parameters["input_norm"],
                        q_norm_weight=parameters["q_norm"],
                        k_norm_weight=parameters["k_norm"],
                        max_tokens=max_tokens,
                        attention_pool=attention_pool,
                    )
                else:
                    attention = ResidentQwen35LinearAttention(
                        self.runtime,
                        *attention_matrices,  # type: ignore[arg-type]
                        input_norm_weight=parameters["input_norm"],
                        a_weight=parameters["a_weight"],
                        b_weight=parameters["b_weight"],
                        a_log=parameters["a_log"],
                        dt_bias=parameters["dt_bias"],
                        conv_weight=parameters["conv_weight"],
                        gated_norm_weight=parameters["gated_norm"],
                    )
                mlp = ResidentNvFp4Mlp(
                    self.runtime,
                    parameters["post_norm"],
                    *parameters["mlp_matrices"],  # type: ignore[arg-type]
                )
                layers.append((attention, mlp))
            self.cadence = ResidentQwen35DecodeCadence(self.runtime, layers)
            bytes_ = 5120 * np.dtype(np.float32).itemsize
            self.input = self.runtime.create_buffer(bytes_)
            self.output = self.runtime.create_buffer(bytes_)
            self.cos = self.runtime.create_buffer(64 * 4)
            self.sin = self.runtime.create_buffer(64 * 4)
        except Exception:
            for attention, mlp in reversed(layers):
                mlp.close()
                attention.close()
            raise

    def enqueue(
        self,
        hidden: DeviceBuffer,
        cos: DeviceBuffer,
        sin: DeviceBuffer,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        if self._closed:
            raise RuntimeError("cadence session is closed")
        if self.position >= self.max_tokens:
            raise RuntimeError("cadence session reached max_tokens")
        assert self.cadence is not None
        result = self.cadence.enqueue(hidden, cos, sin, out)
        self.position += 1
        return result

    def step(self, hidden: np.ndarray) -> tuple[np.ndarray, Profile]:
        self.prepare_host(hidden)
        assert self.input is not None and self.output is not None
        assert self.cos is not None and self.sin is not None
        self.enqueue(self.input, self.cos, self.sin, self.output)
        profile = self.runtime.synchronize()
        return self.download(), profile

    def prepare_host(self, hidden: np.ndarray) -> None:
        if (
            hidden.shape != (1, 5120)
            or hidden.dtype != np.float32
            or not hidden.flags.c_contiguous
        ):
            raise ValueError("hidden must be contiguous float32 [1, 5120]")
        assert self.input is not None and self.output is not None
        assert self.cos is not None and self.sin is not None
        cos, sin = _rope(self.position)
        self.input.upload(hidden)
        self.cos.upload(cos)
        self.sin.upload(sin)

    def download(self) -> np.ndarray:
        if self._closed or self.output is None:
            raise RuntimeError("cadence session is closed")
        return self.output.download((1, 5120))

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("cadence session is closed")
        assert self.cadence is not None
        self.cadence.reset()
        self.position = 0

    def close(self) -> None:
        if self._closed:
            return
        for buffer in (self.sin, self.cos, self.output, self.input):
            if buffer is not None:
                buffer.close()
        if self.cadence is not None:
            self.cadence.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


class Qwen35PagedScheduler:
    """Small scheduler seam with shared pages and one queue sync per batch."""

    def __init__(
        self,
        weights: Qwen35CadenceWeights,
        max_pages: int,
        max_batch_size: int,
        kv_dtype: str = "fp32",
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.weights = weights
        self.runtime = weights.runtime
        self.pool = self.runtime.create_paged_attention_pool(
            max_pages, kv_dtype=kv_dtype
        )
        self.max_batch_size = max_batch_size
        self.sessions: dict[str, Qwen35CadenceSession] = {}
        self._closed = False
        self.batched_mlps = [
            ResidentBatchedNvFp4Mlp(
                self.runtime,
                parameters["post_norm"],
                *parameters["mlp_matrices"],  # type: ignore[arg-type]
                max_batch_size=max_batch_size,
            )
            for parameters in weights.layer_parameters
        ]
        hidden_bytes = max_batch_size * 5120 * np.dtype(np.float32).itemsize
        self.batched_attention = self.runtime.create_buffer(hidden_bytes)
        self.batched_hidden = self.runtime.create_buffer(hidden_bytes)
        self.batched_source = self.runtime.create_buffer(hidden_bytes)
        self.batched_normalized = self.runtime.create_buffer(hidden_bytes)
        self.batched_projection0 = self.runtime.create_buffer(
            max_batch_size * 12288 * np.dtype(np.float32).itemsize
        )
        self.batched_projection1 = self.runtime.create_buffer(
            max_batch_size * 6144 * np.dtype(np.float32).itemsize
        )
        self.batched_projection2 = self.runtime.create_buffer(
            max_batch_size * 1024 * np.dtype(np.float32).itemsize
        )
        self.batched_state_output = self.runtime.create_buffer(
            max_batch_size * 6144 * np.dtype(np.float32).itemsize
        )
        self.batched_attention_output = self.runtime.create_buffer(hidden_bytes)

    @property
    def free_pages(self) -> int:
        return self.pool.free_pages

    def add_request(self, request_id: str, max_tokens: int) -> None:
        if self._closed:
            raise RuntimeError("paged scheduler is closed")
        if request_id in self.sessions:
            raise ValueError(f"request already exists: {request_id}")
        self.sessions[request_id] = Qwen35CadenceSession(
            self.weights, max_tokens, self.pool
        )

    def remove_request(self, request_id: str) -> None:
        try:
            session = self.sessions.pop(request_id)
        except KeyError as error:
            raise KeyError(f"unknown request: {request_id}") from error
        session.close()

    def reset_request(self, request_id: str) -> None:
        try:
            session = self.sessions[request_id]
        except KeyError as error:
            raise KeyError(f"unknown request: {request_id}") from error
        session.reset()

    def decode_batch(
        self, hidden_by_request: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], Profile]:
        if self._closed:
            raise RuntimeError("paged scheduler is closed")
        if not hidden_by_request:
            raise ValueError("decode batch must not be empty")
        if len(hidden_by_request) > self.max_batch_size:
            raise ValueError("decode batch exceeds scheduler max_batch_size")
        scheduled = []
        for request_id, hidden in hidden_by_request.items():
            try:
                session = self.sessions[request_id]
            except KeyError as error:
                raise KeyError(f"unknown request: {request_id}") from error
            session.prepare_host(hidden)
            scheduled.append((request_id, session))
        row_bytes = 5120 * np.dtype(np.float32).itemsize
        last_layer = len(self.batched_mlps) - 1
        for layer_index, batched_mlp in enumerate(self.batched_mlps):
            for row, (_request_id, session) in enumerate(scheduled):
                if session.position >= session.max_tokens:
                    raise RuntimeError("cadence session reached max_tokens")
                assert session.cadence is not None and session.input is not None
                self.runtime.copy_buffer_device(
                    session.input,
                    self.batched_source,
                    row_bytes,
                    destination_offset=row * row_bytes,
                )
            first_session = scheduled[0][1]
            assert first_session.cadence is not None
            first_attention, _unused_mlp = first_session.cadence.layers[layer_index]
            self.runtime.rmsnorm_device(
                self.batched_source,
                first_attention.input_norm_weight,
                len(scheduled),
                5120,
                first_attention.epsilon,
                self.batched_normalized,
            )
            if isinstance(first_attention, ResidentQwen35FullAttention):
                projections = (
                    (first_attention.q_proj, self.batched_projection0),
                    (first_attention.k_proj, self.batched_projection1),
                    (first_attention.v_proj, self.batched_projection2),
                )
                for matrix, projection in projections:
                    self.runtime.linear_fp8_device(
                        matrix,
                        self.batched_normalized,
                        len(scheduled),
                        out=projection,
                        enqueue=True,
                    )
                for row, (_request_id, session) in enumerate(scheduled):
                    assert session.cadence is not None
                    assert session.cos is not None and session.sin is not None
                    attention, _ = session.cadence.layers[layer_index]
                    assert isinstance(attention, ResidentQwen35FullAttention)
                    self.runtime.copy_buffer_device(
                        self.batched_projection0,
                        attention.q_projected,
                        12288 * 4,
                        source_offset=row * 12288 * 4,
                    )
                    self.runtime.copy_buffer_device(
                        self.batched_projection1,
                        attention.k_projected,
                        1024 * 4,
                        source_offset=row * 1024 * 4,
                    )
                    self.runtime.copy_buffer_device(
                        self.batched_projection2,
                        attention.v_projected,
                        1024 * 4,
                        source_offset=row * 1024 * 4,
                    )
                    attention.enqueue_state_from_projected(
                        session.cos, session.sin
                    )
                    self.runtime.copy_buffer_device(
                        attention.attended,
                        self.batched_state_output,
                        6144 * 4,
                        destination_offset=row * 6144 * 4,
                    )
                self.runtime.linear_fp8_device(
                    first_attention.o_proj,
                    self.batched_state_output,
                    len(scheduled),
                    out=self.batched_attention_output,
                    enqueue=True,
                )
            else:
                self.runtime.linear_fp8_device(
                    first_attention.in_proj_qkv,
                    self.batched_normalized,
                    len(scheduled),
                    out=self.batched_projection0,
                    enqueue=True,
                )
                self.runtime.linear_fp8_device(
                    first_attention.in_proj_z,
                    self.batched_normalized,
                    len(scheduled),
                    out=self.batched_projection1,
                    enqueue=True,
                )
                for row, (_request_id, session) in enumerate(scheduled):
                    assert session.cadence is not None
                    attention, _ = session.cadence.layers[layer_index]
                    assert isinstance(attention, ResidentQwen35LinearAttention)
                    self.runtime.copy_buffer_device(
                        self.batched_normalized,
                        attention.normalized,
                        row_bytes,
                        source_offset=row * row_bytes,
                    )
                    self.runtime.copy_buffer_device(
                        self.batched_projection0,
                        attention.mixed_qkv,
                        10240 * 4,
                        source_offset=row * 10240 * 4,
                    )
                    self.runtime.copy_buffer_device(
                        self.batched_projection1,
                        attention.z,
                        6144 * 4,
                        source_offset=row * 6144 * 4,
                    )
                    attention.enqueue_state_from_projected()
                    self.runtime.copy_buffer_device(
                        attention.gated_output,
                        self.batched_state_output,
                        6144 * 4,
                        destination_offset=row * 6144 * 4,
                    )
                self.runtime.linear_fp8_device(
                    first_attention.out_proj,
                    self.batched_state_output,
                    len(scheduled),
                    out=self.batched_attention_output,
                    enqueue=True,
                )
            self.runtime.add_device(
                self.batched_source,
                self.batched_attention_output,
                len(scheduled) * 5120,
                self.batched_attention,
            )
            batched_mlp.enqueue(
                self.batched_attention, self.batched_hidden, len(scheduled)
            )
            for row, (_request_id, session) in enumerate(scheduled):
                assert session.input is not None and session.output is not None
                destination = (
                    session.output if layer_index == last_layer else session.input
                )
                self.runtime.copy_buffer_device(
                    self.batched_hidden,
                    destination,
                    row_bytes,
                    source_offset=row * row_bytes,
                )
        for _request_id, session in scheduled:
            session.position += 1
        profile = self.runtime.synchronize()
        outputs = {
            request_id: session.download()
            for request_id, session in scheduled
        }
        return outputs, profile

    def close(self) -> None:
        if self._closed:
            return
        for session in reversed(list(self.sessions.values())):
            session.close()
        self.sessions.clear()
        self.batched_attention_output.close()
        self.batched_state_output.close()
        self.batched_projection2.close()
        self.batched_projection1.close()
        self.batched_projection0.close()
        self.batched_normalized.close()
        self.batched_source.close()
        self.batched_hidden.close()
        self.batched_attention.close()
        for mlp in reversed(self.batched_mlps):
            mlp.close()
        self.batched_mlps.clear()
        self.pool.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


__all__ = [
    "Qwen35CadenceSession",
    "Qwen35CadenceWeights",
    "Qwen35PagedScheduler",
]
