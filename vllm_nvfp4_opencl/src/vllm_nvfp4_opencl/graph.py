"""Device-resident graph fragments used by the serving worker boundary."""

from __future__ import annotations

from typing import Self

import numpy as np

from .runtime import (
    DeviceBuffer,
    Fp8Matrix,
    NativeMatrix,
    PagedAttentionPool,
    PagedFullAttentionState,
    Profile,
    Runtime,
)


class ResidentQwen35FullAttention:
    """One-token Qwen3.5 full-attention layer with a device-resident KV cache."""

    def __init__(
        self,
        runtime: Runtime,
        q_proj: Fp8Matrix,
        k_proj: Fp8Matrix,
        v_proj: Fp8Matrix,
        o_proj: Fp8Matrix,
        *,
        input_norm_weight: np.ndarray,
        q_norm_weight: np.ndarray,
        k_norm_weight: np.ndarray,
        max_tokens: int,
        attention_pool: PagedAttentionPool | None = None,
        initial_k: np.ndarray | None = None,
        initial_v: np.ndarray | None = None,
        epsilon: float = 1e-6,
        hidden: int = 5120,
        query_heads: int = 24,
        kv_heads: int = 4,
    ):
        if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        attention_width = query_heads * 256
        expected_matrices = (
            (q_proj, (query_heads * 512, hidden), "q_proj"),
            (k_proj, (kv_heads * 256, hidden), "k_proj"),
            (v_proj, (kv_heads * 256, hidden), "v_proj"),
            (o_proj, (hidden, attention_width), "o_proj"),
        )
        for matrix, shape, name in expected_matrices:
            if (matrix.rows, matrix.cols) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        for name, array, shape in (
            ("input_norm_weight", input_norm_weight, (hidden,)),
            ("q_norm_weight", q_norm_weight, (256,)),
            ("k_norm_weight", k_norm_weight, (256,)),
        ):
            if (
                array.shape != shape
                or array.dtype != np.float32
                or not array.flags.c_contiguous
            ):
                raise ValueError(f"{name} must be contiguous float32 {shape}")
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")

        self.runtime = runtime
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.hidden = hidden
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.epsilon = epsilon
        self._closed = False
        self._buffers: list[DeviceBuffer] = []

        def upload(array: np.ndarray) -> DeviceBuffer:
            buffer = runtime.upload_buffer(array)
            self._buffers.append(buffer)
            return buffer

        def create(elements: int) -> DeviceBuffer:
            buffer = runtime.create_buffer(elements * np.dtype(np.float32).itemsize)
            self._buffers.append(buffer)
            return buffer

        self.input_norm_weight = upload(
            np.ascontiguousarray(input_norm_weight + np.float32(1.0))
        )
        # The fused prepare kernel applies Qwen3.5's 1+weight convention.
        self.q_norm_weight = upload(q_norm_weight)
        self.k_norm_weight = upload(k_norm_weight)
        self.normalized = create(hidden)
        self.q_projected = create(query_heads * 512)
        self.k_projected = create(kv_heads * 256)
        self.v_projected = create(kv_heads * 256)
        self.attended = create(attention_width)
        self.attention_output = create(hidden)
        if attention_pool is None:
            if (hidden, query_heads, kv_heads) != (5120, 24, 4):
                raise ValueError("non-dense attention shapes require a paged pool")
            self.attention_state = runtime.create_full_attention_state(
                max_tokens, initial_k, initial_v
            )
        else:
            if initial_k is not None or initial_v is not None:
                raise ValueError("paged attention does not yet import prefix caches")
            if attention_pool.runtime is not runtime:
                raise ValueError("attention pool belongs to a different runtime")
            if (
                attention_pool.query_heads != query_heads
                or attention_pool.kv_heads != kv_heads
            ):
                raise ValueError("attention pool head shape does not match matrices")
            self.attention_state = runtime.create_paged_full_attention_state(
                attention_pool, max_tokens
            )

    @property
    def tokens(self) -> int:
        return self.attention_state.tokens

    def reset(
        self,
        initial_k: np.ndarray | None = None,
        initial_v: np.ndarray | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("resident full-attention layer is closed")
        if isinstance(self.attention_state, PagedFullAttentionState):
            if initial_k is not None or initial_v is not None:
                raise ValueError("paged attention does not yet import prefix caches")
            self.runtime.reset_paged_full_attention_state(self.attention_state)
        else:
            self.runtime.reset_full_attention_state(
                self.attention_state, initial_k, initial_v
            )

    def enqueue(
        self,
        x: DeviceBuffer,
        cos: DeviceBuffer,
        sin: DeviceBuffer,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        if self._closed:
            raise RuntimeError("resident full-attention layer is closed")
        hidden_bytes = self.hidden * np.dtype(np.float32).itemsize
        rope_bytes = 64 * np.dtype(np.float32).itemsize
        if x.bytes < hidden_bytes or out.bytes < hidden_bytes:
            raise ValueError("input/output buffer is smaller than hidden size")
        if cos.bytes < rope_bytes or sin.bytes < rope_bytes:
            raise ValueError("cos/sin buffers must contain at least 64 float32 values")
        self.runtime.rmsnorm_device(
            x,
            self.input_norm_weight,
            1,
            self.hidden,
            self.epsilon,
            self.normalized,
        )
        for matrix, projection in (
            (self.q_proj, self.q_projected),
            (self.k_proj, self.k_projected),
            (self.v_proj, self.v_projected),
        ):
            self.runtime.linear_fp8_device(
                matrix, self.normalized, 1, out=projection, enqueue=True
            )
        self.enqueue_state_from_projected(cos, sin)
        self.runtime.linear_fp8_device(
            self.o_proj,
            self.attended,
            1,
            out=self.attention_output,
            enqueue=True,
        )
        return self.runtime.add_device(x, self.attention_output, self.hidden, out)

    def enqueue_state_from_projected(
        self, cos: DeviceBuffer, sin: DeviceBuffer
    ) -> DeviceBuffer:
        """Run Q/K preparation and paged attention after batched projections."""
        attention_arguments = (
            self.attention_state,
            self.q_projected,
            self.k_projected,
            self.v_projected,
            self.q_norm_weight,
            self.k_norm_weight,
            cos,
            sin,
            self.epsilon,
            self.attended,
        )
        if isinstance(self.attention_state, PagedFullAttentionState):
            self.runtime.paged_full_attention_decode_device(*attention_arguments)
        else:
            self.runtime.full_attention_decode_device(*attention_arguments)
        return self.attended

    def close(self) -> None:
        if self._closed:
            return
        self.attention_state.close()
        for buffer in reversed(self._buffers):
            buffer.close()
        self._buffers.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


class ResidentQwen35LinearAttention:
    """One-token Qwen3.5 linear-attention layer with persistent device state."""

    def __init__(
        self,
        runtime: Runtime,
        in_proj_qkv: Fp8Matrix,
        in_proj_z: Fp8Matrix,
        out_proj: Fp8Matrix,
        *,
        input_norm_weight: np.ndarray,
        a_weight: np.ndarray,
        b_weight: np.ndarray,
        a_log: np.ndarray,
        dt_bias: np.ndarray,
        conv_weight: np.ndarray,
        gated_norm_weight: np.ndarray,
        recurrent_state: np.ndarray | None = None,
        conv_state: np.ndarray | None = None,
        epsilon: float = 1e-6,
        hidden: int = 5120,
        key_heads: int = 16,
        value_heads: int = 48,
    ):
        if key_heads <= 0 or value_heads <= 0 or value_heads % key_heads:
            raise ValueError("value_heads must be divisible by key_heads")
        value_width = value_heads * 128
        mixed_width = (2 * key_heads + value_heads) * 128
        if (
            (in_proj_qkv.rows, in_proj_qkv.cols) != (mixed_width, hidden)
            or (in_proj_z.rows, in_proj_z.cols) != (value_width, hidden)
            or (out_proj.rows, out_proj.cols) != (hidden, value_width)
        ):
            raise ValueError("FP8 matrices do not match Qwen3.5 linear attention")
        expected = {
            "input_norm_weight": (hidden,),
            "a_weight": (value_heads, hidden),
            "b_weight": (value_heads, hidden),
            "a_log": (value_heads,),
            "dt_bias": (value_heads,),
            "conv_weight": (mixed_width, 4),
            "gated_norm_weight": (128,),
        }
        arrays = {
            "input_norm_weight": input_norm_weight,
            "a_weight": a_weight,
            "b_weight": b_weight,
            "a_log": a_log,
            "dt_bias": dt_bias,
            "conv_weight": conv_weight,
            "gated_norm_weight": gated_norm_weight,
        }
        for name, shape in expected.items():
            array = arrays[name]
            if (
                array.shape != shape
                or array.dtype != np.float32
                or not array.flags.c_contiguous
            ):
                raise ValueError(f"{name} must be contiguous float32 {shape}")
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")

        self.runtime = runtime
        self.in_proj_qkv = in_proj_qkv
        self.in_proj_z = in_proj_z
        self.out_proj = out_proj
        self.hidden = hidden
        self.key_heads = key_heads
        self.value_heads = value_heads
        self.epsilon = epsilon
        self._closed = False
        self._buffers: list[DeviceBuffer] = []
        self._states = []

        def upload(array: np.ndarray) -> DeviceBuffer:
            buffer = runtime.upload_buffer(array)
            self._buffers.append(buffer)
            return buffer

        def create(elements: int) -> DeviceBuffer:
            buffer = runtime.create_buffer(
                elements * np.dtype(np.float32).itemsize
            )
            self._buffers.append(buffer)
            return buffer

        effective_input_norm = np.ascontiguousarray(
            input_norm_weight + np.float32(1.0)
        )
        self.input_norm_weight = upload(effective_input_norm)
        self.a_weight = upload(a_weight)
        self.b_weight = upload(b_weight)
        self.a_log = upload(a_log)
        self.dt_bias = upload(dt_bias)
        self.gated_norm_weight = upload(gated_norm_weight)
        self.normalized = create(hidden)
        self.mixed_qkv = create(mixed_width)
        self.convolved_qkv = create(mixed_width)
        self.z = create(value_width)
        self.a = create(value_heads)
        self.b = create(value_heads)
        self.q = create(value_width)
        self.k = create(value_width)
        self.v = create(value_width)
        self.g = create(value_heads)
        self.beta = create(value_heads)
        self.recurrent_output = create(value_width)
        self.gated_output = create(value_width)
        self.attention_output = create(hidden)
        self.recurrent_state = runtime.create_gated_delta_state(
            value_heads, recurrent_state
        )
        self._states.append(self.recurrent_state)
        self.conv_state = runtime.create_causal_conv_state(
            conv_weight, conv_state
        )
        self._states.append(self.conv_state)

    def reset(
        self,
        recurrent_state: np.ndarray | None = None,
        conv_state: np.ndarray | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("resident linear-attention layer is closed")
        self.runtime.reset_gated_delta_state(
            self.recurrent_state, recurrent_state
        )
        self.runtime.reset_causal_conv_state(self.conv_state, conv_state)

    def enqueue(self, x: DeviceBuffer, out: DeviceBuffer) -> DeviceBuffer:
        if self._closed:
            raise RuntimeError("resident linear-attention layer is closed")
        hidden_bytes = self.hidden * np.dtype(np.float32).itemsize
        if x.bytes < hidden_bytes or out.bytes < hidden_bytes:
            raise ValueError("input/output buffer is smaller than hidden size")
        self.runtime.rmsnorm_device(
            x,
            self.input_norm_weight,
            1,
            self.hidden,
            self.epsilon,
            self.normalized,
        )
        self.runtime.linear_fp8_device(
            self.in_proj_qkv,
            self.normalized,
            1,
            out=self.mixed_qkv,
            enqueue=True,
        )
        self.runtime.linear_fp8_device(
            self.in_proj_z,
            self.normalized,
            1,
            out=self.z,
            enqueue=True,
        )
        self.enqueue_state_from_projected()
        self.runtime.linear_fp8_device(
            self.out_proj,
            self.gated_output,
            1,
            out=self.attention_output,
            enqueue=True,
        )
        return self.runtime.add_device(x, self.attention_output, self.hidden, out)

    def enqueue_state_from_projected(self) -> DeviceBuffer:
        """Run request-specific recurrent work after batched QKV/Z projection."""
        self.runtime.f32_gemv_device(
            self.a_weight,
            self.normalized,
            self.value_heads,
            self.hidden,
            self.a,
        )
        self.runtime.f32_gemv_device(
            self.b_weight,
            self.normalized,
            self.value_heads,
            self.hidden,
            self.b,
        )
        self.runtime.causal_conv_silu_device(
            self.conv_state, self.mixed_qkv, 1, self.convolved_qkv
        )
        self.runtime.prepare_gated_delta_decode_device(
            self.convolved_qkv,
            self.a,
            self.b,
            self.a_log,
            self.dt_bias,
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            self.key_heads,
            self.value_heads,
        )
        self.runtime.gated_delta_device(
            self.recurrent_state,
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            1,
            self.recurrent_output,
        )
        self.runtime.rmsnorm_silu_gate_device(
            self.recurrent_output,
            self.z,
            self.gated_norm_weight,
            self.value_heads,
            128,
            self.epsilon,
            self.gated_output,
        )
        return self.gated_output

    def close(self) -> None:
        if self._closed:
            return
        for state in reversed(self._states):
            state.close()
        self._states.clear()
        for buffer in reversed(self._buffers):
            buffer.close()
        self._buffers.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


class ResidentNvFp4Mlp:
    """One-token RMSNorm + gated Qwen MLP + residual graph.

    Matrix ownership stays with the model registry. This object owns only its
    learned RMSNorm weight and reusable activation buffers, which lets a worker
    compose several layer fragments without returning through NumPy or torch.
    """

    def __init__(
        self,
        runtime: Runtime,
        norm_weight: np.ndarray,
        gate: NativeMatrix,
        up: NativeMatrix,
        down: NativeMatrix,
        epsilon: float = 1e-6,
    ):
        if (
            norm_weight.ndim != 1
            or norm_weight.dtype != np.float32
            or not norm_weight.flags.c_contiguous
        ):
            raise ValueError("norm_weight must be contiguous float32 [hidden]")
        hidden = norm_weight.size
        if gate.cols != hidden or up.cols != hidden or gate.rows != up.rows:
            raise ValueError("gate/up dimensions do not match RMSNorm width")
        if down.cols != gate.rows or down.rows != hidden:
            raise ValueError("down projection dimensions do not match gate/up")
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")

        self.runtime = runtime
        self.gate = gate
        self.up = up
        self.down = down
        self.hidden_size = hidden
        self.intermediate_size = gate.rows
        self.epsilon = epsilon
        self._closed = False
        self._buffers: list[DeviceBuffer] = []
        float_bytes = np.dtype(np.float32).itemsize
        effective_norm_weight = np.ascontiguousarray(
            norm_weight + np.float32(1.0)
        )
        self.norm_weight = runtime.upload_buffer(effective_norm_weight)
        self._buffers.append(self.norm_weight)
        self.norm = runtime.create_buffer(hidden * float_bytes)
        self._buffers.append(self.norm)
        self.gate_output = runtime.create_buffer(gate.rows * float_bytes)
        self._buffers.append(self.gate_output)
        self.up_output = runtime.create_buffer(up.rows * float_bytes)
        self._buffers.append(self.up_output)
        self.activation = runtime.create_buffer(gate.rows * float_bytes)
        self._buffers.append(self.activation)
        self.down_output = runtime.create_buffer(hidden * float_bytes)
        self._buffers.append(self.down_output)
        self.host_input = runtime.create_buffer(hidden * float_bytes)
        self._buffers.append(self.host_input)
        self.host_output = runtime.create_buffer(hidden * float_bytes)
        self._buffers.append(self.host_output)

    def enqueue(self, x: DeviceBuffer, out: DeviceBuffer) -> DeviceBuffer:
        """Append this graph to the runtime queue without synchronizing."""
        if self._closed:
            raise RuntimeError("resident MLP is closed")
        hidden_bytes = self.hidden_size * np.dtype(np.float32).itemsize
        if x.bytes < hidden_bytes or out.bytes < hidden_bytes:
            raise ValueError("input/output buffer is smaller than hidden size")
        self.runtime.rmsnorm_device(
            x,
            self.norm_weight,
            1,
            self.hidden_size,
            self.epsilon,
            self.norm,
        )
        self.runtime.linear_device(
            self.gate, self.norm, 1, out=self.gate_output, enqueue=True
        )
        self.runtime.linear_device(
            self.up, self.norm, 1, out=self.up_output, enqueue=True
        )
        self.runtime.silu_mul_device(
            self.gate_output,
            self.up_output,
            self.intermediate_size,
            self.activation,
        )
        self.runtime.linear_device(
            self.down, self.activation, 1, out=self.down_output, enqueue=True
        )
        return self.runtime.add_device(
            x, self.down_output, self.hidden_size, out
        )

    def execute(self, x: np.ndarray) -> tuple[np.ndarray, Profile]:
        """Host convenience path with one input upload and one output read."""
        if x.shape != (1, self.hidden_size) or x.dtype != np.float32:
            raise ValueError("x must be contiguous float32 [1, hidden]")
        if not x.flags.c_contiguous:
            raise ValueError("x must be contiguous float32 [1, hidden]")
        self.host_input.upload(x)
        self.enqueue(self.host_input, self.host_output)
        profile = self.runtime.synchronize()
        return self.host_output.download(x.shape), profile

    def close(self) -> None:
        if self._closed:
            return
        for buffer in reversed(self._buffers):
            buffer.close()
        self._buffers.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


class ResidentNvFp4LmHead:
    """Final RMSNorm plus checkpoint-native NVFP4 vocabulary projection.

    The registry owns the immutable LM-head matrix. This graph owns only the
    effective norm weight and reusable hidden/logit buffers, matching the
    weight/state lifetime split used by the decoder fragments.
    """

    def __init__(
        self,
        runtime: Runtime,
        norm_weight: np.ndarray,
        lm_head: NativeMatrix,
        epsilon: float = 1e-6,
    ):
        if (
            norm_weight.ndim != 1
            or norm_weight.dtype != np.float32
            or not norm_weight.flags.c_contiguous
        ):
            raise ValueError("norm_weight must be contiguous float32 [hidden]")
        if lm_head.cols != norm_weight.size:
            raise ValueError("LM head input width must match final norm")
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        self.runtime = runtime
        self.lm_head = lm_head
        self.hidden_size = norm_weight.size
        self.vocab_size = lm_head.rows
        self.epsilon = epsilon
        self._closed = False
        self.norm_weight = runtime.upload_buffer(
            np.ascontiguousarray(norm_weight + np.float32(1.0))
        )
        self.normalized = runtime.create_buffer(
            self.hidden_size * np.dtype(np.float32).itemsize
        )
        self.logits = runtime.create_buffer(
            self.vocab_size * np.dtype(np.float32).itemsize
        )

    def enqueue(self, hidden: DeviceBuffer) -> DeviceBuffer:
        if self._closed:
            raise RuntimeError("resident LM head is closed")
        hidden_bytes = self.hidden_size * np.dtype(np.float32).itemsize
        if hidden.bytes < hidden_bytes:
            raise ValueError("hidden buffer is smaller than final norm width")
        self.runtime.rmsnorm_device(
            hidden,
            self.norm_weight,
            1,
            self.hidden_size,
            self.epsilon,
            self.normalized,
        )
        return self.runtime.linear_device(
            self.lm_head,
            self.normalized,
            1,
            out=self.logits,
            enqueue=True,
        )

    def execute(self, hidden: np.ndarray) -> tuple[np.ndarray, Profile]:
        if (
            hidden.shape != (1, self.hidden_size)
            or hidden.dtype != np.float32
            or not hidden.flags.c_contiguous
        ):
            raise ValueError("hidden must be contiguous float32 [1, hidden]")
        source = self.runtime.upload_buffer(hidden)
        try:
            self.enqueue(source)
            profile = self.runtime.synchronize()
            return self.logits.download((1, self.vocab_size)), profile
        finally:
            source.close()

    def close(self) -> None:
        if self._closed:
            return
        self.logits.close()
        self.normalized.close()
        self.norm_weight.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


class ResidentFp8LmHead:
    """Final RMSNorm plus a row-scaled FP8 vocabulary projection."""

    def __init__(
        self,
        runtime: Runtime,
        norm_weight: np.ndarray,
        lm_head: Fp8Matrix,
        epsilon: float = 1e-6,
    ):
        if (
            norm_weight.ndim != 1
            or norm_weight.dtype != np.float32
            or not norm_weight.flags.c_contiguous
        ):
            raise ValueError("norm_weight must be contiguous float32 [hidden]")
        if lm_head.cols != norm_weight.size:
            raise ValueError("LM head input width must match final norm")
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        self.runtime = runtime
        self.lm_head = lm_head
        self.hidden_size = norm_weight.size
        self.vocab_size = lm_head.rows
        self.epsilon = epsilon
        self._closed = False
        self.norm_weight = runtime.upload_buffer(
            np.ascontiguousarray(norm_weight + np.float32(1.0))
        )
        self.normalized = runtime.create_buffer(
            self.hidden_size * np.dtype(np.float32).itemsize
        )
        self.logits = runtime.create_buffer(
            self.vocab_size * np.dtype(np.float32).itemsize
        )

    def enqueue(self, hidden: DeviceBuffer) -> DeviceBuffer:
        if self._closed:
            raise RuntimeError("resident FP8 LM head is closed")
        hidden_bytes = self.hidden_size * np.dtype(np.float32).itemsize
        if hidden.bytes < hidden_bytes:
            raise ValueError("hidden buffer is smaller than final norm width")
        self.runtime.rmsnorm_device(
            hidden,
            self.norm_weight,
            1,
            self.hidden_size,
            self.epsilon,
            self.normalized,
        )
        return self.runtime.linear_fp8_device(
            self.lm_head,
            self.normalized,
            1,
            out=self.logits,
            enqueue=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        self.logits.close()
        self.normalized.close()
        self.norm_weight.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


class ResidentFp8Mlp:
    """One-token RMSNorm + row-scaled FP8 gated MLP + residual graph."""

    def __init__(
        self,
        runtime: Runtime,
        norm_weight: np.ndarray,
        gate: Fp8Matrix,
        up: Fp8Matrix,
        down: Fp8Matrix,
        epsilon: float = 1e-6,
    ):
        if (
            norm_weight.ndim != 1
            or norm_weight.dtype != np.float32
            or not norm_weight.flags.c_contiguous
        ):
            raise ValueError("norm_weight must be contiguous float32 [hidden]")
        hidden = norm_weight.size
        if gate.cols != hidden or up.cols != hidden or gate.rows != up.rows:
            raise ValueError("gate/up dimensions do not match RMSNorm width")
        if down.cols != gate.rows or down.rows != hidden:
            raise ValueError("down projection dimensions do not match gate/up")
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        self.runtime = runtime
        self.gate = gate
        self.up = up
        self.down = down
        self.hidden_size = hidden
        self.intermediate_size = gate.rows
        self.epsilon = epsilon
        self._closed = False
        self._buffers: list[DeviceBuffer] = []

        def create(elements: int) -> DeviceBuffer:
            buffer = runtime.create_buffer(elements * np.dtype(np.float32).itemsize)
            self._buffers.append(buffer)
            return buffer

        self.norm_weight = runtime.upload_buffer(
            np.ascontiguousarray(norm_weight + np.float32(1.0))
        )
        self._buffers.append(self.norm_weight)
        self.norm = create(hidden)
        self.gate_output = create(gate.rows)
        self.up_output = create(up.rows)
        self.activation = create(gate.rows)
        self.down_output = create(hidden)

    def enqueue(self, x: DeviceBuffer, out: DeviceBuffer) -> DeviceBuffer:
        if self._closed:
            raise RuntimeError("resident FP8 MLP is closed")
        hidden_bytes = self.hidden_size * np.dtype(np.float32).itemsize
        if x.bytes < hidden_bytes or out.bytes < hidden_bytes:
            raise ValueError("input/output buffer is smaller than hidden size")
        self.runtime.rmsnorm_device(
            x,
            self.norm_weight,
            1,
            self.hidden_size,
            self.epsilon,
            self.norm,
        )
        self.runtime.linear_fp8_device(
            self.gate, self.norm, 1, out=self.gate_output, enqueue=True
        )
        self.runtime.linear_fp8_device(
            self.up, self.norm, 1, out=self.up_output, enqueue=True
        )
        self.runtime.silu_mul_device(
            self.gate_output,
            self.up_output,
            self.intermediate_size,
            self.activation,
        )
        self.runtime.linear_fp8_device(
            self.down,
            self.activation,
            1,
            out=self.down_output,
            enqueue=True,
        )
        return self.runtime.add_device(x, self.down_output, self.hidden_size, out)

    def close(self) -> None:
        if self._closed:
            return
        for buffer in reversed(self._buffers):
            buffer.close()
        self._buffers.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


class ResidentBatchedNvFp4Mlp:
    """Multi-request RMSNorm + NVFP4 MLP using shared-weight GEMM kernels."""

    def __init__(
        self,
        runtime: Runtime,
        norm_weight: np.ndarray,
        gate: NativeMatrix,
        up: NativeMatrix,
        down: NativeMatrix,
        max_batch_size: int,
        epsilon: float = 1e-6,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if (
            norm_weight.ndim != 1
            or norm_weight.dtype != np.float32
            or not norm_weight.flags.c_contiguous
        ):
            raise ValueError("norm_weight must be contiguous float32 [hidden]")
        hidden = norm_weight.size
        if gate.cols != hidden or up.cols != hidden or gate.rows != up.rows:
            raise ValueError("gate/up dimensions do not match RMSNorm width")
        if down.cols != gate.rows or down.rows != hidden:
            raise ValueError("down projection dimensions do not match gate/up")
        self.runtime = runtime
        self.gate = gate
        self.up = up
        self.down = down
        self.hidden_size = hidden
        self.intermediate_size = gate.rows
        self.max_batch_size = max_batch_size
        self.epsilon = epsilon
        self._closed = False
        self._buffers: list[DeviceBuffer] = []

        def create(elements: int) -> DeviceBuffer:
            buffer = runtime.create_buffer(elements * np.dtype(np.float32).itemsize)
            self._buffers.append(buffer)
            return buffer

        self.norm_weight = runtime.upload_buffer(
            np.ascontiguousarray(norm_weight + np.float32(1.0))
        )
        self._buffers.append(self.norm_weight)
        self.norm = create(max_batch_size * hidden)
        self.gate_output = create(max_batch_size * gate.rows)
        self.up_output = create(max_batch_size * up.rows)
        self.activation = create(max_batch_size * gate.rows)
        self.down_output = create(max_batch_size * hidden)

    def enqueue(
        self,
        x: DeviceBuffer,
        out: DeviceBuffer,
        batch_size: int,
    ) -> DeviceBuffer:
        if self._closed:
            raise RuntimeError("resident batched MLP is closed")
        if batch_size <= 0 or batch_size > self.max_batch_size:
            raise ValueError("batch_size exceeds resident batched MLP capacity")
        hidden_elements = batch_size * self.hidden_size
        hidden_bytes = hidden_elements * np.dtype(np.float32).itemsize
        if x.bytes < hidden_bytes or out.bytes < hidden_bytes:
            raise ValueError("input/output buffer is smaller than batched hidden size")
        self.runtime.rmsnorm_device(
            x,
            self.norm_weight,
            batch_size,
            self.hidden_size,
            self.epsilon,
            self.norm,
        )
        self.runtime.linear_device(
            self.gate,
            self.norm,
            batch_size,
            out=self.gate_output,
            enqueue=True,
        )
        self.runtime.linear_device(
            self.up,
            self.norm,
            batch_size,
            out=self.up_output,
            enqueue=True,
        )
        self.runtime.silu_mul_device(
            self.gate_output,
            self.up_output,
            batch_size * self.intermediate_size,
            self.activation,
        )
        self.runtime.linear_device(
            self.down,
            self.activation,
            batch_size,
            out=self.down_output,
            enqueue=True,
        )
        return self.runtime.add_device(
            x, self.down_output, hidden_elements, out
        )

    def close(self) -> None:
        if self._closed:
            return
        for buffer in reversed(self._buffers):
            buffer.close()
        self._buffers.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


class ResidentQwen35DecodeCadence:
    """Compose a Qwen3.5 attention cadence and MLPs on one in-order queue."""

    def __init__(
        self,
        runtime: Runtime,
        layers: list[
            tuple[
                ResidentQwen35LinearAttention | ResidentQwen35FullAttention,
                ResidentNvFp4Mlp,
            ]
        ],
    ):
        if not layers:
            raise ValueError("cadence must contain at least one layer")
        for attention, mlp in layers:
            if attention.runtime is not runtime or mlp.runtime is not runtime:
                raise ValueError("all cadence layers must share one runtime")
        self.runtime = runtime
        self.layers = layers
        self._closed = False
        bytes_ = 5120 * np.dtype(np.float32).itemsize
        self.attention_residual = runtime.create_buffer(bytes_)
        self.scratch = [runtime.create_buffer(bytes_), runtime.create_buffer(bytes_)]

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("resident decode cadence is closed")
        for attention, _mlp in self.layers:
            attention.reset()

    def enqueue(
        self,
        x: DeviceBuffer,
        cos: DeviceBuffer,
        sin: DeviceBuffer,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        if self._closed:
            raise RuntimeError("resident decode cadence is closed")
        current = x
        last_index = len(self.layers) - 1
        for index, (attention, mlp) in enumerate(self.layers):
            if isinstance(attention, ResidentQwen35FullAttention):
                attention.enqueue(current, cos, sin, self.attention_residual)
            else:
                attention.enqueue(current, self.attention_residual)
            destination = out if index == last_index else self.scratch[index % 2]
            mlp.enqueue(self.attention_residual, destination)
            current = destination
        return out

    def close(self) -> None:
        if self._closed:
            return
        for buffer in reversed(self.scratch):
            buffer.close()
        self.attention_residual.close()
        for attention, mlp in reversed(self.layers):
            mlp.close()
            attention.close()
        self.layers.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


__all__ = [
    "ResidentBatchedNvFp4Mlp",
    "ResidentFp8LmHead",
    "ResidentFp8Mlp",
    "ResidentNvFp4LmHead",
    "ResidentNvFp4Mlp",
    "ResidentQwen35DecodeCadence",
    "ResidentQwen35FullAttention",
    "ResidentQwen35LinearAttention",
]
