#!/usr/bin/env python3
"""Validate the exact Qwen3.5 full-attention decode kernel and KV cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * (
        1.0 + weight
    )


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    result = x.copy()
    rotary = x[..., :64]
    rotated = np.concatenate((-rotary[..., 32:], rotary[..., :32]), axis=-1)
    result[..., :64] = rotary * cos + rotated * sin
    return result


def main() -> int:
    here = Path(__file__).resolve().parent
    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        here / "runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        here / "kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(here.parent / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    rng = np.random.default_rng(20260822)
    q_weight = np.ascontiguousarray(rng.normal(0.0, 0.1, 256).astype(np.float32))
    k_weight = np.ascontiguousarray(rng.normal(0.0, 0.1, 256).astype(np.float32))
    runtime = Runtime(*runtime_paths())
    # 8,193 exceeds the old 32 KiB local-logit ceiling and guards the online
    # softmax path against accidentally reintroducing that context limit.
    state = runtime.create_full_attention_state(8193)
    buffers = []

    def upload(array: np.ndarray):
        buffer = runtime.upload_buffer(array)
        buffers.append(buffer)
        return buffer

    def create(elements: int):
        buffer = runtime.create_buffer(elements * np.dtype(np.float32).itemsize)
        buffers.append(buffer)
        return buffer

    q_weight_buffer = upload(q_weight)
    k_weight_buffer = upload(k_weight)
    q_buffer = create(12288)
    k_buffer = create(1024)
    v_buffer = create(1024)
    cos_buffer = create(64)
    sin_buffer = create(64)
    output_buffer = create(24 * 256)
    k_cache: list[np.ndarray] = []
    v_cache: list[np.ndarray] = []
    maximum_error = 0.0

    try:
        for position in range(8):
            q_projected = np.ascontiguousarray(
                rng.normal(0.0, 0.2, (24, 512)).astype(np.float32)
            )
            k_projected = np.ascontiguousarray(
                rng.normal(0.0, 0.2, (4, 256)).astype(np.float32)
            )
            v_projected = np.ascontiguousarray(
                rng.normal(0.0, 0.2, (4, 256)).astype(np.float32)
            )
            angles = (
                np.arange(64, dtype=np.float32) * np.float32(0.013)
                + np.float32(position * 0.071)
            )
            cos = np.ascontiguousarray(np.cos(angles).astype(np.float32))
            sin = np.ascontiguousarray(np.sin(angles).astype(np.float32))
            q_buffer.upload(q_projected)
            k_buffer.upload(k_projected)
            v_buffer.upload(v_projected)
            cos_buffer.upload(cos)
            sin_buffer.upload(sin)

            q = apply_rope(rmsnorm(q_projected[:, :256], q_weight), cos, sin)
            gate = q_projected[:, 256:]
            k = apply_rope(rmsnorm(k_projected, k_weight), cos, sin)
            k_cache.append(k)
            v_cache.append(v_projected)
            cached_k = np.stack(k_cache)
            cached_v = np.stack(v_cache)
            reference = np.empty((24, 256), dtype=np.float32)
            for head in range(24):
                kv_head = head // 6
                logits = cached_k[:, kv_head] @ q[head] * np.float32(0.0625)
                probabilities = np.exp(logits - np.max(logits))
                probabilities /= np.sum(probabilities)
                value = probabilities @ cached_v[:, kv_head]
                reference[head] = value / (1.0 + np.exp(-gate[head]))

            runtime.full_attention_decode_device(
                state,
                q_buffer,
                k_buffer,
                v_buffer,
                q_weight_buffer,
                k_weight_buffer,
                cos_buffer,
                sin_buffer,
                1e-6,
                output_buffer,
            )
            profile = runtime.synchronize()
            actual = output_buffer.download((24, 256))
            error = float(np.max(np.abs(reference - actual)))
            maximum_error = max(maximum_error, error)
            if not np.allclose(reference, actual, rtol=2e-4, atol=8e-5):
                raise SystemExit(
                    f"token {position} mismatch: max_abs={error:.9g}"
                )
            print(
                f"token={position + 1} kernel_ms={profile.kernel_ns / 1e6:.4f} "
                f"max_abs={error:.9g}"
            )

        prefix_k = np.ascontiguousarray(np.stack(k_cache[:3]))
        prefix_v = np.ascontiguousarray(np.stack(v_cache[:3]))
        runtime.reset_full_attention_state(state, prefix_k, prefix_v)
        if state.tokens != 3:
            raise SystemExit(f"reset token count mismatch: {state.tokens}")
        print(
            f"device={runtime.device_name} capacity={state.max_tokens} "
            f"tokens={state.tokens} "
            f"overall_max_abs={maximum_error:.9g} PASS"
        )
        return 0
    finally:
        state.close()
        for buffer in reversed(buffers):
            buffer.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
