#!/usr/bin/env python3
"""Benchmark one exact Qwen3.5 full-attention decode layer on Adreno."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def rope(position: int) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.float32(position) / np.power(
        np.float32(10_000_000.0), np.arange(0, 64, 2, dtype=np.float32) / 64.0
    )
    angles = np.concatenate((frequencies, frequencies))
    return (
        np.ascontiguousarray(np.cos(angles).astype(np.float32)),
        np.ascontiguousarray(np.sin(angles).astype(np.float32)),
    )


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors",
    )
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.layer < 0 or args.tokens <= 0 or args.iterations <= 0:
        parser.error("layer must be nonnegative and token counts must be positive")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        here / "runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        here / "kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(here.parent / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.graph import ResidentQwen35FullAttention
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    prefix = f"model.language_model.layers.{args.layer}"
    attention = prefix + ".self_attn"
    with safe_open(args.model, framework="pt", device="cpu") as checkpoint:

        def f32(name: str) -> np.ndarray:
            return np.ascontiguousarray(checkpoint.get_tensor(name).float().numpy())

        def load_fp8(name: str) -> tuple[np.ndarray, np.ndarray]:
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

        matrix_hosts = [
            load_fp8(attention + "." + name)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        input_norm_weight = f32(prefix + ".input_layernorm.weight")
        q_norm_weight = f32(attention + ".q_norm.weight")
        k_norm_weight = f32(attention + ".k_norm.weight")

    runtime = Runtime(*runtime_paths())
    matrices = [runtime.upload_fp8(*host) for host in matrix_hosts]
    q_matrix, k_matrix, v_matrix, o_matrix = matrices
    max_tokens = max(args.tokens, args.iterations)
    graph = ResidentQwen35FullAttention(
        runtime,
        q_matrix,
        k_matrix,
        v_matrix,
        o_matrix,
        input_norm_weight=input_norm_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        max_tokens=max_tokens,
    )
    rng = np.random.default_rng(20260822)
    inputs = [
        np.ascontiguousarray(
            rng.standard_normal((1, 5120)).astype(np.float32) * np.float32(0.2)
        )
        for _ in range(max_tokens)
    ]
    input_buffers = [runtime.upload_buffer(x) for x in inputs]
    output_buffer = runtime.create_buffer(inputs[0].nbytes)
    rope_buffers = []
    for position in range(max_tokens):
        cos, sin = rope(position)
        rope_buffers.append((runtime.upload_buffer(cos), runtime.upload_buffer(sin)))

    k_cache: list[np.ndarray] = []
    v_cache: list[np.ndarray] = []
    maximum_error = 0.0
    try:
        for position in range(args.tokens):
            x = inputs[position]
            normalized = np.ascontiguousarray(rmsnorm(x, input_norm_weight))
            q_projected = runtime.linear_fp8(q_matrix, normalized).reshape(24, 512)
            k_projected = runtime.linear_fp8(k_matrix, normalized).reshape(4, 256)
            v_projected = runtime.linear_fp8(v_matrix, normalized).reshape(4, 256)
            cos = rope_buffers[position][0].download((64,))
            sin = rope_buffers[position][1].download((64,))
            q = apply_rope(
                rmsnorm(q_projected[:, :256], q_norm_weight), cos, sin
            )
            gate = q_projected[:, 256:]
            k_cache.append(
                apply_rope(rmsnorm(k_projected, k_norm_weight), cos, sin)
            )
            v_cache.append(v_projected)
            cached_k = np.stack(k_cache)
            cached_v = np.stack(v_cache)
            attended = np.empty((24, 256), dtype=np.float32)
            for head in range(24):
                kv_head = head // 6
                logits = cached_k[:, kv_head] @ q[head] * np.float32(0.0625)
                probability = np.exp(logits - np.max(logits))
                probability /= np.sum(probability)
                attended[head] = (probability @ cached_v[:, kv_head]) / (
                    1.0 + np.exp(-gate[head])
                )
            reference = x + runtime.linear_fp8(
                o_matrix, np.ascontiguousarray(attended.reshape(1, -1))
            )

            graph.enqueue(
                input_buffers[position],
                rope_buffers[position][0],
                rope_buffers[position][1],
                output_buffer,
            )
            profile = runtime.synchronize()
            result = output_buffer.download(x.shape)
            error = float(np.max(np.abs(reference - result)))
            maximum_error = max(maximum_error, error)
            if not np.allclose(reference, result, rtol=1e-4, atol=7e-5):
                raise SystemExit(
                    f"resident full-attention token {position} mismatch: "
                    f"max_abs={error:.9g}"
                )
            print(
                f"check_token={position + 1} kernel_ms={profile.kernel_ns / 1e6:.3f} "
                f"max_abs={error:.9g}"
            )

        graph.reset()
        started = time.perf_counter()
        for position in range(args.iterations):
            graph.enqueue(
                input_buffers[position],
                rope_buffers[position][0],
                rope_buffers[position][1],
                output_buffer,
            )
            runtime.synchronize()
        synchronized_ms = (time.perf_counter() - started) * 1e3 / args.iterations

        graph.reset()
        started = time.perf_counter()
        for position in range(args.iterations):
            graph.enqueue(
                input_buffers[position],
                rope_buffers[position][0],
                rope_buffers[position][1],
                output_buffer,
            )
        batched_profile = runtime.synchronize()
        batched_ms = (time.perf_counter() - started) * 1e3 / args.iterations
        repeated = output_buffer.download((1, 5120))
        if not np.isfinite(repeated).all():
            raise SystemExit("repeated full-attention output is non-finite")

        print(f"device={runtime.device_name} layer={args.layer} type=full_attention")
        print(f"overall_max_abs_err={maximum_error:.9g}")
        print(f"token_synchronized_wall_ms={synchronized_ms:.3f}")
        print(
            f"batched_graph_kernel_ms="
            f"{batched_profile.kernel_ns / args.iterations / 1e6:.3f} "
            f"batched_graph_wall_ms={batched_ms:.3f}"
        )
        print("PASS: exact Qwen3.5 full-attention decoder graph stayed resident")
        return 0
    finally:
        for cos_buffer, sin_buffer in reversed(rope_buffers):
            sin_buffer.close()
            cos_buffer.close()
        output_buffer.close()
        for input_buffer in reversed(input_buffers):
            input_buffer.close()
        graph.close()
        for matrix in matrices:
            matrix.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
