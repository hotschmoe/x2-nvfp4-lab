#!/usr/bin/env python3
"""Benchmark one exact Qwen3.5 linear-attention decode layer on device."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from probe_causal_conv import cpu_causal_conv
from probe_gated_delta import cpu_gated_delta
from safetensors import safe_open


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--with-mlp", action="store_true")
    args = parser.parse_args()
    if args.layer < 0 or args.iterations <= 0:
        parser.error("layer must be nonnegative and iterations must be positive")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        here / "runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        here / "kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(here.parent / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.graph import (
        ResidentNvFp4Mlp,
        ResidentQwen35LinearAttention,
    )
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    prefix = f"model.language_model.layers.{args.layer}"
    linear = prefix + ".linear_attn"
    with safe_open(args.model, framework="pt", device="cpu") as checkpoint:

        def f32(name: str) -> np.ndarray:
            return np.ascontiguousarray(checkpoint.get_tensor(name).float().numpy())

        def load_fp8(name: str) -> tuple[np.ndarray, np.ndarray]:
            weight = np.ascontiguousarray(
                checkpoint.get_tensor(name + ".weight").view(torch.uint8).numpy()
            )
            scale = np.ascontiguousarray(
                checkpoint.get_tensor(name + ".weight_scale")
                .view(torch.uint16)
                .numpy()
            )
            return weight, scale

        qkv_host = load_fp8(linear + ".in_proj_qkv")
        z_host = load_fp8(linear + ".in_proj_z")
        out_host = load_fp8(linear + ".out_proj")
        input_norm_weight = f32(prefix + ".input_layernorm.weight")
        a_weight = f32(linear + ".in_proj_a.weight")
        b_weight = f32(linear + ".in_proj_b.weight")
        a_log = f32(linear + ".A_log")
        dt_bias = f32(linear + ".dt_bias")
        conv_weight = np.ascontiguousarray(
            f32(linear + ".conv1d.weight").reshape(10240, 4)
        )
        gated_norm_weight = f32(linear + ".norm.weight")
        if args.with_mlp:

            def load_nvfp4(name: str) -> tuple[np.ndarray, np.ndarray, float]:
                base = prefix + ".mlp." + name
                packed = np.ascontiguousarray(
                    checkpoint.get_tensor(base + ".weight_packed").numpy()
                )
                scales = np.ascontiguousarray(
                    checkpoint.get_tensor(base + ".weight_scale")
                    .view(torch.uint8)
                    .numpy()
                )
                global_scale = float(
                    checkpoint.get_tensor(
                        base + ".weight_global_scale"
                    ).item()
                )
                return packed, scales, global_scale

            mlp_hosts = [
                load_nvfp4(name)
                for name in ("gate_proj", "up_proj", "down_proj")
            ]
            post_attention_norm_weight = f32(
                prefix + ".post_attention_layernorm.weight"
            )

    rng = np.random.default_rng(20260822)
    x = np.ascontiguousarray(
        rng.standard_normal((1, 5120)).astype(np.float32) * 0.2
    )
    initial_recurrent = np.ascontiguousarray(
        rng.standard_normal((48, 128, 128)).astype(np.float32) * 0.01
    )
    initial_conv = np.ascontiguousarray(
        rng.standard_normal((10240, 4)).astype(np.float32) * 0.05
    )
    epsilon = 1e-6

    runtime = Runtime(*runtime_paths())
    matrices = [runtime.upload_fp8(*host) for host in (qkv_host, z_host, out_host)]
    qkv_matrix, z_matrix, out_matrix = matrices
    if args.with_mlp:
        mlp_matrices = [runtime.upload(*host) for host in mlp_hosts]
        matrices.extend(mlp_matrices)
        gate_matrix, up_matrix, down_matrix = mlp_matrices

    inverse_rms = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon)
    normalized = np.ascontiguousarray(
        x * inverse_rms * (1.0 + input_norm_weight)
    )
    mixed = runtime.linear_fp8(qkv_matrix, normalized)
    z = runtime.linear_fp8(z_matrix, normalized).reshape(48, 128)
    a = a_weight @ normalized[0]
    b = b_weight @ normalized[0]
    convolved = cpu_causal_conv(initial_conv, conv_weight, mixed)[0]
    q = np.repeat(convolved[:2048].reshape(16, 128), 3, axis=0)
    k = np.repeat(convolved[2048:4096].reshape(16, 128), 3, axis=0)
    q = np.ascontiguousarray(
        q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + epsilon)
    )
    k = np.ascontiguousarray(
        k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + epsilon)
    )
    v = np.ascontiguousarray(convolved[4096:].reshape(1, 48, 128))
    g = np.ascontiguousarray(
        (-np.exp(a_log) * np.logaddexp(0.0, a + dt_bias)).reshape(1, 48)
    )
    beta = np.ascontiguousarray((1.0 / (1.0 + np.exp(-b))).reshape(1, 48))
    recurrent = cpu_gated_delta(
        initial_recurrent, q[None], k[None], v, g, beta
    )[0]
    recurrent_inverse_rms = 1.0 / np.sqrt(
        np.mean(recurrent * recurrent, axis=-1, keepdims=True) + epsilon
    )
    gated = np.ascontiguousarray(
        recurrent
        * recurrent_inverse_rms
        * gated_norm_weight
        * (z / (1.0 + np.exp(-z)))
    )
    attention_reference = x + runtime.linear_fp8(
        out_matrix, gated.reshape(1, -1)
    )
    reference = attention_reference
    if args.with_mlp:
        mlp_inverse_rms = 1.0 / np.sqrt(
            np.mean(
                attention_reference * attention_reference,
                axis=-1,
                keepdims=True,
            )
            + epsilon
        )
        mlp_normalized = np.ascontiguousarray(
            attention_reference
            * mlp_inverse_rms
            * (1.0 + post_attention_norm_weight)
        )
        gate_reference = runtime.linear(gate_matrix, mlp_normalized)
        up_reference = runtime.linear(up_matrix, mlp_normalized)
        activation_reference = np.ascontiguousarray(
            gate_reference / (1.0 + np.exp(-gate_reference)) * up_reference
        )
        reference = attention_reference + runtime.linear(
            down_matrix, activation_reference
        )

    attention_graph = ResidentQwen35LinearAttention(
        runtime,
        qkv_matrix,
        z_matrix,
        out_matrix,
        input_norm_weight=input_norm_weight,
        a_weight=a_weight,
        b_weight=b_weight,
        a_log=a_log,
        dt_bias=dt_bias,
        conv_weight=conv_weight,
        gated_norm_weight=gated_norm_weight,
        recurrent_state=initial_recurrent,
        conv_state=initial_conv,
        epsilon=epsilon,
    )
    mlp_graph = (
        ResidentNvFp4Mlp(
            runtime,
            post_attention_norm_weight,
            gate_matrix,
            up_matrix,
            down_matrix,
            epsilon,
        )
        if args.with_mlp
        else None
    )
    x_buffer = runtime.upload_buffer(x)
    output_buffer = runtime.create_buffer(x.nbytes)
    middle_buffer = (
        runtime.create_buffer(x.nbytes) if args.with_mlp else None
    )

    def enqueue_graph() -> None:
        if mlp_graph is None or middle_buffer is None:
            attention_graph.enqueue(x_buffer, output_buffer)
        else:
            attention_graph.enqueue(x_buffer, middle_buffer)
            mlp_graph.enqueue(middle_buffer, output_buffer)

    try:
        enqueue_graph()
        graph_profile = runtime.synchronize()
        result = output_buffer.download(x.shape)
        max_abs = float(np.max(np.abs(reference - result)))
        if not np.allclose(reference, result, rtol=8e-5, atol=5e-5):
            raise SystemExit(
                f"resident linear-attention mismatch: max_abs={max_abs}"
            )

        started = time.perf_counter()
        for _ in range(args.iterations):
            enqueue_graph()
            runtime.synchronize()
        synchronized_seconds = (time.perf_counter() - started) / args.iterations

        started = time.perf_counter()
        for _ in range(args.iterations):
            enqueue_graph()
        batched_profile = runtime.synchronize()
        batched_seconds = (time.perf_counter() - started) / args.iterations
        repeated_result = output_buffer.download(x.shape)
        if not np.isfinite(repeated_result).all():
            raise SystemExit("repeated linear-attention output is non-finite")

        print(
            f"device={runtime.lib.nvfp4_runtime_device_name(runtime.handle).decode()} "
            f"layer={args.layer} type="
            f"{'linear_attention+mlp' if args.with_mlp else 'linear_attention'}"
        )
        print(
            f"graph_kernel_ms={graph_profile.kernel_ns / 1e6:.3f} "
            f"max_abs_err={max_abs:.8g}"
        )
        print(f"token_synchronized_wall_ms={synchronized_seconds * 1e3:.3f}")
        print(
            f"batched_graph_kernel_ms="
            f"{batched_profile.kernel_ns / args.iterations / 1e6:.3f} "
            f"batched_graph_wall_ms={batched_seconds * 1e3:.3f}"
        )
        print("PASS: exact Qwen3.5 linear decoder graph stayed resident")
    finally:
        if middle_buffer is not None:
            middle_buffer.close()
        output_buffer.close()
        x_buffer.close()
        if mlp_graph is not None:
            mlp_graph.close()
        attention_graph.close()
        for matrix in matrices:
            matrix.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
