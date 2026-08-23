#!/usr/bin/env python3
"""Benchmark the exact Qwen3.5 three-linear/one-full attention cadence."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from bench_resident_full_attention import rope


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors",
    )
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    if args.first_layer < 0 or args.iterations <= 0:
        parser.error("first-layer must be nonnegative and iterations must be positive")

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
        ResidentQwen35DecodeCadence,
        ResidentQwen35FullAttention,
        ResidentQwen35LinearAttention,
    )
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    layer_hosts = []
    with safe_open(args.model, framework="pt", device="cpu") as checkpoint:

        def f32(name: str) -> np.ndarray:
            return np.ascontiguousarray(checkpoint.get_tensor(name).float().numpy())

        def fp8(name: str) -> tuple[np.ndarray, np.ndarray]:
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

        def nvfp4(name: str) -> tuple[np.ndarray, np.ndarray, float]:
            return (
                np.ascontiguousarray(checkpoint.get_tensor(name + ".weight_packed").numpy()),
                np.ascontiguousarray(
                    checkpoint.get_tensor(name + ".weight_scale")
                    .view(torch.uint8)
                    .numpy()
                ),
                float(checkpoint.get_tensor(name + ".weight_global_scale").item()),
            )

        for layer_index in range(args.first_layer, args.first_layer + 4):
            prefix = f"model.language_model.layers.{layer_index}"
            common = {
                "input_norm": f32(prefix + ".input_layernorm.weight"),
                "post_norm": f32(prefix + ".post_attention_layernorm.weight"),
                "mlp": [
                    nvfp4(prefix + ".mlp." + name)
                    for name in ("gate_proj", "up_proj", "down_proj")
                ],
            }
            if layer_index % 4 == 3:
                attention = prefix + ".self_attn"
                common["type"] = "full"
                common["attention"] = [
                    fp8(attention + "." + name)
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                ]
                common["q_norm"] = f32(attention + ".q_norm.weight")
                common["k_norm"] = f32(attention + ".k_norm.weight")
            else:
                attention = prefix + ".linear_attn"
                common["type"] = "linear"
                common["attention"] = [
                    fp8(attention + "." + name)
                    for name in ("in_proj_qkv", "in_proj_z", "out_proj")
                ]
                common["a_weight"] = f32(attention + ".in_proj_a.weight")
                common["b_weight"] = f32(attention + ".in_proj_b.weight")
                common["a_log"] = f32(attention + ".A_log")
                common["dt_bias"] = f32(attention + ".dt_bias")
                common["conv_weight"] = np.ascontiguousarray(
                    f32(attention + ".conv1d.weight").reshape(10240, 4)
                )
                common["gated_norm"] = f32(attention + ".norm.weight")
            layer_hosts.append(common)

    runtime = Runtime(*runtime_paths())
    matrices = []
    layers = []
    for host in layer_hosts:
        attention_matrices = [runtime.upload_fp8(*item) for item in host["attention"]]
        mlp_matrices = [runtime.upload(*item) for item in host["mlp"]]
        matrices.extend(attention_matrices)
        matrices.extend(mlp_matrices)
        if host["type"] == "full":
            attention_graph = ResidentQwen35FullAttention(
                runtime,
                *attention_matrices,
                input_norm_weight=host["input_norm"],
                q_norm_weight=host["q_norm"],
                k_norm_weight=host["k_norm"],
                max_tokens=args.iterations,
            )
        else:
            attention_graph = ResidentQwen35LinearAttention(
                runtime,
                *attention_matrices,
                input_norm_weight=host["input_norm"],
                a_weight=host["a_weight"],
                b_weight=host["b_weight"],
                a_log=host["a_log"],
                dt_bias=host["dt_bias"],
                conv_weight=host["conv_weight"],
                gated_norm_weight=host["gated_norm"],
            )
        mlp_graph = ResidentNvFp4Mlp(
            runtime, host["post_norm"], *mlp_matrices
        )
        layers.append((attention_graph, mlp_graph))

    cadence = ResidentQwen35DecodeCadence(runtime, layers)
    rng = np.random.default_rng(20260822)
    inputs = [
        np.ascontiguousarray(
            rng.standard_normal((1, 5120)).astype(np.float32) * np.float32(0.2)
        )
        for _ in range(args.iterations)
    ]
    input_buffers = [runtime.upload_buffer(x) for x in inputs]
    rope_buffers = [
        tuple(runtime.upload_buffer(item) for item in rope(position))
        for position in range(args.iterations)
    ]
    output = runtime.create_buffer(inputs[0].nbytes)
    try:
        started = time.perf_counter()
        kernel_ns = 0
        for position in range(args.iterations):
            cadence.enqueue(
                input_buffers[position],
                rope_buffers[position][0],
                rope_buffers[position][1],
                output,
            )
            kernel_ns += runtime.synchronize().kernel_ns
        synchronized_ms = (time.perf_counter() - started) * 1e3 / args.iterations
        synchronized_result = output.download((1, 5120))

        cadence.reset()
        started = time.perf_counter()
        for position in range(args.iterations):
            cadence.enqueue(
                input_buffers[position],
                rope_buffers[position][0],
                rope_buffers[position][1],
                output,
            )
        batched_profile = runtime.synchronize()
        batched_ms = (time.perf_counter() - started) * 1e3 / args.iterations
        batched_result = output.download((1, 5120))
        error = float(np.max(np.abs(synchronized_result - batched_result)))
        if not np.allclose(synchronized_result, batched_result, rtol=2e-4, atol=1e-4):
            raise SystemExit(f"cadence queue/reset mismatch: max_abs={error:.9g}")
        if not np.isfinite(batched_result).all():
            raise SystemExit("cadence output is non-finite")

        print(
            f"device={runtime.device_name} layers={args.first_layer}-"
            f"{args.first_layer + 3} cadence=linear,linear,linear,full+mlp"
        )
        print(f"reset_queue_max_abs_err={error:.9g}")
        print(f"synchronized_kernel_ms={kernel_ns / args.iterations / 1e6:.3f}")
        print(f"synchronized_wall_ms={synchronized_ms:.3f}")
        print(
            f"batched_kernel_ms={batched_profile.kernel_ns / args.iterations / 1e6:.3f} "
            f"batched_wall_ms={batched_ms:.3f}"
        )
        print("PASS: exact four-layer decode cadence stayed device-resident")
        return 0
    finally:
        output.close()
        for cos_buffer, sin_buffer in reversed(rope_buffers):
            sin_buffer.close()
            cos_buffer.close()
        for input_buffer in reversed(input_buffers):
            input_buffer.close()
        cadence.close()
        for matrix in reversed(matrices):
            matrix.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
