#!/usr/bin/env python3
"""Benchmark an exact Qwen3.5 NVFP4 MLP as one device-resident graph."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
    from vllm_nvfp4_opencl.graph import ResidentNvFp4Mlp
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    prefix = f"model.language_model.layers.{args.layer}"
    with safe_open(args.model, framework="pt", device="cpu") as checkpoint:
        norm_weight = np.ascontiguousarray(
            checkpoint.get_tensor(prefix + ".post_attention_layernorm.weight")
            .float()
            .numpy()
        )

        def load_matrix(name: str) -> tuple[np.ndarray, np.ndarray, float]:
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
                checkpoint.get_tensor(base + ".weight_global_scale").item()
            )
            return packed, scales, global_scale

        gate_host = load_matrix("gate_proj")
        up_host = load_matrix("up_proj")
        down_host = load_matrix("down_proj")

    hidden_size = norm_weight.size
    rng = np.random.default_rng(20260822)
    x = np.ascontiguousarray(
        rng.standard_normal((1, hidden_size)).astype(np.float32)
    )
    epsilon = 1e-6
    normalized = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon)
    normalized = np.ascontiguousarray(normalized * (1.0 + norm_weight))

    runtime = Runtime(*runtime_paths())
    matrices = [runtime.upload(*host) for host in (gate_host, up_host, down_host)]
    gate_matrix, up_matrix, down_matrix = matrices
    if gate_matrix.cols != hidden_size or up_matrix.cols != hidden_size:
        raise SystemExit("checkpoint MLP input dimensions do not match norm weight")
    if down_matrix.cols != gate_matrix.rows or down_matrix.rows != hidden_size:
        raise SystemExit("checkpoint MLP intermediate dimensions are inconsistent")

    # Synchronous calls provide an independent host-staged reference for the
    # graph plumbing. The native packed matrices are identical in both paths.
    gate_reference = runtime.linear(gate_matrix, normalized)
    up_reference = runtime.linear(up_matrix, normalized)
    activation_reference = np.ascontiguousarray(
        gate_reference / (1.0 + np.exp(-gate_reference)) * up_reference
    )
    reference = x + runtime.linear(down_matrix, activation_reference)

    x_buffer = runtime.upload_buffer(x)
    output_buffer = runtime.create_buffer(x.nbytes)
    graph = ResidentNvFp4Mlp(
        runtime,
        norm_weight,
        gate_matrix,
        up_matrix,
        down_matrix,
        epsilon,
    )

    def enqueue_graph() -> None:
        graph.enqueue(x_buffer, output_buffer)

    try:
        enqueue_graph()
        graph_profile = runtime.synchronize()
        result = output_buffer.download((1, hidden_size))
        max_abs = float(np.max(np.abs(reference - result)))
        if not np.allclose(reference, result, rtol=5e-5, atol=5e-5):
            raise SystemExit(f"resident MLP mismatch: max_abs={max_abs}")

        started = time.perf_counter()
        for _ in range(args.iterations):
            enqueue_graph()
            runtime.synchronize()
        latency_seconds = (time.perf_counter() - started) / args.iterations

        started = time.perf_counter()
        for _ in range(args.iterations):
            enqueue_graph()
        batch_profile = runtime.synchronize()
        batched_seconds = (time.perf_counter() - started) / args.iterations

        result = output_buffer.download((1, hidden_size))
        repeated_max_abs = float(np.max(np.abs(reference - result)))
        if not np.allclose(reference, result, rtol=5e-5, atol=5e-5):
            raise SystemExit(
                f"repeated resident MLP mismatch: max_abs={repeated_max_abs}"
            )

        print(
            f"device={runtime.lib.nvfp4_runtime_device_name(runtime.handle).decode()} "
            f"layer={args.layer} hidden={hidden_size} intermediate={gate_matrix.rows}"
        )
        print(
            f"graph_kernel_ms={graph_profile.kernel_ns / 1e6:.3f} "
            f"max_abs_err={max_abs:.8g}"
        )
        print(f"token_synchronized_wall_ms={latency_seconds * 1e3:.3f}")
        print(
            f"batched_graph_kernel_ms="
            f"{batch_profile.kernel_ns / args.iterations / 1e6:.3f} "
            f"batched_graph_wall_ms={batched_seconds * 1e3:.3f}"
        )
        print("PASS: exact checkpoint NVFP4 MLP stayed resident across the graph")
    finally:
        graph.close()
        output_buffer.close()
        x_buffer.close()
        for matrix in matrices:
            matrix.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
