#!/usr/bin/env python3
"""Validate persistent Qwen3.5 gated-delta execution on OpenCL."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def cpu_gated_delta(
    initial_state: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    state = initial_state.copy()
    output = np.empty_like(q)
    scale = np.float32(1.0 / np.sqrt(128.0))
    for token in range(q.shape[0]):
        state *= np.exp(g[token])[:, None, None]
        memory = np.einsum("hkv,hk->hv", state, k[token], optimize=True)
        delta = (v[token] - memory) * beta[token, :, None]
        state += k[token, :, :, None] * delta[:, None, :]
        output[token] = (
            np.einsum("hkv,hk->hv", state, q[token], optimize=True) * scale
        )
    return output


def normalized_vectors(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    values = rng.standard_normal(shape).astype(np.float32)
    norms = np.sqrt(np.sum(values * values, axis=-1, keepdims=True) + 1e-6)
    return np.ascontiguousarray(values / norms)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dll", type=Path, default=here / "runtime/build/nvfp4_runtime.dll"
    )
    parser.add_argument(
        "--kernel", type=Path, default=here / "kernels/nvfp4_gemv.cl"
    )
    parser.add_argument("--heads", type=int, default=48)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.heads <= 0 or args.tokens <= 0 or args.iterations <= 0:
        parser.error("heads, tokens, and iterations must be positive")

    sys.path.insert(0, str(here.parent / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime

    rng = np.random.default_rng(20260822)
    shape = (args.tokens, args.heads, 128)
    initial = np.ascontiguousarray(
        rng.standard_normal((args.heads, 128, 128)).astype(np.float32) * 0.01
    )
    q = normalized_vectors(rng, shape)
    k = normalized_vectors(rng, shape)
    v = np.ascontiguousarray(rng.standard_normal(shape).astype(np.float32) * 0.1)
    g = np.ascontiguousarray(
        -rng.uniform(0.001, 0.05, (args.tokens, args.heads)).astype(np.float32)
    )
    beta = np.ascontiguousarray(
        rng.uniform(0.1, 0.9, (args.tokens, args.heads)).astype(np.float32)
    )

    cpu_started = time.perf_counter()
    reference = cpu_gated_delta(initial, q, k, v, g, beta)
    cpu_elapsed = time.perf_counter() - cpu_started

    runtime = Runtime(args.dll.resolve(), args.kernel.resolve())
    state = runtime.create_gated_delta_state(args.heads, initial)
    buffers = []
    try:
        result = runtime.gated_delta(state, q, k, v, g, beta)
        max_abs = float(np.max(np.abs(reference - result)))
        max_rel = float(
            np.max(np.abs(reference - result) / np.maximum(np.abs(reference), 1e-6))
        )
        if not np.allclose(reference, result, rtol=8e-5, atol=2e-5):
            raise SystemExit(
                f"gated-delta mismatch: max_abs={max_abs} max_rel={max_rel}"
            )

        runtime.reset_gated_delta_state(state, initial)
        q_buffer, k_buffer, v_buffer, g_buffer, beta_buffer = (
            runtime.upload_buffer(array) for array in (q, k, v, g, beta)
        )
        buffers.extend((q_buffer, k_buffer, v_buffer, g_buffer, beta_buffer))
        output_buffer = runtime.create_buffer(q.nbytes)
        buffers.append(output_buffer)
        runtime.gated_delta_device(
            state,
            q_buffer,
            k_buffer,
            v_buffer,
            g_buffer,
            beta_buffer,
            args.tokens,
            output_buffer,
        )
        device_profile = runtime.synchronize()
        device_result = output_buffer.download(q.shape)
        device_max_abs = float(np.max(np.abs(reference - device_result)))
        if not np.allclose(reference, device_result, rtol=8e-5, atol=2e-5):
            raise SystemExit(
                f"device gated-delta mismatch: max_abs={device_max_abs}"
            )

        if args.tokens > 1:
            split = max(1, args.tokens // 2)
            runtime.reset_gated_delta_state(state, initial)
            first = runtime.gated_delta(
                state, q[:split], k[:split], v[:split], g[:split], beta[:split]
            )
            second = runtime.gated_delta(
                state, q[split:], k[split:], v[split:], g[split:], beta[split:]
            )
            split_result = np.concatenate((first, second), axis=0)
            if not np.allclose(reference, split_result, rtol=8e-5, atol=2e-5):
                raise SystemExit("persistent split execution does not match one call")

        runtime.reset_gated_delta_state(state, initial)
        runtime.gated_delta(state, q, k, v, g, beta)
        runtime.reset_gated_delta_state(state, initial)
        started = time.perf_counter()
        for _ in range(args.iterations):
            runtime.gated_delta(state, q, k, v, g, beta)
        elapsed = time.perf_counter() - started
        call_seconds = elapsed / args.iterations
        print(
            f"device={runtime.lib.nvfp4_runtime_device_name(runtime.handle).decode()} "
            f"heads={args.heads} tokens={args.tokens} state_bytes={initial.nbytes}"
        )
        print(
            f"cpu_reference_ms={cpu_elapsed * 1e3:.3f} "
            f"call_us={call_seconds * 1e6:.3f} "
            f"tokens_per_second={args.tokens / call_seconds:.3f}"
        )
        print(f"max_abs_err={max_abs:.8g} max_rel_err={max_rel:.8g}")
        print(
            f"device_resident_kernel_us={device_profile.kernel_ns / 1e3:.3f} "
            f"device_max_abs_err={device_max_abs:.8g}"
        )
        print("PASS: persistent Qwen3.5 gated-delta OpenCL matches CPU reference")
    finally:
        for buffer in reversed(buffers):
            buffer.close()
        state.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
