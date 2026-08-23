#!/usr/bin/env python3
"""Validate persistent Qwen3.5 width-4 causal convolution on OpenCL."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def cpu_causal_conv(
    initial_state: np.ndarray, weights: np.ndarray, x: np.ndarray
) -> np.ndarray:
    state = initial_state.copy()
    output = np.empty_like(x)
    for token in range(x.shape[0]):
        state = np.concatenate((state[:, 1:], x[token, :, None]), axis=1)
        value = np.sum(state * weights, axis=1)
        output[token] = value / (1.0 + np.exp(-value))
    return output


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dll", type=Path, default=here / "runtime/build/nvfp4_runtime.dll"
    )
    parser.add_argument(
        "--kernel", type=Path, default=here / "kernels/nvfp4_gemv.cl"
    )
    parser.add_argument("--channels", type=int, default=10240)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.channels <= 0 or args.tokens <= 0 or args.iterations <= 0:
        parser.error("channels, tokens, and iterations must be positive")

    sys.path.insert(0, str(here.parent / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime

    rng = np.random.default_rng(20260822)
    weights = np.ascontiguousarray(
        rng.standard_normal((args.channels, 4)).astype(np.float32) * 0.1
    )
    initial = np.ascontiguousarray(
        rng.standard_normal((args.channels, 4)).astype(np.float32) * 0.1
    )
    x = np.ascontiguousarray(
        rng.standard_normal((args.tokens, args.channels)).astype(np.float32)
    )
    reference = cpu_causal_conv(initial, weights, x)

    runtime = Runtime(args.dll.resolve(), args.kernel.resolve())
    state = runtime.create_causal_conv_state(weights, initial)
    buffers = []
    try:
        result = runtime.causal_conv_silu(state, x)
        max_abs = float(np.max(np.abs(reference - result)))
        max_rel = float(
            np.max(np.abs(reference - result) / np.maximum(np.abs(reference), 1e-6))
        )
        if not np.allclose(reference, result, rtol=2e-5, atol=2e-6):
            raise SystemExit(
                f"causal-convolution mismatch: max_abs={max_abs} max_rel={max_rel}"
            )

        runtime.reset_causal_conv_state(state, initial)
        input_buffer = runtime.upload_buffer(x)
        output_buffer = runtime.create_buffer(x.nbytes)
        buffers.extend((input_buffer, output_buffer))
        runtime.causal_conv_silu_device(
            state, input_buffer, args.tokens, output_buffer
        )
        device_profile = runtime.synchronize()
        device_result = output_buffer.download(x.shape)
        device_max_abs = float(np.max(np.abs(reference - device_result)))
        if not np.allclose(reference, device_result, rtol=2e-5, atol=2e-6):
            raise SystemExit(
                f"device causal-convolution mismatch: max_abs={device_max_abs}"
            )

        if args.tokens > 1:
            split = max(1, args.tokens // 2)
            runtime.reset_causal_conv_state(state, initial)
            first = runtime.causal_conv_silu(state, x[:split])
            second = runtime.causal_conv_silu(state, x[split:])
            split_result = np.concatenate((first, second), axis=0)
            if not np.allclose(reference, split_result, rtol=2e-5, atol=2e-6):
                raise SystemExit("persistent split execution does not match one call")

        runtime.reset_causal_conv_state(state, initial)
        runtime.causal_conv_silu(state, x)
        runtime.reset_causal_conv_state(state, initial)
        started = time.perf_counter()
        for _ in range(args.iterations):
            runtime.causal_conv_silu(state, x)
        call_seconds = (time.perf_counter() - started) / args.iterations
        print(
            f"device={runtime.lib.nvfp4_runtime_device_name(runtime.handle).decode()} "
            f"channels={args.channels} tokens={args.tokens} "
            f"state_and_weight_bytes={initial.nbytes + weights.nbytes}"
        )
        print(
            f"call_us={call_seconds * 1e6:.3f} "
            f"tokens_per_second={args.tokens / call_seconds:.3f}"
        )
        print(f"max_abs_err={max_abs:.8g} max_rel_err={max_rel:.8g}")
        print(
            f"device_resident_kernel_us={device_profile.kernel_ns / 1e3:.3f} "
            f"device_max_abs_err={device_max_abs:.8g}"
        )
        print("PASS: persistent Qwen3.5 causal convolution matches CPU reference")
    finally:
        for buffer in reversed(buffers):
            buffer.close()
        state.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
