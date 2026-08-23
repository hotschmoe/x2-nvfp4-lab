#!/usr/bin/env python3
"""Validate device-only Qwen3.5 linear-attention layout primitives."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def check(label: str, expected: np.ndarray, actual: np.ndarray) -> float:
    error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=5e-5, atol=3e-5):
        raise SystemExit(f"{label} mismatch: max_abs={error}")
    return error


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
    mixed = np.ascontiguousarray(
        rng.standard_normal(10240).astype(np.float32) * 0.2
    )
    a = np.ascontiguousarray(rng.standard_normal(48).astype(np.float32) * 0.2)
    b = np.ascontiguousarray(rng.standard_normal(48).astype(np.float32) * 0.2)
    a_log = np.ascontiguousarray(
        rng.uniform(-2.0, 1.0, 48).astype(np.float32)
    )
    dt_bias = np.ascontiguousarray(
        rng.standard_normal(48).astype(np.float32) * 0.2
    )
    q_source = mixed[:2048].reshape(16, 128)
    k_source = mixed[2048:4096].reshape(16, 128)
    q_reference = np.repeat(q_source, 3, axis=0)
    k_reference = np.repeat(k_source, 3, axis=0)
    q_reference = q_reference / np.sqrt(
        np.sum(q_reference * q_reference, axis=1, keepdims=True) + 1e-6
    )
    k_reference = k_reference / np.sqrt(
        np.sum(k_reference * k_reference, axis=1, keepdims=True) + 1e-6
    )
    v_reference = mixed[4096:].reshape(48, 128)
    beta_reference = 1.0 / (1.0 + np.exp(-b))
    g_reference = -np.exp(a_log) * np.logaddexp(0.0, a + dt_bias)

    hidden = np.ascontiguousarray(rng.standard_normal(5120).astype(np.float32))
    small_weights = np.ascontiguousarray(
        rng.standard_normal((48, 5120)).astype(np.float32) * 0.01
    )
    small_reference = small_weights @ hidden

    recurrent = np.ascontiguousarray(
        rng.standard_normal((48, 128)).astype(np.float32) * 0.1
    )
    gate = np.ascontiguousarray(
        rng.standard_normal((48, 128)).astype(np.float32) * 0.2
    )
    norm_weight = np.ascontiguousarray(
        rng.standard_normal(128).astype(np.float32)
    )
    inverse_rms = 1.0 / np.sqrt(
        np.mean(recurrent * recurrent, axis=1, keepdims=True) + 1e-6
    )
    gated_reference = (
        recurrent
        * inverse_rms
        * norm_weight
        * (gate / (1.0 + np.exp(-gate)))
    )

    runtime = Runtime(*runtime_paths())
    buffers = []

    def upload(array: np.ndarray):
        buffer = runtime.upload_buffer(array)
        buffers.append(buffer)
        return buffer

    def create(elements: int):
        buffer = runtime.create_buffer(elements * np.dtype(np.float32).itemsize)
        buffers.append(buffer)
        return buffer

    try:
        mixed_buffer = upload(mixed)
        a_buffer = upload(a)
        b_buffer = upload(b)
        a_log_buffer = upload(a_log)
        dt_bias_buffer = upload(dt_bias)
        q_buffer = create(48 * 128)
        k_buffer = create(48 * 128)
        v_buffer = create(48 * 128)
        g_buffer = create(48)
        beta_buffer = create(48)
        runtime.prepare_gated_delta_decode_device(
            mixed_buffer,
            a_buffer,
            b_buffer,
            a_log_buffer,
            dt_bias_buffer,
            q_buffer,
            k_buffer,
            v_buffer,
            g_buffer,
            beta_buffer,
        )
        prepare_profile = runtime.synchronize()
        errors = {
            "q": check("q", q_reference, q_buffer.download((48, 128))),
            "k": check("k", k_reference, k_buffer.download((48, 128))),
            "v": check("v", v_reference, v_buffer.download((48, 128))),
            "g": check("g", g_reference, g_buffer.download((48,))),
            "beta": check(
                "beta", beta_reference, beta_buffer.download((48,))
            ),
        }

        weights_buffer = upload(small_weights)
        hidden_buffer = upload(hidden)
        small_output = create(48)
        runtime.f32_gemv_device(
            weights_buffer, hidden_buffer, 48, 5120, small_output
        )
        gemv_profile = runtime.synchronize()
        gemv_error = check(
            "f32_gemv", small_reference, small_output.download((48,))
        )

        recurrent_buffer = upload(recurrent)
        gate_buffer = upload(gate)
        norm_weight_buffer = upload(norm_weight)
        gated_output = create(48 * 128)
        runtime.rmsnorm_silu_gate_device(
            recurrent_buffer,
            gate_buffer,
            norm_weight_buffer,
            48,
            128,
            1e-6,
            gated_output,
        )
        gated_profile = runtime.synchronize()
        gated_error = check(
            "gated_rmsnorm", gated_reference, gated_output.download((48, 128))
        )

        print(
            f"device={runtime.lib.nvfp4_runtime_device_name(runtime.handle).decode()}"
        )
        print(
            f"prepare_qkv_us={prepare_profile.kernel_ns / 1e3:.3f} "
            f"max_abs_err={max(errors.values()):.8g}"
        )
        print(
            f"small_f32_gemv_us={gemv_profile.kernel_ns / 1e3:.3f} "
            f"max_abs_err={gemv_error:.8g}"
        )
        print(
            f"gated_rmsnorm_us={gated_profile.kernel_ns / 1e3:.3f} "
            f"max_abs_err={gated_error:.8g}"
        )
        print("PASS: device-only Qwen3.5 linear-attention transforms match NumPy")
    finally:
        for buffer in reversed(buffers):
            buffer.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
