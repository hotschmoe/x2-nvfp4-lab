#!/usr/bin/env python3
"""Validate interleaved Qwen3.5 paged attention and page reclamation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from probe_full_attention import apply_rope, rmsnorm


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
    runtime = Runtime(*runtime_paths())
    pool = runtime.create_paged_attention_pool(4)
    states = [runtime.create_paged_full_attention_state(pool, 32) for _ in range(2)]
    q_weight = np.ascontiguousarray(rng.normal(0, 0.1, 256).astype(np.float32))
    k_weight = np.ascontiguousarray(rng.normal(0, 0.1, 256).astype(np.float32))
    buffers = []

    def upload(array: np.ndarray):
        result = runtime.upload_buffer(array)
        buffers.append(result)
        return result

    def create(elements: int):
        result = runtime.create_buffer(elements * 4)
        buffers.append(result)
        return result

    q_weight_buffer = upload(q_weight)
    k_weight_buffer = upload(k_weight)
    q_buffer = create(12288)
    k_buffer = create(1024)
    v_buffer = create(1024)
    cos_buffer = create(64)
    sin_buffer = create(64)
    output_buffer = create(24 * 256)
    references = [([], []), ([], [])]
    maximum_error = 0.0

    try:
        for position in range(18):
            for request in range(2):
                q_projected = np.ascontiguousarray(
                    rng.normal(0, 0.2, (24, 512)).astype(np.float32)
                )
                k_projected = np.ascontiguousarray(
                    rng.normal(0, 0.2, (4, 256)).astype(np.float32)
                )
                v_projected = np.ascontiguousarray(
                    rng.normal(0, 0.2, (4, 256)).astype(np.float32)
                )
                angles = np.arange(64, dtype=np.float32) * np.float32(0.013)
                angles += np.float32(position * 0.071)
                cos = np.ascontiguousarray(np.cos(angles).astype(np.float32))
                sin = np.ascontiguousarray(np.sin(angles).astype(np.float32))
                q_buffer.upload(q_projected)
                k_buffer.upload(k_projected)
                v_buffer.upload(v_projected)
                cos_buffer.upload(cos)
                sin_buffer.upload(sin)

                q = apply_rope(
                    rmsnorm(q_projected[:, :256], q_weight), cos, sin
                )
                gate = q_projected[:, 256:]
                key = apply_rope(rmsnorm(k_projected, k_weight), cos, sin)
                references[request][0].append(key)
                references[request][1].append(v_projected)
                cached_k = np.stack(references[request][0])
                cached_v = np.stack(references[request][1])
                expected = np.empty((24, 256), dtype=np.float32)
                for head in range(24):
                    kv_head = head // 6
                    logits = cached_k[:, kv_head] @ q[head] * np.float32(0.0625)
                    probability = np.exp(logits - np.max(logits))
                    probability /= np.sum(probability)
                    expected[head] = (probability @ cached_v[:, kv_head]) / (
                        1.0 + np.exp(-gate[head])
                    )

                runtime.paged_full_attention_decode_device(
                    states[request], q_buffer, k_buffer, v_buffer,
                    q_weight_buffer, k_weight_buffer, cos_buffer, sin_buffer,
                    1e-6, output_buffer,
                )
                runtime.synchronize()
                actual = output_buffer.download((24, 256))
                error = float(np.max(np.abs(expected - actual)))
                maximum_error = max(maximum_error, error)
                if not np.allclose(expected, actual, rtol=2e-4, atol=8e-5):
                    raise SystemExit(
                        f"request={request} token={position} mismatch: {error:.9g}"
                    )

        if [state.pages for state in states] != [2, 2] or pool.free_pages != 0:
            raise SystemExit("unexpected page allocation after crossing block boundary")
        runtime.reset_paged_full_attention_state(states[0])
        if states[0].tokens != 0 or states[0].pages != 0 or pool.free_pages != 2:
            raise SystemExit("reset did not return request pages to the pool")
        runtime.reset_paged_full_attention_state(states[1])
        if pool.free_pages != 4:
            raise SystemExit("page pool did not fully recover")
        print(
            f"device={runtime.device_name} requests=2 tokens_each=18 "
            f"pages=4 page_tokens=16 max_abs={maximum_error:.9g}"
        )
        print("PASS: interleaved block tables and page reclamation are exact")
        return 0
    finally:
        for state in reversed(states):
            state.close()
        for buffer in reversed(buffers):
            buffer.close()
        pool.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
