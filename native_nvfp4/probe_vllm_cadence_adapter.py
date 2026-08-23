#!/usr/bin/env python3
"""Exercise the torch/request lifecycle seam used by a vLLM model runner."""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors",
    )
    parser.add_argument("--kv-dtype", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()
    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        here / "runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        here / "kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(here.parent / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths
    from vllm_nvfp4_opencl.serving import Qwen35CadenceWeights
    from vllm_nvfp4_opencl.vllm_adapter import VllmCadenceAdapter

    runtime = Runtime(*runtime_paths())
    weights = Qwen35CadenceWeights.load(runtime, args.model)
    adapter = VllmCadenceAdapter(
        weights,
        max_pages=4,
        default_max_tokens=32,
        kv_dtype=args.kv_dtype,
    )
    oracle = {
        request_id: weights.create_session(32) for request_id in ("a", "b")
    }
    rng = np.random.default_rng(20260822)

    def batch() -> torch.Tensor:
        values = rng.standard_normal((2, 5120)).astype(np.float32) * np.float32(0.2)
        return torch.from_numpy(values).to(torch.bfloat16)

    maximum_error = 0.0
    try:
        first = batch()
        output, first_profile = adapter.execute(
            ["a", "b"], first, new_max_tokens={"a": 32, "b": 32}
        )
        expected = []
        for index, request_id in enumerate(("a", "b")):
            host = np.ascontiguousarray(first[index : index + 1].float().numpy())
            result, _ = oracle[request_id].step(host)
            expected.append(torch.from_numpy(result).to(torch.bfloat16))
        expected_tensor = torch.cat(expected)
        error = float(torch.max(torch.abs(output.float() - expected_tensor.float())))
        maximum_error = max(maximum_error, error)
        if not torch.allclose(
            output,
            expected_tensor,
            rtol=0 if args.kv_dtype == "fp32" else 1e-2,
            atol=0 if args.kv_dtype == "fp32" else 1e-2,
        ):
            raise SystemExit(f"initial adapter batch mismatch: {error:.9g}")

        second = batch()
        reordered, second_profile = adapter.execute(["b", "a"], second)
        for index, request_id in enumerate(("b", "a")):
            host = np.ascontiguousarray(second[index : index + 1].float().numpy())
            result, _ = oracle[request_id].step(host)
            expected_item = torch.from_numpy(result).to(torch.bfloat16)
            error = float(
                torch.max(torch.abs(reordered[index].float() - expected_item[0].float()))
            )
            maximum_error = max(maximum_error, error)
            if not torch.allclose(
                reordered[index],
                expected_item[0],
                rtol=0 if args.kv_dtype == "fp32" else 1e-2,
                atol=0 if args.kv_dtype == "fp32" else 1e-2,
            ):
                raise SystemExit(f"reordered request mismatch: {request_id} {error:.9g}")

        single = batch()[:1]
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={"b": 1},
            finished_req_ids={"a"},
            preempted_req_ids=None,
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        )
        adapter.execute_scheduler_output(scheduler_output, single)
        if adapter.request_ids != ("b",):
            raise SystemExit(f"finish lifecycle mismatch: {adapter.request_ids}")
        adapter.execute(["b", "c"], batch(), new_max_tokens={"c": 32})
        adapter.abort(["b", "c"])
        if adapter.request_ids or adapter.scheduler.free_pages != 4:
            raise SystemExit("abort did not reclaim all request pages")

        oracle["d"] = weights.create_session(32)
        oracle["e"] = weights.create_session(32)
        chunk_values = (
            rng.standard_normal((5, 5120)).astype(np.float32) * np.float32(0.2)
        )
        chunk = torch.from_numpy(chunk_values).to(torch.bfloat16)
        chunk_scheduler_output = SimpleNamespace(
            num_scheduled_tokens={"d": 3, "e": 2},
            finished_req_ids=set(),
            preempted_req_ids=None,
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        )
        chunk_output, chunk_profile = adapter.execute_scheduler_output(
            chunk_scheduler_output, chunk
        )
        chunk_expected = []
        for request_id, start, count in (("d", 0, 3), ("e", 3, 2)):
            for index in range(start, start + count):
                host = np.ascontiguousarray(chunk[index : index + 1].float().numpy())
                result, _ = oracle[request_id].step(host)
                chunk_expected.append(torch.from_numpy(result).to(torch.bfloat16))
        chunk_expected_tensor = torch.cat(chunk_expected)
        chunk_error = float(
            torch.max(torch.abs(chunk_output.float() - chunk_expected_tensor.float()))
        )
        maximum_error = max(maximum_error, chunk_error)
        if not torch.allclose(
            chunk_output,
            chunk_expected_tensor,
            rtol=0 if args.kv_dtype == "fp32" else 1e-2,
            atol=0 if args.kv_dtype == "fp32" else 1e-2,
        ):
            raise SystemExit(f"request-major prompt chunk mismatch: {chunk_error:.9g}")
        adapter.abort(["d", "e"])
        print(
            f"device={runtime.device_name} dtype={output.dtype} "
            f"kv_dtype={args.kv_dtype} pool_storage_bytes={adapter.scheduler.pool.storage_bytes} "
            f"max_abs={maximum_error:.9g} pages_reclaimed=4"
        )
        print(
            f"first_batch_kernel_ms={first_profile.kernel_ns / 1e6:.3f} "
            f"reordered_batch_kernel_ms={second_profile.kernel_ns / 1e6:.3f} "
            f"five_token_chunk_kernel_ms={chunk_profile.kernel_ns / 1e6:.3f}"
        )
        print("PASS: vLLM request lifecycle adapter preserved order and state")
        return 0
    finally:
        for session in oracle.values():
            session.close()
        adapter.close()
        weights.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
