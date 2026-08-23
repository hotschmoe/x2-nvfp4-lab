#!/usr/bin/env python3
"""Validate shared cadence weights with independent per-request state."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors",
    )
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

    runtime = Runtime(*runtime_paths())
    weights = Qwen35CadenceWeights.load(runtime, args.model)
    first = weights.create_session(max_tokens=2)
    second = weights.create_session(max_tokens=2)
    rng = np.random.default_rng(20260822)
    token0 = np.ascontiguousarray(
        rng.standard_normal((1, 5120)).astype(np.float32) * np.float32(0.2)
    )
    token1 = np.ascontiguousarray(
        rng.standard_normal((1, 5120)).astype(np.float32) * np.float32(0.2)
    )
    try:
        first0, first_profile = first.step(token0)
        second0, second_profile = second.step(token0)
        independent_error = float(np.max(np.abs(first0 - second0)))
        if independent_error != 0.0:
            raise SystemExit(
                f"independent sessions diverged: max_abs={independent_error:.9g}"
            )
        first1, _ = first.step(token1)
        if first.position != 2 or second.position != 1:
            raise SystemExit("per-request token positions are not independent")
        second.reset()
        replay0, replay_profile = second.step(token0)
        replay_error = float(np.max(np.abs(first0 - replay0)))
        if replay_error != 0.0 or not np.isfinite(first1).all():
            raise SystemExit(f"reset replay mismatch: max_abs={replay_error:.9g}")
        print(
            f"device={runtime.device_name} shared_matrices={len(weights.matrices)} "
            f"sessions=2 independent_max_abs={independent_error:.9g} "
            f"reset_max_abs={replay_error:.9g}"
        )
        print(
            f"first_kernel_ms={first_profile.kernel_ns / 1e6:.3f} "
            f"second_kernel_ms={second_profile.kernel_ns / 1e6:.3f} "
            f"replay_kernel_ms={replay_profile.kernel_ns / 1e6:.3f}"
        )
        print("PASS: shared weights and per-request decode state are isolated")
        return 0
    finally:
        second.close()
        first.close()
        weights.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
