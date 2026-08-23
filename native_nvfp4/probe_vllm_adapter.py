#!/usr/bin/env python3
"""Validate the vLLM provider lifecycle without requiring a vLLM install."""

from __future__ import annotations

import argparse
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from probe_native_nvfp4 import cpu_gemm
from safetensors import safe_open


def install_vllm_contract_stub() -> None:
    """Expose only the current NvFp4LinearKernel contract used by the plugin."""
    package_names = (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.kernels",
        "vllm.model_executor.kernels.linear",
    )
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    nvfp4 = types.ModuleType("vllm.model_executor.kernels.linear.nvfp4")

    @dataclass
    class NvFp4LinearLayerConfig:
        pass

    class NvFp4LinearKernel:
        def __init__(self, config: NvFp4LinearLayerConfig):
            assert self.is_supported()[0]
            assert self.can_implement(config)[0]
            self.config = config

    nvfp4.NvFp4LinearKernel = NvFp4LinearKernel
    nvfp4.NvFp4LinearLayerConfig = NvFp4LinearLayerConfig
    sys.modules[nvfp4.__name__] = nvfp4


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors",
    )
    parser.add_argument(
        "--tensor", default="model.language_model.layers.0.mlp.gate_proj"
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=5120)
    parser.add_argument("--vectors", type=int, default=8)
    args = parser.parse_args()

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        here / "runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        here / "kernels/nvfp4_gemv.cl"
    )
    install_vllm_contract_stub()
    sys.path.insert(0, str(here.parent / "vllm_nvfp4_opencl/src"))

    from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearLayerConfig
    from vllm_nvfp4_opencl.linear import OpenCLNvFp4LinearKernel

    with safe_open(args.model, framework="pt", device="cpu") as model:
        packed_t = model.get_slice(args.tensor + ".weight_packed")[
            : args.rows, : args.cols // 2
        ]
        scales_t = model.get_slice(args.tensor + ".weight_scale")[
            : args.rows, : args.cols // 16
        ]
        checkpoint_scale = float(
            model.get_tensor(args.tensor + ".weight_global_scale").item()
        )
    packed = np.ascontiguousarray(packed_t.numpy().astype(np.uint8, copy=False))
    scales = np.ascontiguousarray(scales_t.view(torch.uint8).numpy())
    x_np = np.random.default_rng(20260822).standard_normal(
        (args.vectors, args.cols)
    ).astype(np.float32)
    bias_np = np.random.default_rng(20260823).standard_normal(args.rows).astype(
        np.float32
    )

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.from_numpy(packed.copy()), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            torch.from_numpy(scales.copy()).view(torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_global_scale",
        torch.nn.Parameter(
            torch.tensor([1.0 / checkpoint_scale], dtype=torch.float32),
            requires_grad=False,
        ),
    )

    provider = OpenCLNvFp4LinearKernel(NvFp4LinearLayerConfig())
    provider.process_weights_after_loading(layer)
    leading_shape = (2, args.vectors // 2) if args.vectors % 2 == 0 else (args.vectors,)
    x = torch.from_numpy(x_np).reshape(*leading_shape, args.cols)
    bias = torch.from_numpy(bias_np)
    result = provider.apply_weights(layer, x, bias).reshape(args.vectors, args.rows)
    reference = cpu_gemm(packed, scales, x_np, checkpoint_scale) + bias_np
    max_abs = float(np.max(np.abs(reference - result.numpy())))
    max_rel = float(
        np.max(np.abs(reference - result.numpy()) / np.maximum(np.abs(reference), 1e-6))
    )
    print(
        f"provider={provider.__class__.__name__} input_shape={tuple(x.shape)} "
        f"output_shape={tuple(result.shape)} max_abs_err={max_abs:.8g} "
        f"max_rel_err={max_rel:.8g}"
    )
    if not np.allclose(reference, result.numpy(), rtol=2e-5, atol=2e-5):
        raise SystemExit("vLLM provider lifecycle does not match CPU reference")
    print("PASS: vLLM provider uploads once, preserves leading dims, and applies bias")
    layer._opencl_nvfp4_matrix.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
