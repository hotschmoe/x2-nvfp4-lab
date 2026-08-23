"""vLLM modular NVFP4 linear provider backed by the persistent C ABI."""

from __future__ import annotations

import numpy as np
import torch
from vllm.model_executor.kernels.linear.nvfp4 import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)

from .runtime import NativeMatrix, get_runtime, provider_enabled


class OpenCLNvFp4LinearKernel(NvFp4LinearKernel):
    """Native compressed-tensors W4A16 linear on Qualcomm OpenCL."""

    @classmethod
    def supports_a16(cls) -> bool:
        return True

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return provider_enabled()

    @classmethod
    def can_implement(
        cls, config: NvFp4LinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.device.type != "cpu" or layer.weight_scale.device.type != "cpu":
            raise RuntimeError("OpenCL NVFP4 currently requires CPU-resident weights")
        packed = np.ascontiguousarray(layer.weight.detach().numpy())
        scales = np.ascontiguousarray(
            layer.weight_scale.detach().view(torch.uint8).numpy()
        )
        scale_multiplier = float(layer.weight_global_scale.item())
        checkpoint_global_scale = 1.0 / scale_multiplier
        layer._opencl_nvfp4_matrix = get_runtime().upload(
            packed,
            scales,
            checkpoint_global_scale,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        matrix: NativeMatrix = layer._opencl_nvfp4_matrix
        if x.device.type != "cpu":
            raise RuntimeError("OpenCL NVFP4 currently requires CPU activations")
        if x.shape[-1] != matrix.cols:
            raise ValueError(
                f"input width {x.shape[-1]} does not match matrix width {matrix.cols}"
            )

        leading_shape = x.shape[:-1]
        x_f32 = x.detach().reshape(-1, matrix.cols).to(torch.float32).contiguous()
        result = get_runtime().linear(matrix, x_f32.numpy())
        out = torch.from_numpy(result).reshape(*leading_shape, matrix.rows)
        out = out.to(dtype=x.dtype)
        if bias is not None:
            out = out + bias
        return out


__all__ = ["OpenCLNvFp4LinearKernel"]
