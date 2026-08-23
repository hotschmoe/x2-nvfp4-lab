"""vLLM registration entry point for the Qualcomm OpenCL NVFP4 provider."""

from __future__ import annotations

_registered = False


def register() -> None:
    """Register the provider in every vLLM process, idempotently."""
    global _registered
    if _registered:
        return

    from vllm.model_executor.kernels.linear import register_linear_kernel
    from vllm.platforms import PlatformEnum

    from .linear import OpenCLNvFp4LinearKernel

    register_linear_kernel(
        OpenCLNvFp4LinearKernel,
        PlatformEnum.CPU,
        kernel_type="nvfp4",
    )
    register_linear_kernel(
        OpenCLNvFp4LinearKernel,
        PlatformEnum.OOT,
        kernel_type="nvfp4",
    )
    _registered = True


__all__ = ["register"]
