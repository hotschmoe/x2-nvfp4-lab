"""Thin ctypes ownership layer for the framework-neutral NVFP4 C ABI."""

from __future__ import annotations

import ctypes as C
import os
import platform
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class Profile(C.Structure):
    _fields_ = [
        ("upload_ns", C.c_uint64),
        ("kernel_ns", C.c_uint64),
        ("download_ns", C.c_uint64),
    ]


class _NativeTraceEvent(C.Structure):
    _fields_ = [
        ("scope", C.c_char * 96),
        ("operation", C.c_char * 64),
        ("queued_ns", C.c_uint64),
        ("submit_ns", C.c_uint64),
        ("start_ns", C.c_uint64),
        ("end_ns", C.c_uint64),
    ]


@dataclass(frozen=True)
class TraceEvent:
    scope: str
    operation: str
    queued_ns: int
    submit_ns: int
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def queue_delay_ns(self) -> int:
        return self.start_ns - self.queued_ns


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_artifact(env_name: str, packaged: Path, development: Path) -> Path:
    if configured := os.environ.get(env_name):
        return Path(configured).resolve()
    if packaged.exists():
        return packaged
    return development


def runtime_paths() -> tuple[Path, Path]:
    package = Path(__file__).resolve().parent
    root = _workspace_root()
    dll = _resolve_artifact(
        "VLLM_NVFP4_OPENCL_DLL",
        package / "lib/nvfp4_runtime.dll",
        root / "native_nvfp4/runtime/build/nvfp4_runtime.dll",
    )
    kernel = _resolve_artifact(
        "VLLM_NVFP4_OPENCL_KERNEL",
        package / "kernels/nvfp4_gemv.cl",
        root / "native_nvfp4/kernels/nvfp4_gemv.cl",
    )
    return dll, kernel


def provider_enabled() -> tuple[bool, str | None]:
    if os.environ.get("VLLM_NVFP4_OPENCL") != "1":
        return False, "set VLLM_NVFP4_OPENCL=1 to opt into the research backend"
    if platform.system() != "Windows" or platform.machine().lower() not in {
        "arm64",
        "aarch64",
    }:
        return False, "the current prototype requires Windows ARM64"
    missing = [str(path) for path in runtime_paths() if not path.is_file()]
    if missing:
        return False, "missing runtime artifacts: " + ", ".join(missing)
    return True, None


class NativeMatrix:
    def __init__(
        self,
        runtime: Runtime,
        handle: C.c_void_p,
        rows: int,
        cols: int,
        shared_svm: bool = False,
    ):
        self.runtime = runtime
        self.handle = handle
        self.rows = rows
        self.cols = cols
        self.shared_svm = shared_svm

    def close(self) -> None:
        if self.handle:
            self.runtime.lib.nvfp4_matrix_destroy(self.handle)
            self.handle = C.c_void_p()

    def __del__(self) -> None:
        self.close()


class Fp8Matrix:
    def __init__(self, runtime: Runtime, handle: C.c_void_p, rows: int, cols: int):
        self.runtime = runtime
        self.handle = handle
        self.rows = rows
        self.cols = cols

    def close(self) -> None:
        if self.handle:
            self.runtime.lib.fp8_matrix_destroy(self.handle)
            self.handle = C.c_void_p()

    def __del__(self) -> None:
        self.close()


class GatedDeltaState:
    def __init__(self, runtime: Runtime, handle: C.c_void_p, heads: int):
        self.runtime = runtime
        self.handle = handle
        self.heads = heads

    def close(self) -> None:
        if self.handle:
            self.runtime.lib.qwen35_gated_delta_state_destroy(self.handle)
            self.handle = C.c_void_p()

    def __del__(self) -> None:
        self.close()


class FullAttentionState:
    def __init__(self, runtime: Runtime, handle: C.c_void_p, max_tokens: int):
        self.runtime = runtime
        self.handle = handle
        self.max_tokens = max_tokens

    @property
    def tokens(self) -> int:
        if not self.handle:
            return 0
        return int(self.runtime.lib.qwen35_full_attention_state_tokens(self.handle))

    def close(self) -> None:
        if self.handle:
            self.runtime.lib.qwen35_full_attention_state_destroy(self.handle)
            self.handle = C.c_void_p()

    def __del__(self) -> None:
        self.close()


class PagedAttentionPool:
    def __init__(
        self,
        runtime: Runtime,
        handle: C.c_void_p,
        max_pages: int,
        kv_dtype: str,
        query_heads: int,
        kv_heads: int,
    ):
        self.runtime = runtime
        self.handle = handle
        self.max_pages = max_pages
        self.kv_dtype = kv_dtype
        self.query_heads = query_heads
        self.kv_heads = kv_heads

    @property
    def free_pages(self) -> int:
        if not self.handle:
            return 0
        return int(self.runtime.lib.qwen35_paged_attention_pool_free_pages(self.handle))

    @property
    def storage_bytes(self) -> int:
        if not self.handle:
            return 0
        return int(
            self.runtime.lib.qwen35_paged_attention_pool_storage_bytes(self.handle)
        )

    def close(self) -> None:
        if self.handle:
            self.runtime.lib.qwen35_paged_attention_pool_destroy(self.handle)
            self.handle = C.c_void_p()

    def __del__(self) -> None:
        self.close()


class PagedFullAttentionState:
    def __init__(
        self,
        runtime: Runtime,
        pool: PagedAttentionPool,
        handle: C.c_void_p,
        max_tokens: int,
    ):
        self.runtime = runtime
        self.pool = pool
        self.handle = handle
        self.max_tokens = max_tokens

    @property
    def tokens(self) -> int:
        if not self.handle:
            return 0
        return int(self.runtime.lib.qwen35_paged_attention_state_tokens(self.handle))

    @property
    def pages(self) -> int:
        if not self.handle:
            return 0
        return int(self.runtime.lib.qwen35_paged_attention_state_pages(self.handle))

    def close(self) -> None:
        if self.handle:
            self.runtime.lib.qwen35_paged_attention_state_destroy(self.handle)
            self.handle = C.c_void_p()

    def __del__(self) -> None:
        self.close()


class CausalConvState:
    def __init__(self, runtime: Runtime, handle: C.c_void_p, channels: int):
        self.runtime = runtime
        self.handle = handle
        self.channels = channels

    def close(self) -> None:
        if self.handle:
            self.runtime.lib.qwen35_causal_conv_state_destroy(self.handle)
            self.handle = C.c_void_p()

    def __del__(self) -> None:
        self.close()


class DeviceBuffer:
    def __init__(self, runtime: Runtime, handle: C.c_void_p, bytes_: int):
        self.runtime = runtime
        self.handle = handle
        self.bytes = bytes_

    def upload(self, array: np.ndarray, offset: int = 0) -> None:
        if not array.flags.c_contiguous:
            raise ValueError("upload array must be contiguous")
        self.runtime._check(
            self.runtime.lib.nvfp4_buffer_upload(
                self.handle,
                offset,
                C.c_void_p(array.ctypes.data),
                array.nbytes,
            ),
            "buffer_upload",
        )

    def download(
        self, shape: tuple[int, ...], dtype: np.dtype | None = None
    ) -> np.ndarray:
        if dtype is None:
            dtype = np.dtype(np.float32)
        output = np.empty(shape, dtype=dtype)
        if output.nbytes > self.bytes:
            raise ValueError("download shape exceeds device-buffer capacity")
        self.runtime._check(
            self.runtime.lib.nvfp4_buffer_download(
                self.handle,
                0,
                C.c_void_p(output.ctypes.data),
                output.nbytes,
            ),
            "buffer_download",
        )
        return output

    def close(self) -> None:
        if self.handle:
            self.runtime.lib.nvfp4_buffer_destroy(self.handle)
            self.handle = C.c_void_p()

    def __del__(self) -> None:
        self.close()


class MoeBank:
    def __init__(
        self,
        runtime: Runtime,
        handle: C.c_void_p,
        experts: int,
        hidden: int,
        intermediate: int,
    ):
        self.runtime = runtime
        self.handle = handle
        self.experts = experts
        self.hidden = hidden
        self.intermediate = intermediate

    def upload_projection(
        self,
        expert: int,
        projection: int,
        packed: np.ndarray,
        scales: np.ndarray,
        checkpoint_global_scale: float,
    ) -> None:
        if projection not in (0, 1, 2):
            raise ValueError("projection must be 0=gate, 1=up, or 2=down")
        rows, cols = (
            (self.intermediate, self.hidden)
            if projection < 2
            else (self.hidden, self.intermediate)
        )
        if (
            not 0 <= expert <= self.experts
            or packed.shape != (rows, cols // 2)
            or scales.shape != (rows, cols // 16)
            or packed.dtype != np.uint8
            or scales.dtype != np.uint8
            or not packed.flags.c_contiguous
            or not scales.flags.c_contiguous
        ):
            raise ValueError("invalid contiguous uint8 MoE projection")
        self.runtime._check(
            self.runtime.lib.nvfp4_moe_bank_upload_projection(
                self.handle,
                expert,
                projection,
                C.c_void_p(packed.ctypes.data),
                packed.nbytes,
                C.c_void_p(scales.ctypes.data),
                scales.nbytes,
                checkpoint_global_scale,
            ),
            "nvfp4_moe_bank_upload_projection",
        )

    def upload_expert(
        self,
        expert: int,
        projections: list[tuple[np.ndarray, np.ndarray, float]],
    ) -> None:
        if len(projections) != 3:
            raise ValueError("an MoE expert requires gate, up, and down projections")
        for projection, values in enumerate(projections):
            self.upload_projection(expert, projection, *values)

    def decode_device(self, x: DeviceBuffer, out: DeviceBuffer) -> DeviceBuffer:
        self.runtime._check(
            self.runtime.lib.nvfp4_moe_bank_decode_device_enqueue_f32(
                self.handle, x.handle, out.handle
            ),
            "nvfp4_moe_bank_decode_device_enqueue_f32",
        )
        return out

    def close(self) -> None:
        if self.handle:
            self.runtime.lib.nvfp4_moe_bank_destroy(self.handle)
            self.handle = C.c_void_p()

    def __del__(self) -> None:
        self.close()


class Runtime:
    def __init__(self, dll_path: Path, kernel_path: Path):
        self.lib = C.CDLL(str(dll_path))
        self.lib.nvfp4_last_error.restype = C.c_char_p
        self.lib.nvfp4_runtime_create.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        self.lib.nvfp4_runtime_create.restype = C.c_int
        self.lib.nvfp4_runtime_destroy.argtypes = [C.c_void_p]
        self.lib.nvfp4_runtime_device_name.argtypes = [C.c_void_p]
        self.lib.nvfp4_runtime_device_name.restype = C.c_char_p
        self.lib.nvfp4_runtime_last_profile.argtypes = [
            C.c_void_p,
            C.POINTER(Profile),
        ]
        self.lib.nvfp4_runtime_last_profile.restype = C.c_int
        self.lib.nvfp4_runtime_synchronize.argtypes = [C.c_void_p]
        self.lib.nvfp4_runtime_synchronize.restype = C.c_int
        self.lib.nvfp4_runtime_trace_set_enabled.argtypes = [
            C.c_void_p,
            C.c_int,
        ]
        self.lib.nvfp4_runtime_trace_set_enabled.restype = C.c_int
        self.lib.nvfp4_runtime_trace_set_scope.argtypes = [
            C.c_void_p,
            C.c_char_p,
        ]
        self.lib.nvfp4_runtime_trace_set_scope.restype = C.c_int
        self.lib.nvfp4_runtime_trace_count.argtypes = [C.c_void_p]
        self.lib.nvfp4_runtime_trace_count.restype = C.c_size_t
        self.lib.nvfp4_runtime_trace_read.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.POINTER(_NativeTraceEvent),
        ]
        self.lib.nvfp4_runtime_trace_read.restype = C.c_int
        self.lib.nvfp4_cpu_gemv_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_float,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
        ]
        self.lib.nvfp4_cpu_gemv_f32.restype = C.c_int
        self.lib.nvfp4_buffer_create.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.POINTER(C.c_void_p),
        ]
        self.lib.nvfp4_buffer_create.restype = C.c_int
        self.lib.nvfp4_buffer_destroy.argtypes = [C.c_void_p]
        self.lib.nvfp4_buffer_upload.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
        ]
        self.lib.nvfp4_buffer_upload.restype = C.c_int
        self.lib.nvfp4_buffer_download.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
        ]
        self.lib.nvfp4_buffer_download.restype = C.c_int
        self.lib.nvfp4_buffer_copy_enqueue.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
            C.c_size_t,
        ]
        self.lib.nvfp4_buffer_copy_enqueue.restype = C.c_int
        self.lib.nvfp4_matrix_upload.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
            C.c_int,
            C.c_int,
            C.c_float,
            C.POINTER(C.c_void_p),
        ]
        self.lib.nvfp4_matrix_upload.restype = C.c_int
        self.lib.nvfp4_matrix_upload_shared_svm.argtypes = (
            self.lib.nvfp4_matrix_upload.argtypes
        )
        self.lib.nvfp4_matrix_upload_shared_svm.restype = C.c_int
        self.lib.nvfp4_matrix_is_shared_svm.argtypes = [C.c_void_p]
        self.lib.nvfp4_matrix_is_shared_svm.restype = C.c_int
        self.lib.nvfp4_matrix_cpu_linear_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
        ]
        self.lib.nvfp4_matrix_cpu_linear_f32.restype = C.c_int
        self.lib.nvfp4_matrix_destroy.argtypes = [C.c_void_p]
        self.lib.nvfp4_moe_bank_create.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
            C.c_int,
            C.c_int,
            C.c_int,
            C.POINTER(C.c_void_p),
        ]
        self.lib.nvfp4_moe_bank_create.restype = C.c_int
        self.lib.nvfp4_moe_bank_upload_projection.argtypes = [
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
            C.c_float,
        ]
        self.lib.nvfp4_moe_bank_upload_projection.restype = C.c_int
        self.lib.nvfp4_moe_bank_decode_device_enqueue_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
        ]
        self.lib.nvfp4_moe_bank_decode_device_enqueue_f32.restype = C.c_int
        self.lib.nvfp4_moe_bank_destroy.argtypes = [C.c_void_p]
        self.lib.nvfp4_linear_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_void_p,
            C.c_int,
        ]
        self.lib.nvfp4_linear_f32.restype = C.c_int
        self.lib.nvfp4_linear_device_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_void_p,
            C.c_int,
        ]
        self.lib.nvfp4_linear_device_f32.restype = C.c_int
        self.lib.nvfp4_linear_device_enqueue_f32.argtypes = (
            self.lib.nvfp4_linear_device_f32.argtypes
        )
        self.lib.nvfp4_linear_device_enqueue_f32.restype = C.c_int
        self.lib.nvfp4_linear_device_lab_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_int,
        ]
        self.lib.nvfp4_linear_device_lab_f32.restype = C.c_int
        self.lib.fp8_matrix_upload.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
            C.c_int,
            C.c_int,
            C.POINTER(C.c_void_p),
        ]
        self.lib.fp8_matrix_upload.restype = C.c_int
        self.lib.fp8_matrix_upload_tensor_scaled.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_size_t,
            C.c_float,
            C.c_int,
            C.c_int,
            C.POINTER(C.c_void_p),
        ]
        self.lib.fp8_matrix_upload_tensor_scaled.restype = C.c_int
        self.lib.fp8_matrix_destroy.argtypes = [C.c_void_p]
        self.lib.fp8_linear_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_void_p,
            C.c_int,
        ]
        self.lib.fp8_linear_f32.restype = C.c_int
        self.lib.fp8_linear_device_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_void_p,
            C.c_int,
        ]
        self.lib.fp8_linear_device_f32.restype = C.c_int
        self.lib.fp8_linear_device_enqueue_f32.argtypes = (
            self.lib.fp8_linear_device_f32.argtypes
        )
        self.lib.fp8_linear_device_enqueue_f32.restype = C.c_int
        self.lib.nvfp4_add_device_enqueue_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_void_p,
        ]
        self.lib.nvfp4_add_device_enqueue_f32.restype = C.c_int
        self.lib.nvfp4_weighted_accumulate_device_enqueue_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_float,
            C.c_void_p,
            C.c_int,
            C.c_int,
        ]
        self.lib.nvfp4_weighted_accumulate_device_enqueue_f32.restype = C.c_int
        self.lib.nvfp4_silu_mul_device_enqueue_f32.argtypes = (
            self.lib.nvfp4_add_device_enqueue_f32.argtypes
        )
        self.lib.nvfp4_silu_mul_device_enqueue_f32.restype = C.c_int
        self.lib.nvfp4_rmsnorm_device_enqueue_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_float,
            C.c_void_p,
        ]
        self.lib.nvfp4_rmsnorm_device_enqueue_f32.restype = C.c_int
        self.lib.nvfp4_f32_gemv_device_enqueue.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_void_p,
        ]
        self.lib.nvfp4_f32_gemv_device_enqueue.restype = C.c_int
        self.lib.nvfp4_bf16_gemv_device_enqueue.argtypes = (
            self.lib.nvfp4_f32_gemv_device_enqueue.argtypes
        )
        self.lib.nvfp4_bf16_gemv_device_enqueue.restype = C.c_int
        self.lib.qwen35_prepare_gated_delta_decode_device_enqueue_f32.argtypes = [
            C.c_void_p
        ] * 11
        self.lib.qwen35_prepare_gated_delta_decode_device_enqueue_f32.restype = (
            C.c_int
        )
        self.lib.qwen35_prepare_gated_delta_decode_configured_enqueue_f32.argtypes = [
            *self.lib.qwen35_prepare_gated_delta_decode_device_enqueue_f32.argtypes,
            C.c_int,
            C.c_int,
        ]
        self.lib.qwen35_prepare_gated_delta_decode_configured_enqueue_f32.restype = (
            C.c_int
        )
        self.lib.nvfp4_rmsnorm_silu_gate_device_enqueue_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_float,
            C.c_void_p,
        ]
        self.lib.nvfp4_rmsnorm_silu_gate_device_enqueue_f32.restype = C.c_int
        self.lib.qwen35_full_attention_state_create.argtypes = [
            C.c_void_p,
            C.c_int,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.POINTER(C.c_void_p),
        ]
        self.lib.qwen35_full_attention_state_create.restype = C.c_int
        self.lib.qwen35_full_attention_state_destroy.argtypes = [C.c_void_p]
        self.lib.qwen35_full_attention_state_reset.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
        ]
        self.lib.qwen35_full_attention_state_reset.restype = C.c_int
        self.lib.qwen35_full_attention_state_tokens.argtypes = [C.c_void_p]
        self.lib.qwen35_full_attention_state_tokens.restype = C.c_int
        self.lib.qwen35_full_attention_decode_device_enqueue_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_float,
            C.c_void_p,
        ]
        self.lib.qwen35_full_attention_decode_device_enqueue_f32.restype = C.c_int
        self.lib.qwen35_paged_attention_pool_create.argtypes = [
            C.c_void_p,
            C.c_int,
            C.POINTER(C.c_void_p),
        ]
        self.lib.qwen35_paged_attention_pool_create.restype = C.c_int
        self.lib.qwen35_paged_attention_pool_create_with_dtype.argtypes = [
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.POINTER(C.c_void_p),
        ]
        self.lib.qwen35_paged_attention_pool_create_with_dtype.restype = C.c_int
        self.lib.qwen35_paged_attention_pool_create_configured.argtypes = [
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            C.POINTER(C.c_void_p),
        ]
        self.lib.qwen35_paged_attention_pool_create_configured.restype = C.c_int
        self.lib.qwen35_paged_attention_pool_destroy.argtypes = [C.c_void_p]
        self.lib.qwen35_paged_attention_pool_free_pages.argtypes = [C.c_void_p]
        self.lib.qwen35_paged_attention_pool_free_pages.restype = C.c_int
        self.lib.qwen35_paged_attention_pool_storage_bytes.argtypes = [C.c_void_p]
        self.lib.qwen35_paged_attention_pool_storage_bytes.restype = C.c_size_t
        self.lib.qwen35_paged_attention_state_create.argtypes = [
            C.c_void_p,
            C.c_int,
            C.POINTER(C.c_void_p),
        ]
        self.lib.qwen35_paged_attention_state_create.restype = C.c_int
        self.lib.qwen35_paged_attention_state_destroy.argtypes = [C.c_void_p]
        self.lib.qwen35_paged_attention_state_reset.argtypes = [C.c_void_p]
        self.lib.qwen35_paged_attention_state_reset.restype = C.c_int
        self.lib.qwen35_paged_attention_state_tokens.argtypes = [C.c_void_p]
        self.lib.qwen35_paged_attention_state_tokens.restype = C.c_int
        self.lib.qwen35_paged_attention_state_pages.argtypes = [C.c_void_p]
        self.lib.qwen35_paged_attention_state_pages.restype = C.c_int
        self.lib.qwen35_paged_full_attention_decode_device_enqueue_f32.argtypes = (
            self.lib.qwen35_full_attention_decode_device_enqueue_f32.argtypes
        )
        self.lib.qwen35_paged_full_attention_decode_device_enqueue_f32.restype = C.c_int
        self.lib.qwen35_gated_delta_state_create.argtypes = [
            C.c_void_p,
            C.c_int,
            C.c_void_p,
            C.POINTER(C.c_void_p),
        ]
        self.lib.qwen35_gated_delta_state_create.restype = C.c_int
        self.lib.qwen35_gated_delta_state_destroy.argtypes = [C.c_void_p]
        self.lib.qwen35_gated_delta_state_reset.argtypes = [C.c_void_p, C.c_void_p]
        self.lib.qwen35_gated_delta_state_reset.restype = C.c_int
        self.lib.qwen35_gated_delta_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_void_p,
        ]
        self.lib.qwen35_gated_delta_f32.restype = C.c_int
        self.lib.qwen35_gated_delta_device_enqueue_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_void_p,
        ]
        self.lib.qwen35_gated_delta_device_enqueue_f32.restype = C.c_int
        self.lib.qwen35_causal_conv_state_create.argtypes = [
            C.c_void_p,
            C.c_int,
            C.c_void_p,
            C.c_void_p,
            C.POINTER(C.c_void_p),
        ]
        self.lib.qwen35_causal_conv_state_create.restype = C.c_int
        self.lib.qwen35_causal_conv_state_destroy.argtypes = [C.c_void_p]
        self.lib.qwen35_causal_conv_state_reset.argtypes = [C.c_void_p, C.c_void_p]
        self.lib.qwen35_causal_conv_state_reset.restype = C.c_int
        self.lib.qwen35_causal_conv_silu_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_void_p,
        ]
        self.lib.qwen35_causal_conv_silu_f32.restype = C.c_int
        self.lib.qwen35_causal_conv_silu_device_enqueue_f32.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_void_p,
        ]
        self.lib.qwen35_causal_conv_silu_device_enqueue_f32.restype = C.c_int
        self.handle = C.c_void_p()
        self._check(
            self.lib.nvfp4_runtime_create(
                str(kernel_path).encode(), C.byref(self.handle)
            ),
            "runtime_create",
        )

    def _check(self, status: int, operation: str) -> None:
        if status:
            error = self.lib.nvfp4_last_error().decode(errors="replace")
            raise RuntimeError(f"{operation} failed ({status}): {error}")

    @property
    def device_name(self) -> str:
        value = self.lib.nvfp4_runtime_device_name(self.handle)
        return value.decode(errors="replace") if value else "unknown"

    def last_profile(self) -> Profile:
        profile = Profile()
        self._check(
            self.lib.nvfp4_runtime_last_profile(self.handle, C.byref(profile)),
            "runtime_last_profile",
        )
        return profile

    def synchronize(self) -> Profile:
        self._check(
            self.lib.nvfp4_runtime_synchronize(self.handle),
            "runtime_synchronize",
        )
        return self.last_profile()

    def set_trace_enabled(self, enabled: bool) -> None:
        self._check(
            self.lib.nvfp4_runtime_trace_set_enabled(
                self.handle, int(enabled)
            ),
            "runtime_trace_set_enabled",
        )

    def set_trace_scope(self, scope: str) -> None:
        encoded = scope.encode("utf-8")
        if len(encoded) >= 96:
            raise ValueError("trace scope must be shorter than 96 UTF-8 bytes")
        self._check(
            self.lib.nvfp4_runtime_trace_set_scope(self.handle, encoded),
            "runtime_trace_set_scope",
        )

    def trace_events(self) -> list[TraceEvent]:
        count = int(self.lib.nvfp4_runtime_trace_count(self.handle))
        events = []
        for index in range(count):
            native = _NativeTraceEvent()
            self._check(
                self.lib.nvfp4_runtime_trace_read(
                    self.handle, index, C.byref(native)
                ),
                "runtime_trace_read",
            )
            events.append(
                TraceEvent(
                    scope=bytes(native.scope).split(b"\0", 1)[0].decode(),
                    operation=bytes(native.operation)
                    .split(b"\0", 1)[0]
                    .decode(),
                    queued_ns=int(native.queued_ns),
                    submit_ns=int(native.submit_ns),
                    start_ns=int(native.start_ns),
                    end_ns=int(native.end_ns),
                )
            )
        return events

    def create_buffer(self, bytes_: int) -> DeviceBuffer:
        if bytes_ <= 0:
            raise ValueError("device-buffer size must be positive")
        handle = C.c_void_p()
        self._check(
            self.lib.nvfp4_buffer_create(self.handle, bytes_, C.byref(handle)),
            "buffer_create",
        )
        return DeviceBuffer(self, handle, bytes_)

    def upload_buffer(self, array: np.ndarray) -> DeviceBuffer:
        buffer = self.create_buffer(array.nbytes)
        try:
            buffer.upload(array)
        except Exception:
            buffer.close()
            raise
        return buffer

    def copy_buffer_device(
        self,
        source: DeviceBuffer,
        destination: DeviceBuffer,
        bytes_: int,
        source_offset: int = 0,
        destination_offset: int = 0,
    ) -> None:
        if bytes_ <= 0 or source_offset < 0 or destination_offset < 0:
            raise ValueError("buffer copy ranges must be nonnegative and nonempty")
        self._check(
            self.lib.nvfp4_buffer_copy_enqueue(
                source.handle,
                source_offset,
                destination.handle,
                destination_offset,
                bytes_,
            ),
            "buffer_copy_enqueue",
        )

    def upload(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        checkpoint_global_scale: float,
        *,
        shared_svm: bool | None = None,
    ) -> NativeMatrix:
        # Fine-grained buffer SVM is both the one-backing-store architecture and
        # a measured performance win on the X2-90 driver. None means prefer SVM
        # and retain the copied-buffer path as a capability/allocation fallback.
        use_shared = (
            os.environ.get("VLLM_NVFP4_OPENCL_SVM", "1") != "0"
            if shared_svm is None
            else shared_svm
        )
        handle = C.c_void_p()
        upload = (
            self.lib.nvfp4_matrix_upload_shared_svm
            if use_shared
            else self.lib.nvfp4_matrix_upload
        )
        status = upload(
            self.handle,
            C.c_void_p(packed.ctypes.data),
            packed.nbytes,
            C.c_void_p(scales.ctypes.data),
            scales.nbytes,
            packed.shape[0],
            packed.shape[1] * 2,
            checkpoint_global_scale,
            C.byref(handle),
        )
        if status and shared_svm is None and use_shared:
            handle = C.c_void_p()
            use_shared = False
            status = self.lib.nvfp4_matrix_upload(
                self.handle,
                C.c_void_p(packed.ctypes.data),
                packed.nbytes,
                C.c_void_p(scales.ctypes.data),
                scales.nbytes,
                packed.shape[0],
                packed.shape[1] * 2,
                checkpoint_global_scale,
                C.byref(handle),
            )
        self._check(
            status,
            "matrix_upload_shared_svm" if use_shared else "matrix_upload",
        )
        return NativeMatrix(
            self,
            handle,
            packed.shape[0],
            packed.shape[1] * 2,
            use_shared,
        )

    def create_moe_bank(
        self,
        router_bf16: np.ndarray,
        shared_gate_bf16: np.ndarray,
        intermediate: int,
    ) -> MoeBank:
        if (
            router_bf16.ndim != 2
            or router_bf16.dtype != np.uint16
            or not router_bf16.flags.c_contiguous
            or router_bf16.shape[0] < 8
            or router_bf16.shape[0] > 256
            or intermediate <= 0
            or intermediate % 16 != 0
        ):
            raise ValueError("router must be contiguous BF16 bits [8..256, hidden]")
        shared_gate_bf16 = np.ascontiguousarray(shared_gate_bf16.reshape(-1))
        if (
            shared_gate_bf16.dtype != np.uint16
            or shared_gate_bf16.size != router_bf16.shape[1]
        ):
            raise ValueError("shared gate must be BF16 bits [hidden]")
        experts, hidden = router_bf16.shape
        handle = C.c_void_p()
        self._check(
            self.lib.nvfp4_moe_bank_create(
                self.handle,
                C.c_void_p(router_bf16.ctypes.data),
                router_bf16.nbytes,
                C.c_void_p(shared_gate_bf16.ctypes.data),
                shared_gate_bf16.nbytes,
                experts,
                hidden,
                intermediate,
                C.byref(handle),
            ),
            "nvfp4_moe_bank_create",
        )
        return MoeBank(self, handle, experts, hidden, intermediate)

    def linear_shared_cpu(
        self,
        matrix: NativeMatrix,
        x: np.ndarray,
        threads: int = 0,
    ) -> np.ndarray:
        if not matrix.shared_svm:
            raise ValueError("matrix must use shared SVM storage")
        if (
            x.shape != (1, matrix.cols)
            or x.dtype != np.float32
            or not x.flags.c_contiguous
        ):
            raise ValueError("x must be contiguous float32 [1, cols]")
        if threads < 0:
            raise ValueError("threads must be nonnegative")
        out = np.empty((1, matrix.rows), dtype=np.float32)
        self._check(
            self.lib.nvfp4_matrix_cpu_linear_f32(
                matrix.handle,
                C.c_void_p(x.ctypes.data),
                C.c_void_p(out.ctypes.data),
                threads,
            ),
            "matrix_cpu_linear_f32",
        )
        return out

    def linear_cpu(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        checkpoint_global_scale: float,
        x: np.ndarray,
        threads: int = 0,
    ) -> np.ndarray:
        if (
            packed.ndim != 2
            or scales.ndim != 2
            or packed.dtype != np.uint8
            or scales.dtype != np.uint8
            or not packed.flags.c_contiguous
            or not scales.flags.c_contiguous
        ):
            raise ValueError("packed and scales must be contiguous uint8 matrices")
        rows, packed_cols = packed.shape
        cols = packed_cols * 2
        if scales.shape != (rows, cols // 16):
            raise ValueError("scale shape does not match packed NVFP4 matrix")
        if x.shape != (1, cols) or x.dtype != np.float32 or not x.flags.c_contiguous:
            raise ValueError("x must be contiguous float32 [1, cols]")
        if threads < 0:
            raise ValueError("threads must be nonnegative")
        out = np.empty((1, rows), dtype=np.float32)
        self._check(
            self.lib.nvfp4_cpu_gemv_f32(
                C.c_void_p(packed.ctypes.data),
                C.c_void_p(scales.ctypes.data),
                rows,
                cols,
                checkpoint_global_scale,
                C.c_void_p(x.ctypes.data),
                C.c_void_p(out.ctypes.data),
                threads,
            ),
            "cpu_gemv_f32",
        )
        return out

    def linear(
        self,
        matrix: NativeMatrix,
        x: np.ndarray,
        kernel_kind: int | None = None,
    ) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != matrix.cols or x.dtype != np.float32:
            raise ValueError("x must be contiguous float32 [vectors, cols]")
        out = np.empty((x.shape[0], matrix.rows), dtype=np.float32)
        kind = (
            kernel_kind
            if kernel_kind is not None
            else (2 if x.shape[0] > 1 else (3 if matrix.rows <= 1024 else 1))
        )
        self._check(
            self.lib.nvfp4_linear_f32(
                self.handle,
                matrix.handle,
                C.c_void_p(x.ctypes.data),
                x.shape[0],
                C.c_void_p(out.ctypes.data),
                kind,
            ),
            "linear_f32",
        )
        return out

    def linear_device(
        self,
        matrix: NativeMatrix,
        x: DeviceBuffer,
        vectors: int,
        out: DeviceBuffer | None = None,
        kernel_kind: int | None = None,
        enqueue: bool = False,
    ) -> DeviceBuffer:
        if vectors <= 0:
            raise ValueError("vectors must be positive")
        output_bytes = vectors * matrix.rows * np.dtype(np.float32).itemsize
        output = out if out is not None else self.create_buffer(output_bytes)
        if output.bytes < output_bytes:
            raise ValueError("output device buffer is too small")
        kind = (
            kernel_kind
            if kernel_kind is not None
            else (2 if vectors > 1 else (3 if matrix.rows <= 1024 else 1))
        )
        try:
            operation = (
                self.lib.nvfp4_linear_device_enqueue_f32
                if enqueue
                else self.lib.nvfp4_linear_device_f32
            )
            self._check(
                operation(
                    self.handle,
                    matrix.handle,
                    x.handle,
                    vectors,
                    output.handle,
                    kind,
                ),
                "linear_device_enqueue_f32" if enqueue else "linear_device_f32",
            )
        except Exception:
            if out is None:
                output.close()
            raise
        return output

    def linear_device_lab(
        self,
        matrix: NativeMatrix,
        x: DeviceBuffer,
        *,
        row_tile: int,
        k_tile: int,
        decode_kind: int,
        out: DeviceBuffer | None = None,
    ) -> DeviceBuffer:
        """Run one synchronous, decode-only experimental kernel."""
        output_bytes = matrix.rows * np.dtype(np.float32).itemsize
        output = out if out is not None else self.create_buffer(output_bytes)
        if output.bytes < output_bytes:
            raise ValueError("output device buffer is too small")
        try:
            self._check(
                self.lib.nvfp4_linear_device_lab_f32(
                    self.handle,
                    matrix.handle,
                    x.handle,
                    output.handle,
                    row_tile,
                    k_tile,
                    decode_kind,
                ),
                "linear_device_lab_f32",
            )
        except Exception:
            if out is None:
                output.close()
            raise
        return output

    def upload_fp8(self, weights: np.ndarray, scales_bf16: np.ndarray) -> Fp8Matrix:
        handle = C.c_void_p()
        self._check(
            self.lib.fp8_matrix_upload(
                self.handle,
                C.c_void_p(weights.ctypes.data),
                weights.nbytes,
                C.c_void_p(scales_bf16.ctypes.data),
                scales_bf16.nbytes,
                weights.shape[0],
                weights.shape[1],
                C.byref(handle),
            ),
            "fp8_matrix_upload",
        )
        return Fp8Matrix(self, handle, weights.shape[0], weights.shape[1])

    def upload_fp8_tensor_scaled(
        self, weights: np.ndarray, weight_scale: float
    ) -> Fp8Matrix:
        if (
            weights.ndim != 2
            or weights.dtype != np.uint8
            or not weights.flags.c_contiguous
        ):
            raise ValueError("weights must be contiguous uint8 [rows, cols]")
        if not np.isfinite(weight_scale) or weight_scale == 0:
            raise ValueError("weight_scale must be finite and nonzero")
        handle = C.c_void_p()
        self._check(
            self.lib.fp8_matrix_upload_tensor_scaled(
                self.handle,
                C.c_void_p(weights.ctypes.data),
                weights.nbytes,
                weight_scale,
                weights.shape[0],
                weights.shape[1],
                C.byref(handle),
            ),
            "fp8_matrix_upload_tensor_scaled",
        )
        return Fp8Matrix(self, handle, weights.shape[0], weights.shape[1])

    def linear_fp8(
        self,
        matrix: Fp8Matrix,
        x: np.ndarray,
        kernel_kind: int | None = None,
    ) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != matrix.cols or x.dtype != np.float32:
            raise ValueError("x must be contiguous float32 [vectors, cols]")
        out = np.empty((x.shape[0], matrix.rows), dtype=np.float32)
        kind = kernel_kind if kernel_kind is not None else (2 if x.shape[0] > 1 else 3)
        self._check(
            self.lib.fp8_linear_f32(
                self.handle,
                matrix.handle,
                C.c_void_p(x.ctypes.data),
                x.shape[0],
                C.c_void_p(out.ctypes.data),
                kind,
            ),
            "fp8_linear_f32",
        )
        return out

    def linear_fp8_device(
        self,
        matrix: Fp8Matrix,
        x: DeviceBuffer,
        vectors: int,
        out: DeviceBuffer | None = None,
        kernel_kind: int | None = None,
        enqueue: bool = False,
    ) -> DeviceBuffer:
        if vectors <= 0:
            raise ValueError("vectors must be positive")
        output_bytes = vectors * matrix.rows * np.dtype(np.float32).itemsize
        output = out if out is not None else self.create_buffer(output_bytes)
        if output.bytes < output_bytes:
            raise ValueError("output device buffer is too small")
        kind = kernel_kind if kernel_kind is not None else (2 if vectors > 1 else 3)
        try:
            operation = (
                self.lib.fp8_linear_device_enqueue_f32
                if enqueue
                else self.lib.fp8_linear_device_f32
            )
            self._check(
                operation(
                    self.handle,
                    matrix.handle,
                    x.handle,
                    vectors,
                    output.handle,
                    kind,
                ),
                "fp8_linear_device_enqueue_f32"
                if enqueue
                else "fp8_linear_device_f32",
            )
        except Exception:
            if out is None:
                output.close()
            raise
        return output

    def add_device(
        self,
        a: DeviceBuffer,
        b: DeviceBuffer,
        elements: int,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        self._check(
            self.lib.nvfp4_add_device_enqueue_f32(
                self.handle, a.handle, b.handle, elements, out.handle
            ),
            "add_device_enqueue_f32",
        )
        return out

    def silu_mul_device(
        self,
        gate: DeviceBuffer,
        up: DeviceBuffer,
        elements: int,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        self._check(
            self.lib.nvfp4_silu_mul_device_enqueue_f32(
                self.handle, gate.handle, up.handle, elements, out.handle
            ),
            "silu_mul_device_enqueue_f32",
        )
        return out

    def weighted_accumulate_device(
        self,
        source: DeviceBuffer,
        scale: float,
        out: DeviceBuffer,
        elements: int,
        *,
        reset: bool = False,
    ) -> DeviceBuffer:
        self._check(
            self.lib.nvfp4_weighted_accumulate_device_enqueue_f32(
                self.handle,
                source.handle,
                scale,
                out.handle,
                elements,
                int(reset),
            ),
            "weighted_accumulate_device_enqueue_f32",
        )
        return out

    def rmsnorm_device(
        self,
        x: DeviceBuffer,
        weight: DeviceBuffer,
        rows: int,
        cols: int,
        epsilon: float,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        self._check(
            self.lib.nvfp4_rmsnorm_device_enqueue_f32(
                self.handle,
                x.handle,
                weight.handle,
                rows,
                cols,
                epsilon,
                out.handle,
            ),
            "rmsnorm_device_enqueue_f32",
        )
        return out

    def f32_gemv_device(
        self,
        weights: DeviceBuffer,
        x: DeviceBuffer,
        rows: int,
        cols: int,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        self._check(
            self.lib.nvfp4_f32_gemv_device_enqueue(
                self.handle,
                weights.handle,
                x.handle,
                rows,
                cols,
                out.handle,
            ),
            "f32_gemv_device_enqueue",
        )
        return out

    def bf16_gemv_device(
        self,
        weights: DeviceBuffer,
        x: DeviceBuffer,
        rows: int,
        cols: int,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        self._check(
            self.lib.nvfp4_bf16_gemv_device_enqueue(
                self.handle,
                weights.handle,
                x.handle,
                rows,
                cols,
                out.handle,
            ),
            "bf16_gemv_device_enqueue",
        )
        return out

    def prepare_gated_delta_decode_device(
        self,
        mixed_qkv: DeviceBuffer,
        a: DeviceBuffer,
        b: DeviceBuffer,
        a_log: DeviceBuffer,
        dt_bias: DeviceBuffer,
        q: DeviceBuffer,
        k: DeviceBuffer,
        v: DeviceBuffer,
        g: DeviceBuffer,
        beta: DeviceBuffer,
        key_heads: int = 16,
        value_heads: int = 48,
    ) -> None:
        if (
            key_heads <= 0
            or value_heads <= 0
            or value_heads > 64
            or value_heads % key_heads
        ):
            raise ValueError("value_heads must be divisible by valid key_heads")
        self._check(
            self.lib.qwen35_prepare_gated_delta_decode_configured_enqueue_f32(
                self.handle,
                mixed_qkv.handle,
                a.handle,
                b.handle,
                a_log.handle,
                dt_bias.handle,
                q.handle,
                k.handle,
                v.handle,
                g.handle,
                beta.handle,
                key_heads,
                value_heads,
            ),
            "qwen35_prepare_gated_delta_decode_configured_enqueue_f32",
        )

    def rmsnorm_silu_gate_device(
        self,
        x: DeviceBuffer,
        gate: DeviceBuffer,
        weight: DeviceBuffer,
        rows: int,
        cols: int,
        epsilon: float,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        self._check(
            self.lib.nvfp4_rmsnorm_silu_gate_device_enqueue_f32(
                self.handle,
                x.handle,
                gate.handle,
                weight.handle,
                rows,
                cols,
                epsilon,
                out.handle,
            ),
            "rmsnorm_silu_gate_device_enqueue_f32",
        )
        return out

    @staticmethod
    def _validate_full_attention_cache(
        name: str, array: np.ndarray, max_tokens: int
    ) -> int:
        if (
            array.ndim != 3
            or array.shape[1:] != (4, 256)
            or array.shape[0] > max_tokens
            or array.dtype != np.float32
            or not array.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be contiguous float32 [tokens <= {max_tokens}, 4, 256]"
            )
        return array.shape[0]

    def create_full_attention_state(
        self,
        max_tokens: int,
        initial_k: np.ndarray | None = None,
        initial_v: np.ndarray | None = None,
    ) -> FullAttentionState:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if (initial_k is None) != (initial_v is None):
            raise ValueError("initial_k and initial_v must be supplied together")
        initial_tokens = 0
        k_pointer = C.c_void_p()
        v_pointer = C.c_void_p()
        if initial_k is not None and initial_v is not None:
            initial_tokens = self._validate_full_attention_cache(
                "initial_k", initial_k, max_tokens
            )
            v_tokens = self._validate_full_attention_cache(
                "initial_v", initial_v, max_tokens
            )
            if initial_tokens != v_tokens:
                raise ValueError("initial_k and initial_v token counts must match")
            k_pointer = C.c_void_p(initial_k.ctypes.data)
            v_pointer = C.c_void_p(initial_v.ctypes.data)
        handle = C.c_void_p()
        self._check(
            self.lib.qwen35_full_attention_state_create(
                self.handle,
                max_tokens,
                k_pointer,
                v_pointer,
                initial_tokens,
                C.byref(handle),
            ),
            "qwen35_full_attention_state_create",
        )
        return FullAttentionState(self, handle, max_tokens)

    def reset_full_attention_state(
        self,
        state: FullAttentionState,
        initial_k: np.ndarray | None = None,
        initial_v: np.ndarray | None = None,
    ) -> None:
        if (initial_k is None) != (initial_v is None):
            raise ValueError("initial_k and initial_v must be supplied together")
        initial_tokens = 0
        k_pointer = C.c_void_p()
        v_pointer = C.c_void_p()
        if initial_k is not None and initial_v is not None:
            initial_tokens = self._validate_full_attention_cache(
                "initial_k", initial_k, state.max_tokens
            )
            v_tokens = self._validate_full_attention_cache(
                "initial_v", initial_v, state.max_tokens
            )
            if initial_tokens != v_tokens:
                raise ValueError("initial_k and initial_v token counts must match")
            k_pointer = C.c_void_p(initial_k.ctypes.data)
            v_pointer = C.c_void_p(initial_v.ctypes.data)
        self._check(
            self.lib.qwen35_full_attention_state_reset(
                state.handle, k_pointer, v_pointer, initial_tokens
            ),
            "qwen35_full_attention_state_reset",
        )

    def full_attention_decode_device(
        self,
        state: FullAttentionState,
        q_proj: DeviceBuffer,
        k_proj: DeviceBuffer,
        v_proj: DeviceBuffer,
        q_norm_weight: DeviceBuffer,
        k_norm_weight: DeviceBuffer,
        cos: DeviceBuffer,
        sin: DeviceBuffer,
        epsilon: float,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        self._check(
            self.lib.qwen35_full_attention_decode_device_enqueue_f32(
                self.handle,
                state.handle,
                q_proj.handle,
                k_proj.handle,
                v_proj.handle,
                q_norm_weight.handle,
                k_norm_weight.handle,
                cos.handle,
                sin.handle,
                epsilon,
                out.handle,
            ),
            "qwen35_full_attention_decode_device_enqueue_f32",
        )
        return out

    def create_paged_attention_pool(
        self,
        max_pages: int,
        kv_dtype: str = "fp32",
        query_heads: int = 24,
        kv_heads: int = 4,
    ) -> PagedAttentionPool:
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        dtype_ids = {"fp32": 0, "bf16": 1}
        if kv_dtype not in dtype_ids:
            raise ValueError("kv_dtype must be 'fp32' or 'bf16'")
        if (
            query_heads <= 0
            or query_heads > 64
            or kv_heads <= 0
            or kv_heads > query_heads
            or query_heads % kv_heads
        ):
            raise ValueError("query_heads must be divisible by valid kv_heads")
        handle = C.c_void_p()
        self._check(
            self.lib.qwen35_paged_attention_pool_create_configured(
                self.handle,
                max_pages,
                dtype_ids[kv_dtype],
                query_heads,
                kv_heads,
                C.byref(handle),
            ),
            "qwen35_paged_attention_pool_create_configured",
        )
        return PagedAttentionPool(
            self, handle, max_pages, kv_dtype, query_heads, kv_heads
        )

    def create_paged_full_attention_state(
        self, pool: PagedAttentionPool, max_tokens: int
    ) -> PagedFullAttentionState:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        handle = C.c_void_p()
        self._check(
            self.lib.qwen35_paged_attention_state_create(
                pool.handle, max_tokens, C.byref(handle)
            ),
            "qwen35_paged_attention_state_create",
        )
        return PagedFullAttentionState(self, pool, handle, max_tokens)

    def reset_paged_full_attention_state(
        self, state: PagedFullAttentionState
    ) -> None:
        self._check(
            self.lib.qwen35_paged_attention_state_reset(state.handle),
            "qwen35_paged_attention_state_reset",
        )

    def paged_full_attention_decode_device(
        self,
        state: PagedFullAttentionState,
        q_proj: DeviceBuffer,
        k_proj: DeviceBuffer,
        v_proj: DeviceBuffer,
        q_norm_weight: DeviceBuffer,
        k_norm_weight: DeviceBuffer,
        cos: DeviceBuffer,
        sin: DeviceBuffer,
        epsilon: float,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        self._check(
            self.lib.qwen35_paged_full_attention_decode_device_enqueue_f32(
                self.handle,
                state.handle,
                q_proj.handle,
                k_proj.handle,
                v_proj.handle,
                q_norm_weight.handle,
                k_norm_weight.handle,
                cos.handle,
                sin.handle,
                epsilon,
                out.handle,
            ),
            "qwen35_paged_full_attention_decode_device_enqueue_f32",
        )
        return out

    def create_gated_delta_state(
        self, heads: int, initial_state: np.ndarray | None = None
    ) -> GatedDeltaState:
        if heads <= 0:
            raise ValueError("heads must be positive")
        initial_pointer = C.c_void_p()
        if initial_state is not None:
            expected = (heads, 128, 128)
            if (
                initial_state.shape != expected
                or initial_state.dtype != np.float32
                or not initial_state.flags.c_contiguous
            ):
                raise ValueError(f"initial_state must be contiguous float32 {expected}")
            initial_pointer = C.c_void_p(initial_state.ctypes.data)
        handle = C.c_void_p()
        self._check(
            self.lib.qwen35_gated_delta_state_create(
                self.handle, heads, initial_pointer, C.byref(handle)
            ),
            "qwen35_gated_delta_state_create",
        )
        return GatedDeltaState(self, handle, heads)

    def reset_gated_delta_state(
        self,
        state: GatedDeltaState,
        initial_state: np.ndarray | None = None,
    ) -> None:
        initial_pointer = C.c_void_p()
        if initial_state is not None:
            expected = (state.heads, 128, 128)
            if (
                initial_state.shape != expected
                or initial_state.dtype != np.float32
                or not initial_state.flags.c_contiguous
            ):
                raise ValueError(f"initial_state must be contiguous float32 {expected}")
            initial_pointer = C.c_void_p(initial_state.ctypes.data)
        self._check(
            self.lib.qwen35_gated_delta_state_reset(state.handle, initial_pointer),
            "qwen35_gated_delta_state_reset",
        )

    def gated_delta(
        self,
        state: GatedDeltaState,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        g: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        expected_vectors = (q.shape[0], state.heads, 128)
        expected_scalars = (q.shape[0], state.heads)
        if q.shape != expected_vectors or q.shape[0] == 0:
            raise ValueError(f"q must have shape [tokens, {state.heads}, 128]")
        for name, array in (("q", q), ("k", k), ("v", v)):
            if (
                array.shape != expected_vectors
                or array.dtype != np.float32
                or not array.flags.c_contiguous
            ):
                raise ValueError(
                    f"{name} must be contiguous float32 {expected_vectors}"
                )
        for name, array in (("g", g), ("beta", beta)):
            if (
                array.shape != expected_scalars
                or array.dtype != np.float32
                or not array.flags.c_contiguous
            ):
                raise ValueError(
                    f"{name} must be contiguous float32 {expected_scalars}"
                )
        out = np.empty(expected_vectors, dtype=np.float32)
        self._check(
            self.lib.qwen35_gated_delta_f32(
                self.handle,
                state.handle,
                C.c_void_p(q.ctypes.data),
                C.c_void_p(k.ctypes.data),
                C.c_void_p(v.ctypes.data),
                C.c_void_p(g.ctypes.data),
                C.c_void_p(beta.ctypes.data),
                q.shape[0],
                C.c_void_p(out.ctypes.data),
            ),
            "qwen35_gated_delta_f32",
        )
        return out

    def gated_delta_device(
        self,
        state: GatedDeltaState,
        q: DeviceBuffer,
        k: DeviceBuffer,
        v: DeviceBuffer,
        g: DeviceBuffer,
        beta: DeviceBuffer,
        tokens: int,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        self._check(
            self.lib.qwen35_gated_delta_device_enqueue_f32(
                self.handle,
                state.handle,
                q.handle,
                k.handle,
                v.handle,
                g.handle,
                beta.handle,
                tokens,
                out.handle,
            ),
            "qwen35_gated_delta_device_enqueue_f32",
        )
        return out

    def create_causal_conv_state(
        self,
        weights: np.ndarray,
        initial_state: np.ndarray | None = None,
    ) -> CausalConvState:
        if (
            weights.ndim != 2
            or weights.shape[1] != 4
            or weights.dtype != np.float32
            or not weights.flags.c_contiguous
        ):
            raise ValueError("weights must be contiguous float32 [channels, 4]")
        channels = weights.shape[0]
        initial_pointer = C.c_void_p()
        if initial_state is not None:
            expected = (channels, 4)
            if (
                initial_state.shape != expected
                or initial_state.dtype != np.float32
                or not initial_state.flags.c_contiguous
            ):
                raise ValueError(f"initial_state must be contiguous float32 {expected}")
            initial_pointer = C.c_void_p(initial_state.ctypes.data)
        handle = C.c_void_p()
        self._check(
            self.lib.qwen35_causal_conv_state_create(
                self.handle,
                channels,
                C.c_void_p(weights.ctypes.data),
                initial_pointer,
                C.byref(handle),
            ),
            "qwen35_causal_conv_state_create",
        )
        return CausalConvState(self, handle, channels)

    def reset_causal_conv_state(
        self,
        state: CausalConvState,
        initial_state: np.ndarray | None = None,
    ) -> None:
        initial_pointer = C.c_void_p()
        if initial_state is not None:
            expected = (state.channels, 4)
            if (
                initial_state.shape != expected
                or initial_state.dtype != np.float32
                or not initial_state.flags.c_contiguous
            ):
                raise ValueError(f"initial_state must be contiguous float32 {expected}")
            initial_pointer = C.c_void_p(initial_state.ctypes.data)
        self._check(
            self.lib.qwen35_causal_conv_state_reset(state.handle, initial_pointer),
            "qwen35_causal_conv_state_reset",
        )

    def causal_conv_silu(
        self, state: CausalConvState, x: np.ndarray
    ) -> np.ndarray:
        if (
            x.ndim != 2
            or x.shape[0] == 0
            or x.shape[1] != state.channels
            or x.dtype != np.float32
            or not x.flags.c_contiguous
        ):
            raise ValueError(
                f"x must be contiguous float32 [tokens, {state.channels}]"
            )
        out = np.empty_like(x)
        self._check(
            self.lib.qwen35_causal_conv_silu_f32(
                self.handle,
                state.handle,
                C.c_void_p(x.ctypes.data),
                x.shape[0],
                C.c_void_p(out.ctypes.data),
            ),
            "qwen35_causal_conv_silu_f32",
        )
        return out

    def causal_conv_silu_device(
        self,
        state: CausalConvState,
        x: DeviceBuffer,
        tokens: int,
        out: DeviceBuffer,
    ) -> DeviceBuffer:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        self._check(
            self.lib.qwen35_causal_conv_silu_device_enqueue_f32(
                self.handle,
                state.handle,
                x.handle,
                tokens,
                out.handle,
            ),
            "qwen35_causal_conv_silu_device_enqueue_f32",
        )
        return out

    def close(self) -> None:
        if self.handle:
            self.lib.nvfp4_runtime_destroy(self.handle)
            self.handle = C.c_void_p()


_runtime: Runtime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> Runtime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            enabled, reason = provider_enabled()
            if not enabled:
                raise RuntimeError(reason)
            _runtime = Runtime(*runtime_paths())
        return _runtime
