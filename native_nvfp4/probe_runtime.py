#!/usr/bin/env python3
"""Exercise the persistent C ABI with an exact native safetensors matrix."""

from __future__ import annotations

import argparse
import ctypes as C
import time
from pathlib import Path

import numpy as np
import torch
from probe_native_nvfp4 import cpu_gemm
from safetensors import safe_open


class Runtime:
    def __init__(self, dll_path: Path, kernel_path: Path):
        self.lib = C.CDLL(str(dll_path))
        self.lib.nvfp4_last_error.restype = C.c_char_p
        self.lib.nvfp4_runtime_create.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        self.lib.nvfp4_runtime_create.restype = C.c_int
        self.lib.nvfp4_runtime_destroy.argtypes = [C.c_void_p]
        self.lib.nvfp4_runtime_device_name.argtypes = [C.c_void_p]
        self.lib.nvfp4_runtime_device_name.restype = C.c_char_p
        self.lib.nvfp4_matrix_upload.argtypes = [
            C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
            C.c_int, C.c_int, C.c_float, C.POINTER(C.c_void_p),
        ]
        self.lib.nvfp4_matrix_upload.restype = C.c_int
        self.lib.nvfp4_matrix_upload_shared_svm.argtypes = (
            self.lib.nvfp4_matrix_upload.argtypes
        )
        self.lib.nvfp4_matrix_upload_shared_svm.restype = C.c_int
        self.lib.nvfp4_matrix_is_shared_svm.argtypes = [C.c_void_p]
        self.lib.nvfp4_matrix_is_shared_svm.restype = C.c_int
        self.lib.nvfp4_matrix_cpu_linear_f32.argtypes = [
            C.c_void_p, C.c_void_p, C.c_void_p, C.c_int,
        ]
        self.lib.nvfp4_matrix_cpu_linear_f32.restype = C.c_int
        self.lib.nvfp4_matrix_destroy.argtypes = [C.c_void_p]
        self.lib.nvfp4_linear_f32.argtypes = [
            C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int,
        ]
        self.lib.nvfp4_linear_f32.restype = C.c_int
        self.lib.fp8_matrix_upload.argtypes = [
            C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
            C.c_int, C.c_int, C.POINTER(C.c_void_p),
        ]
        self.lib.fp8_matrix_upload.restype = C.c_int
        self.lib.fp8_matrix_destroy.argtypes = [C.c_void_p]
        self.lib.fp8_linear_f32.argtypes = [
            C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int,
        ]
        self.lib.fp8_linear_f32.restype = C.c_int
        self.handle = C.c_void_p()
        self._check(self.lib.nvfp4_runtime_create(
            str(kernel_path).encode(), C.byref(self.handle)), "runtime_create")

    def _check(self, status: int, operation: str) -> None:
        if status:
            error = self.lib.nvfp4_last_error().decode(errors="replace")
            raise RuntimeError(f"{operation} failed ({status}): {error}")

    @property
    def device_name(self) -> str:
        return self.lib.nvfp4_runtime_device_name(self.handle).decode(errors="replace")

    def upload(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        global_scale: float,
        shared_svm: bool = False,
    ) -> C.c_void_p:
        matrix = C.c_void_p()
        upload = (
            self.lib.nvfp4_matrix_upload_shared_svm
            if shared_svm
            else self.lib.nvfp4_matrix_upload
        )
        self._check(upload(
            self.handle, C.c_void_p(packed.ctypes.data), packed.nbytes,
            C.c_void_p(scales.ctypes.data), scales.nbytes,
            packed.shape[0], packed.shape[1] * 2, global_scale, C.byref(matrix)),
            "matrix_upload")
        return matrix

    def linear_shared_cpu(
        self, matrix: C.c_void_p, x: np.ndarray, rows: int, threads: int
    ) -> np.ndarray:
        out = np.empty((1, rows), dtype=np.float32)
        self._check(
            self.lib.nvfp4_matrix_cpu_linear_f32(
                matrix,
                C.c_void_p(x.ctypes.data),
                C.c_void_p(out.ctypes.data),
                threads,
            ),
            "matrix_cpu_linear_f32",
        )
        return out

    def linear(self, matrix: C.c_void_p, x: np.ndarray, rows: int, kernel: int) -> np.ndarray:
        out = np.empty((x.shape[0], rows), dtype=np.float32)
        self._check(self.lib.nvfp4_linear_f32(
            self.handle, matrix, C.c_void_p(x.ctypes.data), x.shape[0],
            C.c_void_p(out.ctypes.data), kernel), "linear_f32")
        return out

    def upload_fp8(self, weights: np.ndarray, scales: np.ndarray) -> C.c_void_p:
        matrix = C.c_void_p()
        self._check(self.lib.fp8_matrix_upload(
            self.handle, C.c_void_p(weights.ctypes.data), weights.nbytes,
            C.c_void_p(scales.ctypes.data), scales.nbytes,
            weights.shape[0], weights.shape[1], C.byref(matrix)),
            "fp8_matrix_upload")
        return matrix

    def linear_fp8(self, matrix: C.c_void_p, x: np.ndarray, rows: int, kernel: int) -> np.ndarray:
        out = np.empty((x.shape[0], rows), dtype=np.float32)
        self._check(self.lib.fp8_linear_f32(
            self.handle, matrix, C.c_void_p(x.ctypes.data), x.shape[0],
            C.c_void_p(out.ctypes.data), kernel), "fp8_linear_f32")
        return out

    def destroy_fp8_matrix(self, matrix: C.c_void_p) -> None:
        self.lib.fp8_matrix_destroy(matrix)

    def destroy_matrix(self, matrix: C.c_void_p) -> None:
        self.lib.nvfp4_matrix_destroy(matrix)

    def close(self) -> None:
        if self.handle:
            self.lib.nvfp4_runtime_destroy(self.handle)
            self.handle = C.c_void_p()


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=Path, default=here / "runtime/build/nvfp4_runtime.dll")
    parser.add_argument("--model", type=Path, default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors")
    parser.add_argument("--tensor", default="model.language_model.layers.0.mlp.gate_proj")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=5120)
    parser.add_argument("--vectors", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--kernel",
        choices=("scalar", "subgroup", "tiled", "row-tiled"),
        default="subgroup",
    )
    parser.add_argument("--shared-svm", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()
    if args.rows < 1 or args.cols < 1 or args.cols % 16 or args.vectors < 1 or args.iterations < 1:
        parser.error("invalid matrix dimensions or iteration count")

    with safe_open(args.model, framework="pt", device="cpu") as model:
        packed_t = model.get_slice(args.tensor + ".weight_packed")[:args.rows, :args.cols // 2]
        scales_t = model.get_slice(args.tensor + ".weight_scale")[:args.rows, :args.cols // 16]
        global_scale = float(model.get_tensor(args.tensor + ".weight_global_scale").item())
    packed = np.ascontiguousarray(packed_t.numpy().astype(np.uint8, copy=False))
    scales = np.ascontiguousarray(scales_t.view(torch.uint8).numpy())
    x = np.random.default_rng(20260822).standard_normal((args.vectors, args.cols)).astype(np.float32)
    reference = cpu_gemm(packed, scales, x, global_scale)

    runtime = Runtime(args.dll.resolve(), here / "kernels/nvfp4_gemv.cl")
    matrix = runtime.upload(packed, scales, global_scale, args.shared_svm)
    try:
        kind = {"scalar": 0, "subgroup": 1, "tiled": 2, "row-tiled": 3}[
            args.kernel
        ]
        runtime.linear(matrix, x, args.rows, kind)
        started = time.perf_counter()
        for _ in range(args.iterations):
            result = runtime.linear(matrix, x, args.rows, kind)
        elapsed = time.perf_counter() - started
        max_abs = float(np.max(np.abs(reference - result)))
        max_rel = float(np.max(np.abs(reference - result) / np.maximum(np.abs(reference), 1e-6)))
        print(f"device={runtime.device_name} rows={args.rows} cols={args.cols} "
              f"vectors={args.vectors} kernel={args.kernel}")
        print(f"persistent_weight_bytes={packed.nbytes + scales.nbytes} "
              f"iterations={args.iterations} call_us={elapsed/args.iterations*1e6:.3f} "
              f"max_abs_err={max_abs:.8g} max_rel_err={max_rel:.8g}")
        if not np.allclose(reference, result, rtol=2e-5, atol=2e-5):
            raise SystemExit("persistent runtime result does not match CPU reference")
        if args.shared_svm:
            cpu_result = runtime.linear_shared_cpu(
                matrix, x, args.rows, args.cpu_threads
            )
            cpu_max_abs = float(np.max(np.abs(reference - cpu_result)))
            if not np.allclose(reference, cpu_result, rtol=2e-5, atol=2e-5):
                raise SystemExit(
                    f"shared-SVM CPU result mismatch: max_abs={cpu_max_abs}"
                )
            print(
                f"shared_svm=true cpu_threads={args.cpu_threads} "
                f"cpu_max_abs_err={cpu_max_abs:.8g}"
            )
        print("PASS: persistent C ABI consumed native NVFP4 without GGUF or dequantized weights")
    finally:
        runtime.destroy_matrix(matrix)
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
