#!/usr/bin/env python3
"""Validate direct compressed-tensors NVFP4 GEMV on CPU and Qualcomm OpenCL."""

from __future__ import annotations

import argparse
import ctypes as C
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

CL_SUCCESS = 0
CL_DEVICE_TYPE_GPU = 1 << 2
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_WRITE_ONLY = 1 << 1
CL_MEM_COPY_HOST_PTR = 1 << 5
CL_TRUE = 1
CL_PLATFORM_NAME = 0x0902
CL_DEVICE_NAME = 0x102B
CL_PROGRAM_BUILD_LOG = 0x1183

cl_int = C.c_int32
cl_uint = C.c_uint32
cl_ulong = C.c_uint64
cl_bool = cl_uint
cl_bitfield = cl_ulong
cl_device_type = cl_bitfield
cl_platform_id = C.c_void_p
cl_device_id = C.c_void_p
cl_context = C.c_void_p
cl_command_queue = C.c_void_p
cl_program = C.c_void_p
cl_kernel = C.c_void_p
cl_mem = C.c_void_p


def check(err: int, operation: str) -> None:
    if err != CL_SUCCESS:
        raise RuntimeError(f"{operation} failed with OpenCL error {err}")


def bind_opencl() -> C.WinDLL:
    cl = C.WinDLL("OpenCL.dll")
    cl.clGetPlatformIDs.argtypes = [cl_uint, C.POINTER(cl_platform_id), C.POINTER(cl_uint)]
    cl.clGetPlatformIDs.restype = cl_int
    cl.clGetPlatformInfo.argtypes = [cl_platform_id, cl_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
    cl.clGetPlatformInfo.restype = cl_int
    cl.clGetDeviceIDs.argtypes = [cl_platform_id, cl_device_type, cl_uint, C.POINTER(cl_device_id), C.POINTER(cl_uint)]
    cl.clGetDeviceIDs.restype = cl_int
    cl.clGetDeviceInfo.argtypes = [cl_device_id, cl_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
    cl.clGetDeviceInfo.restype = cl_int
    cl.clCreateContext.argtypes = [C.c_void_p, cl_uint, C.POINTER(cl_device_id), C.c_void_p, C.c_void_p, C.POINTER(cl_int)]
    cl.clCreateContext.restype = cl_context
    cl.clCreateCommandQueue.argtypes = [cl_context, cl_device_id, cl_bitfield, C.POINTER(cl_int)]
    cl.clCreateCommandQueue.restype = cl_command_queue
    cl.clCreateProgramWithSource.argtypes = [cl_context, cl_uint, C.POINTER(C.c_char_p), C.POINTER(C.c_size_t), C.POINTER(cl_int)]
    cl.clCreateProgramWithSource.restype = cl_program
    cl.clBuildProgram.argtypes = [cl_program, cl_uint, C.POINTER(cl_device_id), C.c_char_p, C.c_void_p, C.c_void_p]
    cl.clBuildProgram.restype = cl_int
    cl.clGetProgramBuildInfo.argtypes = [cl_program, cl_device_id, cl_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
    cl.clGetProgramBuildInfo.restype = cl_int
    cl.clCreateKernel.argtypes = [cl_program, C.c_char_p, C.POINTER(cl_int)]
    cl.clCreateKernel.restype = cl_kernel
    cl.clCreateBuffer.argtypes = [cl_context, cl_bitfield, C.c_size_t, C.c_void_p, C.POINTER(cl_int)]
    cl.clCreateBuffer.restype = cl_mem
    cl.clSetKernelArg.argtypes = [cl_kernel, cl_uint, C.c_size_t, C.c_void_p]
    cl.clSetKernelArg.restype = cl_int
    cl.clEnqueueNDRangeKernel.argtypes = [cl_command_queue, cl_kernel, cl_uint, C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t), cl_uint, C.c_void_p, C.c_void_p]
    cl.clEnqueueNDRangeKernel.restype = cl_int
    cl.clEnqueueReadBuffer.argtypes = [cl_command_queue, cl_mem, cl_bool, C.c_size_t, C.c_size_t, C.c_void_p, cl_uint, C.c_void_p, C.c_void_p]
    cl.clEnqueueReadBuffer.restype = cl_int
    cl.clFinish.argtypes = [cl_command_queue]
    cl.clFinish.restype = cl_int
    for name, typ in (("clReleaseMemObject", cl_mem), ("clReleaseKernel", cl_kernel),
                      ("clReleaseProgram", cl_program), ("clReleaseCommandQueue", cl_command_queue),
                      ("clReleaseContext", cl_context)):
        fn = getattr(cl, name)
        fn.argtypes = [typ]
        fn.restype = cl_int
    return cl


def info_string(cl: C.WinDLL, obj: C.c_void_p, param: int, device: bool) -> str:
    fn = cl.clGetDeviceInfo if device else cl.clGetPlatformInfo
    size = C.c_size_t()
    check(fn(obj, param, 0, None, C.byref(size)), "query info size")
    buf = C.create_string_buffer(size.value)
    check(fn(obj, param, size.value, buf, None), "query info")
    return buf.value.decode(errors="replace")


def select_qualcomm_gpu(cl: C.WinDLL) -> tuple[cl_platform_id, cl_device_id]:
    count = cl_uint()
    check(cl.clGetPlatformIDs(0, None, C.byref(count)), "clGetPlatformIDs")
    platforms = (cl_platform_id * count.value)()
    check(cl.clGetPlatformIDs(count, platforms, None), "clGetPlatformIDs")

    fallback = None
    for platform in platforms:
        ndev = cl_uint()
        err = cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, None, C.byref(ndev))
        if err != CL_SUCCESS or not ndev.value:
            continue
        devices = (cl_device_id * ndev.value)()
        check(cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, ndev, devices, None), "clGetDeviceIDs")
        for device in devices:
            pair = (platform, device)
            fallback = fallback or pair
            label = (info_string(cl, platform, CL_PLATFORM_NAME, False) + " " +
                     info_string(cl, device, CL_DEVICE_NAME, True)).lower()
            if "qualcomm" in label or "adreno" in label:
                return pair
    if fallback:
        return fallback
    raise RuntimeError("No OpenCL GPU device was found")


def decode_e4m3_scale(raw: np.ndarray) -> np.ndarray:
    raw = raw & np.uint8(0x7F)
    exp = (raw >> np.uint8(3)).astype(np.int32)
    man = (raw & np.uint8(7)).astype(np.float32)
    out = np.where(exp == 0, man / 512.0, (1.0 + man / 8.0) * np.exp2(exp - 7))
    out[(raw == 0) | (raw == 0x7F)] = 0.0
    return out.astype(np.float32)


E2M1 = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def cpu_gemv(packed: np.ndarray, scales: np.ndarray, x: np.ndarray, weight_global_scale: float) -> np.ndarray:
    rows, packed_cols = packed.shape
    cols = packed_cols * 2
    blocks = cols // 16
    q = packed.reshape(rows, blocks, 8)
    xv = x.reshape(blocks, 8, 2)
    local = decode_e4m3_scale(scales).reshape(rows, blocks) / np.float32(weight_global_scale)
    dots = E2M1[q & 0x0F] * xv[None, :, :, 0] + E2M1[q >> 4] * xv[None, :, :, 1]
    return np.sum(np.sum(dots, axis=2, dtype=np.float32) * local, axis=1, dtype=np.float32)


def cpu_gemm(packed: np.ndarray, scales: np.ndarray, x: np.ndarray, weight_global_scale: float) -> np.ndarray:
    return np.stack([cpu_gemv(packed, scales, vector, weight_global_scale) for vector in x])


def opencl_gemv(kernel_path: Path, packed: np.ndarray, scales: np.ndarray, x: np.ndarray,
                 weight_global_scale: float, iterations: int, warmup: int,
                 implementation: str) -> tuple[np.ndarray, str, float]:
    cl = bind_opencl()
    _, device = select_qualcomm_gpu(cl)
    device_name = info_string(cl, device, CL_DEVICE_NAME, True)
    err = cl_int()
    devs = (cl_device_id * 1)(device)
    context = cl.clCreateContext(None, 1, devs, None, None, C.byref(err))
    check(err.value, "clCreateContext")
    queue = cl.clCreateCommandQueue(context, device, 0, C.byref(err))
    check(err.value, "clCreateCommandQueue")

    program = kernel = None
    buffers: list[cl_mem] = []
    try:
        source_bytes = kernel_path.read_bytes()
        source = C.c_char_p(source_bytes)
        length = C.c_size_t(len(source_bytes))
        program = cl.clCreateProgramWithSource(context, 1, C.byref(source), C.byref(length), C.byref(err))
        check(err.value, "clCreateProgramWithSource")
        build_err = cl.clBuildProgram(program, 1, devs, b"-cl-std=CL3.0 -cl-mad-enable", None, None)
        if build_err != CL_SUCCESS:
            log_size = C.c_size_t()
            cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, None, C.byref(log_size))
            log = C.create_string_buffer(log_size.value + 1)
            cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size.value, log, None)
            raise RuntimeError(f"OpenCL build failed ({build_err}):\n{log.value.decode(errors='replace')}")
        vectors = x.shape[0]
        operation = "gemv" if vectors == 1 else "gemm"
        suffix = f"_{implementation}" if implementation != "scalar" else ""
        kernel_name = f"nvfp4_native_{operation}{suffix}".encode()
        kernel = cl.clCreateKernel(program, kernel_name, C.byref(err))
        check(err.value, "clCreateKernel")

        out = np.empty((vectors, packed.shape[0]), dtype=np.float32)
        arrays = (np.ascontiguousarray(packed), np.ascontiguousarray(scales), np.ascontiguousarray(x))
        for array in arrays:
            mem = cl.clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    array.nbytes, array.ctypes.data_as(C.c_void_p), C.byref(err))
            check(err.value, "clCreateBuffer(input)")
            buffers.append(mem)
        out_mem = cl.clCreateBuffer(context, CL_MEM_WRITE_ONLY, out.nbytes, None, C.byref(err))
        check(err.value, "clCreateBuffer(output)")
        buffers.append(out_mem)

        cols = cl_int(packed.shape[1] * 2)
        rows = cl_int(packed.shape[0])
        vector_count = cl_int(vectors)
        inv_global = C.c_float(1.0 / weight_global_scale)
        mem_args = [cl_mem(value) for value in buffers]
        args = [(C.byref(mem_args[0]), C.sizeof(cl_mem)), (C.byref(mem_args[1]), C.sizeof(cl_mem)),
                (C.byref(mem_args[2]), C.sizeof(cl_mem)), (C.byref(mem_args[3]), C.sizeof(cl_mem)),
                (C.byref(cols), C.sizeof(cols)), (C.byref(rows), C.sizeof(rows))]
        if vectors > 1:
            args.append((C.byref(vector_count), C.sizeof(vector_count)))
        args.append((C.byref(inv_global), C.sizeof(inv_global)))
        if implementation == "tiled":
            args.extend(((None, packed.shape[1]), (None, scales.shape[1])))
        for index, (value, size) in enumerate(args):
            check(cl.clSetKernelArg(kernel, index, size, value), f"clSetKernelArg({index})")

        work_dim = 1 if vectors == 1 else 2
        local_vectors = 4 if implementation == "tiled" else 1
        local = (C.c_size_t * work_dim)(64, *([local_vectors] if work_dim == 2 else []))
        global_rows = rows.value * 64 if implementation != "scalar" else ((rows.value + 63) // 64) * 64
        global_size = (C.c_size_t * work_dim)(global_rows,
                                              *([((vectors + local_vectors - 1) // local_vectors) * local_vectors]
                                                if work_dim == 2 else []))
        for _ in range(warmup):
            check(cl.clEnqueueNDRangeKernel(queue, kernel, work_dim, None, global_size, local,
                                           0, None, None), "clEnqueueNDRangeKernel(warmup)")
        check(cl.clFinish(queue), "clFinish(warmup)")
        started = time.perf_counter()
        for _ in range(iterations):
            check(cl.clEnqueueNDRangeKernel(queue, kernel, work_dim, None, global_size, local,
                                           0, None, None), "clEnqueueNDRangeKernel")
        check(cl.clFinish(queue), "clFinish")
        elapsed = time.perf_counter() - started
        check(cl.clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, out.nbytes,
                                     out.ctypes.data_as(C.c_void_p), 0, None, None), "clEnqueueReadBuffer")
        return out, device_name, elapsed
    finally:
        for mem in reversed(buffers):
            cl.clReleaseMemObject(mem)
        if kernel:
            cl.clReleaseKernel(kernel)
        if program:
            cl.clReleaseProgram(program)
        cl.clReleaseCommandQueue(queue)
        cl.clReleaseContext(context)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors")
    parser.add_argument("--tensor", default="model.language_model.layers.0.mlp.gate_proj")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--vectors", type=int, default=1)
    parser.add_argument("--kernel", choices=("scalar", "subgroup", "tiled", "both"), default="scalar")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if args.cols % 16 or args.rows < 1 or args.vectors < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("--cols must be a positive multiple of 16; rows, vectors, and iterations must be positive; warmup cannot be negative")
    if args.kernel == "tiled" and args.vectors == 1:
        parser.error("--kernel tiled requires --vectors greater than 1")

    with safe_open(args.model, framework="pt", device="cpu") as model:
        packed_t = model.get_slice(args.tensor + ".weight_packed")[:args.rows, :args.cols // 2]
        scales_t = model.get_slice(args.tensor + ".weight_scale")[:args.rows, :args.cols // 16]
        weight_global = float(model.get_tensor(args.tensor + ".weight_global_scale").item())
    packed = packed_t.numpy().astype(np.uint8, copy=False)
    scales = scales_t.view(torch.uint8).numpy()
    x = np.random.default_rng(20260822).standard_normal((args.vectors, args.cols)).astype(np.float32)

    cpu_started = time.perf_counter()
    cpu = cpu_gemm(packed, scales, x, weight_global)
    cpu_elapsed = time.perf_counter() - cpu_started
    print(f"tensor={args.tensor} rows={args.rows} cols={args.cols} vectors={args.vectors} "
          f"native_weight_bytes={packed.nbytes + scales.nbytes}")
    print(f"weight_global_scale={weight_global:g} cpu_sample={cpu[0, :4].tolist()} "
          f"cpu_seconds={cpu_elapsed:.6f}")
    if args.cpu_only:
        return 0

    implementations = ("scalar", "subgroup", "tiled") if args.kernel == "both" and args.vectors > 1 else (
        ("scalar", "subgroup") if args.kernel == "both" else (args.kernel,)
    )
    op_name = "GEMV" if args.vectors == 1 else "GEMM"
    for implementation in implementations:
        gpu, device, gpu_elapsed = opencl_gemv(
            here / "kernels/nvfp4_gemv.cl", packed, scales, x,
            weight_global, args.iterations, args.warmup, implementation)
        abs_err = np.max(np.abs(cpu - gpu))
        rel_err = np.max(np.abs(cpu - gpu) / np.maximum(np.abs(cpu), 1e-6))
        operations = 2.0*args.rows*args.cols*args.vectors*args.iterations
        print(f"implementation={implementation} device={device} warmup={args.warmup} "
              f"iterations={args.iterations} kernel_seconds={gpu_elapsed:.6f} "
              f"us_per_iteration={gpu_elapsed/args.iterations*1e6:.3f} "
              f"effective_gflops={operations/gpu_elapsed/1e9:.3f}")
        print(f"gpu_sample={gpu[0, :4].tolist()}")
        print(f"max_abs_err={abs_err:.8g} max_rel_err={rel_err:.8g}")
        if not np.allclose(cpu, gpu, rtol=2e-5, atol=2e-5):
            raise SystemExit(
                f"CPU/OpenCL native NVFP4 results do not match for {implementation}")
        print(f"PASS: {implementation} native safetensors NVFP4 {op_name} matches CPU reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
