"""Bandwidth-island profiler for the native Snapdragon NVFP4 runtime.

Each invocation measures one isolation-friendly cell and writes one versioned
JSON result.  Missing denominators stay null: a raw GPU read is a candidate
island ceiling, while a physical-DRAM claim requires counters we do not yet
have.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
import platform
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
KERNEL = ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
RESULTS = ROOT / "campaign_results/bandwidth-first"
NOMINAL_GBS = 228.0


def system_model() -> str:
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\BIOS",
            ) as key:
                manufacturer = winreg.QueryValueEx(key, "SystemManufacturer")[0]
                product = winreg.QueryValueEx(key, "SystemProductName")[0]
                return f"{manufacturer} {product}".strip()
        except OSError:
            pass
    return platform.uname().machine


class DeviceInfo(C.Structure):
    _fields_ = [
        ("platform_name", C.c_char * 256),
        ("device_name", C.c_char * 256),
        ("device_version", C.c_char * 256),
        ("driver_version", C.c_char * 256),
        ("global_memory_bytes", C.c_uint64),
        ("max_allocation_bytes", C.c_uint64),
        ("global_cache_bytes", C.c_uint64),
        ("local_memory_bytes", C.c_uint64),
        ("compute_units", C.c_uint32),
        ("max_clock_mhz", C.c_uint32),
        ("svm_capabilities", C.c_uint32),
    ]


class MemoryStatus(C.Structure):
    _fields_ = [
        ("length", C.c_uint32),
        ("memory_load", C.c_uint32),
        ("total_physical", C.c_uint64),
        ("available_physical", C.c_uint64),
        ("total_page_file", C.c_uint64),
        ("available_page_file", C.c_uint64),
        ("total_virtual", C.c_uint64),
        ("available_virtual", C.c_uint64),
        ("available_extended_virtual", C.c_uint64),
    ]


class PowerStatus(C.Structure):
    _fields_ = [
        ("ac_line_status", C.c_ubyte),
        ("battery_flag", C.c_ubyte),
        ("battery_percent", C.c_ubyte),
        ("system_status_flag", C.c_ubyte),
        ("battery_life_time", C.c_uint32),
        ("battery_full_life_time", C.c_uint32),
    ]


def decoded(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8", "replace")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cpu_sets() -> list[dict[str, int]]:
    if os.name != "nt":
        return []
    kernel32 = C.WinDLL("kernel32", use_last_error=True)
    function = kernel32.GetSystemCpuSetInformation
    function.argtypes = [C.c_void_p, C.c_ulong, C.POINTER(C.c_ulong), C.c_void_p, C.c_ulong]
    function.restype = C.c_bool
    needed = C.c_ulong()
    function(None, 0, C.byref(needed), None, 0)
    if needed.value == 0:
        return []
    storage = C.create_string_buffer(needed.value)
    if not function(storage, needed, C.byref(needed), None, 0):
        return []
    result: list[dict[str, int]] = []
    offset = 0
    while offset + 20 <= needed.value:
        size = int.from_bytes(storage[offset : offset + 4], "little")
        kind = int.from_bytes(storage[offset + 4 : offset + 8], "little")
        if size < 20 or offset + size > needed.value:
            break
        if kind == 0:
            group = int.from_bytes(storage[offset + 12 : offset + 14], "little")
            logical = int.from_bytes(storage[offset + 14 : offset + 15], "little")
            result.append(
                {
                    "id": int.from_bytes(storage[offset + 8 : offset + 12], "little"),
                    "group": group,
                    "logical_processor": logical,
                    "global_index": group * 64 + logical,
                    "core": int.from_bytes(storage[offset + 15 : offset + 16], "little"),
                    "last_level_cache": int.from_bytes(
                        storage[offset + 16 : offset + 17], "little"
                    ),
                    "numa_node": int.from_bytes(storage[offset + 17 : offset + 18], "little"),
                    "efficiency_class": int.from_bytes(
                        storage[offset + 18 : offset + 19], "little"
                    ),
                }
            )
        offset += size
    return result


def memory_status() -> MemoryStatus:
    status = MemoryStatus()
    status.length = C.sizeof(status)
    if os.name == "nt" and C.windll.kernel32.GlobalMemoryStatusEx(C.byref(status)):
        return status
    return status


def power_status() -> dict[str, Any]:
    status = PowerStatus()
    if os.name != "nt" or not C.windll.kernel32.GetSystemPowerStatus(C.byref(status)):
        return {"power_source": "unknown", "battery_percent": None}
    source = {0: "battery", 1: "AC"}.get(status.ac_line_status, "unknown")
    percent = None if status.battery_percent == 255 else int(status.battery_percent)
    return {"power_source": source, "battery_percent": percent}


def affinity_mask(cpu_set: str) -> tuple[int, list[int]]:
    if not cpu_set:
        return 0, []
    cpus = sorted({int(value) for value in cpu_set.split(",") if value.strip()})
    if any(cpu < 0 or cpu >= 64 for cpu in cpus):
        raise ValueError("CPU indices must be in [0, 63]")
    return sum(1 << cpu for cpu in cpus), cpus


class NativeRuntime:
    def __init__(self, dll_path: Path = RUNTIME, kernel_path: Path = KERNEL):
        self.lib = C.CDLL(str(dll_path))
        self.lib.nvfp4_last_error.restype = C.c_char_p
        self.lib.nvfp4_runtime_create.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        self.lib.nvfp4_runtime_create.restype = C.c_int
        self.lib.nvfp4_runtime_destroy.argtypes = [C.c_void_p]
        self.lib.nvfp4_runtime_query_device.argtypes = [C.c_void_p, C.POINTER(DeviceInfo)]
        self.lib.nvfp4_runtime_query_device.restype = C.c_int
        self.lib.nvfp4_cpu_stream_read.argtypes = [
            C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_uint64,
            C.POINTER(C.c_uint64), C.POINTER(C.c_uint64),
        ]
        self.lib.nvfp4_cpu_stream_read.restype = C.c_int
        self.lib.nvfp4_bandwidth_buffer_create.argtypes = [
            C.c_void_p, C.c_size_t, C.c_int, C.POINTER(C.c_void_p),
        ]
        self.lib.nvfp4_bandwidth_buffer_create.restype = C.c_int
        self.lib.nvfp4_bandwidth_buffer_destroy.argtypes = [C.c_void_p]
        self.lib.nvfp4_bandwidth_gpu_read.argtypes = [
            C.c_void_p, C.c_int, C.c_int,
            C.POINTER(C.c_uint64), C.POINTER(C.c_uint64),
        ]
        self.lib.nvfp4_bandwidth_gpu_read.restype = C.c_int
        self.lib.nvfp4_bandwidth_cpu_read.argtypes = [
            C.c_void_p, C.c_int, C.c_int, C.c_uint64,
            C.POINTER(C.c_uint64), C.POINTER(C.c_uint64),
        ]
        self.lib.nvfp4_bandwidth_cpu_read.restype = C.c_int
        self.handle = C.c_void_p()
        self.check(
            self.lib.nvfp4_runtime_create(
                os.fsencode(kernel_path), C.byref(self.handle)
            ),
            "runtime_create",
        )

    def check(self, status: int, operation: str) -> None:
        if status:
            detail = self.lib.nvfp4_last_error()
            message = detail.decode("utf-8", "replace") if detail else "unknown error"
            raise RuntimeError(f"{operation}: {message}")

    def device_info(self) -> DeviceInfo:
        result = DeviceInfo()
        self.check(
            self.lib.nvfp4_runtime_query_device(self.handle, C.byref(result)),
            "query_device",
        )
        return result

    def create_buffer(self, bytes_: int, shared: bool) -> C.c_void_p:
        result = C.c_void_p()
        self.check(
            self.lib.nvfp4_bandwidth_buffer_create(
                self.handle, bytes_, int(shared), C.byref(result)
            ),
            "bandwidth_buffer_create",
        )
        return result

    def gpu_read(self, buffer: C.c_void_p, vector_bytes: int, passes: int) -> tuple[int, int]:
        duration = C.c_uint64()
        checksum = C.c_uint64()
        self.check(
            self.lib.nvfp4_bandwidth_gpu_read(
                buffer, vector_bytes, passes, C.byref(duration), C.byref(checksum)
            ),
            "bandwidth_gpu_read",
        )
        return duration.value, checksum.value

    def cpu_read(
        self,
        array: np.ndarray,
        passes: int,
        threads: int,
        mask: int,
    ) -> tuple[int, int]:
        duration = C.c_uint64()
        checksum = C.c_uint64()
        self.check(
            self.lib.nvfp4_cpu_stream_read(
                C.c_void_p(array.ctypes.data), array.nbytes, passes, threads, mask,
                C.byref(duration), C.byref(checksum),
            ),
            "cpu_stream_read",
        )
        return duration.value, checksum.value

    def shared_cpu_read(
        self,
        buffer: C.c_void_p,
        passes: int,
        threads: int,
        mask: int,
    ) -> tuple[int, int]:
        duration = C.c_uint64()
        checksum = C.c_uint64()
        self.check(
            self.lib.nvfp4_bandwidth_cpu_read(
                buffer, passes, threads, mask, C.byref(duration), C.byref(checksum)
            ),
            "bandwidth_cpu_read",
        )
        return duration.value, checksum.value

    def close(self) -> None:
        if self.handle:
            self.lib.nvfp4_runtime_destroy(self.handle)
            self.handle = C.c_void_p()


def stable_samples(
    operation: Callable[[], dict[str, float | int]],
    warmups: int,
    samples: int,
) -> list[dict[str, float | int]]:
    for _ in range(warmups):
        operation()
    values = [operation() for _ in range(samples)]
    checksum_keys = [key for key in values[0] if key.endswith("checksum")]
    for key in checksum_keys:
        if len({value[key] for value in values}) != 1:
            raise RuntimeError(f"unstable {key}: {[value[key] for value in values]}")
    return values


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    runtime = NativeRuntime(args.dll, args.kernel)
    buffer = C.c_void_p()
    cpu_array: np.ndarray | None = None
    try:
        info = runtime.device_info()
        bytes_ = args.mib * 1024 * 1024
        mask, cpus = affinity_mask(args.cpu_set)
        threads = args.threads or (len(cpus) if cpus else os.cpu_count() or 1)
        payloads = 1

        if args.case == "cpu":
            cpu_array = np.full(bytes_, 0xA5, dtype=np.uint8)

            def operation() -> dict[str, float | int]:
                duration, checksum = runtime.cpu_read(cpu_array, args.passes, threads, mask)
                return {"wall_ns": duration, "cpu_ns": duration, "cpu_checksum": checksum}

            allocation = "host"
        elif args.case == "gpu":
            buffer = runtime.create_buffer(bytes_, args.shared_svm)

            def operation() -> dict[str, float | int]:
                started = time.perf_counter_ns()
                duration, checksum = runtime.gpu_read(buffer, args.vector_bytes, args.passes)
                return {
                    "wall_ns": time.perf_counter_ns() - started,
                    "gpu_ns": duration,
                    "gpu_checksum": checksum,
                }

            allocation = "shared_svm" if args.shared_svm else "opencl_buffer"
        else:
            shared = args.case == "concurrent-shared"
            buffer = runtime.create_buffer(bytes_, shared)
            if not shared:
                cpu_array = np.full(bytes_, 0xA5, dtype=np.uint8)
            payloads = 2

            def operation() -> dict[str, float | int]:
                barrier = threading.Barrier(3)
                result: dict[str, tuple[int, int]] = {}
                errors: list[BaseException] = []

                def cpu_worker() -> None:
                    try:
                        barrier.wait()
                        if shared:
                            result["cpu"] = runtime.shared_cpu_read(
                                buffer, args.passes, threads, mask
                            )
                        else:
                            assert cpu_array is not None
                            result["cpu"] = runtime.cpu_read(
                                cpu_array, args.passes, threads, mask
                            )
                    except BaseException as error:
                        errors.append(error)

                def gpu_worker() -> None:
                    try:
                        barrier.wait()
                        result["gpu"] = runtime.gpu_read(buffer, args.vector_bytes, args.passes)
                    except BaseException as error:
                        errors.append(error)

                cpu_thread = threading.Thread(target=cpu_worker, name="cpu-stream")
                gpu_thread = threading.Thread(target=gpu_worker, name="gpu-stream")
                cpu_thread.start()
                gpu_thread.start()
                started = time.perf_counter_ns()
                barrier.wait()
                cpu_thread.join()
                gpu_thread.join()
                wall = time.perf_counter_ns() - started
                if errors:
                    raise errors[0]
                return {
                    "wall_ns": wall,
                    "cpu_ns": result["cpu"][0],
                    "gpu_ns": result["gpu"][0],
                    "cpu_checksum": result["cpu"][1],
                    "gpu_checksum": result["gpu"][1],
                }

            allocation = "shared_svm_same_range" if shared else "separate_allocations"

        values = stable_samples(operation, args.warmups, args.samples)
        wall_ms = [float(value["wall_ns"]) / 1e6 for value in values]
        gpu_ms = [float(value["gpu_ns"]) / 1e6 for value in values if "gpu_ns" in value]
        cpu_ms = [float(value["cpu_ns"]) / 1e6 for value in values if "cpu_ns" in value]
        logical_bytes = bytes_ * args.passes * payloads
        bandwidths = [logical_bytes / float(value["wall_ns"]) for value in values]
        gpu_bandwidths = [
            bytes_ * args.passes / float(value["gpu_ns"])
            for value in values
            if "gpu_ns" in value
        ]
        cpu_bandwidths = [
            bytes_ * args.passes / float(value["cpu_ns"])
            for value in values
            if "cpu_ns" in value
        ]
        memory = memory_status()
        power = power_status()
        topology = cpu_sets()
        runtime_hash = hashlib.sha256(
            (ROOT / "native_nvfp4/runtime/nvfp4_runtime.cpp").read_bytes()
            + (ROOT / "native_nvfp4/runtime/nvfp4_cpu.cpp").read_bytes()
        ).hexdigest()[:16]
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        record: dict[str, Any] = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": timestamp,
            "hardware": {
                "system_model": system_model(),
                "soc": platform.processor(),
                "cpu_affinity": cpus,
                "cpu_threads": threads,
                "cpu_sets": topology,
                "gpu": decoded(info.device_name),
                "npu": "Qualcomm Hexagon v81 (not exercised)",
                "physical_memory_bytes": memory.total_physical,
            },
            "software": {
                "os_build": platform.platform(),
                "gpu_driver": decoded(info.driver_version),
                "opencl_version": decoded(info.device_version),
                "runtime_revision": runtime_hash,
                "kernel_id": f"stream-read-v1-{args.vector_bytes}B",
                "command": sys.argv,
            },
            "environment": {
                **power,
                "thermal_regime": args.thermal_regime,
                "free_memory_bytes": memory.available_physical,
                "opencl_budget_bytes": info.global_memory_bytes,
                "opencl_max_allocation_bytes": info.max_allocation_bytes,
                "opencl_cache_bytes": info.global_cache_bytes,
                "opencl_local_memory_bytes": info.local_memory_bytes,
                "opencl_compute_units": info.compute_units,
                "opencl_max_clock_mhz": info.max_clock_mhz,
                "svm_capabilities": info.svm_capabilities,
            },
            "workload": {
                "operation": f"raw_stream_read_{args.case}",
                "format": "raw",
                "allocation": allocation,
                "rows": 0,
                "cols": 0,
                "vectors": 0,
                "vector_bytes": args.vector_bytes if args.case != "cpu" else 16,
                "passes": args.passes,
                "logical_payload_bytes": logical_bytes,
            },
            "timing": {
                "warmups": args.warmups,
                "samples": args.samples,
                "kernel_ms_median": statistics.median(gpu_ms) if gpu_ms else None,
                "cpu_ms_median": statistics.median(cpu_ms) if cpu_ms else None,
                "wall_ms_median": statistics.median(wall_ms),
                "p10_ms": percentile(wall_ms, 0.10),
                "p90_ms": percentile(wall_ms, 0.90),
                "minimum_ms": min(wall_ms),
                "maximum_ms": max(wall_ms),
            },
            "bandwidth": {
                "logical_gbs": statistics.median(bandwidths),
                "gpu_kernel_logical_gbs": statistics.median(gpu_bandwidths)
                if gpu_bandwidths
                else None,
                "cpu_logical_gbs": statistics.median(cpu_bandwidths)
                if cpu_bandwidths
                else None,
                "p10_gbs": percentile(bandwidths, 0.10),
                "p90_gbs": percentile(bandwidths, 0.90),
                "physical_gbs": None,
                "matched_island_ceiling_gbs": None,
                "island_utilization": None,
                "nominal_system_utilization": statistics.median(bandwidths) / NOMINAL_GBS,
            },
            "correctness": {
                "passed": True,
                "max_abs_error": None,
                "checksums": {
                    key: values[0][key]
                    for key in values[0]
                    if key.endswith("checksum")
                },
                "finite_timings": all(value > 0 for value in wall_ms),
                "explicit_completion_marker": True,
            },
            "samples": values,
        }
        args.results.mkdir(parents=True, exist_ok=True)
        slug = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = args.results / f"{slug}-{args.case}.json"
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record, output
    finally:
        if buffer:
            runtime.lib.nvfp4_bandwidth_buffer_destroy(buffer)
        runtime.close()


def summarize(directory: Path) -> None:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]
    print(
        "| Operation | Allocation | CPU threads | Vector | Samples | "
        "Median wall | Wall GB/s | GPU kernel GB/s | Status |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for record in records:
        timing = record.get("timing", {})
        bandwidth = record.get("bandwidth", {})
        logical = bandwidth.get("logical_gbs")
        logical_text = "missing" if logical is None else f"{logical:.2f}"
        gpu_logical = bandwidth.get("gpu_kernel_logical_gbs")
        gpu_text = "n/a" if gpu_logical is None else f"{gpu_logical:.2f}"
        wall = timing.get("wall_ms_median")
        wall_text = "missing" if wall is None else f"{wall:.3f} ms"
        passed = record.get("correctness", {}).get("passed") is True
        print(
            f"| {record.get('workload', {}).get('operation', 'missing')} "
            f"| {record.get('workload', {}).get('allocation', 'missing')} "
            f"| {record.get('hardware', {}).get('cpu_threads', 'missing')} "
            f"| {record.get('workload', {}).get('vector_bytes', 'missing')} B "
            f"| {timing.get('samples', 'missing')} | {wall_text} "
            f"| {logical_text} | {gpu_text} | {'PASS' if passed else 'FAIL/MISSING'} |"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--case",
        choices=("cpu", "gpu", "concurrent-different", "concurrent-shared"),
        default="gpu",
    )
    result.add_argument("--mib", type=int, default=64)
    result.add_argument("--passes", type=int, default=1)
    result.add_argument("--vector-bytes", type=int, choices=(1, 4, 16, 64), default=16)
    result.add_argument("--threads", type=int, default=0)
    result.add_argument("--cpu-set", default="")
    result.add_argument("--shared-svm", action="store_true")
    result.add_argument("--warmups", type=int, default=5)
    result.add_argument("--samples", type=int, default=30)
    result.add_argument("--thermal-regime", default="warm-burst")
    result.add_argument("--dll", type=Path, default=RUNTIME)
    result.add_argument("--kernel", type=Path, default=KERNEL)
    result.add_argument("--results", type=Path, default=RESULTS)
    result.add_argument("--summarize", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.summarize:
        summarize(args.results)
        return 0
    if (
        args.mib <= 0
        or args.passes <= 0
        or args.warmups < 0
        or args.samples <= 0
        or args.threads < 0
    ):
        raise ValueError("sizes, passes, and samples must be positive")
    record, output = run(args)
    print(json.dumps({
        "operation": record["workload"]["operation"],
        "allocation": record["workload"]["allocation"],
        "wall_ms_median": record["timing"]["wall_ms_median"],
        "logical_gbs": record["bandwidth"]["logical_gbs"],
        "result": str(output),
    }))
    print("CAMPAIGN_BANDWIDTH_FIRST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
