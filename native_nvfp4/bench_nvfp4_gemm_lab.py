#!/usr/bin/env python3
"""Correctness-gated NVFP4 multi-vector GEMM/prefill structure sweep."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from bench_nvfp4_kernel_lab import LOADERS, RESULTS, HostMatrix, describe


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = ("dense-gate", "dense-down", "moe-gate", "moe-down")


def config_name(vector_tile: int, k_tile: int, kind: int) -> str:
    path = (
        "local"
        if kind < 2
        else (
            "direct"
            if kind < 4
            else ("register" if kind == 4 else "register-transposed")
        )
    )
    decode = "vector" if kind in (1, 3) else "scalar"
    suffix = f"-k{k_tile}" if path == "local" else ""
    return f"lab-{path}-{decode}-v{vector_tile}{suffix}"


def measure_case(
    runtime: Any,
    host: HostMatrix,
    vectors: int,
    vector_tiles: list[int],
    k_tiles: list[int],
    register_kinds: list[int],
    warmups: int,
    samples: int,
    rng: random.Random,
) -> dict[str, Any]:
    x = np.ascontiguousarray(
        np.random.default_rng(20260822 + host.rows + host.cols + vectors)
        .standard_normal((vectors, host.cols))
        .astype(np.float32)
        * np.float32(0.2)
    )
    matrix = runtime.upload(host.packed, host.scales, host.global_scale)
    input_buffer = runtime.upload_buffer(x)
    input_transposed_buffer = runtime.upload_buffer(np.ascontiguousarray(x.T))
    output_buffer = runtime.create_buffer(vectors * host.rows * 4)
    configurations = [
        (config_name(vector_tile, k_tile, kind), vector_tile, k_tile, kind)
        for vector_tile in vector_tiles
        for kind in range(4)
        for k_tile in (k_tiles if kind < 2 else k_tiles[:1])
    ]
    configurations.extend(
        (config_name(vector_tile, k_tiles[0], kind), vector_tile, k_tiles[0], kind)
        for kind in register_kinds
        for vector_tile in vector_tiles
        if vector_tile in (2, 4, 8, 16)
    )
    timings: dict[str, list[float]] = {"production-dispatch": []}
    correctness: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []

    def run_production() -> float:
        runtime.linear_device(
            matrix, input_buffer, vectors, out=output_buffer, kernel_kind=2
        )
        return runtime.last_profile().kernel_ns / 1e6

    def run_lab(vector_tile: int, k_tile: int, kind: int) -> float:
        runtime.gemm_device_lab(
            matrix,
            input_transposed_buffer if kind == 5 else input_buffer,
            vectors=vectors,
            vector_tile=vector_tile,
            k_tile=k_tile,
            implementation_kind=kind,
            out=output_buffer,
        )
        return runtime.last_profile().kernel_ns / 1e6

    try:
        run_production()
        reference = output_buffer.download((vectors, host.rows))
        accepted = []
        for name, vector_tile, k_tile, kind in configurations:
            try:
                run_lab(vector_tile, k_tile, kind)
                result = output_buffer.download((vectors, host.rows))
                difference = np.abs(reference - result)
                denominator = np.maximum(np.abs(reference), np.float32(1e-6))
                passed = bool(
                    np.isfinite(result).all()
                    and np.allclose(reference, result, rtol=5e-5, atol=5e-5)
                )
                correctness[name] = {
                    "passed": passed,
                    "bit_exact": bool(np.array_equal(reference, result)),
                    "max_abs_error": float(np.max(difference)),
                    "max_rel_error": float(np.max(difference / denominator)),
                }
                if passed:
                    accepted.append((name, vector_tile, k_tile, kind))
                    timings[name] = []
                else:
                    failures.append({"configuration": name, "reason": "oracle_mismatch"})
            except RuntimeError as error:
                failures.append({"configuration": name, "reason": str(error)})

        active: list[tuple[str, int | None, int | None, int | None]] = [
            ("production-dispatch", None, None, None),
            *accepted,
        ]
        for _ in range(warmups):
            order = active.copy()
            rng.shuffle(order)
            for _name, vector_tile, k_tile, kind in order:
                if vector_tile is None:
                    run_production()
                else:
                    run_lab(vector_tile, k_tile, kind)
        for _ in range(samples):
            order = active.copy()
            rng.shuffle(order)
            for name, vector_tile, k_tile, kind in order:
                elapsed_ms = (
                    run_production()
                    if vector_tile is None
                    else run_lab(vector_tile, k_tile, kind)
                )
                timings[name].append(elapsed_ms)

        flop = 2 * host.rows * host.cols * vectors
        useful_model_bytes = host.weight_bytes * vectors + x.nbytes + output_buffer.bytes
        compulsory_bytes = host.weight_bytes + x.nbytes + output_buffer.bytes
        ranking = []
        for name, values in timings.items():
            stats = describe(values)
            seconds = stats["median"] / 1000.0
            ranking.append(
                {
                    "configuration": name,
                    "kernel_ms": stats,
                    "gflops": flop / seconds / 1e9,
                    "effective_model_gbs": useful_model_bytes / seconds / 1e9,
                    "compulsory_gbs": compulsory_bytes / seconds / 1e9,
                    "effective_percent_of_129_gbs":
                        useful_model_bytes / seconds / 1e9 / 129.0 * 100.0,
                    "effective_percent_of_228_gbs":
                        useful_model_bytes / seconds / 1e9 / 228.0 * 100.0,
                    "samples_ms": values,
                }
            )
        ranking.sort(key=lambda item: item["kernel_ms"]["median"])
        baseline = next(
            item for item in ranking
            if item["configuration"] == "production-dispatch"
        )
        for item in ranking:
            item["speedup_vs_production"] = (
                baseline["kernel_ms"]["median"] / item["kernel_ms"]["median"]
            )
        return {
            "shape": host.name,
            "rows": host.rows,
            "cols": host.cols,
            "vectors": vectors,
            "weight_bytes": host.weight_bytes,
            "correctness_oracle": "current production dispatch output",
            "correctness": correctness,
            "failures": failures,
            "ranking": ranking,
        }
    finally:
        output_buffer.close()
        input_transposed_buffer.close()
        input_buffer.close()
        matrix.close()


def parse_csv(parser: argparse.ArgumentParser, value: str, label: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError:
        parser.error(f"{label} must be a comma-separated integer list")
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        parser.error(f"{label} values must be unique and positive")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--vectors", default="2,4,8,16,32")
    parser.add_argument("--vector-tiles", default="1,2,4,8,16")
    parser.add_argument("--k-tiles", default="512,1024,2048,4096,8192,16384,32768")
    parser.add_argument(
        "--register-kinds",
        default="4,5",
        help="register treatments: 4=vector-major, 5=K-major input",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    shapes = args.shapes.split(",")
    unknown = sorted(set(shapes) - set(DEFAULT_SHAPES))
    if unknown:
        parser.error(f"unsupported GEMM shapes: {unknown}")
    vectors = parse_csv(parser, args.vectors, "vectors")
    vector_tiles = parse_csv(parser, args.vector_tiles, "vector tiles")
    k_tiles = parse_csv(parser, args.k_tiles, "K tiles")
    register_kinds = parse_csv(parser, args.register_kinds, "register kinds")
    if not set(register_kinds) <= {4, 5}:
        parser.error("register kinds must be drawn from 4,5")
    if any(value <= 1 for value in vectors):
        parser.error("GEMM vector counts must be greater than one")
    if args.warmups < 0 or args.samples <= 0:
        parser.error("warmups must be nonnegative and samples must be positive")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(ROOT / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    runtime = Runtime(*runtime_paths())
    rng = random.Random(20260822)
    cases = []
    try:
        for shape in shapes:
            host = LOADERS[shape]()
            print(
                f"shape={host.name} rows={host.rows} cols={host.cols} "
                f"weight_bytes={host.weight_bytes}",
                flush=True,
            )
            for vector_count in vectors:
                result = measure_case(
                    runtime,
                    host,
                    vector_count,
                    vector_tiles,
                    k_tiles,
                    register_kinds,
                    args.warmups,
                    args.samples,
                    rng,
                )
                cases.append(result)
                winner = result["ranking"][0]
                print(
                    f"vectors={vector_count} winner={winner['configuration']} "
                    f"kernel_ms={winner['kernel_ms']['median']:.6f} "
                    f"gflops={winner['gflops']:.2f} "
                    f"speedup={winner['speedup_vs_production']:.4f} "
                    f"rejected={len(result['failures'])}",
                    flush=True,
                )
    finally:
        runtime.close()

    record = {
        "campaign": "nvfp4-gemm-lab",
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hardware": {"gpu": "Qualcomm(R) Adreno(TM) X2-90 GPU"},
        "software": {"command": sys.argv},
        "method": {
            "sample_order": "randomized within each shape/vector round",
            "warmups": args.warmups,
            "samples": args.samples,
            "vectors": vectors,
            "vector_tiles": vector_tiles,
            "k_tiles": k_tiles,
            "register_kinds": register_kinds,
            "implementations": [
                "local-scalar",
                "local-vector",
                "direct-global-scalar",
                "direct-global-vector",
                "cross-vector-register-scalar",
                "cross-vector-register-scalar-k-major-input",
            ],
            "correctness_rtol": 5e-5,
            "correctness_atol": 5e-5,
            "production_shape_tuning_enabled": (
                os.environ.get("VLLM_NVFP4_OPENCL_SHAPE_TUNING", "1") != "0"
            ),
        },
        "cases": cases,
        "limitations": [
            "OpenCL event time isolates each synchronous GEMM kernel",
            "effective model bandwidth counts each matrix-vector use and may exceed DRAM bandwidth through reuse",
            "compulsory bandwidth assumes one ideal weight read for the whole launch",
            "physical traffic and register/occupancy counters remain unavailable",
            "the full serving prefill path remains sequential until a winner is promoted",
            "K-major-input kernel timings exclude the activation transpose/layout conversion",
        ],
    }
    args.results.mkdir(parents=True, exist_ok=True)
    slug = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.results / f"{slug}-nvfp4-gemm-lab.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"result={output}")
    print("NVFP4_GEMM_LAB_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
