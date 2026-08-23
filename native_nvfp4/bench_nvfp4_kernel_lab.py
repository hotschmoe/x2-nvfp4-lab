#!/usr/bin/env python3
"""Correctness-gated NVFP4 GEMV structure sweep on real checkpoint shapes."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "campaign_results/bandwidth-first"
DENSE_MODEL = ROOT / "models/Qwen3.8-27B-NVFP4-Unsloth/model.safetensors"
MOE_MODEL = ROOT / "models/Ornith-1.5-35B-A3B-NVFP4"


@dataclass(frozen=True)
class HostMatrix:
    name: str
    packed: np.ndarray
    scales: np.ndarray
    global_scale: float

    @property
    def rows(self) -> int:
        return int(self.packed.shape[0])

    @property
    def cols(self) -> int:
        return int(self.packed.shape[1] * 2)

    @property
    def weight_bytes(self) -> int:
        return self.packed.nbytes + self.scales.nbytes


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "median": statistics.median(values),
        "p10": percentile(0.10),
        "p90": percentile(0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def dense_matrix(projection: str) -> HostMatrix:
    base = f"model.language_model.layers.0.mlp.{projection}_proj"
    with safe_open(DENSE_MODEL, framework="pt", device="cpu") as checkpoint:
        packed = np.ascontiguousarray(checkpoint.get_tensor(base + ".weight_packed").numpy())
        scales = np.ascontiguousarray(
            checkpoint.get_tensor(base + ".weight_scale").view(torch.uint8).numpy()
        )
        global_scale = float(
            checkpoint.get_tensor(base + ".weight_global_scale").item()
        )
    return HostMatrix(f"dense-{projection}", packed, scales, global_scale)


def indexed_tensors(names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    index = json.loads(
        (MOE_MODEL / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(index[name], []).append(name)
    tensors: dict[str, torch.Tensor] = {}
    for shard_name, shard_names in by_shard.items():
        with safe_open(MOE_MODEL / shard_name, framework="pt", device="cpu") as shard:
            for name in shard_names:
                tensors[name] = shard.get_tensor(name)
    return tensors


def moe_expert_matrix(projection: str) -> HostMatrix:
    base = f"model.language_model.layers.0.mlp.experts.0.{projection}_proj"
    names = (base + ".weight", base + ".weight_scale", base + ".weight_scale_2")
    tensors = indexed_tensors(names)
    packed = np.ascontiguousarray(
        tensors[names[0]].numpy().astype(np.uint8, copy=False)
    )
    scales = np.ascontiguousarray(tensors[names[1]].view(torch.uint8).numpy())
    multiplier = float(tensors[names[2]].item())
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(f"invalid {names[2]}: {multiplier}")
    return HostMatrix(f"moe-{projection}", packed, scales, 1.0 / multiplier)


def moe_head_matrix() -> HostMatrix:
    names = ("lm_head.weight", "lm_head.weight_scale", "lm_head.weight_scale_2")
    tensors = indexed_tensors(names)
    packed = np.ascontiguousarray(
        tensors[names[0]].numpy().astype(np.uint8, copy=False)
    )
    scales = np.ascontiguousarray(tensors[names[1]].view(torch.uint8).numpy())
    multiplier = float(tensors[names[2]].item())
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(f"invalid {names[2]}: {multiplier}")
    return HostMatrix("moe-head", packed, scales, 1.0 / multiplier)


LOADERS: dict[str, Callable[[], HostMatrix]] = {
    "dense-gate": lambda: dense_matrix("gate"),
    "dense-down": lambda: dense_matrix("down"),
    "moe-gate": lambda: moe_expert_matrix("gate"),
    "moe-down": lambda: moe_expert_matrix("down"),
    "moe-head": moe_head_matrix,
}


def config_name(row_tile: int, k_tile: int, decode_kind: int) -> str:
    path = "local" if decode_kind < 2 else "direct"
    decode = "scalar" if decode_kind % 2 == 0 else "vector"
    suffix = f"-k{k_tile}" if path == "local" else ""
    return f"lab-{path}-{decode}-r{row_tile}{suffix}"


def measure_shape(
    runtime: Any,
    host: HostMatrix,
    row_tiles: list[int],
    k_tiles: list[int],
    warmups: int,
    samples: int,
    rng: random.Random,
) -> dict[str, Any]:
    x = np.ascontiguousarray(
        np.random.default_rng(20260822 + host.rows + host.cols)
        .standard_normal((1, host.cols))
        .astype(np.float32)
        * np.float32(0.2)
    )
    matrix = runtime.upload(host.packed, host.scales, host.global_scale)
    input_buffer = runtime.upload_buffer(x)
    output_buffer = runtime.create_buffer(host.rows * np.dtype(np.float32).itemsize)
    configurations: list[tuple[str, int, int, int]] = [
        (config_name(row_tile, k_tile, decode_kind), row_tile, k_tile, decode_kind)
        for row_tile in row_tiles
        for decode_kind in range(4)
        for k_tile in (k_tiles if decode_kind < 2 else k_tiles[:1])
    ]
    failures: list[dict[str, Any]] = []
    correctness: dict[str, dict[str, Any]] = {}
    timings: dict[str, list[float]] = {"production-dispatch": []}

    def run_production() -> float:
        runtime.linear_device(
            matrix, input_buffer, 1, out=output_buffer, kernel_kind=3
        )
        return runtime.last_profile().kernel_ns / 1e6

    def run_lab(row_tile: int, k_tile: int, decode_kind: int) -> float:
        runtime.linear_device_lab(
            matrix,
            input_buffer,
            row_tile=row_tile,
            k_tile=k_tile,
            decode_kind=decode_kind,
            out=output_buffer,
        )
        return runtime.last_profile().kernel_ns / 1e6

    try:
        run_production()
        reference = output_buffer.download((1, host.rows))
        accepted: list[tuple[str, int, int, int]] = []
        for name, row_tile, k_tile, decode_kind in configurations:
            try:
                run_lab(row_tile, k_tile, decode_kind)
                result = output_buffer.download((1, host.rows))
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
                    accepted.append((name, row_tile, k_tile, decode_kind))
                    timings[name] = []
                else:
                    failures.append({"configuration": name, "reason": "oracle_mismatch"})
            except RuntimeError as error:
                failures.append({"configuration": name, "reason": str(error)})

        active: list[tuple[str, int, int, int] | tuple[str, None, None, None]] = [
            ("production-dispatch", None, None, None),
            *accepted,
        ]
        for _ in range(warmups):
            order = active.copy()
            rng.shuffle(order)
            for _name, row_tile, k_tile, decode_kind in order:
                if row_tile is None:
                    run_production()
                else:
                    run_lab(row_tile, k_tile, decode_kind)
        for _ in range(samples):
            order = active.copy()
            rng.shuffle(order)
            for name, row_tile, k_tile, decode_kind in order:
                elapsed_ms = (
                    run_production()
                    if row_tile is None
                    else run_lab(row_tile, k_tile, decode_kind)
                )
                timings[name].append(elapsed_ms)

        logical_bytes = host.weight_bytes + x.nbytes + output_buffer.bytes
        results = []
        for name, values in timings.items():
            stats = describe(values)
            logical_gbs = logical_bytes / (stats["median"] * 1e6)
            results.append(
                {
                    "configuration": name,
                    "kernel_ms": stats,
                    "logical_bytes": logical_bytes,
                    "logical_gbs": logical_gbs,
                    "percent_of_measured_129_gbs": logical_gbs / 129.0 * 100.0,
                    "percent_of_nominal_228_gbs": logical_gbs / 228.0 * 100.0,
                    "samples_ms": values,
                }
            )
        results.sort(key=lambda item: item["kernel_ms"]["median"])
        baseline = next(
            item for item in results if item["configuration"] == "production-dispatch"
        )
        for item in results:
            item["speedup_vs_production"] = (
                baseline["kernel_ms"]["median"] / item["kernel_ms"]["median"]
            )
        return {
            "shape": host.name,
            "rows": host.rows,
            "cols": host.cols,
            "weight_bytes": host.weight_bytes,
            "correctness_oracle": "current production dispatch output",
            "correctness": correctness,
            "failures": failures,
            "ranking": results,
        }
    finally:
        output_buffer.close()
        input_buffer.close()
        matrix.close()


def parse_positive_csv(parser: argparse.ArgumentParser, value: str, label: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError:
        parser.error(f"{label} must be a comma-separated integer list")
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        parser.error(f"{label} values must be unique and positive")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default=",".join(LOADERS))
    parser.add_argument("--row-tiles", default="1,2,4,8")
    parser.add_argument("--k-tiles", default="256,512,1024,2048,4096")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    shapes = args.shapes.split(",")
    unknown = sorted(set(shapes) - LOADERS.keys())
    if unknown:
        parser.error(f"unknown shapes: {unknown}; choices are {sorted(LOADERS)}")
    row_tiles = parse_positive_csv(parser, args.row_tiles, "row tiles")
    k_tiles = parse_positive_csv(parser, args.k_tiles, "K tiles")
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
    shape_results = []
    try:
        for shape in shapes:
            host = LOADERS[shape]()
            print(
                f"shape={host.name} rows={host.rows} cols={host.cols} "
                f"weight_bytes={host.weight_bytes}",
                flush=True,
            )
            result = measure_shape(
                runtime, host, row_tiles, k_tiles, args.warmups, args.samples, rng
            )
            shape_results.append(result)
            winner = result["ranking"][0]
            print(
                f"winner={winner['configuration']} "
                f"kernel_ms={winner['kernel_ms']['median']:.6f} "
                f"logical_gbs={winner['logical_gbs']:.3f} "
                f"speedup={winner['speedup_vs_production']:.4f} "
                f"rejected={len(result['failures'])}",
                flush=True,
            )
    finally:
        runtime.close()

    record = {
        "campaign": "nvfp4-kernel-lab",
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hardware": {"gpu": "Qualcomm(R) Adreno(TM) X2-90 GPU"},
        "software": {"command": sys.argv},
        "method": {
            "sample_order": "randomized within each round",
            "warmups": args.warmups,
            "samples": args.samples,
            "row_tiles": row_tiles,
            "k_tiles": k_tiles,
            "decode_kinds": [
                "local-scalar",
                "local-vector",
                "direct-global-scalar",
                "direct-global-vector",
            ],
            "correctness_rtol": 5e-5,
            "correctness_atol": 5e-5,
            "measured_bandwidth_ceiling_gbs": 129.0,
            "nominal_bandwidth_gbs": 228.0,
            "production_shape_tuning_enabled": (
                os.environ.get("VLLM_NVFP4_OPENCL_SHAPE_TUNING", "1") != "0"
            ),
        },
        "shapes": shape_results,
        "limitations": [
            "OpenCL event time isolates each synchronous GEMV kernel",
            "logical bandwidth counts native weights, activation input, and output once",
            "no hardware counter data is available yet",
            "this milestone covers decode GEMV; multi-vector GEMM follows separately",
            "vision and MTP are intentionally outside the current text-only baseline",
        ],
    }
    args.results.mkdir(parents=True, exist_ok=True)
    slug = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.results / f"{slug}-nvfp4-kernel-lab.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"result={output}")
    print("NVFP4_KERNEL_LAB_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
