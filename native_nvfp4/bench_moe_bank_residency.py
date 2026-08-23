"""Stage real Qwen3.5 MoE banks through bounded cumulative residency gates."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from bench_islands import memory_status, power_status, system_model
from bench_moe_device_bank import stream_experts_into_bank
from bench_moe_experts import load_experts
from bench_moe_routed_layer import (
    MODEL,
    RESULTS,
    expert_reference,
    load_layer_tensors,
    route,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--gates", default="3,5,10,19")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    gates = [int(value) for value in args.gates.split(",")]
    if (
        args.first_layer < 0
        or not gates
        or gates != sorted(set(gates))
        or gates[0] <= 0
        or gates[-1] > 40 - args.first_layer
    ):
        parser.error("gates must be sorted unique layer counts within the checkpoint")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        ROOT / "native_nvfp4/runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        ROOT / "native_nvfp4/kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(ROOT / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths

    x = np.ascontiguousarray(
        np.random.default_rng(20260822).standard_normal((1, 2048)).astype(np.float32)
        * np.float32(0.2)
    )
    runtime = Runtime(*runtime_paths())
    input_buffer = runtime.upload_buffer(x)
    output_buffer = runtime.create_buffer(2048 * 4)
    banks: list[object] = []
    baseline_available = memory_status().available_physical
    layer_records: list[dict[str, object]] = []
    gate_records: list[dict[str, object]] = []
    resident_payload = 0
    started = time.perf_counter()
    try:
        for relative_layer in range(gates[-1]):
            layer = args.first_layer + relative_layer
            layer_started = time.perf_counter()
            router_bf16, router_f32, shared_gate_bf16, shared_host = (
                load_layer_tensors(args.model, layer)
            )
            expected_ids, expected_weights = route(router_f32 @ x[0], 8)
            selected_hosts = load_experts(args.model, layer, expected_ids)
            shared_gate_f32 = (
                np.left_shift(shared_gate_bf16.astype(np.uint32), 16)
                .view(np.float32)
                .reshape(-1)
            )
            shared_weight = float(
                1.0
                / (
                    1.0
                    + np.exp(-float((shared_gate_f32 @ x[0]).item()))
                )
            )
            reference = np.zeros((1, 2048), dtype=np.float32)
            for weight, expert in zip(
                expected_weights, selected_hosts, strict=True
            ):
                reference += np.float32(weight) * expert_reference(expert, x)
            reference += np.float32(shared_weight) * expert_reference(shared_host, x)

            bank = runtime.create_moe_bank(router_bf16, shared_gate_bf16, 512)
            routed_payload = stream_experts_into_bank(bank, args.model, layer)
            bank.upload_expert(256, shared_host)
            shared_payload = sum(
                packed.nbytes + scales.nbytes
                for packed, scales, _divisor in shared_host
            )
            layer_payload = routed_payload + shared_payload
            resident_payload += layer_payload
            banks.append(bank)

            bank.decode_device(input_buffer, output_buffer)
            profile = runtime.synchronize()
            result = output_buffer.download((1, 2048))
            max_abs = float(np.max(np.abs(reference - result)))
            if not np.allclose(reference, result, rtol=1e-4, atol=1e-4):
                raise SystemExit(
                    f"layer {layer} residency oracle mismatch: max_abs={max_abs}"
                )
            del selected_hosts
            gc.collect()
            available = memory_status().available_physical
            layer_record: dict[str, object] = {
                "layer": layer,
                "banks_resident": len(banks),
                "resident_payload_bytes": resident_payload,
                "available_physical_bytes": available,
                "available_delta_bytes": available - baseline_available,
                "load_and_validate_seconds": time.perf_counter() - layer_started,
                "validation_kernel_ms": profile.kernel_ns / 1e6,
                "max_abs_error": max_abs,
                "selected_experts": expected_ids,
            }
            layer_records.append(layer_record)
            if len(banks) in gates:
                gate_records.append(dict(layer_record))
                print(
                    f"gate_banks={len(banks)} layer={layer} "
                    f"resident_bytes={resident_payload} "
                    f"available_bytes={available} max_abs_err={max_abs:.8g}"
                )

        before_release = memory_status().available_physical
        for bank in reversed(banks):
            bank.close()
        after_release = memory_status().available_physical
        record = {
            "campaign": "bandwidth-first",
            "schema_version": 1,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hardware": {"system_model": system_model(), "gpu": runtime.device_name},
            "software": {"command": sys.argv},
            "environment": {**power_status(), "thermal_regime": "warm-burst"},
            "workload": {
                "operation": "qwen35_moe_svm_bank_residency_ladder",
                "format": "nvfp4_expert_bank_bf16_router",
                "first_layer": args.first_layer,
                "gate_bank_counts": gates,
                "maximum_resident_payload_bytes": resident_payload,
            },
            "memory": {
                "baseline_available_physical_bytes": baseline_available,
                "before_release_available_physical_bytes": before_release,
                "after_release_available_physical_bytes": after_release,
                "recovered_on_release_bytes": after_release - before_release,
                "reported_opencl_global_budget_bytes": 24379 * 1024 * 1024,
            },
            "timing": {"total_seconds": time.perf_counter() - started},
            "gates": gate_records,
            "layers": layer_records,
            "correctness": {
                "passed": True,
                "layers_validated": len(layer_records),
                "maximum_abs_error": max(
                    float(item["max_abs_error"]) for item in layer_records
                ),
                "finite_outputs": True,
                "explicit_completion_marker": True,
            },
            "limitations": [
                "bounded at the requested 19 banks rather than attempting full residency",
                "available physical memory is a system-wide sample, not an OpenCL budget counter",
                "each layer is validated independently; a 19-layer token graph is not timed",
                "non-expert model weights, KV, recurrent state, and serving scratch are excluded",
            ],
        }
        args.results.mkdir(parents=True, exist_ok=True)
        path = args.results / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-moe-bank-residency.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"max_resident_bytes={resident_payload} "
            f"max_abs_err={record['correctness']['maximum_abs_error']:.8g} "
            f"released_bytes={after_release - before_release} result={path}"
        )
        print("MOE_NVFP4_BANK_RESIDENCY_PASS")
    finally:
        for bank in reversed(banks):
            bank.close()
        output_buffer.close()
        input_buffer.close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
