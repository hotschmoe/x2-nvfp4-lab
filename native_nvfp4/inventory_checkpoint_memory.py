"""Inventory safetensors storage and derive Qwen3.5 serving-state budgets.

This reads tensor metadata only: checkpoint payloads are never materialized.  It
is intentionally useful before a full model loader exists, because residency
decisions must account for every tensor rather than extrapolate from one layer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open


DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def tensor_category(name: str) -> str:
    # MTP can contain names that otherwise look like ordinary attention/MLP.
    if name.startswith("mtp."):
        return "optional_mtp"
    if name.startswith("model.visual."):
        return "optional_vision"
    if "embed_tokens" in name:
        return "text_embedding"
    if name.startswith("lm_head"):
        return "text_lm_head"
    if ".linear_attn." in name:
        return "text_linear_attention"
    if ".self_attn." in name:
        return "text_full_attention"
    if ".mlp.shared_expert." in name:
        return "text_moe_shared_expert"
    if ".mlp.experts." in name:
        return "text_moe_routed_experts"
    if name.endswith(".mlp.gate.weight") or ".mlp.shared_expert_gate." in name:
        return "text_moe_router"
    if ".mlp." in name:
        return "text_dense_mlp"
    if "norm" in name:
        return "text_norm"
    return "unclassified"


def storage_class(name: str, dtype: str) -> str:
    if dtype == "U8" and name.endswith((".weight", ".weight_packed")):
        return "nvfp4_packed"
    if dtype.startswith("F8_") and name.endswith(".weight_scale"):
        return "nvfp4_block_scale"
    if dtype.startswith("F8_") and name.endswith(".weight"):
        return "fp8_weight"
    if name.endswith(("input_scale", "weight_scale_2", "global_scale")):
        return "quant_scalar"
    return dtype.lower()


def add(bucket: dict[str, dict[str, int]], key: str, size: int) -> None:
    row = bucket.setdefault(key, {"tensors": 0, "bytes": 0})
    row["tensors"] += 1
    row["bytes"] += size


def inventory(model_dir: Path) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}
    by_dtype: dict[str, dict[str, int]] = {}
    by_storage: dict[str, dict[str, int]] = {}
    by_file: dict[str, dict[str, int]] = {}
    unclassified: list[str] = []
    tensor_bytes = 0
    tensor_count = 0

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files under {model_dir}")
    for path in files:
        file_tensor_bytes = 0
        file_tensors = 0
        with safe_open(path, framework="np") as handle:
            for name in handle.keys():
                tensor = handle.get_slice(name)
                dtype = tensor.get_dtype()
                if dtype not in DTYPE_BYTES:
                    raise ValueError(f"unsupported safetensors dtype {dtype}: {name}")
                elements = math.prod(tensor.get_shape())
                size = elements * DTYPE_BYTES[dtype]
                category = tensor_category(name)
                add(by_category, category, size)
                add(by_dtype, dtype, size)
                add(by_storage, storage_class(name, dtype), size)
                if category == "unclassified":
                    unclassified.append(name)
                tensor_bytes += size
                tensor_count += 1
                file_tensor_bytes += size
                file_tensors += 1
        by_file[path.name] = {
            "tensors": file_tensors,
            "tensor_bytes": file_tensor_bytes,
            "file_bytes": path.stat().st_size,
            "container_overhead_bytes": path.stat().st_size - file_tensor_bytes,
        }

    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    layer_types = text_config.get("layer_types", [])
    full_layers = layer_types.count("full_attention")
    linear_layers = layer_types.count("linear_attention")
    profile = {
        "hidden_size": text_config.get("hidden_size"),
        "layers": text_config.get("num_hidden_layers"),
        "full_attention_layers": full_layers,
        "linear_attention_layers": linear_layers,
        "attention_heads": text_config.get("num_attention_heads"),
        "kv_heads": text_config.get("num_key_value_heads"),
        "head_dim": text_config.get("head_dim"),
        "linear_key_heads": text_config.get("linear_num_key_heads"),
        "linear_value_heads": text_config.get("linear_num_value_heads"),
        "linear_key_head_dim": text_config.get("linear_key_head_dim"),
        "linear_value_head_dim": text_config.get("linear_value_head_dim"),
        "linear_conv_kernel_dim": text_config.get("linear_conv_kernel_dim"),
        "experts": text_config.get("num_experts"),
        "experts_per_token": text_config.get("num_experts_per_tok"),
    }
    return {
        "model_dir": str(model_dir),
        "tensor_count": tensor_count,
        "tensor_bytes": tensor_bytes,
        "safetensors_file_bytes": sum(row["file_bytes"] for row in by_file.values()),
        "by_category": dict(sorted(by_category.items())),
        "by_dtype": dict(sorted(by_dtype.items())),
        "by_storage": dict(sorted(by_storage.items())),
        "by_file": by_file,
        "unclassified": unclassified,
        "profile": profile,
    }


def state_budget(profile: dict[str, Any], context_tokens: int, concurrency: int) -> dict[str, int]:
    full_layers = profile["full_attention_layers"]
    linear_layers = profile["linear_attention_layers"]
    kv_heads = profile["kv_heads"]
    head_dim = profile["head_dim"]
    key_heads = profile["linear_key_heads"]
    value_heads = profile["linear_value_heads"]
    key_dim = profile["linear_key_head_dim"]
    value_dim = profile["linear_value_head_dim"]
    conv_kernel = profile["linear_conv_kernel_dim"]

    kv_elements_per_token = full_layers * 2 * kv_heads * head_dim
    recurrent_per_request = linear_layers * value_heads * key_dim * value_dim * 4
    conv_channels = 2 * key_heads * key_dim + value_heads * value_dim
    conv_per_request = linear_layers * conv_channels * conv_kernel * 4
    block_table_per_request = full_layers * math.ceil(context_tokens / 16) * 4
    query_gate_scratch_per_request = (
        full_layers * 2 * profile["attention_heads"] * head_dim * 4
    )
    return {
        "context_tokens": context_tokens,
        "concurrency": concurrency,
        "kv_fp32_bytes": concurrency * context_tokens * kv_elements_per_token * 4,
        "kv_bf16_bytes": concurrency * context_tokens * kv_elements_per_token * 2,
        "kv_fp8_bytes": concurrency * context_tokens * kv_elements_per_token,
        "gated_delta_fp32_bytes": concurrency * recurrent_per_request,
        "causal_conv_fp32_bytes": concurrency * conv_per_request,
        "paged_block_tables_bytes": concurrency * block_table_per_request,
        "attention_query_gate_scratch_bytes": concurrency * query_gate_scratch_per_request,
        # Current C ABI keeps a private FP32 copy of width-4 conv weights in each
        # request state. A full server should make this once-per-layer instead.
        "current_runtime_extra_conv_weight_bytes": concurrency * conv_per_request,
    }


def residency_plan(
    row: dict[str, Any],
    device_budget_bytes: int,
    safety_reserve_bytes: int,
) -> dict[str, Any]:
    categories = row["by_category"]
    lazy_embedding = categories.get("text_embedding", {}).get("bytes", 0)
    resident_weights = sum(
        value["bytes"]
        for key, value in categories.items()
        if key.startswith("text_") and key != "text_embedding"
    )
    plans = []
    for state in row["state_budgets"]:
        fixed_state = sum(
            state[key]
            for key in (
                "gated_delta_fp32_bytes",
                "causal_conv_fp32_bytes",
                "paged_block_tables_bytes",
                "attention_query_gate_scratch_bytes",
                "current_runtime_extra_conv_weight_bytes",
            )
        )
        variants = {}
        for dtype in ("fp32", "bf16", "fp8"):
            known = resident_weights + fixed_state + state[f"kv_{dtype}_bytes"]
            variants[dtype] = {
                "known_runtime_bytes": known,
                "headroom_before_reserve_bytes": device_budget_bytes - known,
                "headroom_after_reserve_bytes": (
                    device_budget_bytes - known - safety_reserve_bytes
                ),
                "fits_with_reserve": known + safety_reserve_bytes <= device_budget_bytes,
            }
        plans.append(
            {
                "context_tokens": state["context_tokens"],
                "concurrency": state["concurrency"],
                "fixed_state_and_known_scratch_bytes": fixed_state,
                "kv_variants": variants,
            }
        )
    is_moe = bool(row["profile"].get("experts"))
    return {
        "reported_opencl_global_budget_bytes": device_budget_bytes,
        "safety_reserve_bytes": safety_reserve_bytes,
        "resident_text_compute_weights_bytes": resident_weights,
        "lazy_cpu_embedding_bytes": lazy_embedding,
        "excluded_optional_vision_bytes": categories.get("optional_vision", {}).get("bytes", 0),
        "excluded_optional_mtp_bytes": categories.get("optional_mtp", {}).get("bytes", 0),
        "scenarios": plans,
        "qualification_policy": {
            "load_categories": [
                key
                for key in categories
                if key.startswith("text_") and key != "text_embedding"
            ],
            "lazy_categories": ["text_embedding"],
            "excluded_categories": ["optional_vision", "optional_mtp"],
            "current_kv_dtype": "fp32",
            "current_context_tokens": 32_768 if is_moe else 16_384,
            "current_max_concurrency": 1,
            "next_kv_dtype": "bf16",
            "next_context_tokens": 65_536 if is_moe else 32_768,
            "moe_bank_gates": [24, 30, 35, 40] if is_moe else [],
            "admission_rule": (
                f"known runtime bytes + {safety_reserve_bytes} safety-reserve "
                "bytes must not exceed the reported OpenCL global budget; "
                "full-load qualification may lower the advertised context"
            ),
        },
        "assumptions": [
            "embedding lookup stays memory-mapped on CPU and only touched rows enter the working set",
            "all other text compute weights are resident in their checkpoint precision",
            "vision and MTP are not loaded by the initial coding service",
            "the reserve covers allocator overhead and model-wide activation/scheduler scratch not yet measured",
            "current runtime causal-convolution objects duplicate FP32 conv weights per request",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dirs", nargs="+", type=Path)
    parser.add_argument("--contexts", default="16384,32768,65536,131072")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--device-budget-bytes", type=int, default=25_563_234_304)
    parser.add_argument("--safety-reserve-gib", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    contexts = [int(value) for value in args.contexts.split(",")]
    concurrencies = [int(value) for value in args.concurrency.split(",")]
    result = {"models": []}
    for model_dir in args.model_dirs:
        row = inventory(model_dir)
        row["state_budgets"] = [
            state_budget(row["profile"], context, concurrency)
            for concurrency in concurrencies
            for context in contexts
        ]
        row["residency_plan"] = residency_plan(
            row,
            args.device_budget_bytes,
            int(args.safety_reserve_gib * 1024**3),
        )
        result["models"].append(row)

    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
