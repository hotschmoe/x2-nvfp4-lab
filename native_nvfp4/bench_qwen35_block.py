#!/usr/bin/env python3
"""Benchmark real Qwen3.5 decoder blocks on native quantized weights."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_modeling
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)


class OpenCLLinear(torch.nn.Module):
    def __init__(self, name: str, runtime, matrix, fp8: bool):
        super().__init__()
        self.name = name
        self.runtime = runtime
        self.matrix = matrix
        self.fp8 = fp8
        self.in_features = matrix.cols
        self.out_features = matrix.rows
        self.calls = 0
        self.seconds = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        started = time.perf_counter()
        leading = x.shape[:-1]
        x_f32 = x.reshape(-1, self.in_features).to(torch.float32).contiguous()
        x_np = x_f32.numpy()
        result = (
            self.runtime.linear_fp8(self.matrix, x_np)
            if self.fp8
            else self.runtime.linear(self.matrix, x_np)
        )
        out = torch.from_numpy(result).reshape(*leading, self.out_features)
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return out

    def reset_stats(self) -> None:
        self.calls = 0
        self.seconds = 0.0


def causal_mask(length: int) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), dtype=torch.float32)
    return torch.triu(mask, diagonal=1)[None, None, :, :]


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=here.parent / "models/Qwen3.8-27B-NVFP4-Unsloth",
    )
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--layer-count", type=int, default=1)
    parser.add_argument("--prompt", default="The future of efficient local AI is")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--prefill-iterations", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--gpu-gated-delta", action="store_true")
    parser.add_argument("--gpu-causal-conv", action="store_true")
    args = parser.parse_args()
    if args.prefill_iterations < 1 or args.decode_tokens < 1 or args.layer_count < 1:
        parser.error("iteration counts must be positive")

    os.environ["VLLM_NVFP4_OPENCL"] = "1"
    os.environ["VLLM_NVFP4_OPENCL_DLL"] = str(
        here / "runtime/build/nvfp4_runtime.dll"
    )
    os.environ["VLLM_NVFP4_OPENCL_KERNEL"] = str(
        here / "kernels/nvfp4_gemv.cl"
    )
    sys.path.insert(0, str(here.parent / "vllm_nvfp4_opencl/src"))
    from vllm_nvfp4_opencl.runtime import get_runtime

    config = AutoConfig.from_pretrained(args.model, local_files_only=True).text_config
    layer_indices = list(range(args.layer, args.layer + args.layer_count))
    if not layer_indices or layer_indices[-1] >= config.num_hidden_layers:
        parser.error("requested layer range is outside the model")
    config._attn_implementation = "eager"
    with torch.device("meta"):
        layers = torch.nn.ModuleList(
            [Qwen3_5DecoderLayer(config, index) for index in layer_indices]
        )
    layers.eval()
    runtime = get_runtime()
    matrices = []
    native_bytes = 0
    upload_started = time.perf_counter()

    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        for layer_index, layer in zip(layer_indices, layers):
            prefix = f"model.language_model.layers.{layer_index}"
            label_prefix = f"layer{layer_index}."

            def upload_fp8(
                name: str,
                prefix: str = prefix,
                label_prefix: str = label_prefix,
            ):
                nonlocal native_bytes
                weight = np.ascontiguousarray(
                    checkpoint.get_tensor(prefix + "." + name + ".weight")
                    .view(torch.uint8)
                    .numpy()
                )
                scale = np.ascontiguousarray(
                    checkpoint.get_tensor(prefix + "." + name + ".weight_scale")
                    .view(torch.uint16)
                    .numpy()
                )
                native_bytes += weight.nbytes + scale.nbytes
                matrix = runtime.upload_fp8(weight, scale)
                matrices.append(matrix)
                return OpenCLLinear(label_prefix + name, runtime, matrix, fp8=True)

            def upload_nvfp4(
                name: str,
                prefix: str = prefix,
                label_prefix: str = label_prefix,
            ):
                nonlocal native_bytes
                base = prefix + "." + name
                packed = np.ascontiguousarray(
                    checkpoint.get_tensor(base + ".weight_packed").numpy()
                )
                scales = np.ascontiguousarray(
                    checkpoint.get_tensor(base + ".weight_scale")
                    .view(torch.uint8)
                    .numpy()
                )
                global_scale = float(
                    checkpoint.get_tensor(base + ".weight_global_scale").item()
                )
                native_bytes += packed.nbytes + scales.nbytes
                matrix = runtime.upload(packed, scales, global_scale)
                matrices.append(matrix)
                return OpenCLLinear(label_prefix + name, runtime, matrix, fp8=False)

            if layer.block_type == "full_attention":
                layer.self_attn.q_proj = upload_fp8("self_attn.q_proj")
                layer.self_attn.k_proj = upload_fp8("self_attn.k_proj")
                layer.self_attn.v_proj = upload_fp8("self_attn.v_proj")
                layer.self_attn.o_proj = upload_fp8("self_attn.o_proj")
            else:
                layer.linear_attn.in_proj_qkv = upload_fp8("linear_attn.in_proj_qkv")
                layer.linear_attn.in_proj_z = upload_fp8("linear_attn.in_proj_z")
                layer.linear_attn.out_proj = upload_fp8("linear_attn.out_proj")
            layer.mlp.gate_proj = upload_nvfp4("mlp.gate_proj")
            layer.mlp.up_proj = upload_nvfp4("mlp.up_proj")
            layer.mlp.down_proj = upload_nvfp4("mlp.down_proj")

            for name, parameter in list(layer.named_parameters()):
                if not parameter.is_meta:
                    continue
                parent = layer
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                value = checkpoint.get_tensor(prefix + "." + name).float()
                setattr(
                    parent,
                    parts[-1],
                    torch.nn.Parameter(value, requires_grad=False),
                )

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
        if not token_ids:
            raise RuntimeError("prompt tokenized to an empty sequence")
        if args.sequence_length > 0:
            repeats = (args.sequence_length + len(token_ids) - 1) // len(token_ids)
            token_ids = (token_ids * repeats)[: args.sequence_length]
        embedding = checkpoint.get_slice("model.language_model.embed_tokens.weight")
        hidden = torch.cat(
            [embedding[token_id : token_id + 1, :].float() for token_id in token_ids],
            dim=0,
        )[None, :, :]

    upload_elapsed = time.perf_counter() - upload_started
    rotary = Qwen3_5TextRotaryEmbedding(config)
    native_linears = [
        module for module in layers.modules() if isinstance(module, OpenCLLinear)
    ]
    device_states = {}
    device_conv_states = {}
    gated_delta_calls = 0
    gated_delta_seconds = 0.0
    causal_conv_calls = 0
    causal_conv_seconds = 0.0
    original_recurrent = qwen35_modeling.torch_recurrent_gated_delta_rule
    original_causal_conv_update = qwen35_modeling.causal_conv1d_update

    def opencl_recurrent_gated_delta(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        nonlocal gated_delta_calls, gated_delta_seconds
        del kwargs
        if initial_state is None or query.shape[0] != 1 or query.shape[-1] != 128:
            raise ValueError("OpenCL gated-delta requires batch 1 and 128-wide heads")
        started = time.perf_counter()
        initial_dtype = query.dtype
        q_np = np.ascontiguousarray(query[0].float().numpy())
        k_np = np.ascontiguousarray(key[0].float().numpy())
        if use_qk_l2norm_in_kernel:
            q_np = np.ascontiguousarray(
                q_np
                / np.sqrt(np.sum(q_np * q_np, axis=-1, keepdims=True) + 1e-6)
            )
            k_np = np.ascontiguousarray(
                k_np
                / np.sqrt(np.sum(k_np * k_np, axis=-1, keepdims=True) + 1e-6)
            )
        v_np = np.ascontiguousarray(value[0].float().numpy())
        g_np = np.ascontiguousarray(g[0].float().numpy())
        beta_np = np.ascontiguousarray(beta[0].float().numpy())
        state_key = initial_state.data_ptr()
        device_state = device_states.get(state_key)
        if device_state is None:
            initial_np = np.ascontiguousarray(initial_state[0].float().numpy())
            device_state = runtime.create_gated_delta_state(
                query.shape[2], initial_np
            )
            device_states[state_key] = device_state
        output = runtime.gated_delta(
            device_state, q_np, k_np, v_np, g_np, beta_np
        )
        gated_delta_calls += 1
        gated_delta_seconds += time.perf_counter() - started
        output_tensor = torch.from_numpy(output)[None].to(initial_dtype)
        return output_tensor, initial_state if output_final_state else None

    if args.gpu_gated_delta:
        qwen35_modeling.torch_recurrent_gated_delta_rule = (
            opencl_recurrent_gated_delta
        )

    def opencl_causal_conv_update(
        hidden_states,
        conv_state,
        weight,
        bias=None,
        activation=None,
    ):
        nonlocal causal_conv_calls, causal_conv_seconds
        if (
            hidden_states.shape[0] != 1
            or conv_state.shape[0] != 1
            or conv_state.shape[-1] != 4
            or bias is not None
            or activation != "silu"
        ):
            raise ValueError("OpenCL causal convolution requires Qwen3.5 decode")
        started = time.perf_counter()
        initial_dtype = hidden_states.dtype
        state_key = conv_state.data_ptr()
        device_state = device_conv_states.get(state_key)
        if device_state is None:
            weights_np = np.ascontiguousarray(weight.float().numpy())
            initial_np = np.ascontiguousarray(conv_state[0].float().numpy())
            device_state = runtime.create_causal_conv_state(weights_np, initial_np)
            device_conv_states[state_key] = device_state
        x_np = np.ascontiguousarray(
            hidden_states[0].transpose(0, 1).float().numpy()
        )
        output = runtime.causal_conv_silu(device_state, x_np)
        causal_conv_calls += 1
        causal_conv_seconds += time.perf_counter() - started
        return torch.from_numpy(output).transpose(0, 1)[None].to(initial_dtype)

    if args.gpu_causal_conv:
        qwen35_modeling.causal_conv1d_update = opencl_causal_conv_update

    def reset_linear_stats() -> None:
        for module in native_linears:
            module.reset_stats()

    def capture_linear_stats() -> list[tuple[str, int, float]]:
        return [
            (module.name, module.calls, module.seconds) for module in native_linears
        ]

    def run_stack(
        input_hidden: torch.Tensor, cache=None, start_position: int = 0
    ) -> torch.Tensor:
        length = input_hidden.shape[1]
        positions = torch.arange(
            start_position, start_position + length, dtype=torch.long
        )[None, :]
        position_embeddings = rotary(input_hidden, positions)
        full_mask = causal_mask(length) if length > 1 else None
        state = input_hidden
        for layer in layers:
            state = layer(
                state,
                position_embeddings=position_embeddings,
                attention_mask=full_mask
                if layer.block_type == "full_attention"
                else None,
                position_ids=positions,
                past_key_values=cache,
            )
        return state

    with torch.inference_mode():
        warmup = run_stack(hidden)
        reset_linear_stats()
        prefill_started = time.perf_counter()
        for _ in range(args.prefill_iterations):
            run_stack(hidden)
        prefill_elapsed = time.perf_counter() - prefill_started
        prefill_linear_stats = capture_linear_stats()

        cache = DynamicCache(config=config)
        state = run_stack(hidden, cache=cache)[:, -1:, :]
        decode_input = hidden[:, -1:, :]
        reset_linear_stats()
        gated_delta_calls = 0
        gated_delta_seconds = 0.0
        causal_conv_calls = 0
        causal_conv_seconds = 0.0
        decode_started = time.perf_counter()
        for step in range(args.decode_tokens):
            position = hidden.shape[1] + step
            state = run_stack(
                decode_input,
                cache=cache,
                start_position=position,
            )
        decode_elapsed = time.perf_counter() - decode_started
        decode_linear_stats = capture_linear_stats()

    prompt_tokens = len(token_ids)
    prefill_seconds = prefill_elapsed / args.prefill_iterations
    layer_types = ",".join(config.layer_types[index] for index in layer_indices)
    print(f"device={runtime.lib.nvfp4_runtime_device_name(runtime.handle).decode()} "
          f"layers={args.layer}:{args.layer + args.layer_count} types={layer_types}")
    print(f"prompt_tokens={prompt_tokens} native_matrix_bytes={native_bytes} "
          f"upload_seconds={upload_elapsed:.3f}")
    print(f"prefill_seconds={prefill_seconds:.6f} "
          f"prefill_tokens_per_second={prompt_tokens/prefill_seconds:.3f}")
    print(f"decode_seconds={decode_elapsed:.6f} decode_tokens={args.decode_tokens} "
          f"decode_tokens_per_second={args.decode_tokens/decode_elapsed:.3f}")
    print(f"output_rms={state.square().mean().sqrt().item():.8g} "
          f"finite={bool(torch.isfinite(state).all())} warmup_rms={warmup.square().mean().sqrt().item():.8g}")
    print(
        f"gated_delta_backend={'opencl' if args.gpu_gated_delta else 'torch'} "
        f"calls={gated_delta_calls} total_ms={gated_delta_seconds * 1e3:.3f}"
    )
    print(
        f"causal_conv_backend={'opencl' if args.gpu_causal_conv else 'torch'} "
        f"calls={causal_conv_calls} total_ms={causal_conv_seconds * 1e3:.3f}"
    )
    for phase, stats in (("prefill", prefill_linear_stats), ("decode", decode_linear_stats)):
        for name, calls, seconds in stats:
            print(f"linear_phase={phase} op={name} calls={calls} "
                  f"total_ms={seconds*1e3:.3f} avg_us={seconds/calls*1e6:.3f}")
    if not torch.isfinite(state).all():
        raise SystemExit("decoder block produced non-finite output")
    print("PASS: real Qwen3.5 decoder stack ran with native NVFP4+FP8 OpenCL linears")

    qwen35_modeling.torch_recurrent_gated_delta_rule = original_recurrent
    qwen35_modeling.causal_conv1d_update = original_causal_conv_update
    for device_state in device_states.values():
        device_state.close()
    for device_state in device_conv_states.values():
        device_state.close()
    for matrix in matrices:
        matrix.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
