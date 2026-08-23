# Native NVFP4 serving direction

The serving backend should keep model execution framework-neutral and attach it
to vLLM as a worker/plugin boundary. vLLM remains responsible for request
scheduling, tokenization, sampling, API compatibility, and observability. The
native runtime owns quantized matrices, device buffers, KV/recurrent state, and
operator dispatch.

## Current executable slice

The complete 40-layer Ornith coding model now executes:

- checkpoint-native NVFP4 MLP projections on Adreno;
- checkpoint-native row-scaled FP8 attention projections on Adreno;
- persistent gated-delta state on Adreno;
- persistent causal-convolution state and weights on Adreno;
- real attention, normalization, residual, cache, and layer flow;
- 40 independent 256+shared expert banks and all 18.448 GiB of planned text
  compute weights resident at once;
- a shared 32K BF16 paged-KV allocation, persistent recurrent/conv state, final
  RMSNorm, the 248,320-row NVFP4 LM head, and lazy embedding-row gathers.

The original optimized cached-decode result was 55.78 ms per four-layer cadence.
The row-tiled NVFP4 decode kernel subsequently reached 47.07 ms (21.25 cadence
tokens/s) with the same output RMS. Sustained results vary with Adreno DVFS, so
serving qualification must report repeated distributions rather than one peak.

The full 40-layer plus LM-head token measures 75.884 ms kernel / 79.381 ms wall.
A tokenizer-backed coding request prefills sequentially at 13.86 tok/s and
decodes at 11.75 end-to-end tok/s. It generates valid Python, stops on the
official `<|im_end|>` token, and matches a layer-synchronized replay exactly at
every generated position.

## Runtime contract to build next

Completed foundations:

1. Typed reusable device buffers, profiling events, queued submission, and one
   synchronization point per graph.
2. Device-input/device-output NVFP4, FP8, and float32 control projections.
3. RMSNorm, gated RMSNorm, SiLU-multiply, residual addition, Q/K repeat and
   normalization, causal convolution, and gated-delta device operators.
4. Reusable resident MLP and linear-attention graph fragments.
5. Direct multithreaded ARM64 NEON NVFP4 GEMV for hybrid CPU placement.

Next serving boundary:

1. Move the complete registry/session boundary out of the benchmark into the
   packaged provider with explicit per-request position and cancellation state.
2. Replace sequential prompt execution with batched shared-weight prefill GEMM.
3. Add the official top-k 20/top-p 0.95/temperature sampler and stream token IDs
   without downloading all 248,320 logits when possible.
4. Add `prefill(sequence, token_ids)` and complete-model
   `decode_batch(sequences)` worker calls.
5. Connect those calls to a vLLM out-of-tree worker while retaining vLLM's
   scheduler and OpenAI-compatible server.

Decode batching should group active sequences at the worker boundary, not inside
individual ctypes calls. The first supported configuration remains batch size 1
for correctness; continuous batching follows after device-buffer numerics pass.

## Device allocation

| Processor | Initial responsibility |
|---|---|
| CPU | tokenizer, scheduler, sampler, request lifecycle, reference kernels |
| Adreno GPU | direct NVFP4/FP8 linears, attention/KV work, recurrent state |
| Hexagon v81 NPU | supported FP16/INT8 partitions, then custom packed NVFP4 op |

The NPU platform and FastRPC unit test pass with QAIRT 2.45.40. The custom QHPI
`NvFp4Linear` definition now shares the OpenCL logical tensor contract and
validates against QAIRT's official schema. Code generation still requires an
x86-64 Python 3.10 host: the installed ARM64 generator module omits its serializer
and the complete x86-64 binary cannot execute in ARM64 WSL.

## Safe scale-up gates

- Keep all accelerator probes in `scripts/run-isolated.ps1`.
- Require an explicit `PASS` and finite outputs at each layer-count increase.
- Increase resident model payload in small steps: 1 GiB, 2 GiB, 4 GiB, then 8 GiB.
- Do not attempt a full Vulkan offload.
- Measure committed memory and device allocation failures before advancing.
- Treat any driver reset or LiveKernelEvent 141 as a hard stop for that path.

The 24/30/35/40-bank gates and complete 32K BF16 text-model load now pass. The
next safety tests are sustained thermal generation, multi-request allocation,
request cancellation/reclamation, and dense-27B full residency. The measured
11.75 tok/s Ornith number is a real full-model greedy decode result, not an
isolated-layer extrapolation.
