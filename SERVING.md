# Native NVFP4 serving direction

The serving backend should keep model execution framework-neutral and attach it
to vLLM as a worker/plugin boundary. vLLM remains responsible for request
scheduling, tokenization, sampling, API compatibility, and observability. The
native runtime owns quantized matrices, device buffers, KV/recurrent state, and
operator dispatch.

## Current executable slice

The four-layer Qwen3.5 cadence already executes:

- checkpoint-native NVFP4 MLP projections on Adreno;
- checkpoint-native row-scaled FP8 attention projections on Adreno;
- persistent gated-delta state on Adreno;
- persistent causal-convolution state and weights on Adreno;
- real Transformers attention, normalization, residual, cache, and layer flow.

The original optimized cached-decode result was 55.78 ms per four-layer cadence.
The row-tiled NVFP4 decode kernel subsequently reached 47.07 ms (21.25 cadence
tokens/s) with the same output RMS. Sustained results vary with Adreno DVFS, so
serving qualification must report repeated distributions rather than one peak.

The runtime can now queue a complete linear-attention decoder layer without a
host boundary. The exact layer-0 graph includes input RMSNorm, FP8 projections,
the two small control projections, persistent convolution, Q/K layout and
normalization, persistent gated-delta, gated RMSNorm, output projection,
NVFP4 MLP, and both residuals. Its combined result is 12.18 ms of kernel time
and 12.28 ms queued wall with 1.43e-6 maximum error against the staged oracle.

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

1. Add one sequence-state object per request: full-attention KV cache, existing
   gated-delta state, causal
   convolution history, position, and cancellation flag.
2. Implement resident full-attention decode (RoPE, KV append, grouped-query
   softmax/value reduction) to complete the four-layer cadence.
3. Add `prefill(sequence, token_ids)` and `decode_batch(sequences)` worker calls.
4. Connect those calls to a vLLM out-of-tree worker while retaining vLLM's
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

The next scale test remains a four-layer resident cadence including the new
full-attention graph. No isolated-layer number should be extrapolated into a
claimed full-model serving rate.
