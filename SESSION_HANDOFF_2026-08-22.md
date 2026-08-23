# X2 NVFP4 lab — end-of-session handoff

Date: 2026-08-22  
Hardware: Snapdragon X2 Elite Extreme, Adreno X2-90, Hexagon v81  
Repository: <https://github.com/hotschmoe/x2-nvfp4-lab>

This research session is intentionally finished. The repository is left at a
reproducible milestone: decode shape tuning is promoted and end-to-end proven;
the new register-microtile GEMM is correctness/performance proven in isolation
but remains experimental. Current scope is text-only. Vision and multi-token
prediction (MTP) are important future coding features, not part of this close.

## Measured end state

| Path | Current measured result | Bandwidth interpretation |
|---|---:|---:|
| Dense 27B retained decode | 2.243 end-to-end tok/s | full useful 44.53 GB/s = 34.5% of 129 GB/s / 19.5% of 228 GB/s |
| Dense 27B sequential prefill | 2.376 tok/s | intentionally not multi-token GEMM yet |
| Dense complete-token kernel/wall | 430.741 / 436.060 ms | NVFP4 linears 42.79 GB/s = 33.2% / 18.8% |
| MoE 35B tokenizer coding decode | 11.75 end-to-end tok/s | active useful bytes about 29.02 GB/s = 22.5% / 12.7% |
| MoE complete-token kernel/wall | 74.521 / 78.582 ms (12.726 tok/s) | standalone head tuned; bank experts still separate |

All production promotions passed complete-model A/B with bit-identical logits.
The dense trace contains 1,042 OpenCL events and attributes 196.858 ms, or
45.89% of kernel time, to NVFP4 linears. Inter-kernel gaps are small; the dense
bottleneck is kernel work, not Python launch gaps.

The final lab result adds true cross-vector weight-decode reuse:

- dense gate register-v8: 185.71 GFLOP/s, 1.222x at eight vectors;
- MoE gate register-v8: 212.07 GFLOP/s, 1.095x at 32 vectors;
- MoE down register-v8: 184.37 GFLOP/s, 1.716x at 32 vectors;
- dense down is shape-sensitive and usually retains direct-vector at large
  batches;
- register-v16 loses badly, consistent with register pressure/spilling;
- K-major input loses badly even before charging transpose cost;
- non-power-of-two tails require measured dispatch rather than a global v8 rule.

The effective 50–61 GB/s values in the GEMM reports count one useful model use
per prompt vector. They are not physical DRAM measurements; true reuse means the
register kernel requests a weight tile once per microtile.

## What would 80% bandwidth mean for MoE decode?

Use the measured 129 GB/s Adreno streaming ceiling, not the 228 GB/s nominal SoC
pin rate. Eighty percent is 103.2 GB/s.

The absolute whole-token useful-byte roof is:

```text
2.265 GB active bytes / 103.2 GB/s = 21.95 ms/token = 45.6 tok/s
```

That 45.6 tok/s is an intentionally optimistic roof: it assumes every active
weight/state byte in attention, experts, and the head sustains 80%, with no
router, reduction, synchronization, or other compute floor.

If only the MoE expert projections reach 80%, the trace gives a more realistic
estimate. Gate/up moves about 0.425 GB in 14.526 ms and down about 0.212 GB in
11.464 ms. At 103.2 GB/s those projections take about 6.17 ms instead of 25.99
ms. Holding the rest of the current 74.521 ms kernel fixed yields about 54.7
ms/token, or **18.3 kernel tok/s**. Preserving the current roughly 4.1 ms host/
queue overhead gives about **17.0 end-to-end tok/s**. If routing/reduction were
also nearly eliminated, the expert-only ceiling rises toward 19–21 tok/s.

So the useful planning answer is **~17–18 tok/s from expert-kernel bandwidth
work alone**, with **~45.6 tok/s as the all-active-bytes physical upper bound**
at 80% of the measured GPU ceiling. Reaching 80% of nominal 228 GB/s would imply
80.5 tok/s, but current hardware measurements do not justify that denominator.

## Exact pickup point

Start from the final commit on `main`, then read these in order:

1. `README.md`
2. `NVFP4_KERNEL_LAB.md`
3. `BENCHMARKS.md`
4. `UNIFIED_MEMORY_RESEARCH.md`
5. `SERVING.md`

The relevant implementation is:

- `native_nvfp4/kernels/nvfp4_gemv.cl`: experimental register and transposed
  register kernels;
- `native_nvfp4/runtime/nvfp4_runtime.cpp`: lab ABI, corrected one-subgroup
  register launch geometry, and a fixed 16-subgroup cap for 32-vector direct
  dispatch;
- `native_nvfp4/bench_nvfp4_gemm_lab.py`: randomized correctness-gated sweeps;
- `vllm_nvfp4_opencl/src/vllm_nvfp4_opencl/runtime.py`: Python lab binding.

Canonical final artifacts are:

- `campaign_results/bandwidth-first/20260822-233031-739112-nvfp4-gemm-lab.json`
- `campaign_results/bandwidth-first/20260822-233403-749729-nvfp4-gemm-lab.json`
- `campaign_results/bandwidth-first/20260822-233640-979831-nvfp4-gemm-lab.json`
- `campaign_results/bandwidth-first/20260822-233805-716364-nvfp4-gemm-lab.json`

Do not promote register-v8 directly into token-sequential prefill. First build a
semantically correct layer-major prompt graph: execute recurrent/full-attention
state transitions in token order, batch only dependency-safe projection/MLP
regions, preserve per-token state/KV writes, and compare final logits. Then A/B
tokenizer-backed time-to-first-token with transpose/layout, uploads, and all
kernel passes included. The experiment survives only if TTFT improves.

## Most exciting next experiments

1. **Real batched prefill.** Integrate register microtiles into safe MLP regions,
   add arbitrary-tail dispatch, and measure TTFT rather than isolated GFLOP/s.
2. **MoE bank reuse and fusion.** Port the short-K win to contiguous expert
   banks; fuse gate+up+SiLU and down+expert weighting/reduction. This is the most
   promising route to move the 11.75 tok/s coding path.
3. **Hybrid Hexagon + Adreno.** Measure whole-op NPU placement first, then a
   double-buffered custom-HTP dequant/Adreno-dot split and its inverse. Use
   registered shared buffers where QAIRT permits. Count expanded bytes,
   coherency transitions, fences, overlap, power, and end-to-end token latency.
   A dequant handoff can easily lose by expanding NVFP4 before the GPU reads it.
4. **New lane mappings.** Test K-striped lanes with cross-vector reuse,
   split-K reductions, full-wave versus half-wave, and small 2D row×vector
   microtiles. Preserve coalescing; the measured K-major/block-stride layout is
   a dead end.
5. **Decode arithmetic.** Try branchless nibble decode, packed integer-dot
   reformulations, image/texture reads, and fused scale/decode/FMA. Require
   compiler ISA/register/spill evidence and hardware counters where available.
6. **Thermal and energy envelopes.** Compare cold, sustained, and power-limited
   performance. Tokens/joule may choose a different GPU/NPU partition than raw
   tok/s.
7. **Client-visible serving trace.** Extend timestamps through HTTP, tokenizer,
   scheduler, embedding, sampling, detokenization, and SSE so kernel changes map
   directly to coding-client latency.
8. **Vision and MTP, later.** Once text TTFT is genuinely batched, add vision as
   a separately budgeted residency/attention phase and MTP as a speculative
   multi-head acceptance pipeline. Neither should obscure the current text
   baseline.

## Reproduction and safety

Build the runtime with:

```powershell
cmake --build native_nvfp4/runtime/build --config Release
```

Run the final power-of-two register sweep with:

```powershell
python native_nvfp4/bench_nvfp4_gemm_lab.py `
  --shapes dense-gate,dense-down,moe-gate,moe-down `
  --vectors 2,4,8,16,32 --vector-tiles 1,2,4,8,16 `
  --k-tiles 8192 --register-kinds 4 --warmups 3 --samples 15
```

The register kernels are lab-only. `VLLM_NVFP4_OPENCL_SHAPE_TUNING=0` disables
the promoted decode dispatcher. Continue to use isolated subprocess gates for
driver-risky experiments, close safetensors mappings immediately after upload,
and never infer physical bandwidth from useful-byte accounting alone.

The hardware has plenty left to reveal. The strongest lesson so far is that
phase, shape, tail occupancy, layout, and register pressure all change the
winner—and this X2 rewards measuring the weird ideas.
