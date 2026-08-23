# NVFP4 kernel laboratory

Date: 2026-08-22

This laboratory tests decode GEMV structures on the actual Snapdragon X2 Elite
Extreme / Adreno X2-90. It does not assume that a published mobile-GPU recipe is
optimal for this newer device, and it does not change the production kernel
until a candidate passes exact-shape correctness, repeated isolated timing, and
full-model A/B gates.

The current phase is text-only. Vision and MTP remain future coding requirements,
but neither is allowed to broaden this kernel campaign yet.

## Falsifiable hypotheses

Qualcomm's OpenCL optimization guide recommends manually sweeping work-group
shape, keeping the first dimension wave-aligned, preferring 128-bit vector
access, avoiding private arrays/spills, and using local memory only when reuse
repays its load and barrier costs. It also warns that no work-group size is
universally optimal. The matrix-multiply case study motivates bounded tiling and
per-work-item microtiles, but its published shapes are hypotheses here, not X2-90
results.

The first treatment matrix therefore changes one independent choice at a time:

| Axis | Treatments | Question |
|---|---|---|
| output rows/work-group | 1, 2, 4, 8, 16 half-wave subgroups | how much activation reuse is worth occupancy/work-group cost? |
| activation K tile | 256, 512, 1024, 2048, 4096, 8192 FP32 values | where do barriers and local-memory footprint cross over? |
| activation path | staged local memory or direct global reads | is explicit sharing better than Adreno's cache path? |
| inner block | scalar-unrolled or explicit `uchar8`/`float8` loads and vector dot arithmetic | does nominal vectorization survive compiler/register pressure? |
| shape | dense gate/up, dense down, MoE expert gate/up, MoE expert down, MoE LM head | is any policy actually shape-independent? |

Every experimental entry point consumes the original packed E2M1 values and
E4M3 block scales. It creates no persistent expanded weights. The production
row-tiled kernel remains the correctness oracle and baseline.

## Method

`native_nvfp4/bench_nvfp4_kernel_lab.py` loads complete real checkpoint matrices,
uploads each through the shared-SVM runtime, and checks every candidate against
the production output at `rtol=5e-5, atol=5e-5`. Only accepted candidates enter
timing. Within every timing round the candidate order is randomized; reported
latencies are OpenCL event medians rather than Python wall time.

Logical bandwidth counts native packed weights, block scales, one FP32 input,
and one FP32 output exactly once. It is useful-work bandwidth, not a claim about
physical DRAM traffic. Results report both the calibrated 129 GB/s Adreno
streaming ceiling and the nominal 228 GB/s SoC pin rate.

Two complete runs used opposite shape order. The first used 3 warmups and 15
samples; the repeat used 4 warmups and 25 samples. All 350 candidate/shape
correctness gates passed in each run.

## Reproduced results

| Real shape | Production | Repeated winner | Winner | Speedup | Logical GB/s | % of 129 / 228 |
|---|---:|---|---:|---:|---:|---:|
| dense gate/up 17408x5120 | 1.463 ms | local scalar, rows 16, K 8192 | 1.117 ms | 1.310x | 44.95 | 34.8% / 19.7% |
| dense down 5120x17408 | 1.471 ms | local scalar, rows 16, K 8192 | 1.113 ms | 1.322x | 45.14 | 35.0% / 19.8% |
| MoE expert gate/up 512x2048 | 0.0263 ms | local scalar, rows 16, K 2048 | 0.0171 ms | 1.538x | 35.09 | 27.2% / 15.4% |
| MoE expert down 2048x512 | 0.0285 ms | local scalar, rows 4, K 512 | 0.0256 ms | 1.113x | 23.44 | 18.2% / 10.3% |
| MoE LM head 248320x2048 | 8.604 ms | local scalar, rows 8, K 4096 | 7.025 ms | 1.225x | 40.86 | 31.7% / 17.9% |

The adjacent K=2048 LM-head treatment was effectively tied at 7.027 ms in the
first run, so 2048 versus 4096 is not yet a meaningful distinction. Likewise,
K tiles larger than a short shape do not imply more useful activation data; they
change dynamic local-memory reservation and must be interpreted as an occupancy
treatment.

Canonical artifacts:

- `20260822-223952-546712-nvfp4-kernel-lab.json`
- `20260822-224106-732597-nvfp4-kernel-lab.json`

## What the controls say

The result is not “vectorize everything.” Explicit vector decode was slower than
scalar unrolling for every large winning shape. On dense gate, the best scalar
treatment was 1.116 ms versus 1.209 ms for the corresponding vector treatment.
This confirms the earlier private-array failure was not merely bad syntax: inner
decode/register structure is a first-class bottleneck.

The result is also not “trust the cache and skip barriers.” Direct-global scalar
paths stayed near 1.63-1.65 ms on the dense shapes and 9.17-9.21 ms on the head,
slower than production. Local staging is clearly useful when many rows consume a
nontrivial activation. The exception is the tiny 512x2048 expert gate: a
one-row direct-vector treatment reached about 0.0175 ms, close to the 0.0171 ms
local winner. At very small launch durations, fixed scheduling and barrier costs
can dominate.

There is no universal row tile. Sixteen rows win dense and the small expert
gate, eight win the enormous-row-count head, and four win the short-K expert
down. A dispatcher keyed by `(rows, cols, vectors, format)` is justified; one
global compile-time constant is not.

If the isolated 1.31-1.32x dense projection gains survive composition, the
284.3 ms dense NVFP4-linear trace component would fall by roughly 68 ms. That
would project the current 515 ms full-token kernel to about 447 ms and sustained
decode from 1.91 toward roughly 2.18 tok/s. This is a planning estimate only;
cache pressure, power state, and the expert-bank kernels require full-model A/B.

## Next experiments we want

1. Promote separate dense gate/up/down candidates behind a runtime shape
   dispatcher, then run exact full-layer and full-model A/B traces.
2. Port the shape findings into the contiguous MoE expert-bank kernels rather
   than assuming standalone-matrix results transfer.
3. Sweep prefill GEMM vector counts 2/4/8/16/32/64 and weight-reuse microtiles;
   report TTFT and not just GEMM GFLOP/s.
4. Test full-wave versus half-wave subgroups, subgroup count tails, explicit
   `float4` loads, branchless arithmetic nibble decode, and image/texture reads.
5. Obtain compiler ISA/register/spill evidence and Qualcomm profiler counters;
   work-group size alone cannot explain occupancy.
6. Test fused NVFP4 gate+up+SiLU and expert down+weighted reduction, with exact
   intermediate/output oracles and byte accounting.
7. Test a fused 256-way router/top-8 kernel; the current top-8 sequence costs
   4.137 ms per MoE token.
8. Keep hybrid placement open: characterize QNN registered-memory behavior and
   dispatch cost, then test a double-buffered Hexagon-dequant/Adreno-dot pipeline,
   its inverse, and whole-operation placement. Count expanded bytes, coherency
   transitions, synchronization, overlap, energy, and end-to-end token latency.
9. Extend the serving trace through HTTP, tokenizer, scheduler, embedding,
   sampling, detokenization, and SSE so kernel wins are visible to coding clients.

Hybrid is an experimental treatment, not an architectural assumption. The NPU
split survives only if measured overlap exceeds format expansion and cross-runtime
handoff cost.

## Primary references

- [Qualcomm Snapdragon Mobile Platform OpenCL General Programming and Optimization Guide](https://docs.qualcomm.com/bundle/publicresource/80-NB295-11_REV_C_Qualcomm_Snapdragon_Mobile_Platform_Opencl_General_Programming_and_Optimization.pdf)
- [Qualcomm: Matrix Multiply on Adreno GPUs, Part 1](https://www.qualcomm.com/news/onq/2016/10/matrix-multiply-adreno-gpus-part-1-opencl-optimization)
- [Qualcomm: Better OpenCL Performance through Memory Optimization](https://www.qualcomm.com/news/onq/2016/06/better-opencl-performance-qualcomm-adreno-gpu-memory-optimization)
- [Khronos OpenCL C specification](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html)
