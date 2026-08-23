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

## Multi-vector GEMM / prefill sweep

The second laboratory flips the reuse direction. For GEMV, multiple output rows
share one activation tile. For GEMM, multiple prompt vectors may share one native
weight row. The sweep tests 1/2/4/8/16 vector subgroups per work-group, local
weight tiles from 512 through 32768 K values, direct-global controls, scalar and
explicit vector decode, and prompt batches of 2/4/8/16/32 vectors.

The complete run covered four checkpoint shapes and 1,600 candidate/case pairs;
all passed. A reversed-shape-order repeat retained the direct-vector candidates,
used 20 timing samples, and passed another 400 candidate/case gates.

| Shape | Vector range | Repeated policy | Performance range | Speedup range |
|---|---:|---|---:|---:|
| dense gate/up 17408x5120 | 2-32 | direct vector, tile tracks active vectors up to 16 | 131.6-159.1 GFLOP/s | 1.11-1.27x |
| dense down 5120x17408 | 2-32 | direct vector, tile tracks active vectors up to 16 | 132.2-159.6 GFLOP/s | 1.32-1.49x |
| MoE expert gate/up 512x2048 | 2-32 | direct vector, one subgroup/work-group | 145.4-187.1 GFLOP/s | 1.15-1.29x |
| MoE expert down 2048x512 | 2-4 direct vector; production from 8 onward | 93.3-110.1 GFLOP/s | 1.15x down to parity |

Dense effective model bandwidth rises from about 37 GB/s at two vectors to
about 45 GB/s at 16-32 vectors, or roughly 29-35% of the 129 GB/s streaming
ceiling and 16-20% of nominal 228 GB/s. MoE gate reaches about 54 GB/s effective
model bandwidth at 32 vectors. These effective rates count one useful matrix use
per vector; the JSON also reports the ideal compulsory-byte lower bound. Neither
is a hardware-counter measurement of physical traffic.

The vector result reverses the decode finding. Explicit vector decode loses on
large batch-one GEMV but wins multi-vector GEMM, while local weight staging loses
to direct reads on dense shapes. That is evidence for phase-specific kernels,
not a contradiction: prefill exposes more independent activation streams and a
different cache/register/occupancy balance. The short-K expert down remains a
counterexample and keeps the production local path at eight or more vectors.

Canonical artifacts:

- `20260822-225440-604739-nvfp4-gemm-lab.json`
- `20260822-225726-770266-nvfp4-gemm-lab.json`

The next gate is not another isolated GFLOP/s number. A winner must enter a
multi-token recurrent/attention-aware prefill graph and reduce tokenizer-backed
TTFT without changing logits.

## Production promotion

The decode dispatcher now selects r16/k8192 scalar-local kernels for both dense
projection directions, r8/k2048 for the MoE LM head, r16/k2048 for standalone
expert gate/up, and r4/k512 for standalone expert down. The GEMM dispatcher uses
direct-vector dense and expert-gate treatments while retaining production local
expert down from eight vectors onward. Set
`VLLM_NVFP4_OPENCL_SHAPE_TUNING=0` to restore the previous paths.

Complete dense A/B improves 514.155 to 430.741 ms kernel and 520.621 to 436.060
ms wall (1.194x), with a reverse-order pair reproducing the result. The tuned
trace shows NVFP4 linears at 196.858 ms rather than 284.313 ms. Complete MoE A/B
improves 76.468 to 74.521 ms kernel and 80.611 to 78.582 ms wall. Every complete
model comparison returns bit-identical logits.

Dense retained-state generation now reaches 2.243 decode tok/s and 2.376
sequential-prefill tok/s. The latter does not use the multi-vector GEMM yet;
integrating a semantically correct prompt graph remains the next TTFT gate.

## Final register-microtile experiment

The last experiment tested true cross-vector decode reuse rather than grouping
independent GEMVs. One subgroup owns one output row and a 2/4/8/16-vector
register microtile, decodes each packed E2M1 value and E4M3 scale once, and
applies it to every live vector accumulator. The production output remains the
oracle at `rtol=5e-5, atol=5e-5`.

An experimental launch error initially created `vector_tile` duplicate
subgroups, all computing the same output. Those invalid artifacts were removed.
After correcting register kernels to one 64-work-item subgroup per row/tile,
opposite-order sweeps passed every gate and reproduced the shape-dependent
result:

| Shape / vectors | Repeated winner | GFLOP/s | Speedup vs production | Effective model GB/s | % of 129 / 228 |
|---|---|---:|---:|---:|---:|
| dense gate, 8 | register scalar v8 | 185.71 | 1.222x | 52.32 | 40.6% / 22.9% |
| dense gate, 16 | register scalar v8 | 181.35 | 1.143x | 51.10 | 39.6% / 22.4% |
| dense gate, 32 | register scalar v8 | 180.31 | 1.166x | 50.80 | 39.4% / 22.3% |
| MoE gate, 32 | register scalar v8 | 212.07 | 1.095x | 60.68 | 47.0% / 26.6% |
| MoE down, 8 | register scalar v8 | 160.09 | 1.566x | 45.81 | 35.5% / 20.1% |
| MoE down, 32 | register scalar v8 | 184.37 | 1.716x | 52.75 | 40.9% / 23.1% |

“Effective model GB/s” counts one useful native matrix use per vector. It is a
model-throughput metric, not measured DRAM traffic: register v8 requests each
weight tile once per eight vectors, and cache behavior is not exposed by the
available counters. Dense-down's long K dimension is a counterexample: v4 wins
at some small batches, but the direct production path ties or wins at 16/32.

Arbitrary-vector tail tests at 3/5/6/7/9/12/24 also reject a simplistic rule.
Masked register tiles win when sufficiently occupied, but direct kernels often
win at 3/5/6/9. A future dispatcher must use `(rows, cols, vectors/tail)` rather
than applying v8 globally. Register v16 is consistently poor, consistent with
private-register pressure or spills.

The K-major/transposed treatment is a verified negative control. Even excluding
the transpose itself, it takes roughly 64-83 ms on the 16-vector dense-gate case
versus about 15.7 ms for vector-major register v8 and 17.8 ms for production.
The subgroup's lane-wise K stride matters more than making values across prompt
vectors adjacent. Keep this result; do not retry the same layout without a
different lane mapping.

Canonical artifacts:

- `20260822-233031-739112-nvfp4-gemm-lab.json` (layout control)
- `20260822-233403-749729-nvfp4-gemm-lab.json` (forward order)
- `20260822-233640-979831-nvfp4-gemm-lab.json` (reverse order repeat)
- `20260822-233805-716364-nvfp4-gemm-lab.json` (non-power-of-two tails)

The experiment is intentionally not promoted. The existing model prefill is
token-sequential, and the older GEMM merely batches launches without reusing
decoded weights. The next valid gate is a layer-major, recurrent/attention-aware
prefill graph that batches safe MLP regions and reports tokenizer-backed TTFT.

## Primary references

- [Qualcomm Snapdragon Mobile Platform OpenCL General Programming and Optimization Guide](https://docs.qualcomm.com/bundle/publicresource/80-NB295-11_REV_C_Qualcomm_Snapdragon_Mobile_Platform_Opencl_General_Programming_and_Optimization.pdf)
- [Qualcomm: Matrix Multiply on Adreno GPUs, Part 1](https://www.qualcomm.com/news/onq/2016/10/matrix-multiply-adreno-gpus-part-1-opencl-optimization)
- [Qualcomm: Better OpenCL Performance through Memory Optimization](https://www.qualcomm.com/news/onq/2016/06/better-opencl-performance-qualcomm-adreno-gpu-memory-optimization)
- [Khronos OpenCL C specification](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html)
