# Native mixed-quant Qwen3.5 benchmarks

Measured 2026-08-22 on the Snapdragon X2 Elite Extreme / Adreno X2-90 Windows
ARM64 machine. Every GPU run used `scripts/run-isolated.ps1` and required an
explicit `PASS` line. Weights came directly from
`Qwen3.8-27B-NVFP4-Unsloth/model.safetensors`.

## Direct linear baselines

| Format and shape | Mode | Result |
|---|---|---:|
| NVFP4 256x5120 | decode subgroup | 44.0 GFLOP/s |
| NVFP4 256x5120x8 | four-vector tile | 90.1 GFLOP/s |
| FP8 256x5120 | decode subgroup, including C ABI copies | 17.8 GFLOP/s |
| FP8 256x5120x8 | four-vector tile, including C ABI copies | 113.4 GFLOP/s |

The FP8 path consumes signed E4M3 bytes and BF16 per-output-row scales directly.
The NVFP4 path consumes packed E2M1 values, E4M3 block scales, and the checkpoint
global divisor directly. Neither path creates dequantized persistent weights.

## Real decoder execution

`native_nvfp4/bench_qwen35_block.py` instantiates Hugging Face's Qwen3.5 decoder
classes on the meta device, replaces large linear modules with persistent OpenCL
matrices, and materializes only the actual BF16 norms, recurrent parameters,
convolution weights, and small 48x5120 gates on CPU.

Controlled benchmark: 32 prompt tokens, layers 0 through 3, three repeated
prefills, and eight sequential cached decode steps.

| Metric | Four-layer result |
|---|---:|
| Native matrix payload | 1,052,676,096 bytes |
| Upload/load time | 1.067 s |
| Prefill latency | 1.291 s |
| Prefill throughput | 24.78 stack-tokens/s |
| Cached decode latency | 238.05 ms/token |
| Cached decode throughput | 4.20 stack-tokens/s |

The four layers are the model's real cadence: linear attention, linear attention,
linear attention, then full attention. Outputs remained finite. The run exercises
native NVFP4 MLP projections, native FP8 attention projections, causal convolution,
gated-delta recurrent state, rotary attention, KV cache, RMS normalization,
residuals, and SiLU gating.

This is not yet a full 64-layer generation number. Repeating this original,
unoptimized cadence 16 times projected roughly 0.26 decode tokens/s and 1.55
prefill tokens/s before the final norm/lm-head and before accounting for the last
eight layers' FP8 MLPs. The optimized projection is recorded below. Both are
planning estimates, not full-model benchmarks.

## Stateful decode optimization

The first serving-oriented step moved the two Qwen3.5 recurrent operators onto
the persistent OpenCL runtime:

| Exact model shape | Result including C ABI copies |
|---|---:|
| gated-delta, 48 heads x 128, one token | 364.8 us |
| gated-delta, 48 heads x 128, 32 tokens | 1.226 ms |
| causal convolution, 10240 channels x width 4, one token | 133.6 us |
| causal convolution, 10240 channels x width 4, 32 tokens | 159.1 us |

Both operators retain state on Adreno across calls. Independent CPU oracles and
split-call persistence checks pass.

Repeating the controlled four-layer benchmark with both operators enabled:

| Metric | CPU stateful ops | OpenCL stateful ops |
|---|---:|---:|
| Cached decode latency | 238.05 ms/token | 55.78 ms/token |
| Cached decode throughput | 4.20 stack-tokens/s | 17.93 stack-tokens/s |

This is a 4.27x decode improvement. The output RMS is unchanged at the printed
precision (`0.43193221`). The optimized run is
`logs/20260822-155929.stdout.log`.

Repeating the optimized cadence 16 times gives a planning estimate near 1.12
decode tokens/s, with the same full-model exclusions described above.

## Pre-resident bottleneck (historical)

Across the optimized four-layer cached-decode run:

- all 25 native linear calls: 52.47 ms/token;
- OpenCL gated-delta and causal convolution: 1.70 ms/token;
- all other attention/norm/residual/orchestration work: about 1.61 ms/token.

At this stage native linears dominated decode. The device-buffer ABI and complete
four-layer resident cadence described below have now removed these per-op host
round trips. Continuous-batch scheduling remains serving work. The verified
Hexagon v81 path is documented in `native_nvfp4/npu/README.md`; direct NPU NVFP4
requires a custom HTP op package.

Primary logs:

- `logs/20260822-152251.stdout.log`: full-attention block, stable run
- `logs/20260822-152430.stdout.log`: linear-attention block, stable run
- `logs/20260822-152631.stdout.log`: real four-layer cadence
- `logs/20260822-155929.stdout.log`: cadence with persistent GPU recurrent state

## Device-resident decode graphs

The runtime now exposes reusable buffers and enqueued operators. OpenCL event
profiling separates transfer, kernel, and host synchronization time.

The four-row NVFP4 decode kernel shares each activation tile across four output
rows. On exact checkpoint matrices it reduced kernel time by 26-28%:

| Exact NVFP4 projection | Original subgroup | Four-row tile |
|---|---:|---:|
| gate/up, 17408x5120 | 3.096 ms | 2.299 ms |
| down, 5120x17408 | 3.163 ms | 2.284 ms |

Queued submission removes almost all per-call host wait overhead. For a
256x5120 slice, blocking resident execution cost 184.2 us wall while 100 queued
calls averaged 34.6 us wall and 31.9 us kernel.

Exact resident graph results:

| Graph | Kernel time | Queued wall | Max absolute error |
|---|---:|---:|---:|
| layer-0 NVFP4 MLP, including norms/residual | 9.855 ms | 9.704 ms | `7.63e-6` |
| layer-0 linear attention, including recurrent state | 2.472 ms | 2.526 ms | `3.34e-6` |
| complete layer-0 linear attention + NVFP4 MLP | 11.963 ms | 12.249 ms | `1.53e-5` |
| layer-3 full attention, FP8 Q/K/V/O and KV cache | 2.091 ms | 2.159 ms | `7.63e-6` |
| layers 0-3, all attention and MLP work resident | 48.122 ms | 48.537 ms | reset replay `0` |

The combined graph is deliberately reported even though it is slower than the
sum of separately sampled fragments. With both weight groups live, Adreno cache,
power, and DVFS behavior changes. Future serving benchmarks must include warmup,
temperature/frequency context, and multiple samples.

The layer-3 full-attention kernel implements the checkpoint's 24 query heads,
four KV heads, 256-wide head dimension, interleaved query gate, and 64-wide
partial RoPE. K/V state remains on device and grows across decode calls. Its
online stable softmax does not allocate a context-sized local-memory logits
array; an 8,193-token-capacity regression probe passes, beyond the earlier
32-KiB local-memory ceiling. Cache capacity is therefore limited by allocatable
device memory rather than workgroup local memory.

The exact layer 0-3 cadence is now 20.61 four-layer cadence-tokens/s in the
reported queued run, about 4.90x faster than the earlier 238.05 ms host-mediated
baseline. Repeating this measured group 16 times is a planning estimate near
1.29 decode tokens/s before final norm/lm-head and serving overhead; it is not a
full-model benchmark.

Device-only recurrent kernels, excluding host copies:

| Operator | Decode | 32-token prefill |
|---|---:|---:|
| gated-delta, 48x128 heads | 173.8 us | 809.8 us |
| causal convolution, 10240x4 | 9.1 us | 22.4 us |

The row-tiled NVFP4 default improved one controlled four-layer run from 17.93 to
21.25 cadence-tokens/s with identical output RMS (`logs/20260822-162523.stdout.log`).
A later sustained run reached 18.74 cadence-tokens/s, demonstrating the DVFS
variance described above.

Current resident logs:

- `logs/20260822-174044.stdout.log`: online-softmax/KV-cache oracle
- `logs/20260822-174104.stdout.log`: exact full-attention checkpoint graph
- `logs/20260822-174105.stdout.log`: complete layers 0-3 resident cadence
- `logs/20260822-174431.stdout.log`: two independent request states sharing 25 matrices

## Paged multi-request serving

Full attention now also has a vLLM-shaped 16-token paged-cache path. A shared
pool owns physical K/V pages while each request owns a device block table and
its recurrent/conv state. Pages are allocated lazily, returned on reset or
request completion, and may immediately be reused by another request. One page
pair for one full-attention layer is 128 KiB; across all 16 full-attention layers
it represents 2 MiB per 16 logical tokens.

The scheduler batches shared FP8 attention projections and NVFP4 MLP projections,
while stateful convolution, gated-delta, and attention-cache updates remain
request-specific. Every result below matched independent contiguous-cache
sessions exactly through the 16-token page boundary:

| Active requests | Four-layer kernel/step | Scheduler wall/step | Aggregate throughput |
|---:|---:|---:|---:|
| 1 | 48.328 ms | 52.674 ms | 18.98 req-tokens/s |
| 2 | 78.505 ms | 85.781 ms | 23.32 req-tokens/s |
| 4 | 135.270 ms | 145.069 ms | 27.57 req-tokens/s |

Batch four improves aggregate scheduler throughput by 45.2% over the paged
single-request path. This is continuous-batch throughput, not latency: an
individual request waits for the whole batch. The next optimization is batching
the remaining small control projections and state kernels, plus asynchronous
host staging.

The adapter accepts the stable request lifecycle fields from vLLM V1
`SchedulerOutput`. Decode batches, request reordering, finish/abort, and
request-major multi-token prompt chunks pass. Preemption/resume is deliberately
rejected until recurrent and KV state-copy semantics are implemented.

Paged serving logs:

- `logs/20260822-175238.stdout.log`: synthetic two-request block-table oracle
- `logs/20260822-180405.stdout.log`: exact two-request checkpoint scheduler
- `logs/20260822-180424.stdout.log`: exact four-request batched scheduler
- `logs/20260822-180445.stdout.log`: paged single-request regression
- `logs/20260822-180536.stdout.log`: vLLM lifecycle and prompt-chunk adapter

## ARM64 CPU hybrid path

The NEON fallback directly decodes the packed E2M1 nibbles and E4M3 block scales;
it does not build GGUF or expanded weights.

| Exact projection | Threads | Latency | Throughput | Max absolute error |
|---|---:|---:|---:|---:|
| gate/up, 17408x5120 | automatic | 2.845 ms | 62.65 GFLOP/s | `2.86e-6` |
| down, 5120x17408 | automatic | 2.705 ms | 65.89 GFLOP/s | `2.48e-5` |

Thread creation is visible on tiny matrices, so the serving worker should use a
persistent CPU pool. At full MLP shapes, CPU performance is close enough to the
Adreno path to support memory-budget spill and cross-request heterogeneous work.

## Bandwidth campaign and shared SVM

The first matched raw-read ceiling is approximately 129 GB/s by OpenCL event
timing with 16-byte vector loads. A 256 MiB conventional buffer measured
127.17 GB/s and a 256 MiB fine-grained SVM buffer measured 129.29 GB/s across 30
samples. These are logical kernel read rates, not physical DRAM-counter results.

The exact 17408x5120 native NVFP4 gate/up projection was then measured with both
backings interleaved in one runtime:

| Backing | Median kernel | p10-p90 | Logical bandwidth | Raw-ceiling use |
|---|---:|---:|---:|---:|
| copied buffer | 2.3258 ms | 2.2935-2.3385 ms | 21.56 GB/s | 16.7% |
| shared SVM | 1.4651 ms | 1.4621-1.4675 ms | 34.22 GB/s | 26.5% |

The SVM path is 1.59x faster by kernel time and 1.53x faster by synchronous call
wall time. GPU outputs are bit-identical across backings. Maximum absolute error
is `8.34e-7` on GPU and `2.86e-6` on the NEON executor reading the same SVM
allocation. The raw SVM probe also passed 30 separate isolated-process runs.

Primary artifacts:

- `campaign_results/bandwidth-first/20260822-193648-695518-nvfp4-svm-comparison.json`
- `logs/20260822-193647-176.stdout.log`
- `native_nvfp4/bench_islands.py`
- `native_nvfp4/bench_svm_matrix.py`

Promoting SVM to the runtime's default NVFP4 backing carried the improvement
through the resident graphs:

| Graph | Previous copied result | Shared-SVM result |
|---|---:|---:|
| exact layer-0 MLP kernel | 9.855 ms | 5.051 ms |
| layer-0 linear attention + MLP kernel | 11.963 ms | 7.848 ms |
| layers 0-3 cadence kernel | 48.122 ms | 30.151 ms |
| layers 0-3 cadence queued wall | 48.537 ms | 30.465 ms |

The new four-layer planning projection is about 2.05 full-model decode tokens/s
when repeated 16 times, versus 1.29 before SVM. It still excludes the final
norm, LM head, sampling, final eight-layer format difference, and serving
overhead, so it is not a full-model benchmark.

Paged continuous-batch results with shared SVM:

| Active requests | Four-layer kernel/step | Scheduler wall/step | Aggregate throughput |
|---:|---:|---:|---:|
| 1 | 29.559 ms | 31.273 ms | 31.98 req-tokens/s |
| 4 | 100.975 ms | 104.947 ms | 38.11 req-tokens/s |

Relative to the prior copied-buffer scheduler, batch-one throughput improves
68.5% and batch-four aggregate throughput improves 38.2%. Both cells matched
independent per-request contiguous-cache oracles exactly.

## Qwen3.5-MoE 35B expert micrograph

`Ornith-1.5-35B-A3B-NVFP4` has 40 decoder layers, hidden width 2048,
256 experts per layer, top-8 routing, and expert intermediate width 512. Its
expert tensors use the same packed E2M1 and E4M3-block representation under the
names `weight`, `weight_scale`, and `weight_scale_2`. The last value is a
multiplier, so the native runtime receives its reciprocal as the equivalent
global divisor.

Eight real layer-0 experts were loaded into shared SVM and executed as serial
gate/up, SiLU-multiply, and down graphs on one in-order queue:

| Metric | Top-8 result |
|---|---:|
| Active native payload | 14,155,776 bytes |
| Median kernel time | 0.6477 ms |
| p10-p90 kernel time | 0.6452-0.6537 ms |
| Median queued wall | 0.7742 ms |
| Per-expert kernel time | 0.0810 ms |
| Logical native bandwidth | 21.86 GB/s |
| Expert-0 maximum absolute error | `4.66e-10` |

Repeating only this top-8 expert work across 40 layers is a planning estimate of
25.9 ms, or 38.6 tokens/s. It deliberately excludes router selection and
weights, shared-expert execution/gating, expert-output reduction, attention,
normalization, LM head, MTP, and serving overhead. The immediate result proves
that the checkpoint's sparse expert shapes work natively; it is not an
end-to-end MoE throughput claim.

Artifacts:

- `native_nvfp4/bench_moe_experts.py`
- `campaign_results/bandwidth-first/20260822-195228-287591-moe-experts.json`
- `logs/20260822-195227-227.stdout.log`

### Checkpoint-routed layer micrograph

The next gate adds the checkpoint BF16 router and shared-expert gate, exact
softmax/top-8 selection and renormalization, the always-on NVFP4 shared expert,
eight routed NVFP4 experts, and device-side weighted accumulation. Layer 0 chose
experts `[60, 111, 188, 144, 224, 92, 105, 252]` for the deterministic test
activation.

| Stage | Median time |
|---|---:|
| BF16 router + shared gate + shared NVFP4 expert | 0.1162 ms kernel |
| Eight selected experts + weighted reductions | 0.7064 ms kernel |
| Complete routed MoE micrograph | 0.8230 ms kernel |
| Host softmax/top-8 work | 0.0192 ms |
| Two-stage end-to-end wall | 1.2767 ms |

The 30-sample run consumed 16,977,920 logical native bytes per step. GPU router
IDs matched the independent BF16 CPU route exactly, selected routing weights
had `7.45e-9` maximum absolute error, and the final 2048-element output had
`1.16e-9` maximum absolute error. The host top-k forces a device materialization
boundary; moving selection and indirect dispatch onto the device is the next
graph-level optimization.

This remains a one-token MoE-layer micrograph. Attention, input/output norms,
residuals, sampling, complete-model residency, and serving overhead are not
included. Reproduction artifact:
`campaign_results/bandwidth-first/20260822-200044-231748-moe-routed-layer.json`.

### Device-routed contiguous SVM expert bank

The device-bank successor keeps all 256 routed layer-0 experts plus the shared
expert in six contiguous SVM weight/scale arrays. A device top-8 kernel writes
expert IDs and normalized weights; row-tiled gate/up and down kernels derive
the selected matrix offsets inside the bank. The shared expert is slot 8, so one
device reduction completes the layer without router readback.

| Metric | Device-bank result | Host-dispatch result |
|---|---:|---:|
| Resident layer bank | 454,754,304 bytes | selected matrices only |
| Steady GPU launches | 7 | approximately 47 |
| Queue materializations | 1 | 2 |
| Median kernel time | 0.7563 ms | 0.8230 ms |
| Median wall time | 0.9231 ms | 1.2767 ms |
| Final maximum absolute error | `1.16e-9` | `1.16e-9` |

The 30-sample device-bank run improves kernel time by 8.1% and wall time by
27.7%. Its first one-subgroup-per-row form was a negative control at 1.4037 ms;
applying the proven four-row/1,024-K activation tile reduced it to 0.7563 ms.
This isolates activation reuse and launch count as material effects.

Streaming expert tensors from safetensors into the bank avoids assembling a
454.8 MB host concatenation. Bank creation and streaming took 0.2220 seconds in
the canonical cached-file run. Available physical memory fell by approximately
475 MB after allocation, close to one bank rather than two. Forty such banks are
16.94 GiB before router, attention, recurrent, embedding, LM-head, KV, and
scratch storage; the full checkpoint is 21.81 GiB of safetensors against an
observed 23.81 GiB OpenCL global budget. Full residency is therefore plausible
but too close to the budget to attempt without the staged 1/2/4/8 GiB safety
ladder and explicit KV/headroom accounting.

Canonical artifact:
`campaign_results/bandwidth-first/20260822-201826-627077-moe-device-bank.json`.

### Cumulative real-bank residency ladder

One isolated process streamed actual checkpoint experts for layers 0-18 into
independent banks, retained every earlier bank, and validated the newly added
layer before advancing. It stopped at the predefined 8 GiB-class gate.

| Banks/layers resident | Native bank payload | Available physical memory | Newest-layer error |
|---:|---:|---:|---:|
| 3 | 1,364,262,912 bytes | 38,026,776,576 bytes | `1.40e-9` |
| 5 | 2,273,771,520 bytes | 37,138,202,624 bytes | `8.15e-10` |
| 10 | 4,547,543,040 bytes | 34,837,250,048 bytes | `2.56e-9` |
| 19 | 8,640,331,776 bytes | 30,696,697,856 bytes | `9.31e-10` |

The worst error over all 19 independently routed layer validations was
`2.56e-9`. Available physical memory decreased by 8,706,265,088 bytes at the
maximum gate, close to the 8,640,331,776-byte native payload. Bank destruction
immediately recovered 8,540,033,024 bytes and returned the system to within
166 MB of its pre-run available-memory sample. This is evidence against a large
per-bank leak through 8.64 GB; it is not authorization to jump to 40 banks.

Artifact:
`campaign_results/bandwidth-first/20260822-202356-768080-moe-bank-residency.json`.

## Exact checkpoint and serving-state memory ledger

`inventory_checkpoint_memory.py` reads safetensors metadata without loading
tensor payloads, classifies every tensor by serving ownership, and derives KV,
gated-delta, causal-convolution, block-table, and known scratch allocations from
each model config. Both inventories have zero unclassified tensors.

| Model | All tensor payloads | Text compute weights resident | Lazy CPU embedding | Vision omitted | MTP omitted |
|---|---:|---:|---:|---:|---:|
| Ornith 35B MoE | 21.800 GiB | 18.448 GiB | 0.947 GiB | 0.832 GiB | 1.573 GiB |
| Qwen 27B dense | 21.809 GiB | 17.792 GiB | 2.368 GiB | 0.858 GiB | 0.791 GiB |

"Text compute weights" includes the LM head, every attention/MLP weight, norms,
and MoE routing/shared weights, but not the token embedding table. The initial
text-only server can memory-map the embedding and touch only requested token
rows; it must not eagerly copy the entire table into an OpenCL allocation.

The planner uses the observed 25,563,234,304-byte OpenCL global budget and holds
back 2 GiB for allocator overhead plus model-wide scratch not yet measured. At
concurrency one:

| Model / KV format | 32K known runtime | Headroom after 2 GiB reserve | 64K known runtime | Headroom after reserve |
|---|---:|---:|---:|---:|
| 35B MoE / current FP32 | 19.764 GiB | 2.044 GiB | 21.014 GiB | 0.794 GiB |
| 35B MoE / BF16 target | 19.139 GiB | 2.669 GiB | 19.764 GiB | 2.044 GiB |
| 27B dense / current FP32 | 21.948 GiB | -0.140 GiB | 25.948 GiB | -4.140 GiB |
| 27B dense / BF16 target | 19.948 GiB | 1.860 GiB | 21.948 GiB | -0.140 GiB |
| 27B dense / FP8 target | 18.948 GiB | 2.860 GiB | 19.948 GiB | 1.860 GiB |

These are allocation ledgers, not successful full-model residency results. The
current FP32 policy is therefore 32K for 35B MoE and 16K for dense 27B until the
full staged load passes. BF16 KV is the next high-value memory implementation:
it makes dense 32K fit the reserve policy and gives MoE 64K comfortable room.
FP8 KV remains a later correctness-qualified option for dense 64K or higher
concurrency.

Canonical artifact:
`campaign_results/bandwidth-first/checkpoint-memory-inventory.json`.

## Dense BF16 paged KV gate

The dense Qwen paged-attention pool now accepts `fp32` or `bf16`. BF16 K/V is
rounded to nearest-even on device; query/gate storage, softmax, and value
accumulation remain FP32. Four pages occupy 262,144 bytes in BF16 versus 524,288
bytes in FP32. Interleaved two-request decoding crossed the 16-token page
boundary, matched a separately rounded BF16 cache oracle within `1.19e-7`, and
returned every page on reset.

The exact four-layer cadence (one full-attention layer) was then compared with
its FP32-cache oracle over 18 tokens:

| KV / batch | Pool bytes | Scheduler kernel/step | Wall/step | Max abs vs FP32 | Relative RMSE |
|---|---:|---:|---:|---:|---:|
| FP32 / 1 | 262,144 | 30.484 ms | 33.122 ms | 0 | 0 |
| BF16 / 1 | 131,072 | 30.428 ms | 33.055 ms | `0.00257` | `4.08e-5` |
| FP32 / 4 | 1,048,576 | 102.037 ms | 110.112 ms | 0 | 0 |
| BF16 / 4 | 524,288 | 102.266 ms | 111.346 ms | `0.00642` | `5.64e-5` |

This warm-burst gate shows a 2x capacity reduction with no meaningful cadence
change. The vLLM adapter also passes request reorder, request-major prompt
chunks, abort, and full page reclamation using BF16 KV. A dense 32K allocation
projects from exactly 4 GiB to 2 GiB across all 16 full-attention layers, but the
complete 64-layer model and long-context quality have not yet been tested.

Canonical artifacts:

- `20260822-204038-388368-paged-fp32-batch1.json`
- `20260822-204046-722776-paged-fp32-batch4.json`
- `20260822-204050-233641-paged-bf16-batch1.json`
- `20260822-204058-576796-paged-bf16-batch4.json`

## Exact 35B MoE attention and complete decoder layer

The paged-attention ABI is now head-profile aware. Dense Qwen uses 24 query
heads/four KV heads; Ornith MoE uses 16/two. Both retain 256-wide heads, the
interleaved query/gate projection, 64 rotary dimensions, and 16-token pages.
The four-page/two-request shape gate produced:

| Profile / KV | Pool bytes | Attention-state kernel | Error vs storage oracle | Relative RMSE vs FP32 cache |
|---|---:|---:|---:|---:|
| Dense 24/4 FP32 | 524,288 | 0.02915 ms | `8.94e-8` | `1.59e-7` |
| Dense 24/4 BF16 | 262,144 | 0.02925 ms | `1.19e-7` | `0.00198` |
| MoE 16/2 FP32 | 262,144 | 0.02875 ms | `8.94e-8` | `1.57e-7` |
| MoE 16/2 BF16 | 131,072 | 0.02955 ms | `1.04e-7` | `0.00203` |

The real Ornith checkpoint then exposed and closed another format gap: its
attention matrices are tensor-scaled FP8 E4M3 with one exact FP32 scale, unlike
the dense model's BF16 row scales. The native FP8 matrix ABI now represents both
forms without expanding or rounding the scalar scale.

An independent CPU E4M3 decode of a real 256x2048 Q-projection slice agrees with
the native tensor-scaled kernel within `6.56e-7` for GEMV and `1.31e-6` for four
vectors. This prevents the composed graph from serving as its own FP8 oracle.

Real layer-3 attention—including input norm, tensor-scaled FP8 Q/K/V, RoPE,
paged GQA, gating, tensor-scaled FP8 output projection, and residual—passes over
18 tokens:

| KV | Pool bytes | Median kernel | Median wall | Max error vs storage oracle | Relative RMSE vs FP32 cache |
|---|---:|---:|---:|---:|---:|
| FP32 | 131,072 | 0.6433 ms | 0.8505 ms | `4.77e-7` | `2.24e-7` |
| BF16 | 65,536 | 0.6309 ms | 0.8407 ms | `3.32e-5` | `0.00134` |

Finally, the BF16 graph was composed with post-attention RMSNorm and the real
layer-3 256-expert bank. The measured token step includes device top-8 routing,
eight routed experts, the shared expert/gate, weighted reduction, and the second
residual. All work is queued before one synchronization.

| Complete layer / KV | Expert-bank payload | KV pool | Median kernel | Median wall | Max error |
|---|---:|---:|---:|---:|---:|
| FP32 | 454,754,304 B | 65,536 B | 1.3810 ms | 1.5914 ms | `1.19e-7` |
| BF16 | 454,754,304 B | 32,768 B | 1.3923 ms | 1.5549 ms | `5.96e-8` |

This is the first exact complete sparse decoder layer, not a full-model token.
Linear-attention layers, final norm/head, sampling, and 40-layer residency are
still excluded. Canonical artifacts:

- `20260822-204914-258266-paged-attention-moe-bf16.json`
- `20260822-205331-851639-moe-full-attention-bf16.json`
- `20260822-205343-046114-moe-full-attention-fp32.json`
- `20260822-205622-626682-moe-full-layer.json`
- `20260822-205634-208981-moe-full-layer.json`

## Exact 35B MoE linear layer and four-layer cadence

The gated-delta preparation and resident graph are now parameterized for both
Qwen profiles: dense uses 16 key heads and 48 value heads, while Ornith uses 16
key heads and 32 value heads. A real Ornith layer-0 decode step now includes
input RMSNorm, tensor-scaled FP8 projections, causal convolution, recurrent
gated-delta update, output projection, the first residual, post-attention norm,
device top-8 routing, the shared and routed NVFP4 experts, and the second
residual.

| Complete layer | Expert-bank payload | Median kernel | Median wall | Max abs vs independent CPU oracle |
|---|---:|---:|---:|---:|
| Linear attention + MoE, layer 0 | 454,754,304 B | 1.6442 ms | 1.8166 ms | `3.58e-7` |

Layers 0 through 3 were then composed in their checkpoint cadence: three
linear-attention layers followed by one BF16-KV full-attention layer. Each layer
owns an independent 256+shared expert bank; the four bank payloads total
1,819,017,216 bytes. All recurrent, convolution, and KV state remains resident,
and the complete block is queued before one synchronization.

| Real layers | Bank load | Median kernel | Median queued wall | Error vs synchronized device-layer oracle |
|---|---:|---:|---:|---:|
| 0-3: linear, linear, linear, full | 1.786 s | 6.4166 ms | 6.8675 ms | `0` |

Repeating that measured kernel time ten times gives a planning-only arithmetic
projection of 64.166 ms, or about 15.58 decode tokens/s. It is not a full-model
benchmark: the final norm, LM head, sampling, complete 40-layer registry, memory
pressure, and scheduler overhead are still excluded. Each layer operator also
has an independent CPU oracle; the zero-error cadence comparison specifically
checks that queued composition matches those same proven layers synchronized
one at a time.

Canonical artifacts:

- `20260822-210348-345022-moe-linear-full-layer.json`
- `20260822-210541-006599-moe-cadence.json`
