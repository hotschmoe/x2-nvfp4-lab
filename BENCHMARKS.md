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

## Exact 35B MoE final norm and LM head

Ornith's vocabulary projection is checkpoint-native NVFP4 rather than FP8:
248,320 rows by 2,048 columns, stored as a 248,320x1,024 packed-U8 matrix plus a
248,320x128 E4M3 block-scale matrix and one FP32 global multiplier. The native
payload is 286,064,640 bytes and is uploaded without expansion.

The complete final RMSNorm and vocabulary projection matches an independent,
chunked CPU E2M1/E4M3 decoder within `2.86e-6`; all 248,320 logits are compared
and the greedy argmax token agrees exactly.

| Operation | Median kernel | Median wall | Independent oracle | Max abs | Argmax token |
|---|---:|---:|---:|---:|---:|
| Final RMSNorm + full NVFP4 LM head | 9.1450 ms | 11.0537 ms | 1.197 s | `2.86e-6` | 169,213 |

This isolates a synthetic hidden state. It proves the final arithmetic and full
vocabulary allocation, but not yet a 40-layer checkpoint token or sampling
policy. Canonical artifact:
`20260823-041604-638698-moe-lm-head.json`.

## Complete 35B MoE text-model residency and token

The full coding-endpoint registry now loads every real text-compute tensor in
the Ornith checkpoint while deliberately omitting vision and MTP and touching
only the requested BF16 embedding row. Immutable matrices are uploaded once;
all 40 expert banks remain resident; the 30 linear-attention layers own their
recurrent and convolution state; and the 10 full-attention layers share a BF16
paged pool sized for 32K context. Model-wide activation buffers are reused on
the in-order queue.

The staged isolated-process gates all independently validated their last loaded
bank:

| Resident banks | Checkpoint-native text payload | Available physical memory | Last-bank max abs |
|---:|---:|---:|---:|
| 24 | 12,514,953,780 B | 25,934,319,616 B | `6.40e-10` |
| 30 | 15,249,814,140 B | 23,164,715,008 B | `9.31e-10` |
| 35 | 17,528,864,440 B | 20,867,960,832 B | `2.33e-9` |
| 40 | 19,807,914,740 B | 18,773,078,016 B | `9.31e-10` |

The 40-bank payload exactly equals the metadata inventory's complete resident
text-compute set (18.448 GiB). The 32K BF16 pool contributes 671,088,640 bytes
of KV capacity beyond checkpoint payload. The full allocation returned
20,427,419,648 bytes of physical memory on teardown without a driver reset.

A BF16 embedding row for token 248044 was then queued through all 40 real
checkpoint layers, final RMSNorm, and all 248,320 LM-head rows. One
synchronization covers the model graph; logits are downloaded for host argmax.

| Complete token | Median kernel | Median wall | Wall throughput | Output token | Composition error |
|---|---:|---:|---:|---:|---:|
| 40 layers + final head | 75.8837 ms | 79.3810 ms | 12.60 tok/s | 95,726 | `0` |

The composition oracle resets the same proven states and synchronizes after
every layer. Its logits are bit-identical to the single-queue result. The
individual attention, expert-bank, FP8, NVFP4, and LM-head operators retain
their independent CPU oracles. This is a real complete-text-model token, but it
is still a one-token warm-burst gate: autoregressive state retention, tokenizer
assets, device/partial-logit sampling, and sustained thermal behavior remain to
be measured.

Canonical artifact:
`20260823-042544-384217-moe-full-model-token.json`.

## Retained-state 35B MoE generation

The same full registry now keeps all causal-convolution, gated-delta, and paged
KV state across 32 greedy steps. Each output token selects one BF16 embedding
row from the still-lazy CPU mapping; the full vocabulary logits are downloaded
for host argmax. Token 17 forces every full-attention layer across the first
16-token KV page boundary.

| Stateful decode | Median kernel/token | Median device wall/token | Median end-to-end/token | Mean end-to-end throughput | Replay error |
|---|---:|---:|---:|---:|---:|
| 32 generated tokens | 76.1513 ms | 80.1610 ms | 81.3112 ms | 12.17 tok/s | `0` |

The complete sequence and every one of the 32 full logit vectors are
bit-identical when the model is reset and replayed with a synchronization after
each layer. The BOS-only raw-ID seed repeats token 95,726, so this is a state,
page-lifecycle, and sustained-throughput gate rather than a language-quality
claim. A tokenizer-backed coding prompt is the next qualification.

Canonical artifact:
`20260823-043021-424329-moe-full-model-generation.json`.

## Tokenizer-backed coding generation

The official Ornith tokenizer and adjusted chat template turn the request
"Write a Python function fibonacci(n) with type hints and a short docstring.
Return only code." into 30 prompt tokens ending at the assistant's `<think>`
prefix. Prefill is currently correctness-first sequential execution; the LM
head is skipped for every prompt position except the last.

The complete resident model then greedily generates a reasoning trace followed
by a fenced, typed iterative implementation with input validation, a docstring,
and the correct recurrence. It emits checkpoint stop token 248046
(`<|im_end|>`) after 183 tokens and the loop terminates immediately.

```python
def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number (0-indexed)."""
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

| Prompt/generation | Result |
|---|---:|
| Sequential prefill | 13.8581 tok/s |
| Time to first token | 2,165.1 ms |
| Decode kernel median | 78.0368 ms/token |
| Decode end-to-end median | 84.5639 ms/token |
| Decode end-to-end mean | 11.7542 tok/s |
| Stateful positions | 212 |
| Queued vs synchronized replay | bit-identical logits and tokens |

An earlier length-capped probe exposed that generation must stop at token
248046; continuing beyond it produced unrelated text. The canonical loop reads
the official `generation_config.json` stop set (248046 and 248044) and records
`finish_reason=stop`. Sampling remains greedy even though the official default
requests top-k 20, top-p 0.95, temperature 1.0.

Canonical artifact:
`20260823-044101-661462-moe-full-model-generation.json`.

## Complete dense 27B text-model residency and token

The dense checkpoint's exact mixed policy is now represented end to end. Layers
0-55 retain checkpoint-native NVFP4 MLPs, layers 56-63 retain row-scaled FP8
MLPs, every attention projection is row-scaled FP8, and the 248,320x5,120 LM
head is row-scaled FP8. No matrix is expanded merely to normalize formats.

The loader maps and closes one layer at a time. Keeping the entire 22.6 GB
safetensors source mapping open during an initial successful probe drove
available memory down to 465 MB at layer 64. Per-layer source lifetimes leave
19.60 GB available at the same gate while device residency is unchanged. This
is the direct practical payoff from the MLX-style lazy/lifetime research.

| Resident layers | Checkpoint-native text payload | Available physical memory | Last-layer format | Composition error |
|---:|---:|---:|---|---:|
| 16 | 5,495,726,736 B | 33,519,509,504 B | NVFP4 MLP | `0` |
| 32 | 9,719,548,192 B | 29,185,019,904 B | NVFP4 MLP | `0` |
| 48 | 13,943,369,648 B | 24,874,496,000 B | NVFP4 MLP | `0` |
| 56 | 16,055,280,376 B | 22,718,398,464 B | NVFP4 MLP | `0` |
| 60 | 17,579,482,172 B | 21,159,673,856 B | FP8 MLP | `0` |
| 64 | 19,103,683,968 B | 19,604,447,232 B | FP8 MLP | `0` |

The complete payload exactly matches the metadata inventory's 17.792 GiB text
compute set. The shared 32K BF16 pool adds exactly 2,147,483,648 bytes of KV
capacity across 16 full-attention layers. Recurrent/conv state and reusable
activations are also live. Closing the registry returns 21,683,474,432 bytes.

| Complete token | Median kernel | Median wall | Wall throughput | Output token | Composition error |
|---|---:|---:|---:|---:|---:|
| Dense 64 layers + FP8 head | 511.9609 ms | 519.9734 ms | 1.923 tok/s | 17 | `0` |

This is a real complete checkpoint token from a lazy BF16 embedding row, not a
four-layer projection. The token uses a raw BOS seed. Canonical artifact:
`20260823-045207-005011-dense-full-model-token.json`.

## Retained-state dense 27B generation

The official tokenizer turns the request "Write a Python function add(a, b)
with type hints. Return only code." into 27 prompt tokens. The complete model
then retains all recurrent, convolution, and BF16 KV state for 32 greedy output
tokens. The short cap ends after the opening of the requested function, so this
is a state/throughput gate rather than a complete-answer quality claim.

| Prompt/generation | Result |
|---|---:|
| Sequential prefill | 2.0034 tok/s |
| Time to first token | 13,477.3 ms |
| Decode kernel median | 512.5330 ms/token |
| Decode end-to-end median | 525.9123 ms/token |
| Decode end-to-end mean | 1.9079 tok/s |
| Stateful positions | 58 |
| Queued vs synchronized replay | bit-identical logits and tokens |

Canonical artifact:
`20260823-045725-102478-dense-full-model-generation.json`.

## Full-model bandwidth attribution and profiling coverage

Useful model bandwidth counts checkpoint-native bytes used by one token. It is
not a hardware-counter measurement of DRAM traffic. Dense activates essentially
all 19,103,683,968 resident text bytes each token. At the 512.533 ms sustained
decode kernel median that is 37.27 GB/s: 28.9% of the calibrated 129 GB/s
Adreno raw-read ceiling and 16.3% of the nominal 228 GB/s SoC pin rate. Using
end-to-end throughput gives 36.45 GB/s, or 28.3% and 16.0% respectively.

The MoE denominator must not include inactive experts. Eight routed experts plus
the shared expert, routers, attention/recurrent matrices, and the LM head total
approximately 2.265 GB of useful active payload per token even though 19.808 GB
is resident. At 78.0368 ms median kernel time this is approximately 29.02 GB/s,
22.5% of the calibrated ceiling and 12.7% of nominal. End to end it is about
26.62 GB/s, 20.6% and 11.7% respectively. Small norms, scalar scales, cache
effects, activation/KV traffic, and repeated physical reads keep this an
attribution estimate rather than a bus measurement.

The inference core is substantially profiled, but the complete client request
path is not yet traced as one labeled timeline:

| Stage | Current evidence | Remaining visibility |
|---|---|---|
| HTTP/API, tokenization, streaming | tokenizer/template and stop-token correctness | separate HTTP, tokenizer, detokenizer, and SSE timings |
| Admission and scheduling | paged batch-one/four scheduler microbenchmarks | live vLLM request trace and queue delay |
| Embedding | lazy row gather and exact row-count accounting | gather and upload timings split from host wall |
| Decoder layers | operator oracles plus labeled, barrier-free full-token command traces | hardware memory/cache/occupancy counters |
| KV/recurrent state | labeled attention scopes, page-boundary replay, and independent operator oracles | physical traffic counters inside each scope |
| LM head | isolated Ornith head plus labeled complete-model heads | isolated dense FP8 head and device-side partial-logit sampling |
| Sampling/output | full-logit download plus host argmax included in wall | split download, sampler, detokenization, and stream flush |

At batch one the full-token data already identifies the decoder as the primary
bottleneck: kernels are 92.3% of MoE median decode wall and 97.5% of dense
median decode wall. Ten measured MoE four-layer cadences plus its isolated head
account for about 96.6% of the complete MoE kernel. Dense's early-layer cadence
projects to roughly 94% of its complete kernel, although its final eight FP8
MLPs prevent treating that projection as an exact decomposition. Sequential
single-token prefill is separately the dominant TTFT limitation.

## Barrier-free full-token command traces

The runtime now retains a logical scope and OpenCL queued/submit/start/end
timestamps for every event when tracing is enabled. Scope changes do not enqueue
barriers; the same complete graph is submitted to the same in-order queue. Each
canonical capture executes three exact replays, selects the trace nearest the
median total kernel time as the raw timeline, and stores cross-sample stage and
operation distributions. Every traced replay is bit-identical to the untraced
logits.

| Dense 27B stage | Events | Median kernel | Kernel share |
|---|---:|---:|---:|
| 56 NVFP4 MLPs | 336 | 287.452 ms | 55.7% |
| 48 linear-attention layers | 528 | 128.855 ms | 25.0% |
| Eight FP8 MLPs | 48 | 41.030 ms | 8.0% |
| 16 full-attention layers | 128 | 33.681 ms | 6.5% |
| FP8 final norm/head | 2 | 24.843 ms | 4.8% |

The dense trace contains 1,042 events. Median summed kernel time is 515.306 ms
(514.926-517.101 ms); median first-start to final-end device span is 518.042 ms,
leaving only 2.204 ms between kernels. Quantized matrix kernels account for
95.3% of total kernel time.

The most important format comparison is within the same model and MLP shape.
NVFP4 MLPs average 5.133 ms/layer while the last eight row-scaled FP8 MLPs
average 5.129 ms/layer. NVFP4 consumes only about 150.4 MB of native checkpoint
payload per layer versus 267.5 MB for FP8, yet currently gains no latency. Its
8.423 GB of MLP payload is delivered at 29.62 GB/s, while the FP8 MLP tail
delivers 2.140 GB at 52.76 GB/s. The packed E2M1 decode/instruction path, not
LPDDR capacity, is therefore the primary dense optimization target.

| MoE 35B stage | Events | Median kernel | Kernel share |
|---|---:|---:|---:|
| Routers and active experts | 360 | 32.457 ms | 42.4% |
| 30 linear-attention layers | 330 | 28.093 ms | 36.7% |
| NVFP4 final norm/head | 2 | 9.969 ms | 12.8% |
| Ten full-attention layers | 80 | 6.065 ms | 7.9% |

The MoE trace contains 772 events. Median summed kernel time is 77.079 ms
(76.550-77.759 ms), the device span is 78.674 ms, and only 0.915 ms lies between
kernels. Expert gate/up projections consume 14.526 ms, expert down projections
11.464 ms, and the 256-way top-8 kernels 4.137 ms. Active gate/up payload reaches
29.24 GB/s, but down projection reaches only 18.52 GB/s. The down kernel, NVFP4
head, and top-8 selection are the clearest MoE-specific targets; FP8 attention
already delivers roughly 47-48 GB/s of useful matrix bytes.

Reading hundreds of trace records through ctypes occurs after queue completion
and inflates trace-replay wall time, so throughput remains based on the untraced
generation runs. The OpenCL kernel timestamps themselves agree exactly with the
runtime aggregate.

Canonical artifacts:

- `20260823-052018-140022-dense-full-model-trace.json`
- `20260823-052128-636267-moe-full-model-trace.json`

## Correctness-gated NVFP4 GEMV structure sweep

The experimental lab keeps production dispatch unchanged and sweeps local versus
direct-global activation access, scalar-unrolled versus explicit vector decode,
1/2/4/8/16 output-row subgroups, and 256-8192-element K tiles. It uses complete
real checkpoint matrices and rejects a treatment before timing if it differs
from the production output beyond `rtol=5e-5, atol=5e-5`.

Two runs in opposite shape order reproduced the winners. Each run accepted all
350 candidate/shape combinations.

| Shape | Repeated best kernel | Median | Speedup | Logical bandwidth |
|---|---|---:|---:|---:|
| dense gate/up 17408x5120 | local scalar r16/k8192 | 1.117 ms | 1.310x | 44.95 GB/s |
| dense down 5120x17408 | local scalar r16/k8192 | 1.113 ms | 1.322x | 45.14 GB/s |
| expert gate/up 512x2048 | local scalar r16/k2048 | 0.0171 ms | 1.538x | 35.09 GB/s |
| expert down 2048x512 | local scalar r4/k512 | 0.0256 ms | 1.113x | 23.44 GB/s |
| LM head 248320x2048 | local scalar r8/k4096 | 7.025 ms | 1.225x | 40.86 GB/s |

These are isolated kernel results. The dense projection gains project to about
68 ms less work in the current 515 ms full-token trace, but only a full-model A/B
can turn that estimate into a throughput claim. Method, controls, both artifacts,
and the next experiment matrix are documented in `NVFP4_KERNEL_LAB.md`.

## Multi-vector NVFP4 GEMM sweep

The prefill laboratory sweeps 80 treatments per `(shape, vectors)` case across
2/4/8/16/32 input vectors. Its first complete run passed all 1,600 correctness
gates; a narrower 20-sample reverse-order repeat passed another 400.

| Shape | Vectors | Repeated winner | Median | GFLOP/s | Speedup |
|---|---:|---|---:|---:|---:|
| dense gate/up | 2 | direct vector v2 | 2.710 ms | 131.6 | 1.205x |
| dense gate/up | 16 | direct vector v16 | 17.972 ms | 158.7 | 1.248x |
| dense gate/up | 32 | direct vector v16 | 35.865 ms | 159.1 | 1.266x |
| dense down | 2 | direct vector v2 | 2.697 ms | 132.2 | 1.456x |
| dense down | 16 | direct vector v16 | 17.867 ms | 159.6 | 1.491x |
| dense down | 32 | direct vector v16 | 36.173 ms | 157.7 | 1.484x |
| expert gate/up | 32 | direct vector v1 | 0.3587 ms | 187.1 | 1.148x |
| expert down | 2 | direct vector v4 | 0.0450 ms | 93.3 | 1.150x |
| expert down | 8-32 | production local v4 | 0.161-0.609 ms | 104.5-110.1 | 1.000x |

Direct vector decode winning here while scalar local decode wins GEMV is the
strongest evidence so far that prefill and decode require different kernel
families. The results remain isolated linears; TTFT claims wait for a correct
multi-token model graph.

## Promoted shape dispatch: full-model A/B

The verified GEMV winners now sit behind a shape/phase dispatcher. Setting
`VLLM_NVFP4_OPENCL_SHAPE_TUNING=0` restores the previous production selection.
The result JSON records that switch explicitly.

| Complete model | Dispatch | Kernel median | Wall median | Wall tok/s | Exact logits |
|---|---|---:|---:|---:|---:|
| dense 27B | previous | 514.155 ms | 520.621 ms | 1.921 | yes |
| dense 27B | shape tuned | 430.741 ms | 436.060 ms | 2.293 | yes |
| MoE 35B | previous | 76.468 ms | 80.611 ms | 12.405 | yes |
| MoE 35B | shape-tuned head | 74.521 ms | 78.582 ms | 12.726 | yes |

Dense improves 1.194x at both kernel and queued-wall boundaries. A second pair
before the provenance-correct canonical run measured 512.11/518.28 ms previous
and 429.11/435.36 ms tuned, reproducing the result in reverse order.

The tuned dense command trace keeps the same 1,042 events and exact logits:

| Dense stage | Previous trace | Tuned trace | Change |
|---|---:|---:|---:|
| NVFP4 linear kernels | 284.313 ms | 196.858 ms | -30.8% |
| complete NVFP4 MLP stage | 287.452 ms | 199.966 ms | -30.4% |
| summed full-token kernels | 515.306 ms | 429.024 ms | -16.7% |
| first-start to last-end span | 518.042 ms | 430.480 ms | -16.9% |
| inter-kernel gaps | 2.204 ms | 1.742 ms | still negligible |

The 8.423 GB NVFP4 MLP payload now delivers 42.79 GB/s: 33.2% of the
calibrated 129 GB/s ceiling and 18.8% of nominal 228 GB/s. Same-shape NVFP4 MLP
layers now average 3.571 ms versus 5.207 ms for the FP8 tail, so the packed
format finally converts its lower byte count into latency.

Retained-state dense generation over the same 27-token prompt and 32-token cap
reaches 2.243 end-to-end decode tok/s and 2.376 sequential-prefill tok/s, up from
1.908 and 2.003. The measured 159-160 GFLOP/s GEMM winner is not yet integrated,
so sequential prefill remains the TTFT bottleneck.

Canonical artifacts:

- `20260823-060651-814543-dense-full-model-token.json` (previous)
- `20260823-060723-551631-dense-full-model-token.json` (tuned)
- `20260823-060836-924873-moe-full-model-token.json` (previous)
- `20260823-060913-864839-moe-full-model-token.json` (tuned)
- `20260823-061008-508194-dense-full-model-trace.json`
- `20260823-061211-329751-dense-full-model-generation.json`

A separate 64-token BOS-only MoE retained-state stress run measures 11.700
end-to-end tok/s with a 77.653 ms median decode kernel and exact replay
(`20260823-061338-224847-moe-full-model-generation.json`). It is not directly
comparable to the earlier 183-token coding prompt because prompt, positions,
length, and thermal duration differ; the 12.726 tok/s single-token A/B is the
controlled head-promotion result.
