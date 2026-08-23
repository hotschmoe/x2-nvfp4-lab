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
