# Campaign: Bandwidth First

Start date: 2026-08-22

Motto: **make it work, then make it right, then make it fast**

Companion research: [Unified-memory heterogeneous NVFP4 inference research](UNIFIED_MEMORY_RESEARCH.md)

## Mission

Build a safe, checkpoint-native NVFP4 serving runtime for Snapdragon X2 Elite
Extreme that:

1. sustains 75-90% of the measured workload-relevant memory-bandwidth ceiling;
2. uses unified memory to permit direct CPU and GPU consumption of one native
   weight backing store;
3. exploits CPU/GPU/NPU concurrency only where measured island behavior improves
   latency, throughput, or memory capacity;
4. serves real requests through the existing vLLM scheduler boundary;
5. never requires GGUF or expanded FP16/BF16 weight copies.

The campaign is successful only when bandwidth claims state their denominator,
correctness remains gated by exact or bounded-error oracles, and the serving
result survives repeated warm and sustained runs without driver resets.

## Starting point

Completed before this campaign:

- direct safetensors NVFP4 and FP8 checkpoint loading;
- Adreno OpenCL NVFP4 GEMV/GEMM and FP8 projections;
- ARM64 NEON native NVFP4 GEMV;
- exact four-layer Qwen3.5 cadence with three linear-attention layers and one
  full-attention layer;
- persistent recurrent and convolution state;
- paged 16-token KV blocks and per-request block tables;
- continuous batches of one, two, and four requests;
- vLLM `SchedulerOutput` lifecycle adapter;
- isolated-process accelerator test wrapper;
- validated QHPI schema for a future HTP `NvFp4Linear` custom operation.

Current reference measurements:

| Measurement | Baseline |
|---|---:|
| Four-layer native matrix payload | 1,052,676,096 bytes |
| Paged batch-one kernel time | 48.328 ms |
| Paged batch-one scheduler wall | 52.674 ms |
| Paged batch-one logical weight bandwidth | 21.78 GB/s |
| Paged batch-four aggregate throughput | 27.57 request-tokens/s |
| Exact gate/up GPU time | 2.299 ms |
| Exact gate/up GPU logical bandwidth | 21.82 GB/s |
| Exact gate/up CPU time | 2.845 ms |
| Exact gate/up CPU logical bandwidth | 17.63 GB/s |

The nominal platform bandwidth is 228 GB/s. The campaign must not interpret the
current 21.78 GB/s as 9.6% utilization until the Adreno island ceiling and actual
DRAM traffic are characterized.

## Campaign principles

- Native checkpoint bytes are the correctness source.
- One physical weight backing store is preferred over CPU and GPU duplicates.
- Measure actual kernel shapes and contended execution, not only memcpy or
  specification-sheet rates.
- Optimize bandwidth delivery before chasing peak arithmetic throughput.
- Prefill and decode are different scheduling regimes.
- A heterogeneous mode must beat an adjacent GPU-only or CPU-only baseline.
- The baseline execution mode remains a policy arm and fallback.
- Runtime tuning is keyed by hardware, driver, shape, batch, power, and thermal
  regime.
- Under memory pressure, calibrate once and lock; do not continuously evict the
  active working set to probe alternatives.
- Every accelerator expansion passes an isolated safety gate.

## Definition of bandwidth success

Every reported cell includes:

- nominal system bandwidth percentage;
- engine/island streaming ceiling;
- logical native model bytes per second;
- physical DRAM bandwidth when counters are available;
- kernel time and end-to-end wall time;
- output error or exact-match result;
- power source, thermal state, free physical memory, and device budget;
- driver, OpenCL, runtime, and kernel identifiers.

The primary campaign target is:

```text
logical native model bytes/s >= 75% of the matched island ceiling
```

The stretch target is 90%. For batch reuse, report both:

```text
effective model bytes = payload * request-vectors served
estimated physical weight bytes = payload reads after measured reuse
```

No single percentage may silently mix these definitions.

## Workstreams and milestone gates

### Phase 0 - Campaign harness and safety ledger

Goal: produce trustworthy, repeatable experiments without destabilizing the
interactive development session.

Tasks:

- [x] Add a campaign results directory with one JSON result per subprocess run.
- [x] Record machine, SKU, OS build, GPU driver, OpenCL version, power source,
      free memory, and advertised OpenCL budgets.
- [x] Record warmup count, sample count, median, p10, p90, minimum, and maximum.
- [x] Add an explicit completion marker and finite-output check to every probe.
- [ ] Preserve stdout, stderr, command line, exit code, and timeout state.
- [x] Add a campaign summary generator that never treats missing samples as zero.
- [ ] Define a cooldown and sustained-run protocol.

Exit gate:

- The same safe GPU probe runs at least 30 times without a timeout, silent exit,
  driver reset, missing completion marker, or correctness failure.
- Repeated median kernel time is stable enough to distinguish a 5% change.

Stop conditions:

- `LiveKernelEvent 141` or another driver reset;
- incomplete output despite exit code zero;
- a newly introduced system or Codex process termination;
- device allocation growth inconsistent with the requested size.

### Phase 1 - Bandwidth-island profiler

Goal: establish raw and production-kernel ceilings for each accessible compute
island and their interference relationships.

Planned deliverables:

- [x] `native_nvfp4/bench_islands.py`
- [x] native CPU streaming-read implementation;
- [x] OpenCL raw sequential-read kernel;
- [x] CPU/GPU concurrent launch barrier;
- versioned JSON schema and Markdown report.

CPU modes:

- [ ] one prime core;
- [ ] all prime cores;
- [ ] one performance core;
- [ ] all performance cores;
- [ ] all 18 cores;
- [ ] all cores minus one coordinator core.

GPU modes:

- [x] raw byte/word sequential read;
- [x] raw read with multiple vector widths;
- [ ] NVFP4 17408x5120 GEMV;
- [ ] NVFP4 5120x17408 GEMV;
- [ ] row-scaled FP8 projection;
- [ ] vector counts 1, 2, 4, 8, 16, and 32 where safe.

Concurrent modes:

- [x] CPU and GPU on different allocations;
- [ ] CPU and GPU on disjoint ranges of one allocation;
- [x] CPU and GPU on the same read-only range;
- [ ] prime CPU plus GPU;
- [ ] performance CPU plus GPU;
- [ ] all CPU plus GPU;
- [ ] cold, warm, and sustained samples.

Profiler interpretation:

- Additive island: concurrent aggregate exceeds the fastest solo path by at
  least 10% with an end-to-end wall-time opportunity.
- Shared bottleneck: aggregate remains within 5% of the fastest solo path while
  both branches slow down.
- Ambiguous: result falls between those thresholds or changes with thermal state;
  retain both modes for further controlled testing.

Exit gate:

- A complete interference matrix exists for CPU and GPU.
- Results identify the best CPU affinity set and GPU raw-read configuration.
- The direct NVFP4 kernel has a named, matched island ceiling.
- At least three adjacent reruns agree within the campaign variance threshold.

### Phase 2 - Fine-grained SVM native weight store

Goal: give ARM64 and OpenCL direct access to one physical allocation containing
the original packed checkpoint representation.

Tasks:

- [x] Add an SVM capability query to the runtime API.
- [x] Add fine-grained SVM allocation/free ownership with explicit lifetime
      ordering.
- [x] Wrap SVM pointers in `cl_mem` using `CL_MEM_USE_HOST_PTR` for current kernels.
- [x] Retain the CPU pointer in each native matrix handle.
- [ ] Load packed values and scales directly from safetensors into SVM.
- [x] Add initialization and CPU/GPU visibility synchronization.
- [ ] Add allocation accounting and enforce the 2 GiB per-allocation limit.
- [ ] Fall back to the existing copied-buffer path when SVM validation fails.
- [ ] Verify committed physical memory and OpenCL budget before and after upload.

Validation ladder:

1. 64 MiB read-only allocation;
2. one small real checkpoint slice;
3. exact 17408x5120 matrix;
4. one complete MLP;
5. one complete decoder layer;
6. the existing four-layer cadence;
7. 1, 2, 4, and 8 GiB cumulative payload gates.

The cumulative gate is now complete through 8 GiB-class residency. One isolated
process retained real layer banks at 1.36, 2.27, 4.55, and 8.64 GB, validated
all 19 routed outputs with `2.56e-9` worst error, and recovered 8.54 GB when the
banks were destroyed. Larger residency remains gated on full non-expert/KV and
headroom accounting.

Exit gate:

- CPU and GPU consume the same SVM-backed matrix and match their existing
  correctness oracles.
- There is no second persistent host/device weight copy after loading.
- Thirty isolated four-layer runs complete without a reset or leak.
- SVM performance is no worse than the copied path by more than 3%, or the
  difference is understood and documented.

### Phase 3 - Direct Adreno bandwidth optimization

Goal: move native NVFP4 and FP8 kernels from the current approximately 21.8 GB/s
toward 75-90% of the matched GPU-island ceiling.

Tasks:

- [ ] Capture the current kernel as the immutable campaign baseline.
- [ ] Vectorize packed NVFP4 loads.
- [ ] Replace scalar nibble table accesses with a vector or branchless decode.
- [ ] Vectorize E4M3 scale decoding.
- [ ] Test FP16/`half2` products with FP32 accumulation.
- [ ] Compile K=5120 and K=17408 shape-specialized variants.
- [ ] Sweep row tiles 1, 2, 4, and 8.
- [ ] Sweep K tiles 256, 512, and 1024.
- [ ] Measure occupancy/local-memory tradeoffs rather than assuming the widest
      tile is best.
- [ ] Fuse gate and up projections when the shared activation and error gates pass.
- [ ] Apply the successful load/decode strategy to row-scaled FP8.
- [ ] Inspect compiler output and available Qualcomm counters.

Per-variant gate:

- exact or accepted bounded error against the existing oracle;
- no regression at either native matrix shape;
- adjacent baseline/treatment measurements;
- no new allocation or synchronization inside the steady-state kernel path;
- no silent failure in 30 isolated repetitions.

Phase exit gate:

- Both NVFP4 matrix families reach at least 75% of their matched GPU-island
  ceiling in batch-one decode, or a documented compute/ISA ceiling explains why
  memory saturation is impossible with native W4A32 arithmetic.
- Prefill/vector-tiled kernels demonstrate weight reuse separately from physical
  bandwidth.

### Phase 4 - Persistent ARM64 executor and CPU island tuning

Goal: make the CPU path a stable runtime engine rather than a fallback benchmark.

Tasks:

- [ ] Replace per-call worker creation with a persistent pool.
- [ ] Discover and record Windows CPU-set/efficiency-class topology.
- [ ] Implement prime, performance, combined, and coordinator-reserved affinities.
- [ ] Vectorize nibble decoding with NEON table operations where profitable.
- [ ] Use FP16 products with FP32 accumulation when the accuracy gate passes.
- [ ] Measure each projection with the exact SVM layout.
- [ ] Measure scheduler responsiveness while the CPU pool is saturated.
- [ ] Add cancellation and clean shutdown semantics.

Exit gate:

- CPU latency distributions are stable across repeated serving calls.
- The best affinity policy is recorded per shape.
- The scheduler retains a reserved execution context/core.
- CPU execution from shared SVM matches the current NEON oracle.

### Phase 5 - Contention-calibrated CPU+GPU execution

Goal: obtain real heterogeneous wins where the measured topology permits them.

Implementation order:

1. prefill output-row split;
2. independent-request CPU/GPU scheduling;
3. batch-one decode row split only if the island profile predicts a win.

Tasks:

- [ ] Materialize the input activation into shared SVM before branch launch.
- [ ] Release the persistent CPU pool and GPU queue from one barrier.
- [ ] Partition output rows on CPU/GPU tile boundaries.
- [ ] Concatenate disjoint outputs without a cross-engine reduction.
- [ ] Calibrate CPU fractions `{0, .1, .2, .3, .4, .5, 1}` per shape/vector count.
- [ ] Key profiles by hardware, driver, power, thermal regime, and CPU affinity.
- [ ] Add GPU-only, CPU-only, and split policy arms.
- [ ] Add switching hysteresis and a no-regression fallback.
- [ ] Use probation-then-lock when candidate working sets cannot co-reside.
- [ ] Test CPU execution of independent requests alongside GPU tiled batches.

Exit gates:

- Prefill/TTFT improves by at least 10% on a real checkpoint and is neutral or
  better in every supported prompt-size cell.
- Outputs remain exact where execution order is unchanged or within the declared
  numerical tolerance where backend accumulation differs.
- Decode splitting is enabled only if it improves adjacent end-to-end results by
  at least 5% after synchronization and scheduler cost.
- A losing heterogeneous mode automatically falls back to the fastest baseline.

### Phase 6 - Full serving campaign

Goal: turn the executor into a useful vLLM-backed local serving runtime.

Tasks:

- [ ] Preserve vLLM ownership of request scheduling, tokenization, sampling,
      API behavior, and observability.
- [ ] Keep native weights, SVM handles, kernels, paged KV, and recurrent state
      below the worker/plugin boundary.
- [ ] Add prefill execution to the current `SchedulerOutput` adapter.
- [ ] Batch the remaining small control projections and state kernels.
- [ ] Add asynchronous host staging and eliminate avoidable blocking waits.
- [ ] Implement preemption/resume copies for KV, convolution, and recurrent state.
- [ ] Add semantic recurrent-state checkpoints at tool/turn boundaries.
- [ ] Add an elastic budget for weights, KV, recurrent checkpoints, activations,
      scratch, and optional backend layouts.
- [ ] Expand resident model payload only through 1, 2, 4, and 8 GiB safety gates.
- [ ] Complete final norm, LM head, sampling, and end-to-end generation.
- [ ] Expose throughput, latency, cache, memory, and policy-arm metrics.

Serving qualification matrix:

| Dimension | Required cells |
|---|---|
| Prompt length | 32, 512, 2K, 8K, then larger if safe |
| Active requests | 1, 2, 4, 8, then capacity limit |
| Phase | prefill, decode, mixed continuous batch |
| Backend policy | GPU, CPU, split/heterogeneous |
| Thermal state | warm burst and sustained |
| Correctness | greedy token match plus tensor oracle probes |

Exit gate:

- A real checkpoint serves multiple concurrent requests through the vLLM
  boundary without GGUF.
- Thirty-minute sustained serving completes without leaks, reset, silent exit,
  or correctness drift.
- Latency and throughput distributions are published for every supported policy.

### Phase 7 - NPU island and speculative decode

Goal: use the NPU where persistent graph execution adds useful work without
stealing more memory bandwidth than it saves.

Tasks:

- [ ] Obtain a supported x86-64 Windows generator environment for the HTP custom
      op package.
- [ ] First measure a supported FP16/INT8 QNN graph's dispatch, bandwidth, and
      sustained behavior.
- [ ] Measure QNN memory registration and determine whether it copies.
- [ ] Add NPU-alone and CPU/GPU/NPU interference cells to `bench_islands`.
- [ ] Generate, build, and validate the custom packed `NvFp4Linear` package.
- [ ] Compare long-prefill, large-batch, and persistent-graph placements.
- [ ] Evaluate a small CPU- or NPU-hosted speculative draft model for decode.
- [ ] Count accepted tokens per full main-model weight stream, not just draft
      tokens per second.

Exit gate:

- NPU placement improves an end-to-end serving metric after registration,
  dispatch, contention, and power effects.
- If it does not, the NPU remains disabled by default with the negative result
  documented.

### Phase 8 - Compute utilization after bandwidth

Goal: use additional arithmetic capability once weight delivery is no longer the
dominant avoidable loss.

Candidate work:

- [ ] mixed W4A8 integer-dot kernels if quantization error is acceptable;
- [ ] cooperative-matrix paths for prefill and larger continuous batches;
- [ ] fused projection/activation/down-projection kernels where dependencies allow;
- [ ] fused attention preparation and output paths;
- [ ] shape-specific command-buffer or graph capture;
- [ ] speculative decoding and multi-token prediction;
- [ ] scheduler policies that trade latency for weight reuse intentionally.

Exit gate:

- Every compute optimization is evaluated against the bandwidth-optimized
  runtime, not the pre-campaign baseline.

## Benchmark protocol

All campaign benchmark reports follow this protocol unless a result explicitly
documents an exception.

1. Run inside `scripts/run-isolated.ps1` with an operation-specific timeout.
2. Record the exact command and code/kernel identity.
3. Verify the output against the closest independent oracle before timing.
4. Warm page mappings, JIT compilation, buffers, and clocks.
5. Interleave baseline and treatment in the same session.
6. Take at least 30 samples for sub-second cells and at least 10 for expensive
   full-depth cells.
7. Report median, p10, p90, min, and max; do not publish only the best sample.
8. Separate kernel event time, queued wall time, and scheduler/end-to-end time.
9. Record free memory and device budgets before and after the run.
10. Record power and thermal context, and distinguish burst from sustained runs.
11. Require an explicit completion marker; exit code zero is insufficient.
12. Preserve logs for every claimed headline result.

## Result schema

Each JSON result should contain at least:

```json
{
  "campaign": "bandwidth-first",
  "schema_version": 1,
  "timestamp": "ISO-8601",
  "hardware": {
    "system_model": "",
    "soc": "",
    "cpu_affinity": [],
    "gpu": "",
    "npu": "",
    "physical_memory_bytes": 0
  },
  "software": {
    "os_build": "",
    "gpu_driver": "",
    "opencl_version": "",
    "runtime_revision": "",
    "kernel_id": ""
  },
  "environment": {
    "power_source": "",
    "thermal_regime": "",
    "free_memory_bytes": 0,
    "opencl_budget_bytes": 0
  },
  "workload": {
    "operation": "",
    "format": "nvfp4",
    "rows": 0,
    "cols": 0,
    "vectors": 0,
    "logical_payload_bytes": 0
  },
  "timing": {
    "warmups": 0,
    "samples": 0,
    "kernel_ms_median": 0,
    "wall_ms_median": 0,
    "p10_ms": 0,
    "p90_ms": 0
  },
  "bandwidth": {
    "logical_gbs": 0,
    "physical_gbs": null,
    "matched_island_ceiling_gbs": 0,
    "island_utilization": 0,
    "nominal_system_utilization": 0
  },
  "correctness": {
    "passed": false,
    "max_abs_error": null,
    "explicit_completion_marker": false
  }
}
```

## Immediate first sprint

### Progress checkpoint: 2026-08-22 evening

The new profiler and SVM path have crossed the first safe integration gate.
These are warm-burst measurements, not the final sustained interference matrix:

| Cell | Allocation | Samples | Median wall GB/s | GPU event GB/s |
|---|---|---:|---:|---:|
| GPU raw read, 16-byte vectors, 64 MiB | conventional | 30 | 106.59 | 129.60 |
| GPU raw read, 16-byte vectors, 256 MiB | conventional | 30 | 117.94 | 127.17 |
| GPU raw read, 16-byte vectors, 256 MiB | shared SVM | 30 | 119.19 | 129.29 |
| CPU raw read, cores 12-17, 64 MiB | host | 30 | 69.12 | n/a |
| Concurrent CPU+GPU, separate 64 MiB allocations | conventional | 30 | 110.72 aggregate | 109.22 GPU |
| Concurrent CPU+GPU, same 64 MiB range | shared SVM | 30 | 108.49 aggregate logical | 108.01 GPU |

The raw ceiling selected for the exact K=5120 decode comparison is 129 GB/s.
Physical DRAM bandwidth is still unknown, so the same-range concurrent result
must not be interpreted as 108 GB/s of physical traffic: both readers may reuse
the same cache lines. Separate-allocation CPU+GPU throughput is within 5% of the
comparable 64 MiB GPU-only wall result and both engines slow under contention;
there is no evidence yet for an additive decode island.

Fine-grained SVM capability bits are `11`: coarse-grained buffer,
fine-grained buffer, and atomics are present; fine-grained system SVM is absent.
The 64 MiB SVM GPU probe passed 30 separate isolated processes with explicit
completion markers. A real 16x5120 checkpoint slice then matched independent
oracles from both GPU and CPU using the same SVM allocations.

The first exact 17408x5120 gate/up comparison interleaved 30 copied-buffer and
30 shared-SVM calls in one runtime:

| Matrix backing | Median kernel | Logical native bandwidth | Raw-ceiling utilization |
|---|---:|---:|---:|
| `CL_MEM_COPY_HOST_PTR` | 2.3258 ms | 21.56 GB/s | 16.7% |
| fine-grained SVM + `CL_MEM_USE_HOST_PTR` | 1.4651 ms | 34.22 GB/s | 26.5% |

SVM is therefore a measured **go** for exact-matrix integration: 1.59x kernel
speedup and 1.53x call-wall speedup, exact GPU agreement between backing modes,
GPU maximum absolute error `8.34e-7`, and CPU maximum absolute error `2.86e-6`.
At this shape, source-array release recovered approximately one payload's worth
of available physical memory while the SVM and copied matrices remained live.

The staged integration then passed one complete MLP, one complete decoder layer,
and the existing four-layer cadence:

| Resident graph | Copied baseline | Shared-SVM result | Improvement |
|---|---:|---:|---:|
| layer-0 NVFP4 MLP | 9.855 ms kernel | 5.051 ms kernel | 1.95x |
| layer-0 linear attention + MLP | 11.963 ms kernel | 7.848 ms kernel | 1.52x |
| layers 0-3 exact cadence | 48.122 ms kernel | 30.151 ms kernel | 1.60x |
| layers 0-3 queued wall | 48.537 ms | 30.465 ms | 1.59x |

Paged scheduler wall time fell from 52.674 to 31.273 ms at batch one and from
145.069 to 104.947 ms at batch four. Aggregate four-request throughput is now
38.11 request-tokens/s, up from 27.57. Every request remained exact against its
independent contiguous-cache session through the 16-token page boundary.

The same shared-SVM kernel path also passed the 35B MoE checkpoint's native
expert shapes. Eight fixed real layer-0 experts (2048->512->2048) execute in a
0.6477 ms median kernel interval. The next routed micrograph adds native BF16
router/shared-gate GEMV, exact checkpoint top-8 selection and renormalization,
the always-on shared NVFP4 expert, and device weighted reduction. Its 30-sample
median is 0.8230 ms kernel / 1.2767 ms wall with exact selected IDs and
`1.16e-9` final-output maximum error. Device-resident selection/indirect
dispatch, attention, and full-model expert residency remain open.

A subsequent device-bank treatment removes the selection boundary. All 256
routed experts and the shared expert occupy one 454,754,304-byte contiguous SVM
bank per layer; GPU top-8 output directly indexes row-tiled bank kernels. The
30-sample layer result is 0.7563 ms kernel / 0.9231 ms wall with `1.16e-9`
maximum error, an 8.1% kernel and 27.7% wall improvement over host dispatch.
The initial untiled bank kernel regressed to 1.4037 ms and was replaced after
the controlled result confirmed activation staging was missing.

An explicit `uchar8` packed-load treatment was also tested. Qualcomm's compiler
rejected dynamic vector indexing; materializing the vector through a private
array compiled but regressed the exact SVM kernel from about 1.47 ms to 12.60 ms.
The treatment was removed. The next kernel experiment should avoid private-array
materialization and target nibble/scale decode or a compiler-verified vector form.

Reproduction entry points:

- `scripts/run-bandwidth-sprint.ps1`
- `native_nvfp4/bench_islands.py`
- `native_nvfp4/bench_svm_matrix.py`
- `native_nvfp4/bench_moe_experts.py`
- `native_nvfp4/bench_moe_routed_layer.py`
- `native_nvfp4/bench_moe_device_bank.py`
- `native_nvfp4/bench_moe_bank_residency.py`
- `native_nvfp4/bench_moe_full_attention.py`
- `native_nvfp4/bench_moe_full_layer.py`
- `native_nvfp4/probe_tensor_scaled_fp8.py`
- `native_nvfp4/inventory_checkpoint_memory.py`
- `campaign_results/bandwidth-first/*.json`
- `logs/20260822-194606-767.stdout.log` (complete SVM MLP)
- `logs/20260822-194623-106.stdout.log` (complete SVM decoder layer)
- `logs/20260822-194645-268.stdout.log` (four-layer SVM cadence)
- `logs/20260822-194711-776.stdout.log` and
  `logs/20260822-194715-577.stdout.log` (paged batch 1/4)

The first sprint ends before any full-model expansion.

1. Implement campaign result metadata and the CPU/GPU raw bandwidth probes.
2. Produce the CPU/GPU interference matrix on conventional OpenCL buffers.
3. Implement a 64 MiB fine-grained SVM allocation and shared CPU/GPU read probe.
4. Wrap the SVM pointer as `cl_mem` and run the existing native NVFP4 slice.
5. Advance to one exact 17408x5120 matrix only after the small probe passes 30
   isolated repetitions.
6. Compare copied-buffer and shared-SVM bandwidth, latency, committed memory, and
   device-budget accounting.
7. Use those results to choose the first Adreno kernel transformation and the
   first CPU/GPU prefill split experiment.

Sprint completion artifacts:

- raw CPU/GPU solo and concurrent bandwidth table;
- SVM correctness and stability report;
- conventional-buffer versus SVM memory-accounting comparison;
- selected GPU island ceiling and initial utilization percentage;
- explicit go/no-go decision for SVM-backed exact-matrix integration.

## Campaign completion criteria

The campaign is complete when all of the following are true:

- Native NVFP4 and FP8 weights are consumed directly from safetensors without
  GGUF or expanded persistent copies.
- CPU and GPU can consume one validated shared weight backing store.
- NVFP4 kernels sustain at least 75% of their matched practical bandwidth ceiling,
  or a proven compute/ISA boundary replaces the bandwidth hypothesis.
- CPU+GPU prefill is automatically selected only where it wins.
- Decode uses the fastest measured policy and does not assume heterogeneous
  execution is beneficial.
- A real model serves concurrent requests through vLLM with paged KV and correct
  recurrent state.
- The runtime survives sustained qualification without a driver reset, leak,
  silent exit, or correctness failure.
- NPU results are measured and documented whether positive or negative.
- All headline results are reproducible from preserved commands, JSON, and logs.
