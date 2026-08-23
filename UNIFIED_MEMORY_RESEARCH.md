# Unified-memory heterogeneous NVFP4 inference research

Date: 2026-08-22

Execution plan: [Campaign: Bandwidth First](CAMPAIGN_BANDWIDTH_FIRST.md)

## Executive conclusion

The runtime now has measured, budget-qualified complete tokens and retained
state generation for both Ornith 35B MoE and dense 27B. The next milestone is
to turn those proven executor loops into profiled client-facing serving:

1. extend the completed native decoder-event trace through HTTP intake,
   tokenizer, scheduler, embedding upload, logits/sampling, detokenization, and
   streamed response;
2. replace sequential prefill with shared-weight GEMM and add top-k/top-p
   sampling without downloading the entire vocabulary when the policy permits;
3. measure sustained thermal and multi-request behavior under the proven
   full-residency policy;
4. preserve the one-queue device cadence through the existing vLLM adapter;
5. compare vLLM, SGLang, and Atlas control-plane behavior on the same native
   executor.

The machine has a nominal 228 GB/s memory interface, but that number is the
aggregate LPDDR pin rate for the SoC. It is not evidence that one Adreno kernel,
one CPU cluster, or one NPU graph can individually reach 228 GB/s. The campaign
must define 75-90% utilization against a calibrated, workload-relevant ceiling
for each execution island and also report the percentage of the nominal system
rate.

## Current platform facts

The development machine is an ASUS Zenbook A16 UX3607OA with:

- Snapdragon X2 Elite Extreme X2E-94-100;
- 18 Oryon CPU cores: 12 prime and six performance cores;
- Adreno X2-90 GPU;
- Hexagon v81 NPU;
- 51,127,103,488 physical bytes, or 47.61 GiB;
- 48 GB configured unified memory;
- 192-bit LPDDR5x at 9523 MT/s and a nominal 228 GB/s bandwidth;
- Windows 11 ARM64.

Official X2 platform specifications are in Qualcomm's
[Snapdragon X2 Elite product brief](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Snapdragon-X2-Elite-Product-Brief.pdf).

Unified physical memory does not require an API or driver to expose all physical
RAM as one allocatable device heap. The observed budgets are:

| Interface | Observed budget/capability |
|---|---:|
| Vulkan device budget | 28,490 MiB |
| OpenCL `CL_DEVICE_GLOBAL_MEM_SIZE` | 24,379 MiB |
| OpenCL `CL_DEVICE_MAX_MEM_ALLOC_SIZE` | 2,048 MiB |
| OpenCL free memory during a small probe | approximately 23,355 MiB |
| OpenCL coarse-grained buffer SVM | supported |
| OpenCL fine-grained buffer SVM | supported |
| OpenCL fine-grained system SVM | not supported |
| OpenCL SVM atomics | supported |

These are WDDM/driver allocation policies and capabilities, not measurements of
installed memory. The OS, display, other processes, driver reservations,
eviction policy, and watchdog behavior all affect them. See `FINDINGS.md` and
`logs/20260822-142245.stderr.log` for the local evidence.

## Current inference baseline

The runtime consumes the original compressed-tensors safetensors checkpoint
directly. It does not require GGUF and does not expand NVFP4 weights to FP16 or
BF16.

The exact four-layer Qwen3.5 cadence contains checkpoint-native NVFP4 MLP
projections, row-scaled FP8 attention projections, recurrent linear-attention
state, causal convolution, full attention, residuals, and normalization. Its
current paged serving measurements are:

| Active requests | Kernel time/step | Scheduler wall/step | Aggregate throughput |
|---:|---:|---:|---:|
| 1 | 48.328 ms | 52.674 ms | 18.98 request-tokens/s |
| 2 | 78.505 ms | 85.781 ms | 23.32 request-tokens/s |
| 4 | 135.270 ms | 145.069 ms | 27.57 request-tokens/s |

The batch-one cadence reads 1,052,676,096 logical native matrix bytes in 48.328
ms, or 21.78 GB/s of logical compressed-weight bandwidth. A single exact
17408x5120 NVFP4 projection reaches approximately 21.82 GB/s on Adreno and
17.63 GB/s on the current ARM64 CPU path.

Those are useful-model-byte rates, not hardware-counter DRAM rates. Cache reuse,
repeated loads, activation traffic, and scale traffic can make physical DRAM
traffic differ. Batch four additionally reuses weights across request vectors,
so its effective model bandwidth and physical bandwidth must be reported
separately. Detailed measurements live in `BENCHMARKS.md`.

## Bandwidth terminology

The campaign will distinguish four quantities:

1. **Nominal system bandwidth** is the platform's 228 GB/s LPDDR pin rate.
2. **Island streaming ceiling** is the sustained rate produced by a simple,
   validated streaming kernel on a particular CPU cluster, GPU, or NPU path.
3. **Logical model bandwidth** is native checkpoint payload consumed per unit
   time. It represents useful inference work and can exceed physical bandwidth
   when one weight read is reused across several tokens.
4. **Physical DRAM bandwidth** is the traffic observed by hardware counters. If
   usable counters are unavailable, it is approximated with carefully matched
   streaming controls and clearly labeled as an estimate.

Primary efficiency is:

```text
logical native bytes per second
-----------------------------------------------
same-engine, same-shape island streaming ceiling
```

Physical bandwidth utilization and percentage of 228 GB/s are secondary
metrics. A claim of 75-90% utilization must name its denominator.

## Memory-controller islands

"Island" means a compute unit's effective path through its private caches, NoC
ingress, coherency fabric, controller arbitration, and DRAM channels. It does not
assume that CPU, GPU, or NPU owns private LPDDR channels.

The island topology should be inferred experimentally:

- If CPU+GPU aggregate bandwidth is materially higher than either unit alone,
  there is useful independent ingress or unused controller headroom.
- If aggregate bandwidth stays flat while both units slow down, they share the
  active bottleneck and splitting a decode GEMV cannot help.
- If same-allocation and different-allocation co-runs behave differently, cache,
  page placement, coherency, or memory-controller hashing matters.
- If CPU-cluster affinity changes co-run performance, the 12-core and six-core
  groups have different useful paths or arbitration behavior.
- NPU access must be measured through QNN registered buffers. Unified memory by
  itself does not prove that an OpenCL SVM pointer is directly consumable by HTP.

The minimum characterization matrix is:

| Producer/consumer | Alone | With prime CPU | With performance CPU | With GPU | With NPU |
|---|---:|---:|---:|---:|---:|
| Prime CPU | required | required | required | required | later |
| Performance CPU | required | required | required | required | later |
| All CPU cores | required | required | required | required | later |
| Adreno raw read | required | required | required | n/a | later |
| Adreno NVFP4 | required | required | required | n/a | later |
| HTP graph | later | later | later | later | required |

Each CPU+GPU test must cover separate allocations, disjoint ranges in one shared
allocation, and—where meaningful—the same read-only range.

## What FreeToken contributes

FreeToken is an edge-native MoE serving system for discrete CUDA systems. Its
PCIe-specific policy is not directly transferable to this SoC, but several
design rules apply exactly.

### Measure production paths, including contention

FreeToken separates hardware ceilings from the real CPU MoE kernel and real
host-to-device expert gather. It then measures the CPU and transfer paths while
they run concurrently because solo bandwidths do not predict the hybrid split.
Its implementation is visible in
[`benchbw.py`](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/benchbw.py).

Our profiler should likewise benchmark raw streams and the actual native NVFP4
kernel at deployed shapes. CPU/GPU split ratios must come from contended runs.

### Bind representation, packing, and execution

FreeToken makes a backend own load-time packing and forward execution as one
unit. It keeps one expert layout rather than two incompatible resident copies,
and selects vLLM Marlin, FlashInfer, or its Triton path using architecture and
shape. See
[`nvfp4_backends.py`](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/nvfp4_backends.py).

For this runtime, the checkpoint-native row-major NVFP4 representation should be
the single correctness source and the single CPU/GPU weight allocation. A future
specialized layout is acceptable only if its measured improvement justifies the
memory cost or it replaces, rather than duplicates, the native resident copy.

### Use persistent, affined CPU workers

FreeToken uses one worker per physical core for bandwidth-bound decode and avoids
SMT siblings. The Snapdragon has no SMT, but cluster affinity still matters. The
CPU executor needs a persistent pool with separately measurable prime,
performance, and combined modes. A coordinator core should remain available for
scheduler and GPU submission work.

### Treat accelerator memory as an elastic performance cache

FreeToken's CPU expert banks are the correctness source while GPU memory is a
global LRU performance cache. Misses can be divided between GPU cache fill and
direct CPU execution, then exact partial outputs are merged. Its cache machinery
and hybrid route are implemented in
[`offload_cache.py`](https://github.com/FlashML-org/FreeToken/blob/main/python/freetoken/moe/offload_cache.py).

Dense Qwen layers do not have expert sparsity, so expert LRU locality does not
apply. The transferable mechanism is an elastic budget among resident layer
groups, paged KV, recurrent state, activations, and scratch. Resizing occurs only
at scheduler-safe points.

### Double-buffer prefill and checkpoint recurrent state

FreeToken overlaps expert transfer with prefill computation and stores semantic
recurrent-state checkpoints at tool and turn boundaries. The latter is directly
relevant to Qwen3.5's linear-attention state: radix/paged KV reuse is insufficient
when recurrent state must otherwise be recomputed.

The complete system and policy are described in the
[FreeToken paper](https://arxiv.org/html/2608.16157).

## What MLX and FusionML contribute

### Locationless arrays and operation placement

MLX arrays live in unified memory. The operation, not the allocation, selects a
CPU or GPU stream. Independent operations can run concurrently, and dependencies
between streams are inserted automatically. See the
[MLX unified-memory documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
and [stream API](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html).

The corresponding runtime abstraction is:

```text
Checkpoint-native weight store (fine-grained OpenCL SVM)
    |-- ARM64 NVFP4 executor
    |-- Adreno OpenCL executor
    `-- future QNN/HTP handle or separately registered view

Per-request state
    |-- recurrent state
    |-- paged full-attention KV
    `-- position, cancellation, and scheduler metadata

Execution policy
    |-- phase: prefill or decode
    |-- shape and batch
    |-- measured island profile
    `-- memory, power, and thermal regime
```

### Materialization boundaries are part of correctness and performance

FusionML found that MLX's lazy graph serialized a CPU branch that consumed an
unmaterialized GPU result. Explicit evaluation of the shared activation before
launching both branches restored concurrency. Its row-split prefill improved
block latency by 1.15-1.38x and real-model TTFT by 1.18-1.25x, while decode was
unchanged because both engines shared the memory bottleneck.

See the [FusionML paper](https://arxiv.org/html/2607.22785).

Our equivalent boundary is:

1. produce the layer activation into shared SVM storage;
2. complete the dependency/event that makes it CPU-visible;
3. release the CPU pool and GPU queue from one barrier;
4. wait for both branches;
5. concatenate output rows and continue.

The CPU branch must not be launched only after a GPU-only dependency chain has
already serialized the work.

### Calibrate under contention and avoid harmful re-probing

FusionML selects a CPU output-row fraction per matrix shape by timing the actual
concurrent split. It also found that periodic alternative-mode probes under
memory pressure evicted the active working set and biased both the probe and the
following request.

The runtime should therefore use:

- short startup probation when all candidate working sets co-fit;
- GPU-only, CPU-only, and split as explicit policy arms;
- a switching hysteresis;
- probation-then-lock under memory pressure;
- drift-triggered rather than periodic re-probing when alternatives cannot
  co-reside safely.

### MLX comparison after the SVM measurements

MLX's most transferable idea is not merely that Apple silicon has unified
memory. It makes arrays locationless, places each operation on a CPU or GPU
stream, records dependencies, and evaluates useful groups of operations at one
materialization boundary. Its lazy-evaluation guidance explicitly warns that
too-frequent evaluation pays fixed overhead and that scalar access used for
control flow forces evaluation.

The native runtime now has the same useful ingredients at a lower level:

- SVM-backed weights are locationless between ARM64 and Adreno within the
  capabilities exposed by this OpenCL driver;
- the in-order OpenCL queue records a resident graph and one `synchronize()`
  materializes a measured stage;
- CPU/GPU raw contention tests decide operation placement rather than the UMA
  label;
- the checkpoint-routed MoE graph currently has one unavoidable host control
  boundary for top-k selection.

The routed MoE measurement makes that last point concrete. GPU BF16 routing,
the always-on shared expert, eight selected NVFP4 experts, weighted reduction,
and the shared gate take 0.8230 ms of kernel time, but the two-stage wall time is
1.2767 ms. Host top-k arithmetic itself is only 0.0192 ms median; queue
materialization and small downloads dominate the remaining gap. A device top-k
kernel and indirect expert dispatch are therefore higher priority than
optimizing the host softmax routine.

## What DGX Spark runtimes contribute

DGX Spark is a useful comparison, but not a performance proxy for Snapdragon.
NVIDIA documents 128 GB of LPDDR5x unified system memory at 273 GB/s on GB10,
alongside a Blackwell GPU with 6,144 CUDA cores and fifth-generation Tensor
Cores. The memory topology is analogous at a high level; the compute ISA,
software stack, power envelope, and native NVFP4 acceleration are materially
stronger and different.

### vLLM and SGLang: budget the shared pool explicitly

NVIDIA's current Spark vLLM recipe emphasizes PagedAttention, continuous
batching, `--gpu-memory-utilization`, maximum model length, and maximum sequence
count. Its SGLang recipe similarly recommends lowering
`--mem-fraction-static` to roughly 0.70-0.75 under UMA memory pressure, with
context length and concurrency determining KV allocation. Spark's unified pool
increases the capacity envelope, but these runtimes still reserve and police
weights, KV, graph scratch, and concurrency explicitly.

That reinforces this project's budget-aware registry design:

- reserve headroom for Windows, the display, driver, and transient scratch;
- treat paged KV and recurrent state as first-class consumers beside weights;
- expose maximum context and concurrency as admission-control inputs;
- do not infer allocatable accelerator memory from installed physical memory;
- preserve one native representation, then make residency and cache decisions
  at scheduler-safe points.

### Atlas: specialize the complete hot path

Atlas is a newer open-source Rust/CUDA engine specialized for GB10 and specific
model/quantization pairs. Its documentation describes hand-tuned SM121 kernels
for NVFP4, attention, recurrent layers, MoE routing, and paged FP8 KV, with no
Python, PyTorch, Triton JIT, or runtime compilation in the serving hot path. Its
published throughput comparisons are project-reported and have not been
independently reproduced here, so they are research leads rather than campaign
baselines.

The transferable lesson is the scope of specialization: optimizing one GEMV is
insufficient when routing synchronization, graph dispatch, attention, state,
sampling, and HTTP scheduling remain generic. The current native C ABI plus a
thin vLLM lifecycle adapter follows the same separation: keep the model/hardware
hot path compiled and resident, while retaining a mature external scheduler and
OpenAI-compatible API until measurements justify replacing more of it.

The source audit found four concrete decode mechanisms worth transferring:

1. router top-k stays on device and writes IDs/weights consumed by later kernels;
2. device pointer tables let one kernel select any resident expert;
3. the shared expert is represented as an additional dispatch slot;
4. gate/up and SiLU/down work is grouped so routing does not become dozens of
   per-expert launches.

CUDA device-pointer tables are not a portable OpenCL assumption. The implemented
equivalent is a contiguous per-layer SVM bank: selected IDs produce byte offsets
inside six packed/scale arrays, and shared slot 8 maps to bank expert 256. This
reduced the routed layer from approximately 47 launches and two materialization
boundaries to seven launches and one. The measured result is 0.7563 ms kernel /
0.9231 ms wall, 27.7% lower wall time than host dispatch.

### llama.cpp: a useful deployment and MTP reference

NVIDIA's Spark llama.cpp recipe builds directly for GB10's `sm_121` CUDA target,
offloads the model, exposes an OpenAI-compatible server, and documents MTP
speculative decoding for compatible Qwen checkpoints. This supports the current
plan to investigate the local checkpoint's MTP head after baseline decode is
correct: reducing full-model weight streams per accepted token may be more
valuable than forcing CPU/GPU co-execution across a shared memory bottleneck.

## Shared allocation design

The current runtime creates matrix buffers with `CL_MEM_COPY_HOST_PTR`. The
OpenCL implementation may optimize this on UMA, but the API contract creates a
buffer initialized from a copy; it does not establish a shared CPU/GPU backing
store.

The proposed matrix allocation is:

1. call `clSVMAlloc` with `CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER`;
2. copy the original safetensors packed values or scale bytes into that pointer;
3. create a `cl_mem` view with `CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR`;
4. retain the SVM pointer for ARM64 kernels and the `cl_mem` view for existing
   OpenCL kernels;
5. synchronize initialization before either executor consumes it;
6. release the `cl_mem` object before calling `clSVMFree`.

The OpenCL specification guarantees that a buffer created with
`CL_MEM_USE_HOST_PTR` from a pointer returned by `clSVMAlloc` uses the shared
memory as its underlying storage. See the Khronos reference pages for
[`clSVMAlloc`](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clSVMAlloc.html)
and [`clCreateBuffer`](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clCreateBuffer.html).

Fine-grained system SVM is not available, so an arbitrary safetensors mmap
pointer is not a valid substitute. There is one file-to-SVM load, followed by one
physical backing store for CPU and GPU execution. Individual allocations must
remain at or below the reported 2 GiB maximum, and staged allocation growth must
preserve several GiB of device-budget headroom.

### Measured SVM checkpoint

The proposed design is now implemented for raw campaign buffers and native
NVFP4 matrices. The device reports SVM capability bits `11` (coarse buffer,
fine-grained buffer, atomics; no fine-grained system allocation). Destructors
release each `cl_mem` view before `clSVMFree`, and the matrix retains the SVM
pointers for direct NEON execution.

A 64 MiB shared read passed 30 isolated processes. On an exact checkpoint-native
17408x5120 gate/up projection, a 30-sample interleaved comparison measured:

| Backing | Median kernel | Logical payload bandwidth |
|---|---:|---:|
| copied OpenCL buffer | 2.3258 ms | 21.56 GB/s |
| fine-grained SVM + `CL_MEM_USE_HOST_PTR` | 1.4651 ms | 34.22 GB/s |

The GPU outputs were bit-identical between allocation modes. Both matched the
independent tensor oracle, and NEON consumed the same SVM matrix within the
existing error gate. This is a 1.59x kernel improvement with no persistent
expanded weight representation. Raw 256 MiB reads are approximately 129 GB/s
for both conventional and SVM storage, suggesting the exact-matrix improvement
is allocation/placement behavior specific to host-initialized production
buffers rather than a higher SVM-only DRAM ceiling.

## Heterogeneous execution policy

Output-row splitting is the first intra-operator policy:

- the CPU computes the first aligned fraction of output rows;
- the GPU computes the remaining rows;
- both read the same input activation and disjoint weight rows;
- concatenation requires no numerical reduction across engines;
- partitions align to the GPU row tile and CPU vector blocking.

For a CPU fraction `rho`, select the minimum measured:

```text
T(rho) = max(
    T_cpu(rho | GPU concurrently active),
    T_gpu(1-rho | CPU concurrently active)
) + synchronization and merge cost
```

The candidate grid begins with `rho` in `{0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0}`
and is refined only around a stable winner. Profiles are keyed by matrix shape,
quantization format, vector count, CPU cluster, and thermal/power regime.

Initial policy by workload:

| Workload | Initial execution policy |
|---|---|
| Long prefill | Calibrated CPU+GPU output-row split |
| Batch-one decode | GPU-only unless the island profiler proves an end-to-end win |
| Continuous decode batch | GPU vector tiling first; test independent requests on CPU |
| Device-budget spill | CPU consumes the same native SVM weight allocation |
| NPU | Long-lived prefill/batch graph or speculative draft after dispatch profiling |

If CPU+GPU cannot add decode bandwidth, idle CPU/NPU compute is better directed
to a small speculative draft model that can reduce the number of full main-model
weight streams per accepted token.

## Direct kernel path to higher bandwidth

The existing Adreno NVFP4 kernel uses scalar FP32 lookup and multiply-add work
for every decoded nibble. Its approximately 21.8 GB/s logical payload rate may
therefore be instruction- or occupancy-limited rather than DRAM-limited.

Optimization order after the raw ceiling is known:

1. vectorize packed-byte and scale loads;
2. replace scalar constant-table accesses with vector/branchless nibble decode;
3. use FP16 or `half2` products with FP32 lane accumulation where the error gate
   passes;
4. vectorize E4M3 scale decode;
5. compile shape-specialized K=5120 and K=17408 kernels;
6. sweep row tiles `{1,2,4,8}` and K tiles `{256,512,1024}`;
7. fuse gate and up projections to share activation staging;
8. apply the same method to row-scaled FP8 projections;
9. inspect compiler output and hardware counters where Qualcomm tooling permits;
10. optimize arithmetic throughput only after weight delivery approaches the
    measured island ceiling.

Qualcomm documents OpenCL kernel analysis and CPU/GPU/DSP/memory profiling in
its [OpenCL programming and optimization guide](https://docs.qualcomm.com/bundle/publicresource/80-NB295-11_REV_C_Qualcomm_Snapdragon_Mobile_Platform_Opencl_General_Programming_and_Optimization.pdf).

## NPU boundary

The NPU is a distinct workstream, not an assumed third OpenCL device. It must be
measured through QAIRT/QNN and its supported registered-memory path.

The local platform and FastRPC validation pass, and the custom `NvFp4Linear`
QHPI definition validates. Generating the complete custom HTP op package remains
blocked by the installed ARM64 host generator's missing serializer and the
inability to execute the x86-64 generator in ARM64 WSL.

Before adding direct NPU NVFP4 execution:

1. run a supported FP16 or INT8 graph long enough to amortize dispatch;
2. measure NPU-only bandwidth/throughput and fixed dispatch cost;
3. co-run it with prime CPU, performance CPU, and GPU streaming loads;
4. determine whether QNN registration is zero-copy, mapped, or copied on this
   Windows stack;
5. use an x86-64 Windows host/VM or supported generator environment to build the
   custom HTP op package;
6. prefer a persistent prefill partition, large batch, or small speculative
   draft over per-layer batch-one decode dispatch.

## Serving integration

The native runtime remains framework-neutral. vLLM owns request scheduling,
tokenization, sampling, API compatibility, and observability. The native runtime
owns checkpoint-native weights, CPU/GPU/NPU execution handles, paged KV,
recurrent state, and device dispatch.

The existing vLLM `SchedulerOutput` adapter already handles decode, request
reordering, finish/abort, and request-major prompt chunks. The heterogeneous
policy should sit below that adapter so vLLM does not need to understand SVM,
OpenCL events, or CPU row partitions.

FreeToken's elastic cache policy suggests a scheduler-safe budget controller
for:

- shared resident weights;
- paged full-attention KV;
- recurrent-state checkpoints;
- activation and scratch buffers;
- optional backend-specific packed layouts;
- NPU graph/context memory.

## Safety and experimental discipline

The earlier crashes make benchmark containment part of the architecture:

- run accelerator probes through `scripts/run-isolated.ps1`;
- change one variable per probe;
- require explicit `PASS` and finite outputs;
- never treat a silent exit or exit code zero alone as success;
- allocate in staged steps rather than jumping to a full model;
- record committed memory, device budget, power state, temperature context, and
  driver versions;
- retain several GiB outside the advertised device budget;
- stop a path after a driver reset, `LiveKernelEvent 141`, timeout, or missing
  completion marker;
- do not run full-model Vulkan offload on the current driver;
- use exact CPU or staged-device oracles at every new shape and layer-count gate.

## Research queue: things we want to investigate

Prioritized next questions, including work not yet performed:

1. **Server-facing prefill and sampling.** Official tokenizer-backed coding
   generation now produces correct code and EOS-stops at 13.86 prefill / 11.75
   decode tok/s. Replace sequential prefill with shared-weight GEMM batching,
   add the official top-k 20/top-p 0.95 sampler, and expose request cancellation
   and streaming through the worker boundary.
2. **Complete the serving trace.** Barrier-free native traces now label all
   1,042 dense and 772 MoE device events and separate the dense 56 NVFP4 versus
   eight FP8 MLP layers. Extend the same trace ID through tokenizer, request
   queueing, embedding gather/upload, logits transfer, sampling, detokenization,
   and SSE, then add hardware memory/cache/occupancy counters.
3. **NVFP4 decode and MoE fusion.** Same-shape dense NVFP4 and FP8 MLPs both
   take about 5.13 ms/layer, proving the compressed path is instruction limited.
   Tune packed/scale loads and output tiling, especially the 18.52 GB/s expert
   down path; compare fused SiLU/down/shared reduction and replace the current
   4.137 ms/token top-8 sequence with a fused router/selection design.
4. **Atlas kernel and graph structure.** Audit its open-source Qwen3.5/GB10 MoE,
   recurrent-layer, paged-KV, CUDA-graph, and MTP paths for reusable scheduling
   patterns, while separating CUDA/SM121-only techniques from portable ones.
5. **MLX allocator and scheduling internals.** Trace how locationless arrays,
   stream dependencies, batched `eval`, buffer reuse, and wired-memory limits
   avoid redundant materialization; test equivalent lifetime classes in the
   SVM arena.
6. **vLLM versus SGLang control-plane A/B.** Put the now-complete native token
   executor behind each scheduler and compare request admission, prefix reuse,
   continuous batching, cancellation, structured output, and coding-client
   compatibility with identical prompts and trace spans.
7. **Multi-request memory pressure.** Complete one-request residency now passes
   for both models, including 40 MoE banks and dense 32K BF16 KV. Add a second
   request's KV/recurrent state, cancellation, page reclamation, and admission
   control while keeping the single immutable weight registry.
8. **Expert residency policy.** Measure per-layer routing locality over real
   coding traces, determine whether all 256 experts fit inside the safe OpenCL
   budget, and compare full residency, layer streaming, and frequency-aware
   caching.
9. **Vision and MTP coding expansion.** Both are future coding requirements,
   not permanently optional features: vision supplies screenshot/UI/document
   context and MTP can accelerate interactive generation. Keep both outside the
   current text-kernel baseline, then qualify the vision tower/projector and an
   acceptance-correct MTP device graph behind independent memory, correctness,
   and latency gates. Measure accepted MTP tokens per expensive main-model
   weight stream before enabling it by default.
10. **NPU registered memory.** Measure whether QAIRT/QNN can consume a registered
   view of the same underlying allocation, its dispatch floor, and contention
   with Adreno and Oryon before assigning it any production work.
11. **Sustained thermal behavior and counters.** Add cooldown/sustained protocols
   and obtain Qualcomm compiler/occupancy/memory-counter evidence so the gap
   between 34.22 GB/s useful NVFP4 delivery and the 129 GB/s raw ceiling can be
   attributed rather than guessed.
12. **KV precision and long-context quality.** BF16 now covers dense and MoE
    head profiles. Next compare FP32/BF16/FP8 perplexity and coding-task behavior
    at 32K/64K/128K with prefix reuse and concurrent requests.

The one-request memory question is now answered empirically. All 40 MoE banks
and 19,807,914,740 bytes of Ornith text payload coexist with 32K BF16 state; the
dense model retains 19,103,683,968 bytes of text payload plus exactly 2 GiB of
32K BF16 KV. Both release cleanly. Admission policy for additional requests and
longer contexts remains a separate qualification and must not reuse this
single-request success as proof.

The bounded experiment subsequently retained 19 actual checkpoint layer banks
at once (8,640,331,776 bytes), validated each routed output, and observed
8,540,033,024 bytes return immediately on destruction. Available physical
memory at the maximum gate remained 30.70 GB. That historical gate justified
the later 24/30/35/40 ladder; the subsequent complete model run now supersedes
it as the one-request residency evidence.

## Exact residency policy derived from both checkpoints

The metadata-only ledger in
`campaign_results/bandwidth-first/checkpoint-memory-inventory.json` classifies
all 94,393 MoE tensors and all 1,968 dense/MTP tensors. The important result is
that checkpoint file size is not the device working set:

- omit the 0.83-0.86 GiB vision tower for a coding-only endpoint;
- omit the 0.79-1.57 GiB MTP graph until speculative decode is implemented;
- memory-map token embeddings and gather a single row on CPU, avoiding eager
  residency of 0.95 GiB (MoE) or 2.37 GiB (dense);
- keep all remaining text compute weights in their checkpoint precision;
- reserve 2 GiB below OpenCL's reported 23.81 GiB for allocator behavior and
  activation/scheduler scratch not yet measured.

This yields 18.448 GiB of resident compute weights for MoE and 17.792 GiB
planned for dense. MoE has now passed a complete load and token at 32K BF16:
19,807,914,740 checkpoint bytes plus a 671,088,640-byte KV pool, recurrent
state, and reusable graph scratch. Keep MoE 64K disabled until it passes the
same complete-load and long-context gates. Dense still needs its full-load
qualification; use 16K FP32 or the already proven 32K BF16 storage policy for
that attempt. FP8 KV must pass an end-to-end long-context quality gate before
it can unlock dense 64K or higher concurrency.

The BF16 storage path now passes both exact head profiles: dense 24/4 and MoE
16/2. Dense also passes four-layer batch-one/four cadence and vLLM lifecycle.
MoE now passes both complete layer types and its real three-linear/one-full
cadence with four independent resident expert banks: 6.4166 ms kernel and
6.8675 ms queued wall. Both remain opt-in pending complete-model and
long-context qualification.

This policy borrows MLX's useful lifetime principle rather than its API: keep
locationless/lazy data such as embeddings from being eagerly materialized, and
evaluate the sequential decoder with reusable workspace. It also follows the
Spark runtimes' practical separation between a scheduler/control plane and a
hardware-specific weight/KV allocator. The Snapdragon implementation remains
OpenCL/SVM-native; CUDA pointer tables and graph capture are not portable here.

## Research sources

- [Official Ornith 1.5 35B NVFP4 checkpoint](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4)
- [FreeToken paper](https://arxiv.org/html/2608.16157)
- [FreeToken inference engine](https://github.com/FlashML-org/FreeToken)
- [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- [MLX streams](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html)
- [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA DGX Spark vLLM recipe](https://build.nvidia.com/spark/vllm/i)
- [NVIDIA DGX Spark SGLang recipe](https://build.nvidia.com/spark/sglang/instructions)
- [NVIDIA DGX Spark llama.cpp recipe](https://build.nvidia.com/spark/llama-cpp/instructions)
- [Atlas inference engine](https://atlasinference.io/)
- [Atlas Spark engineering journey](https://github.com/Avarok-Cybersecurity/atlas/blob/main/docs/ATLAS_SPARK_JOURNEY.md)
- [FusionML paper](https://arxiv.org/html/2607.22785)
- [Khronos OpenCL SVM allocation](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clSVMAlloc.html)
- [Khronos OpenCL buffer creation](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clCreateBuffer.html)
- [Qualcomm Snapdragon X2 Elite product brief](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Snapdragon-X2-Elite-Product-Brief.pdf)
- [Qualcomm OpenCL programming and optimization guide](https://docs.qualcomm.com/bundle/publicresource/80-NB295-11_REV_C_Qualcomm_Snapdragon_Mobile_Platform_Opencl_General_Programming_and_Optimization.pdf)
