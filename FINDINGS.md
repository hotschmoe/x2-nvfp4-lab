# NVFP4 inference findings (2026-08-22)

## Problem boundary

There are three separate issues that looked like one failure:

1. **Vulkan full offload is unsafe on this driver.** The Codex session losses
   correlate with large llama.cpp Vulkan offload attempts. Enumeration is safe;
   the failure appears only after allocating/executing the large model. Historical
   `LiveKernelEvent 141` records make a GPU watchdog/reset the leading explanation,
   although Windows did not record a fresh 141 for every terminated test.
2. **llama.cpp's broad OpenCL backend test is not a safe health check.** On this
   Qualcomm Windows driver it silently ends at some multi-vector/broadcast cases,
   with exit code 0 and no OpenCL error. The unmodified MXFP4 path does the same,
   so this behavior is not caused by the new NVFP4 decoder.
3. **The 30.858 GiB GGUF is a conversion artifact, not NVFP4's true size.** The
   converter retains the NVFP4 MLP tensors but expands the checkpoint's separate
   FP8 group to BF16.

The safe direct OpenCL probe repeatedly executes the native NVFP4 kernel without
a reset or process exit. That isolates the immediate OpenCL instability to the
llama backend/test integration and its unsupported shape paths rather than basic
NVFP4 execution on Adreno.

## Memory accounting

The machine has 51,127,103,488 physical bytes, or 47.61 GiB. Unified memory means
CPU and GPU draw from the same physical pool; it does not require every graphics
API to expose the whole pool as allocatable device memory.

Observed driver budgets:

- Vulkan budget reported by llama.cpp: 28,490 MiB
- Qualcomm OpenCL `CL_DEVICE_GLOBAL_MEM_SIZE`: 24,379 MiB
- Qualcomm OpenCL `CL_DEVICE_MAX_MEM_ALLOC_SIZE`: 2,048 MiB
- OpenCL free memory at the small probe: about 23,355 MiB

These are driver/WDDM allocation budgets, not installed RAM. The operating system,
display, CPU processes, driver reservations, eviction policy, and watchdog risk
all reduce what an API advertises. A 21.8 GiB checkpoint therefore does not leave
enough safe GPU headroom for KV cache, activations, command buffers, and scratch;
hybrid placement or demand-paged weights are required even though physical RAM is
48 GB.

## Checkpoint size breakdown

`model.safetensors` is 21.018 GiB and contains:

| Native group | Size (GiB) | Representation |
|---|---:|---|
| NVFP4 packed values | 6.973 | U8, two values per byte |
| NVFP4 E4M3 scales | 0.872 | one byte per 16 weights |
| FP8 weights | 9.895 | E4M3 |
| BF16 tensors | 3.279 | BF16 |
| Metadata/global scales | negligible | F32 |

The native NVFP4 matrices occupy 7.844 GiB total. GGUF preserves roughly this
amount as NVFP4, but turns the 9.895 GiB FP8 group into about 19.79 GiB of BF16.
Together with the original BF16 tensors, the GGUF BF16 payload is about 23 GiB,
which explains the final 30.858 GiB file.

## Direct native result

`native_nvfp4/probe_native_nvfp4.py` memory-maps the original safetensors file.
For a matrix it reads the two original arrays and the global scale:

- `[M, K/2]` packed E2M1 nibbles
- `[M, K/16]` E4M3 local scales
- scalar `weight_global_scale`

`native_nvfp4/kernels/nvfp4_gemv.cl` consumes those buffers directly. It does not
create GGUF blocks or a dequantized FP16/BF16 weight tensor. On a real 16-row,
5,120-column slice of `layers.0.mlp.gate_proj`, three consecutive Adreno executions
matched the independent CPU implementation:

- max absolute error: `2.2649765e-06`
- max relative error: `1.6488722e-05`

The entire native input for that test was 46,080 bytes.

The correctness-first GEMM was also validated on a real 256-row by 5,120-column
slice with eight input vectors and three consecutive launches:

- max absolute error: `5.364418e-06`
- measured kernel time: `0.002234 s` for the three launches
- effective baseline throughput: `28.159 GFLOP/s`

The scalar path remains the numerical oracle. Two optimized paths now pass the
same independent CPU comparison on the 256 by 5,120 slice:

| Shape/path | Time per launch | Effective throughput | Max absolute error |
|---|---:|---:|---:|
| Decode, scalar | 345.0 us | 7.6 GFLOP/s | `3.22e-06` |
| Decode, 64-lane subgroup | 59.5 us | 44.0 GFLOP/s | `4.77e-07` |
| 8-vector prefill, scalar | 362.0 us | 57.9 GFLOP/s | `5.36e-06` |
| 8-vector prefill, subgroup | 338.6 us | 61.9 GFLOP/s | `4.77e-07` |
| 8-vector prefill, 4-vector tile | 232.7 us | 90.1 GFLOP/s | `4.77e-07` |

The tiled path stages one row's packed values and E4M3 scales in OpenCL local
memory and shares them across four subgroups. A five-vector probe also passed,
covering the masked vector-tile tail.

The checkpoint contains only two native NVFP4 packed shape families:
`[17408, 2560]` (K=5120 gate/up) and `[5120, 8704]` (K=17408 down). A 64-row,
K=17408 down-projection slice passed all three paths; tiled prefill reached
83.2 GFLOP/s with `1.58e-06` maximum absolute error. This covers every NVFP4 K
width in the model without allocating a full layer.

`native_nvfp4/runtime` is now a Windows ARM64 shared library with a C ABI. It
creates one context/queue/program, uploads native matrices once, and reuses input
and output buffers. Its synchronous end-to-end call time, including CPU/OpenCL
activation and output copies, was 251.6 us for decode and 326.5 us for tiled
eight-vector prefill at the same slice.

## Current backend choice

OpenCL is the first kernel backend because Qualcomm exposes a native Windows ARM64
OpenCL 3.0 device and llama.cpp already has Qwen 3.5 attention, linear-attention,
SSM, and Adreno plumbing. PyTorch installed here is ARM64 CPU-only (`2.13.0+cpu`),
so vLLM or SGLang cannot reach the Adreno by adding only one quantization op.

vLLM remains the preferred host. `vllm_nvfp4_opencl` now scaffolds a general
plugin around vLLM's modular `NvFp4LinearKernel`: post-load hooks upload native
matrices and `apply_weights` preserves arbitrary leading activation dimensions
and bias semantics. A real `(2, 4, 5120)` adapter probe passed without GGUF or
dequantized weights.

Current vLLM `main` exposes an NVFP4 kernel registry, but its W4A16 selector
hard-codes Marlin and Humming. The local `vllm` checkout contains a capability
patch (`supports_a16`) and unit test so registered CPU/OOT providers can be
selected. Duplicate open-PR searches found no matching work as of 2026-08-22.
The patch is lint-clean but its pytest could not collect in the deliberately
minimal Windows environment because the full vLLM test dependencies are absent.

## Tiptoe rules

- Do not run Vulkan with full model offload (`-ngl 99`, `-ngl 999`) on the current
  driver.
- Keep probes in `scripts/run-isolated.ps1`, with one small, unique shape and a
  short timeout.
- Do not treat a blank/incomplete test line as success even if the Qualcomm ICD
  returns exit code 0; require an explicit `PASS`/`OK` line.
- Do not run the broad OpenCL operation suite yet. Test one named tensor/shape.
- Keep at least several GiB outside the advertised GPU budget. Do not infer GPU
  capacity from 48 GB physical UMA.
- Preserve packed values and scale bytes. Never expand weights merely to adapt an
  execution framework.
- Add decode GEMV first, then a separately tested prefill GEMM. Do not advertise a
  shape in backend capability checks until that exact shape passes.

## Next implementation boundary

Completed since the initial boundary was written:

- profiled reusable device buffers and event-based queued submission;
- direct row-scaled FP8 kernels and shape-aware decode dispatch;
- a four-row NVFP4 decode tile over both exact MLP shape families;
- device norms, gating, residual, Q/K layout, causal-convolution, and
  gated-delta kernels;
- exact resident NVFP4 MLP and linear-attention decoder graphs;
- a multithreaded ARM64 NEON direct-NVFP4 fallback;
- a QAIRT-schema-valid QHPI NVFP4 op definition.

The next implementation boundary is now:

1. Retain all 40 layers' KV, gated-delta, and convolution state across a real
   autoregressive generation loop.
2. Gather each generated token's embedding row lazily and add device or
   partial-logit greedy/top-k/top-p sampling.
3. Measure sustained decode and prefill under the complete 32K BF16 residency
   policy, including thermal behavior and more than one request.
4. Connect the complete token boundary to continuous batching in the vLLM OOT
   worker, then compare its control plane with SGLang and Atlas.
5. Move HTP QHPI skeleton generation to an x86-64 Python 3.10 build host, then
   implement scalar correctness and HVX/HMX optimization for Hexagon v81.

## Bandwidth-first checkpoint

Fine-grained buffer SVM is not merely a memory-capacity mechanism on this
driver. Wrapping checkpoint-native packed/scales allocations with
`CL_MEM_USE_HOST_PTR` reduced the exact 17408x5120 row-tiled kernel from a
2.3258 ms median to 1.4651 ms in a 30-sample interleaved run. That lifts logical
native payload delivery from 21.56 to 34.22 GB/s while CPU and GPU retain access
to one backing store.

The matched 16-byte-vector raw-read ceiling is approximately 129 GB/s, so the
SVM NVFP4 kernel is still only at 26.5% of the raw island ceiling. Decode remains
instruction/decode limited or occupancy limited rather than DRAM saturated.
CPU+GPU raw reads on separate allocations reached only 110.72 aggregate GB/s
and slowed both engines, providing no current reason to split batch-one decode.

A manual `uchar8` packed load was a useful negative result: the compiler forbids
dynamic vector indexing, and spilling through a private array increased exact
SVM latency to roughly 12.6 ms. That variant was removed. Future vectorization
must be verified in compiler output and must avoid private-array materialization.

The allocation change survives composition. The resident layer-0 MLP dropped to
5.051 ms, the full layer to 7.848 ms, and the exact four-layer cadence to
30.151 ms kernel / 30.465 ms queued wall. Paged batch-one scheduling now reaches
31.98 request-tokens/s and batch four reaches 38.11 aggregate request-tokens/s,
with exact request-state oracle matches. SVM is now the plugin default, with
`VLLM_NVFP4_OPENCL_SVM=0` and automatic allocation-failure fallback preserving
the conventional path.

The same runtime now covers the sparse 35B checkpoint's native expert shapes.
Eight fixed real experts consume 14.16 MB of native payload in 0.6477 ms. The
checkpoint-routed successor adds a BF16 router and shared gate, exact top-8
renormalization, the always-on shared NVFP4 expert, and device weighted output
reduction. It measures 0.8230 ms kernel / 1.2767 ms wall over 30 samples, with
exact selected expert IDs and `1.16e-9` final-output maximum error. The remaining
MoE graph gap is device-resident top-k/indirect dispatch plus staged full-model
expert residency, not expert arithmetic correctness.

The device-dispatch gap is now closed for one layer. A contiguous fine-grained
SVM bank holds all 256 routed experts plus the shared expert, a GPU top-8 kernel
produces IDs/weights, and row-tiled bank kernels consume those IDs without a
host boundary. The 30-sample result is 0.7563 ms kernel / 0.9231 ms wall with
`1.16e-9` final-output error, improving the host-dispatch wall by 27.7%. The cost
is 454.75 MB per layer; 40 banks are 16.94 GiB, so staged full-checkpoint budget
validation is now the binding MoE task.

The bounded residency ladder now passes with actual layers at 3/5/10/19 banks,
or 1.36/2.27/4.55/8.64 GB of native payload. Every new layer was routed and
validated while all earlier banks remained live; maximum error was `2.56e-9`.
Closing the 19 banks recovered 8.54 GB immediately. This completes the planned
1/2/4/8 GiB safety ladder, but the remaining full-model budget is still too
tight to skip explicit non-expert, KV, scratch, and Windows headroom accounting.

That accounting is now exact at the tensor-metadata level. The 35B MoE contains
21.800 GiB of tensor payload: 18.448 GiB of resident text compute weights under
the proposed policy, a 0.947 GiB embedding kept lazy on CPU, 0.832 GiB of
optional vision, and 1.573 GiB of optional BF16 MTP. The dense 27B contains
21.809 GiB: 17.792 GiB resident text compute weights, a 2.368 GiB lazy embedding,
0.858 GiB vision, and 0.791 GiB MTP. No tensor is unclassified.

With current FP32 state, the model-derived recurrent/conv/known-scratch cost is
67.9 MiB per 35B request and 159.9 MiB per dense request. FP32 KV costs 1.25 GiB
per 32K request for MoE but 4.00 GiB for dense because it has 16 full-attention
layers and four KV heads. Against the observed 23.81 GiB OpenCL budget and a
2 GiB safety reserve, 35B 32K fits with 2.04 GiB additional headroom. Dense 32K
misses the policy by 0.14 GiB; dense 16K FP32 or dense 32K BF16 are the safe
initial choices. These are planned allocations, not a full-load pass.

The first policy implementation is now live for the dense paged-attention path.
Round-to-nearest-even BF16 K/V halves the exact pool allocation and matches an
independent BF16 cache oracle across a page boundary. On the four-layer cadence,
BF16 measures 30.428 ms at batch one and 102.266 ms at batch four versus 30.484
and 102.037 ms for FP32. Relative RMSE against FP32 is `4.08e-5`/`5.64e-5`.
This validates the storage primitive and vLLM lifecycle, not 32K quality or a
complete dense-model load. FP32 remains the default; the adapter exposes BF16 by
argument or `VLLM_NVFP4_OPENCL_KV_DTYPE=bf16`.

The same cache path is no longer dense-only. Parameterizing query/KV heads
preserves the dense 24/4 path and adds Ornith's exact 16/2 GQA shape. MoE BF16
pages are one quarter the byte size of dense FP32 pages and match an independent
BF16 storage oracle within `1.04e-7` across a page boundary.

Ornith attention also required exact tensor-scaled FP8 support: its E4M3 weights
carry one FP32 scale per matrix rather than BF16 row scales. With that native
format added, real layer-3 full attention measures 0.6309 ms kernel / 0.8407 ms
wall using BF16 KV. Composing it with post-attention norm and the resident
device-routed expert bank produces the first complete sparse decoder layer at
1.3923 ms kernel / 1.5549 ms wall and `5.96e-8` maximum oracle error. This closes
the full-attention MoE layer arithmetic boundary.

Ornith's 16-key/32-value-head gated-delta profile is now represented directly
rather than reusing the dense model's 16/48 shape. A complete real layer-0
linear-attention plus MoE step measures 1.6442 ms kernel / 1.8166 ms wall and
matches its independent CPU oracle within `3.58e-7`. The checkpoint's true
layers 0-3 cadence (linear, linear, linear, full) holds four independent expert
banks totaling 1,819,017,216 bytes plus every recurrent, convolution, and BF16
KV state. Queued behind one synchronization it measures 6.4166 ms kernel /
6.8675 ms wall and exactly matches the same proven device layers synchronized
individually.

Ten measured four-layer kernels arithmetically project to 64.166 ms, or about
15.58 decode tokens/s, but this is not yet a full-model measurement. Final norm,
LM head, sampling, the complete 40-layer registry, full-checkpoint memory
pressure, and serving-control overhead remain open.

The remaining projection format is now closed independently. Ornith's final
248,320x2,048 LM head is native NVFP4 with 286,064,640 bytes of packed weights
and block scales. Final RMSNorm plus all vocabulary logits measures 9.1450 ms
kernel / 11.0537 ms wall, matches an independent chunked CPU decoder within
`2.86e-6`, and returns the same argmax. The open boundary is composition with
the complete 40-layer registry, not LM-head arithmetic.

That registry now passes. All 19,807,914,740 bytes (18.448 GiB) of planned
Ornith text-compute checkpoint payload coexist with the 40-layer graph and a
671,088,640-byte 32K BF16 KV pool. The staged 24/30/35/40-bank gates validate
independently, and teardown recovers 20,427,419,648 bytes without a driver
reset. A lazy BF16 embedding row then traverses all 40 real layers and the full
LM head in 75.8837 ms kernel / 79.3810 ms wall, or 12.60 measured tokens/s.
Single-queue logits are bit-identical to the same graph synchronized after each
layer. The next correctness boundary is retained-state autoregressive decoding,
not a larger residency projection.

That boundary now passes for 32 consecutive greedy steps. All recurrent,
convolution, and BF16 KV state is retained, the sequence crosses the first KV
page boundary, and the replayed full logits are bit-identical at every position.
Median kernel time is 76.1513 ms/token, median host-loop wall is 81.3112 ms, and
mean end-to-end throughput is 12.17 tok/s. Because the local checkpoint lacked
tokenizer files during this gate, a raw BOS-only seed repeats token 95,726; it
is not a coding-quality result. The official tokenizer/chat template is the
next prompt-level gate.

The official tokenizer-backed prompt gate now produces a complete, correct
typed Fibonacci implementation. Thirty prompt tokens prefill sequentially at
13.86 tok/s; retained-state decode averages 11.75 end-to-end tok/s over 183
generated tokens and stops on `<|im_end|>` at position 212. The queued prompt
and generation are bit-identical to a full layer-synchronized replay. A
length-capped precursor demonstrated why EOS handling is mandatory: tokens
after 248046 were unrelated garbage. The current loop loads both official stop
IDs and reports `finish_reason=stop`.

Dense 27B now passes its complete residency boundary as well. Its exact
19,103,683,968-byte text payload mixes 56 native NVFP4 MLP layers with eight
row-scaled FP8 MLP layers, row-scaled FP8 attention, and a full FP8 vocabulary
head. A 2,147,483,648-byte 32K BF16 KV pool and all recurrent state coexist with
the weights. The full token measures 511.9609 ms kernel / 519.9734 ms wall, or
1.923 tok/s, with bit-identical queued and layer-synchronized logits.

Closing each safetensors layer mapping immediately after upload is essential.
One successful whole-file-mapping probe transiently left only 465 MB available;
the per-layer loader leaves 19.60 GB at the same complete device-residency gate
and returns 21.68 GB on teardown. Source lifetime, not final capacity, was the
dangerous part.

Dense retained-state decoding now passes a tokenizer-backed 27-token prompt and
32-token generation gate. Sequential prefill is 2.003 tok/s with 13.477 s TTFT;
31 measured decode steps average 1.908 end-to-end tok/s with a 512.533 ms median
kernel and 525.912 ms median wall. Full replay with a synchronization after each
layer returns bit-identical logits and tokens.

Full-model useful-byte attribution puts dense at 37.27 GB/s during its sustained
decode kernel (28.9% of the calibrated 129 GB/s GPU raw-read ceiling, 16.3% of
the nominal 228 GB/s SoC rate). MoE activates only about 2.265 GB of its 19.808
GB resident payload per token; it reaches about 29.02 GB/s (22.5% calibrated,
12.7% nominal). These are logical checkpoint-byte rates, not memory-controller
counters.

The runtime now retains scope and OpenCL command timestamps for an opt-in trace
without inserting queue barriers. Three-sample exact-replay captures contain
1,042 dense events and 772 MoE events. Dense quantized linears consume 95.3% of
kernel time; its 56 NVFP4 MLPs average 5.133 ms/layer while the eight same-shape
FP8 MLPs average 5.129 ms/layer despite NVFP4 moving roughly 44% fewer bytes.
NVFP4 decode efficiency is therefore the primary dense bottleneck. On MoE,
experts are 42.4%, linear attention 36.7%, the head 12.8%, and full attention
7.9%; expert down reaches only 18.52 GB/s, and top-8 alone costs 4.137 ms/token.
Inter-kernel gaps are only 2.204 ms dense and 0.915 ms MoE. The remaining trace
gap is the client-facing tokenizer/scheduler/upload/download/detokenizer/SSE
path plus hardware memory/cache/occupancy counters.

## First decoder benchmarks

Direct row-scaled FP8 kernels now accompany NVFP4 in the persistent runtime. A
real Qwen3.5 layers 0-3 stack, using the checkpoint's native FP8 attention and
NVFP4 MLP matrices, completed 32-token prefill at 24.78 stack-tokens/s and cached
decode at 4.20 stack-tokens/s. See `BENCHMARKS.md` for methodology and the strict
distinction between this measured four-layer result and the unmeasured full-model
projection.

The persistent runtime now owns both gated-delta and causal-convolution state.
Independent CPU oracles pass at the exact 48-head and 10240-channel model shapes.
Moving both operators to Adreno improves the controlled four-layer cached decode
from 238.1 to 55.8 ms/token (4.20 to 17.93 stack-tokens/s). Native linears now
consume 52.47 ms/token and are the dominant target.

The Hexagon NPU was also validated through QAIRT 2.45.40: FastRPC and the v81
calculator unit test pass. Stock HTP deployment supports FP16/INT8/INT16 rather
than packed NVFP4, so direct E2M1 consumption needs a custom HTP op package. See
`SERVING.md` and `native_nvfp4/npu/README.md` for the worker and device split.
