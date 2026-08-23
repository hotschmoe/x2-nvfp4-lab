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

1. Implement one-token full attention on device: RoPE, per-request grouped-query
   KV cache, stable softmax, value reduction, and output projection.
2. Chain three resident linear-attention layers plus one full-attention layer and
   their MLPs as the real four-layer cadence, synchronizing once per token.
3. Add a budget-aware matrix registry and per-request state arena; validate
   incremental 1/2/4/8 GiB weight residency before attempting the full model.
4. Expose `prefill` and continuous-batch `decode` calls through a vLLM OOT worker.
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
