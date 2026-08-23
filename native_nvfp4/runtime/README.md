# Native NVFP4/FP8 C ABI runtime

The shared library owns one OpenCL context, queue, and compiled program per
process. `nvfp4_matrix_upload` retains exact packed E2M1 values and E4M3 block
scales in device buffers. `nvfp4_linear_f32` reuses activation/output buffers and
dispatches scalar, subgroup, or four-vector tiled kernels synchronously.
The companion FP8 handles retain signed E4M3 weights and BF16 per-row scales.
Qwen3.5 gated-delta and width-4 causal-convolution handles keep their recurrent
state on the device across decode calls.

Queueable graph primitives also include direct BF16 GEMV for checkpoint routers
and scalar gates, plus reset-or-add weighted float32 accumulation for sparse
expert reduction. These preserve BF16 router storage and keep routed expert
outputs resident until the graph's explicit materialization boundary.

`nvfp4_moe_bank` adds a device-routed sparse decode path. It stores all routed
experts plus the shared expert in contiguous fine-grained SVM allocations,
streams one projection at a time into indexed offsets, and runs BF16 routing,
top-8, row-tiled NVFP4 experts, and weighted reduction in one queued stage.

The bandwidth-campaign ABI also exposes OpenCL/SVM capability metadata, raw
checksum-protected CPU/GPU streaming reads, and fine-grained-SVM native matrix
handles. `nvfp4_matrix_upload_shared_svm` copies checkpoint bytes once into SVM,
wraps those allocations with `CL_MEM_USE_HOST_PTR`, and retains the same pointers
for `nvfp4_matrix_cpu_linear_f32`. The `cl_mem` views are released before their
SVM allocations.

Opt-in tracing attaches a caller-defined logical scope to every queued graph
operation and exposes the completed OpenCL queued/submit/start/end timestamps.
It does not add synchronization points. `nvfp4_runtime_trace_set_enabled`,
`nvfp4_runtime_trace_set_scope`, `nvfp4_runtime_trace_count`, and
`nvfp4_runtime_trace_read` provide the C boundary; the Python runtime returns
immutable `TraceEvent` records. The next synchronize replaces the completed
trace, so consumers must read it before submitting another measured graph.

Build on this Windows ARM64 machine:

```powershell
cmake -S native_nvfp4/runtime -B native_nvfp4/runtime/build --fresh -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER='C:/Program Files/LLVM/bin/clang-cl.exe' `
  -DCMAKE_RC_COMPILER='C:/Program Files/LLVM/bin/llvm-rc.exe' `
  -DCMAKE_MT='C:/Program Files/LLVM/bin/llvm-mt.exe' `
  -DOpenCL_INCLUDE_DIR='C:/Qualcomm/OpenCL_SDK/2.3.2/include' `
  -DOpenCL_LIBRARY='C:/Qualcomm/OpenCL_SDK/2.3.2/lib/OpenCL.lib'
cmake --build native_nvfp4/runtime/build
```

The runtime must outlive its matrix and state handles. The current API is deliberately
synchronous and accepts float32 CPU activation/output pointers. Persistent native
weights are solved; zero-copy or staged asynchronous activations are the next ABI
revision.

`nvfp4_linear_device_lab_f32` is an intentionally synchronous experimental ABI.
It sweeps row sharing, dynamic local K tiles, direct-global controls, and
scalar/vector decode without changing production kernel selection. Use
`native_nvfp4/bench_nvfp4_kernel_lab.py`; candidates must pass its output oracle
before timing and still require full-model A/B before promotion.

`nvfp4_gemm_device_lab_f32` provides the corresponding multi-vector treatments:
dynamic vector sharing and weight K tiles, plus direct-global scalar/vector
controls. It is exercised by `native_nvfp4/bench_nvfp4_gemm_lab.py` and is not a
production dispatch promise.

Shape-specific winners are enabled in production dispatch by default. Set
`VLLM_NVFP4_OPENCL_SHAPE_TUNING=0` before runtime creation for the reproducible
previous-path control. The full-model benchmark JSON records this setting.
