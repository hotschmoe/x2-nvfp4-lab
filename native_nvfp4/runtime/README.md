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
