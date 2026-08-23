# Native NVFP4 backend probe

This path consumes the compressed-tensors safetensors representation directly:

- `weight_packed`: two E2M1 values per byte, adjacent along K
- `weight_scale`: one E4M3 scale byte per 16 weights
- `weight_global_scale`: the divisor that returns local scales to model units

No GGUF conversion and no FP16/BF16 weight expansion occurs. The Python probe
memory-maps a small slice of a real tensor, computes an independent CPU result,
then uploads the original packed bytes and scale bytes to the OpenCL kernel.

Run validation in a disposable child process from the workspace root. The
wrapper preserves logs and contains ordinary process faults (though no wrapper
can contain a system-wide GPU driver reset):

```powershell
$python = (Get-Command py).Source

scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_native_nvfp4.py'

# Four-vector prefill/GEMM probe at the real Qwen gate-projection K width
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_native_nvfp4.py --cols 5120 --rows 256 --vectors 4'

# Compare scalar, subgroup, and four-vector tiled paths
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_native_nvfp4.py --cols 5120 --rows 256 --vectors 8 --kernel both'

# Exercise persistent C handles and reusable activation/output buffers
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_runtime.py --cols 5120 --rows 256 --vectors 8 --kernel tiled'

# Exercise the vLLM post-load/apply lifecycle against its current interface
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_vllm_adapter.py --cols 5120 --rows 256 --vectors 8'

# Persistent 48-head gated-delta decode state
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_gated_delta.py --heads 48 --tokens 1 --iterations 20'

# Persistent 10240-channel width-4 convolution state
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_causal_conv.py --channels 10240 --tokens 1 --iterations 50'

# Profile reusable device buffers and queued NVFP4 submission
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_device_runtime.py --rows 17408 --cols 5120 --iterations 20'

# Validate device graph primitives and exact resident NVFP4 MLP
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_device_ops.py'
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/bench_resident_mlp.py --layer 0 --iterations 20'

# Validate direct ARM64 NEON fallback at the exact gate shape
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_cpu_neon.py --rows 17408 --cols 5120 --iterations 3 --threads 1 4 8 0'

# Exact resident linear-attention layer, then the complete layer plus NVFP4 MLP
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/bench_resident_linear_attention.py --layer 0 --iterations 20'
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/bench_resident_linear_attention.py --layer 0 --iterations 20 --with-mlp'

# Full-attention CPU oracle, exact layer-3 checkpoint graph, and resident cadence
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_full_attention.py'
scripts/run-isolated.ps1 -Executable $python -TimeoutSeconds 300 `
  -CommandLine '-3 native_nvfp4/bench_resident_full_attention.py --layer 3 --tokens 8 --iterations 20'
scripts/run-isolated.ps1 -Executable $python -TimeoutSeconds 300 `
  -CommandLine '-3 native_nvfp4/bench_resident_cadence.py --first-layer 0 --iterations 8'
scripts/run-isolated.ps1 -Executable $python -TimeoutSeconds 300 `
  -CommandLine '-3 native_nvfp4/probe_serving_session.py'

# Paged KV/block tables, continuous batches, and vLLM request lifecycle
scripts/run-isolated.ps1 -Executable $python `
  -CommandLine '-3 native_nvfp4/probe_paged_attention.py'
scripts/run-isolated.ps1 -Executable $python -TimeoutSeconds 300 `
  -CommandLine '-3 native_nvfp4/bench_paged_scheduler.py --tokens 18 --requests 4'
scripts/run-isolated.ps1 -Executable $python -TimeoutSeconds 300 `
  -CommandLine '-3 native_nvfp4/probe_vllm_cadence_adapter.py'

# Real Qwen3.5 four-layer cadence: 3 linear-attention + 1 full-attention
scripts/run-isolated.ps1 -Executable $python -TimeoutSeconds 300 `
  -CommandLine '-3 native_nvfp4/bench_qwen35_block.py --layer 0 --layer-count 4 --sequence-length 32 --prefill-iterations 3 --decode-tokens 8 --gpu-gated-delta --gpu-causal-conv'

# Exact sparse layer: BF16 route, shared expert, top-8 NVFP4 experts, reduction
scripts/run-isolated.ps1 -Executable $python -TimeoutSeconds 180 `
  -CompletionMarker 'MOE_NVFP4_ROUTED_LAYER_PASS' `
  -CommandLine '-3 native_nvfp4/bench_moe_routed_layer.py --warmups 5 --samples 30'
```

The scalar GEMV/GEMM kernels assign one work-item to each output row/vector pair
and remain the semantic oracle. The first optimized path gives one Qualcomm
64-lane subgroup to each output pair and reduces the partial dot product inside
the subgroup. The prefill path groups four subgroups in one workgroup and stages
native weight bytes once in local memory for reuse across four vectors. All paths
support arbitrary multiples of 16 along K; tiled prefill masks non-multiple tails.

The checkpoint's `input_global_scale` describes W4A4 activation quantization.
The current A16 path does not quantize the input, matching llama.cpp's treatment
of this converted model, so only `weight_global_scale` is applied here.

The same kernel program and C ABI include row-scaled FP8 E4M3 linears, precise
event profiling, asynchronous device-to-device submission, persistent recurrent
state, and the float32 graph primitives needed between projections. The
framework-neutral graph layer now keeps a full Qwen3.5 four-layer cadence
resident: three linear-attention layers, one full-attention layer with online
softmax and a persistent or paged KV cache, and all four NVFP4 MLPs. The serving
layer now has shared 16-token pages, per-request state, continuous-batch
projection GEMMs, and a vLLM `SchedulerOutput` adapter. Automatic model-runner
attachment, sampling, preemption/state transfer, and long-context attention
tiling remain next.
