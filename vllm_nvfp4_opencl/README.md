# vLLM native NVFP4 OpenCL provider

This research plugin registers a modular `NvFp4LinearKernel` for CPU and
out-of-tree vLLM platforms. It retains compressed-tensors `weight_packed` and
E4M3 `weight_scale` data, uploads each matrix once to the persistent OpenCL C
runtime, and dispatches the Qualcomm subgroup GEMV/GEMM path.

The current boundary is deliberately explicit:

- Persistent native NVFP4 and row-scaled FP8 matrices.
- Reusable device buffers, profiled asynchronous submission, and one queue per
  process.
- Resident Qwen3.5 NVFP4 MLP, linear attention, full attention, and exact
  four-layer decode cadence.
- Device-resident RMSNorm, gated norm, elementwise, layout, convolution, and
  gated-delta operators.
- Native BF16 router/gate GEMV and weighted sparse-expert accumulation.
- Exact tensor-scaled FP8 matrices for the sparse checkpoint alongside the
  dense checkpoint's BF16 row-scaled FP8 representation.
- Contiguous SVM expert banks with device top-8 and direct indexed dispatch.
- Direct ARM64 NEON NVFP4 GEMV for hybrid placement.
- Fine-grained buffer SVM is the default native NVFP4 backing. The model-specific
  loader can release its source arrays after the one file-to-SVM copy; generic
  vLLM layer parameters remain resident until the framework releases them.
- The conventional copied-buffer path remains an automatic fallback. Neither
  path creates a dequantized BF16/FP16 weight copy.
- Full-attention online softmax and persistent K/V cache are implemented.
- Shared 16-token KV pages, per-request block tables, and continuous-batch FP8
  attention/NVFP4 MLP projections are implemented.
- Paged attention accepts both exact Qwen3.5 head profiles: dense 24/4 and MoE
  16/2, with independently selectable FP32 or BF16 K/V storage.
- Gated-delta preparation accepts dense 16-key/48-value heads and MoE
  16-key/32-value heads; the resident graph sizes convolution/recurrent state
  from that profile rather than assuming the dense model.
- Sampling, preemption/state transfer, and automatic OOT model-runner attachment
  are not yet implemented.
- Outside the packaged adapter, the native full-model qualification now retains
  all 40 Ornith layers, 32K BF16 state, final NVFP4 head, and tokenizer-backed
  greedy generation at 11.75 end-to-end tok/s. Moving that registry/session
  boundary into this package is the immediate integration task.
- Dense 27B full residency also passes outside the adapter, including its last
  eight row-scaled-FP8 MLPs and row-scaled-FP8 LM head. The packaged graph now
  exposes resident FP8 MLP and LM-head fragments for that mixed checkpoint.

The backend is opt-in so installing the plugin cannot hijack unrelated CPU
vLLM deployments:

```powershell
$env:VLLM_NVFP4_OPENCL = '1'
$env:VLLM_NVFP4_OPENCL_DLL = 'C:\path\to\nvfp4_runtime.dll'
$env:VLLM_NVFP4_OPENCL_KERNEL = 'C:\path\to\nvfp4_gemv.cl'
```

Set `VLLM_NVFP4_OPENCL_SVM=0` to force conventional copied buffers. Otherwise
the runtime prefers shared SVM and falls back automatically if capability or
allocation validation fails.

The dense-Qwen paged serving seam keeps FP32 KV as the compatibility default.
Set `VLLM_NVFP4_OPENCL_KV_DTYPE=bf16` (or pass `kv_dtype="bf16"` to
`VllmCadenceAdapter`) to use the validated half-size BF16 cache.

Current vLLM `main` needs the adjacent capability patch in this workspace so a
registered out-of-tree kernel can advertise W4A16 support. Once that patch is
upstream, the plugin does not need to modify vLLM's kernel registry.

For the resident model-specific path, shared checkpoint matrices and per-request
state have separate lifetimes:

```python
from vllm_nvfp4_opencl.runtime import Runtime, runtime_paths
from vllm_nvfp4_opencl.serving import Qwen35CadenceWeights

runtime = Runtime(*runtime_paths())
weights = Qwen35CadenceWeights.load(runtime, "model.safetensors", first_layer=0)
session = weights.create_session(max_tokens=8192)
hidden, profile = session.step(hidden_float32)
```

Create one session per active sequence. Sessions share all 25 matrices in a
four-layer cadence but own separate gated-delta, convolution, and K/V state.
Close sessions before their shared weights. Build a wheel containing the exact
DLL and OpenCL source with `scripts/package-opencl-provider.ps1`.

The scheduler-facing path accepts CPU torch hidden states and vLLM V1
`SchedulerOutput` lifecycle metadata:

```python
from vllm_nvfp4_opencl.vllm_adapter import VllmCadenceAdapter

adapter = VllmCadenceAdapter(
    weights,
    max_pages=512,
    default_max_tokens=8192,
    max_batch_size=4,
)
hidden_states, profile = adapter.execute_scheduler_output(
    scheduler_output,
    hidden_states,
)
```

Request-major prompt chunks are replayed temporally as a correctness-first
prefill path. The adapter explicitly rejects preemption/resume rather than
silently losing recurrent state.
