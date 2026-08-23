# X2 NVFP4 Lab

Checkpoint-native NVFP4 inference research for the Snapdragon X2 Elite Extreme:
make it work, make it right, then make it fast.

The project is building a Windows ARM64 runtime that serves dense Qwen3.5 27B
and sparse Qwen3.5-MoE 35B-class checkpoints without GGUF or persistent
dequantized weight copies. vLLM remains the scheduler/API boundary; the native
runtime owns Adreno OpenCL kernels, ARM64 NEON execution, recurrent state, and
paged KV cache.

Current highlights on an Adreno X2-90:

- Direct packed E2M1 + E4M3-scale NVFP4 execution from safetensors.
- Fine-grained buffer SVM gives CPU and GPU one native weight backing store.
- Exact 17408x5120 GEMV: 2.326 ms copied vs 1.465 ms shared SVM (1.59x).
- Exact four-layer Qwen3.5 cadence: 30.151 ms kernel / 30.465 ms queued wall.
- Paged batch-one: 31.98 request-tokens/s; batch-four: 38.11 aggregate.
- Real 35B MoE top-8 expert micrograph: 0.648 ms kernel per layer.
- Independent CPU/GPU tensor oracles and isolated-process accelerator gates.

Start with [CAMPAIGN_BANDWIDTH_FIRST.md](CAMPAIGN_BANDWIDTH_FIRST.md) and
[UNIFIED_MEMORY_RESEARCH.md](UNIFIED_MEMORY_RESEARCH.md). Reproduce the current
bandwidth sprint with:

```powershell
scripts/run-bandwidth-sprint.ps1
```

Hardware-specific safety rules matter here: do not attempt full Vulkan offload
on the current Qualcomm Windows driver, keep experiments behind
`scripts/run-isolated.ps1`, and require explicit completion markers.

This is an experimental research runtime, not a production release—yet.
