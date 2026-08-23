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
- Real checkpoint-routed 35B MoE layer micrograph: 0.823 ms kernel / 1.277 ms
  wall, including BF16 router, shared expert, top-8 experts, and reduction.
- Device-routed 256-expert SVM bank: 0.756 ms kernel / 0.923 ms wall with no
  host routing boundary; one complete layer bank occupies 454.8 MB.
- Real cumulative residency passes at 1.36/2.27/4.55/8.64 GB across 3/5/10/19
  expert banks; all layers pass and closing recovers 8.54 GB immediately.
- Metadata-exact memory plans cover every tensor in both checkpoints. The first
  coding-service policy omits vision/MTP, keeps embedding lookup lazy on CPU,
  and retains a 2 GiB OpenCL safety reserve.
- Opt-in BF16 paged KV halves cache storage for the dense model with unchanged
  four-layer cadence and `5.64e-5` relative RMSE versus FP32 at batch four.
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
