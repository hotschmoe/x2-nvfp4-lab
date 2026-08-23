# Hexagon NPU track

The installed Qualcomm NPU stack is usable on this machine:

- device: Snapdragon X2 Elite Extreme Hexagon NPU;
- architecture: Hexagon v81;
- QAIRT: 2.45.40.260406;
- Hexagon SDK: 6.6.0.0 with tools 19.0.07;
- platform validator: DSP hardware and libraries found, calculator unit test passed.

Re-run the small isolated platform check from the workspace root:

```powershell
native_nvfp4/npu/validate-qnn.ps1
```

`platform_validator/Result.csv` is the durable machine-readable result. This
probe does not load an LLM or allocate model weights.

## NVFP4 route

The stock HTP deployment formats are FP16, INT8, and INT16. Direct packed E2M1
weights therefore require an HTP op package rather than an ordinary converted
QNN MatMul. QAIRT includes the HTP op-package generator and v81 examples; the
installed Hexagon compiler can produce the device-side shared object.

The first custom op should keep the same logical contract as the OpenCL path:

- FP16 activation `[vectors, K]`;
- packed E2M1 weights `[M, K/2]`;
- E4M3 scales `[M, K/16]` and one global divisor;
- FP16 output `[vectors, M]`;
- no persistent dequantized weight tensor.

`NvFp4LinearQhpiHtp.xml` now defines that contract and
`validate_opdef.py` passes against QAIRT 2.45's `OpDef.xsd`. The installed
generator cannot produce the HTP skeleton on this ARM64 host: its ARM64 Python
package lacks `qnn_ir`, while the complete generator libraries are x86-64 and
require Python 3.10. Run generation on an x86-64 Linux/WSL build host, then copy
the generated QHPI package back for v81 compilation.

Start with scalar/HVX decode correctness, then add 128-byte HVX loads and an
HMX-friendly accumulation layout. Until that package is numerically validated,
the NPU is best used for supported FP16/INT8 graph partitions while native
NVFP4/FP8 linears remain on Adreno.
