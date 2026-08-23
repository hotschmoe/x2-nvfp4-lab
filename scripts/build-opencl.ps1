param(
    [string]$SourceDir = (Join-Path $PSScriptRoot '..\llama.cpp'),
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\llama.cpp\build-opencl')
)

$ErrorActionPreference = 'Stop'
$vsDevCmd = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
$llvmBin = 'C:\Program Files\LLVM\bin'
$openclRoot = 'C:\Qualcomm\OpenCL_SDK\2.3.2'

foreach ($path in @($SourceDir, $vsDevCmd, $llvmBin, $openclRoot)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path is missing: $path" }
}

# The OpenCL backend is upstream's Snapdragon/X2-90 path.  It embeds the current
# Adreno GEMM/GEMV kernels and enables loading Qualcomm's optional binary library.
$cmake = @"
call `"$vsDevCmd`" -arch=arm64 -host_arch=arm64 &&
set PATH=$llvmBin;%PATH% &&
cmake -S `"$SourceDir`" -B `"$BuildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=`"$SourceDir\cmake\arm64-windows-llvm.cmake`" -DGGML_OPENCL=ON -DOpenCL_ROOT=`"$openclRoot`" -DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_OPENCL_USE_ADRENO_BIN_KERNELS=ON -DBUILD_SHARED_LIBS=OFF &&
cmake --build `"$BuildDir`" --parallel 18
"@ -replace "`r?`n", ' '

cmd.exe /c $cmake
if ($LASTEXITCODE) { throw "llama.cpp build failed ($LASTEXITCODE)" }

# Clang links this filename when OpenMP is enabled.  It is not part of the normal
# Windows PATH, so colocate the ARM64 runtime with the executables.
$omp = Get-ChildItem 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools' -Recurse -Filter 'libomp140.aarch64.dll' |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $omp) { throw 'ARM64 LLVM OpenMP runtime was not found.' }
Copy-Item -LiteralPath $omp -Destination (Join-Path $BuildDir 'bin\libomp140.aarch64.dll') -Force

& (Join-Path $BuildDir 'bin\llama-bench.exe') --list-devices
