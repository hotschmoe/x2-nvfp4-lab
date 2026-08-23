param(
    [string]$SourceDir = (Join-Path $PSScriptRoot '..\llama.cpp'),
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\llama.cpp\build-cpu')
)

# Deliberately does not enable Vulkan, OpenCL, or Hexagon.  This is the recovery
# build used to establish that model conversion and CPU kernels are sound without
# loading a Qualcomm graphics driver into the inference process.
$ErrorActionPreference = 'Stop'
$vsDevCmd = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
$llvmBin = 'C:\Program Files\LLVM\bin'

foreach ($path in @($SourceDir, $vsDevCmd, $llvmBin)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path is missing: $path" }
}

$cmake = @"
call `"$vsDevCmd`" -arch=arm64 -host_arch=arm64 &&
set PATH=$llvmBin;%PATH% &&
cmake -S `"$SourceDir`" -B `"$BuildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=`"$SourceDir\cmake\arm64-windows-llvm.cmake`" -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.7-a -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_HEXAGON=OFF -DBUILD_SHARED_LIBS=OFF &&
cmake --build `"$BuildDir`" --target llama-cli llama-bench llama-server --parallel 18
"@ -replace "`r?`n", ' '

cmd.exe /c $cmake
if ($LASTEXITCODE) { throw "CPU-only llama.cpp build failed ($LASTEXITCODE)" }

$omp = Get-ChildItem 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools' -Recurse -Filter 'libomp140.aarch64.dll' |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $omp) { throw 'ARM64 LLVM OpenMP runtime was not found.' }
Copy-Item -LiteralPath $omp -Destination (Join-Path $BuildDir 'bin\libomp140.aarch64.dll') -Force

& (Join-Path $BuildDir 'bin\llama-bench.exe') --list-devices
