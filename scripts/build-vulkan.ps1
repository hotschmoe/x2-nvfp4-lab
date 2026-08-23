param(
    [string]$SourceDir = (Join-Path $PSScriptRoot '..\llama.cpp'),
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\llama.cpp\build-vulkan')
)

$ErrorActionPreference = 'Stop'
$vsDevCmd = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
$llvmBin = 'C:\Program Files\LLVM\bin'
$vulkanSdk = 'C:\VulkanSDK\1.4.341.1'

foreach ($path in @($SourceDir, $vsDevCmd, $llvmBin, $vulkanSdk)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path is missing: $path" }
}

# This retains GGML_TYPE_NVFP4 end-to-end.  The current Vulkan backend supplies
# dequant, GEMV and GEMM pipelines for NVFP4; the Adreno driver JITs them for X2-90.
$cmake = @"
call `"$vsDevCmd`" -arch=arm64 -host_arch=arm64 &&
set PATH=$vulkanSdk\Bin;$llvmBin;%PATH% &&
cmake -S `"$SourceDir`" -B `"$BuildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=`"$SourceDir\cmake\arm64-windows-llvm.cmake`" -DGGML_VULKAN=ON -DVulkan_INCLUDE_DIR=`"$vulkanSdk\Include`" -DVulkan_LIBRARY=`"$vulkanSdk\Lib\vulkan-1.lib`" -DGGML_VULKAN_SHADERS_GEN_TOOLCHAIN=`"$SourceDir\cmake\arm64-windows-llvm.cmake`" -DBUILD_SHARED_LIBS=OFF &&
cmake --build `"$BuildDir`" --target llama-cli llama-bench llama-server --parallel 18
"@ -replace "`r?`n", ' '

cmd.exe /c $cmake
if ($LASTEXITCODE) { throw "llama.cpp Vulkan build failed ($LASTEXITCODE)" }

$omp = Get-ChildItem 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools' -Recurse -Filter 'libomp140.aarch64.dll' |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $omp) { throw 'ARM64 LLVM OpenMP runtime was not found.' }
Copy-Item -LiteralPath $omp -Destination (Join-Path $BuildDir 'bin\libomp140.aarch64.dll') -Force

& (Join-Path $BuildDir 'bin\llama-bench.exe') --list-devices
