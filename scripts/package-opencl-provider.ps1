param(
    [string]$Python = (Get-Command python).Source
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$runtimeBuild = Join-Path $workspace 'native_nvfp4\runtime\build'
$runtimeDll = Join-Path $runtimeBuild 'nvfp4_runtime.dll'
$kernelSource = Join-Path $workspace 'native_nvfp4\kernels\nvfp4_gemv.cl'
$project = Join-Path $workspace 'vllm_nvfp4_opencl'
$package = Join-Path $project 'src\vllm_nvfp4_opencl'

cmake --build $runtimeBuild --config Release
if ($LASTEXITCODE -ne 0) {
    throw "OpenCL runtime build failed with exit code $LASTEXITCODE"
}

$libraryDirectory = Join-Path $package 'lib'
$kernelDirectory = Join-Path $package 'kernels'
New-Item -ItemType Directory -Force -Path $libraryDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $kernelDirectory | Out-Null
Copy-Item -LiteralPath $runtimeDll `
    -Destination (Join-Path $libraryDirectory 'nvfp4_runtime.dll') -Force
Copy-Item -LiteralPath $kernelSource `
    -Destination (Join-Path $kernelDirectory 'nvfp4_gemv.cl') -Force

Push-Location $project
try {
    & $Python -m pip wheel . --no-deps --wheel-dir dist
    if ($LASTEXITCODE -ne 0) {
        throw "wheel build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
