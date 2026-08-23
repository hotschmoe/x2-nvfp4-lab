param(
    [string]$QairtRoot = 'C:\Qualcomm\AIStack\QAIRT\2.45.40.260406',
    [ValidateRange(1, 300)]
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = 'Stop'
$validator = Join-Path $QairtRoot 'bin\aarch64-windows-msvc\qnn-platform-validator.exe'
$hostLibraries = Join-Path $QairtRoot 'lib\aarch64-windows-msvc'
$hostBinaries = Join-Path $QairtRoot 'bin\aarch64-windows-msvc'
$hexagonLibraries = Join-Path $QairtRoot 'lib\hexagon-v81\unsigned'
$isolatedRunner = Resolve-Path (Join-Path $PSScriptRoot '..\..\scripts\run-isolated.ps1')
$resultDirectory = Join-Path $PSScriptRoot 'platform_validator'

if (-not (Test-Path -LiteralPath $validator -PathType Leaf)) {
    throw "QAIRT platform validator not found: $validator"
}
if (-not (Test-Path -LiteralPath $hexagonLibraries -PathType Container)) {
    throw "Hexagon v81 libraries not found: $hexagonLibraries"
}

$env:ADSP_LIBRARY_PATH = $hexagonLibraries
$env:PATH = "$hostBinaries;$hostLibraries;$env:PATH"
New-Item -ItemType Directory -Force -Path $resultDirectory | Out-Null

& $isolatedRunner -Executable $validator -TimeoutSeconds $TimeoutSeconds `
    -CommandLine "--backend dsp --coreVersion --libVersion --testBackend --targetPath `"$resultDirectory`""
exit $LASTEXITCODE
