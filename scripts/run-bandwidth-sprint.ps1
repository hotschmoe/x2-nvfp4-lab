param(
    [string]$Python = (Get-Command python).Source,
    [ValidateRange(1, 100)]
    [int]$Samples = 30,
    [ValidateRange(1, 100)]
    [int]$StabilityRuns = 30,
    [ValidateRange(1, 1024)]
    [int]$MiB = 64
)

$ErrorActionPreference = 'Stop'
$isolated = Join-Path $PSScriptRoot 'run-isolated.ps1'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')

function Invoke-BandwidthProbe {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Arguments,
        [string]$Marker = 'CAMPAIGN_BANDWIDTH_FIRST_PASS',
        [int]$TimeoutSeconds = 120
    )
    & $isolated -Executable $Python -TimeoutSeconds $TimeoutSeconds `
        -CompletionMarker $Marker -CommandLine $Arguments
    if ($LASTEXITCODE) {
        throw "Bandwidth probe failed: $Arguments"
    }
}

Push-Location $root
try {
    foreach ($vectorBytes in 1, 4, 16, 64) {
        Invoke-BandwidthProbe `
            "native_nvfp4/bench_islands.py --case gpu --mib $MiB --vector-bytes $vectorBytes --warmups 5 --samples $Samples"
    }

    Invoke-BandwidthProbe `
        "native_nvfp4/bench_islands.py --case gpu --shared-svm --mib $MiB --vector-bytes 16 --warmups 5 --samples $Samples"
    Invoke-BandwidthProbe `
        "native_nvfp4/bench_islands.py --case cpu --mib $MiB --threads 6 --cpu-set 0,1,2,3,4,5 --warmups 5 --samples $Samples"
    Invoke-BandwidthProbe `
        "native_nvfp4/bench_islands.py --case cpu --mib $MiB --threads 6 --cpu-set 6,7,8,9,10,11 --warmups 5 --samples $Samples"
    Invoke-BandwidthProbe `
        "native_nvfp4/bench_islands.py --case cpu --mib $MiB --threads 6 --cpu-set 12,13,14,15,16,17 --warmups 5 --samples $Samples"
    Invoke-BandwidthProbe `
        "native_nvfp4/bench_islands.py --case concurrent-different --mib $MiB --vector-bytes 16 --threads 6 --cpu-set 12,13,14,15,16,17 --warmups 5 --samples $Samples"
    Invoke-BandwidthProbe `
        "native_nvfp4/bench_islands.py --case concurrent-shared --mib $MiB --vector-bytes 16 --threads 6 --cpu-set 12,13,14,15,16,17 --warmups 5 --samples $Samples"

    for ($iteration = 1; $iteration -le $StabilityRuns; $iteration++) {
        Invoke-BandwidthProbe `
            "native_nvfp4/bench_islands.py --case gpu --shared-svm --mib $MiB --vector-bytes 16 --warmups 1 --samples 1"
    }

    Invoke-BandwidthProbe `
        'native_nvfp4/bench_svm_matrix.py --rows 17408 --cols 5120 --warmups 5 --samples 30 --cpu-threads 6' `
        -Marker 'SVM_NVFP4_INTERLEAVED_PASS' -TimeoutSeconds 180

    & $Python native_nvfp4/bench_islands.py --summarize
} finally {
    Pop-Location
}
