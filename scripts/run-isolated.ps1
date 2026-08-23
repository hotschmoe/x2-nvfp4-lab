param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,

    [ValidateRange(1, 3600)]
    [int]$TimeoutSeconds = 120,

    # When set, successful process exit is not sufficient: stdout must contain
    # this exact marker. This catches the Qualcomm ICD's observed silent exits.
    [string]$CompletionMarker = '',

    # Keep the child command line as one string.  This avoids PowerShell
    # interpreting llama switches such as `-t` as abbreviations of this
    # script's own parameters.
    [string]$CommandLine = ''
)

# Runs a probe outside the invoking shell and keeps stdout/stderr in durable
# files. Use only CPU work or small, explicitly validated Adreno shapes; do not
# use this wrapper as justification for a full Vulkan offload. It cannot protect
# Windows from a system-wide driver reset, but it contains normal child faults.
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Executable not found: $Executable"
}

$logDir = Join-Path $PSScriptRoot '..\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$stdout = Join-Path $logDir "$stamp.stdout.log"
$stderr = Join-Path $logDir "$stamp.stderr.log"

$child = Start-Process -FilePath $Executable -ArgumentList $CommandLine -NoNewWindow -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
# Force PowerShell to retain the process handle; otherwise ExitCode may remain
# unset after an ARM64 child faults while output is redirected.
$nativeHandle = $child.Handle
$finished = $child.WaitForExit($TimeoutSeconds * 1000)
if (-not $finished) {
    Stop-Process -Id $child.Id -Force
    throw "Isolated child timed out after $TimeoutSeconds seconds (PID $($child.Id)); logs: $stdout ; $stderr"
}
# Complete asynchronous stdout/stderr draining and refresh the native process
# handle before reading ExitCode.  Without the parameterless WaitForExit call,
# PowerShell can report a blank exit code for very short-lived ARM64 children.
$child.WaitForExit()
$child.Refresh()
$null = $nativeHandle

$markerFound = $true
if ($CompletionMarker) {
    $markerFound = Select-String -LiteralPath $stdout -SimpleMatch $CompletionMarker -Quiet
}

[pscustomobject]@{
    PID = $child.Id
    ExitCode = $child.ExitCode
    StdOut = $stdout
    StdErr = $stderr
    CompletionMarker = if ($CompletionMarker) { $markerFound } else { $null }
}
if ($child.ExitCode -eq 0 -and -not $markerFound) {
    Write-Error "Isolated child exited without completion marker '$CompletionMarker'; logs: $stdout ; $stderr"
    exit 125
}
exit $child.ExitCode
