#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run all 7 micro-benchmarks against the Qwopus 9B model.
    Outputs JSON results for CI tracking.

.DESCRIPTION
    Each benchmark measures one subsystem:
      1. gen           - Token generation speed (PP + TG tok/s)
      2. cache         - Expert cache hit rate and DMA stats
      3. router        - CPU router prediction accuracy
      4. kv-evict      - KV cache eviction rate and VRAM savings
      5. mtp           - MTP speculative decoding acceptance rate
      6. quant         - Format quantization speed
      7. nvfp4-convert - NVFP4 conversion + side-by-side comparison

.PARAMETER Model
    Path to the GGUF model file.
    Default: C:\Den\Models\Qwopus3.5-9B-Coder-MTP-Q4_K_M.gguf

.PARAMETER BuildDir
    Build directory containing the llama-micro-bench binary.
    Default: build_wsl

.PARAMETER Bench
    Specific benchmark to run (default: all).
    Accepted values: all, gen, cache, router, kv-evict, mtp, quant, nvfp4-convert

.PARAMETER NumTokens
    Number of tokens to generate per benchmark (default: 32).

.PARAMETER OutputPath
    Output JSON file path. Default: bench_results.json

.PARAMETER ExpertCacheSlots
    Expert VRAM cache slots (0 = auto, default: 0)

.PARAMETER NcpuMoe
    Number of layers CPU MoE offload (default: 61)

.EXAMPLE
    .\scripts\run-bench.ps1
    .\scripts\run-bench.ps1 -Bench cache
    .\scripts\run-bench.ps1 -Model "D:\models\other-model.gguf" -OutputPath results.json
#>

param(
    [string]$Model = "C:\Den\Models\Qwopus3.5-9B-Coder-MTP-Q4_K_M.gguf",
    [string]$BuildDir = "build_wsl",
    [string]$Bench = "all",
    [int]$NumTokens = 32,
    [string]$OutputPath = "bench_results.json",
    [int]$ExpertCacheSlots = 0,
    [int]$NcpuMoe = 61
)

$ErrorActionPreference = "Stop"

# Resolve paths
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Den Micro-Benchmarks"                      -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Detect build directory
$possibleDirs = @(
    "$ProjectRoot\$BuildDir",
    "$ProjectRoot\build",
    "$ProjectRoot\build_win",
    "$ProjectRoot\build_wsl"
)

$actualBuildDir = $null
foreach ($dir in $possibleDirs) {
    $binPath = Join-Path $dir "bin\llama-micro-bench.exe"
    if (Test-Path $binPath) {
        $actualBuildDir = $dir
        break
    }
    $binPath = Join-Path $dir "bin\llama-micro-bench"
    if (Test-Path $binPath) {
        $actualBuildDir = $dir
        break
    }
}

if (-not $actualBuildDir) {
    Write-Host "ERROR: llama-micro-bench binary not found in any build directory." -ForegroundColor Red
    Write-Host "  Checked:" -ForegroundColor Red
    foreach ($dir in $possibleDirs) {
        Write-Host "    $dir" -ForegroundColor Red
    }
    Write-Host "  Build with: cmake --build $BuildDir --target llama-micro-bench -j8" -ForegroundColor Yellow
    exit 1
}

$Binary = Join-Path $actualBuildDir "bin\llama-micro-bench.exe"
if (-not (Test-Path $Binary)) {
    $Binary = Join-Path $actualBuildDir "bin\llama-micro-bench"
}

if (-not (Test-Path $Binary)) {
    Write-Host "ERROR: llama-micro-bench not found at $Binary" -ForegroundColor Red
    exit 1
}

# Check model exists
if (-not (Test-Path $Model)) {
    Write-Host "WARNING: Default model not found at: $Model" -ForegroundColor Yellow
    Write-Host "  Benchmarks will use model path from -m argument." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Model:     $Model"      -ForegroundColor Gray
Write-Host "Binary:    $Binary"     -ForegroundColor Gray
Write-Host "Build dir: $actualBuildDir" -ForegroundColor Gray
Write-Host "Bench:     $Bench"      -ForegroundColor Gray
Write-Host "Tokens:    $NumTokens"  -ForegroundColor Gray
Write-Host "Output:    $OutputPath" -ForegroundColor Gray
Write-Host ""

# Build the argument list
$args = @(
    "--bench", $Bench,
    "-n", $NumTokens,
    "--output", $OutputPath
)

# Only pass -m if the model file actually exists (otherwise rely on DEFAULT_MODEL in binary)
if (Test-Path $Model) {
    $args = @("-m", $Model) + $args
}

if ($ExpertCacheSlots -gt 0) {
    $args += @("--expert-cache", $ExpertCacheSlots)
}

if ($NcpuMoe -gt 0) {
    $args += @("--n-cpu-moe", $NcpuMoe)
}

Write-Host "Running: $Binary $($args -join ' ')" -ForegroundColor DarkGray
Write-Host ""

# Run the benchmark
$startTime = Get-Date
try {
    & $Binary @args
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host "ERROR: Benchmark execution failed: $_" -ForegroundColor Red
    exit 1
}

$elapsed = (Get-Date) - $startTime

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Benchmark Complete"                         -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

if ($exitCode -ne 0) {
    Write-Host "ERROR: Benchmark exited with code $exitCode" -ForegroundColor Red
    exit $exitCode
}

# Check output file
if (Test-Path $OutputPath) {
    $fileSize = (Get-Item $OutputPath).Length
    Write-Host "Results saved to: $OutputPath ($fileSize bytes)" -ForegroundColor Green

    # Quick JSON summary
    try {
        $results = Get-Content $OutputPath -Raw | ConvertFrom-Json
        Write-Host ""
        Write-Host "Summary:" -ForegroundColor Yellow
        Write-Host ("{0,-18} {1,10} {2,-8} {3}" -f "Benchmark", "Value", "Unit", "Detail")
        Write-Host ("{0,-18} {1,10} {2,-8} {3}" -f "---------", "-----", "----", "------")
        foreach ($r in $results) {
            $val = if ($r.value -is [double]) { "{0,10:0.00}" -f $r.value } else { "{0,10}" -f $r.value }
            Write-Host ("{0,-18} {1} {2,-8} {3}" -f $r.benchmark, $val, $r.unit, $r.detail)
        }
    } catch {
        Write-Host "  (Could not parse JSON for summary — raw file OK)" -ForegroundColor DarkYellow
    }
} else {
    Write-Host "WARNING: Output file not created at $OutputPath" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Elapsed: $($elapsed.TotalSeconds.ToString('0.0'))s" -ForegroundColor Gray
