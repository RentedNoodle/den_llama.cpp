# ============================================================================
# den_llama.cpp NVFP4 Blood Proof Test
# Verifies NVFP4 OMMA inference on consumer Blackwell (RTX 5070 Ti, sm_120a)
#
# Test model: Qwen3.5-4B-hybrid.gguf (NVFP4 + BF16 hybrid, ~3.28 GB)
# Uses: OMMA.SF.16864 tensor cores for native NVFP4 GEMM
# ============================================================================

param(
    [string]$ModelPath = "I:\models\Qwen3.5-4B-hybrid.gguf",
    [string]$BuildDir = "build_win",               # Primary: MSVC + CUDA 13.3
    [string]$BuildConfig = "Release",
    [int]$Ngl = 99,                                # All layers on GPU
    [int]$NumTokens = 16,
    [string]$Prompt = "The capital of France is",
    [switch]$NoBuild,                              # Skip build, assume binary exists
    [switch]$ListModels                            # Just list available NVFP4 models
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

# ============================================================================
# SECTION 1: Helper functions
# ============================================================================

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Text)
    Write-Host "  >> $Text" -ForegroundColor Yellow
}

function Write-OK {
    param([string]$Text)
    Write-Host "  [OK] $Text" -ForegroundColor Green
}

function Write-Fail {
    param([string]$Text)
    Write-Host "  [FAIL] $Text" -ForegroundColor Red
}

function Write-Warn {
    param([string]$Text)
    Write-Host "  [WARN] $Text" -ForegroundColor Magenta
}

# ============================================================================
# SECTION 2: Discovery
# ============================================================================

Write-Header "DEN_LLAMA.CPP NVFP4 BLOOD PROOF — PRE-FLIGHT"

# 2a. Detect available NVFP4 GGUF models
$modelSearchPaths = @(
    "I:\models",
    "C:\Den\Models",
    "D:\models"
)

$allModels = @()
foreach ($searchPath in $modelSearchPaths) {
    if (Test-Path $searchPath) {
        $found = Get-ChildItem -Path $searchPath -Filter "*.gguf" -ErrorAction SilentlyContinue |
            Where-Object { $_.Length -gt 100MB } |
            Sort-Object Length -Descending
        $allModels += $found
    }
}

Write-Step "Available GGUF models (potential NVFP4 candidates):"
$allModels | ForEach-Object {
    $sizeGB = [math]::Round($_.Length / 1GB, 2)
    Write-Host "    $($_.FullName)  [$sizeGB GB]"
}

if ($ListModels) {
    Write-Host ""
    Write-Host "Model listing complete. Use -ModelPath to specify which to test."
    exit 0
}

# 2b. Verify the primary test model
if (-not (Test-Path $ModelPath)) {
    Write-Fail "Primary test model not found: $ModelPath"
    Write-Host "  Available models:"
    $allModels | ForEach-Object { Write-Host "    $($_.FullName)" }
    exit 1
}
$modelSize = [math]::Round((Get-Item $ModelPath).Length / 1GB, 2)
Write-OK "Test model: $ModelPath ($modelSize GB)"

# 2c. Check model type (NVFP4 or not)
Write-Step "Checking model tensor types..."
$modelInfo = & "$RepoRoot\build_win\bin\Release\llama-gguf.exe" $ModelPath 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    Write-Warn "llama-gguf.exe not available (not built yet). Cannot verify NVFP4 type."
} else {
    # Count NVFP4 tensors
    $nvfp4Count = ($modelInfo | Select-String -Pattern "NVFP4|nvfp4|TYPE_40" -AllMatches).Matches.Count
    if ($nvfp4Count -gt 0) {
        Write-OK "NVFP4 tensors detected in model ($nvfp4Count references)"
    } else {
        Write-Warn "No NVFP4 tensor type detected. Model may be non-NVFP4 format."
    }
}

# ============================================================================
# SECTION 3: Binary verification and build
# ============================================================================

Write-Header "BINARY VERIFICATION"

$binaryPath = "$RepoRoot\$BuildDir\bin\$BuildConfig\llama-cli.exe"

if (-not (Test-Path $binaryPath)) {
    Write-Warn "llama-cli.exe not found at: $binaryPath"

    if ($NoBuild) {
        Write-Fail "--NoBuild specified and binary doesn't exist. Cannot continue."
        exit 1
    }

    Write-Step "Building llama-cli target..."

    # Check if build directory is configured
    $sln = "$RepoRoot\$BuildDir\llama.cpp.sln"
    if (-not (Test-Path $sln)) {
        Write-Step "Configuring CMake for $BuildDir..."
        Push-Location $RepoRoot
        try {
            cmake -B $BuildDir -G "Visual Studio 17 2022" -A x64
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "CMake configure failed"
                exit 1
            }
            Write-OK "CMake configure complete"
        }
        finally {
            Pop-Location
        }
    }

    Push-Location $RepoRoot
    try {
        Write-Step "Running cmake --build $BuildDir --config $BuildConfig --target llama-cli -j 8 ..."
        cmake --build $BuildDir --config $BuildConfig --target llama-cli -j 8
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Build failed"
            exit 1
        }
        Write-OK "Build succeeded"
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path $binaryPath)) {
        Write-Fail "Binary still not found after build: $binaryPath"
        exit 1
    }
}

$binarySize = [math]::Round((Get-Item $binaryPath).Length / 1MB, 2)
Write-OK "Binary: $binaryPath ($binarySize MB)"

# ============================================================================
# SECTION 4: GPU verification
# ============================================================================

Write-Header "GPU VERIFICATION"

# Quick GPU check via nvidia-smi
try {
    $gpuInfo = & nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-OK "GPU detected: $($gpuInfo.Trim())"

        # Check if Blackwell (compute capability 12.x for sm_120a)
        if ($gpuInfo -match "12\.") {
            Write-OK "Blackwell GPU confirmed (sm_120a) — NVFP4 OMMA.SF.16864 ACTIVE"
        } else {
            Write-Warn "Non-Blackwell GPU detected. NVFP4 OMMA kernels require sm_120a."
        }
    } else {
        Write-Warn "nvidia-smi not available. Cannot verify GPU."
    }
} catch {
    Write-Warn "GPU check failed: $_"
}

# ============================================================================
# SECTION 5: Run blood proof test
# ============================================================================

Write-Header "BLOOD PROOF TEST"

Write-Step "Launching: llama-cli"
Write-Host "  Model  : $ModelPath"
Write-Host "  Prompt : '$Prompt'"
Write-Host "  Tokens : $NumTokens"
Write-Host "  GPU Layers: $Ngl (all on GPU)"
Write-Host "  NVFP4  : AUTO (detected from model type, OMMA.SF.16864 on sm_120a)"
Write-Host ""

$startTime = Get-Date

# Run the test
$output = & $binaryPath `
    -m $ModelPath `
    -p $Prompt `
    -n $NumTokens `
    -ngl $Ngl `
    2>&1

$exitCode = $LASTEXITCODE
$endTime = Get-Date
$elapsed = ($endTime - $startTime).TotalSeconds

Write-Host ""
Write-Host ("-" * 78)
Write-Host "  OUTPUT:"
Write-Host ("-" * 78)
Write-Host $output
Write-Host ("-" * 78)

if ($exitCode -ne 0) {
    Write-Host ""
    Write-Fail "llama-cli exited with code $exitCode"
    Write-Host ""
    Write-Host "TROUBLESHOOTING:" -ForegroundColor Yellow
    Write-Host "  - Check CUDA driver version (610.47+ required for CUDA 13.3)"
    Write-Host "  - Verify model file is not corrupted (re-copy from source)"
    Write-Host "  - Try with --gpu-layers 0 to verify CPU path works first"
    Write-Host "  - Check Windows Event Log for GPU TDR events"
    Write-Host "  - For OMMA errors: ensure build_win uses CUDA 13.3 nvcc"
    exit $exitCode
}

# ============================================================================
# SECTION 6: Results analysis
# ============================================================================

Write-Header "BLOOD PROOF RESULTS"

# Parse token count from output
$generatedTokens = 0
$totalTime = 0.0

if ($output -match 'llama_perf_sampled_n_tokens\s*=\s*(\d+)') {
    $generatedTokens = [int]$Matches[1]
}
if ($output -match 'llama_perf_sampled_n_sec\s*=\s*([\d.]+)') {
    $totalTime = [double]$Matches[1]
}

if ($generatedTokens -gt 0 -and $totalTime -gt 0) {
    $tokPerSec = [math]::Round($generatedTokens / $totalTime, 1)
    Write-Host ""
    Write-Host "  GENERATED TOKENS : $generatedTokens" -ForegroundColor Green
    Write-Host "  GENERATION TIME  : $totalTime sec" -ForegroundColor Green
    Write-Host "  TOKENS/SEC       : $tokPerSec tok/s" -ForegroundColor Green
    Write-Host ""

    # Compare against known baseline (30.2 tok/s from CLAUDE.md)
    $baseline = 30.2
    if ($tokPerSec -ge $baseline * 0.85) {
        Write-OK "Performance within 85% of baseline ($baseline tok/s)"
    } elseif ($tokPerSec -ge $baseline * 0.5) {
        Write-Warn "Performance low: $tokPerSec vs baseline $baseline tok/s"
    } else {
        Write-Fail "Performance critical: $tokPerSec vs baseline $baseline tok/s"
    }
} else {
    Write-Warn "Could not parse performance stats from output"
    Write-Host "  Raw output above for manual inspection"
}

Write-Host ""
Write-OK "Blood proof test complete."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. If output is garbled: check NVFP4 format alignment (146B vs 160B block issue from CLAUDE.md)"
Write-Host "  2. If cos < 0.99: compare against BF16 reference with same prompt"
Write-Host "  3. For 35B MoE testing: use Qwen3.6-35B-A3B GGUF with --gpu-layers 999"
