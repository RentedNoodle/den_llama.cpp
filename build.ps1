# build.ps1 — One-click Windows native build for den_llama.cpp
# Uses pip CUDA 13.3 nvcc.exe (NOT system CUDA, NOT WSL).
# Hardened 2026-08-02: multi-root pip search, FORCE compiler pins,
# unconditional build_win cleanup (corrupt cache killer), PATH prepend for cl.exe.
param(
    [string]$Config = "Release",
    [int]$Jobs = 8,
    [switch]$Clean,
    [switch]$CPU  # emergency CPU-only fallback — not recommended
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  den_llama.cpp BUILD" -ForegroundColor Cyan
Write-Host "  Config: $Config | Jobs: $Jobs" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# --- Multi-root pip CUDA search (fail loudly if none found) ---
$pipRoots = @(
    "D:\Den\den-pytorch\Lib\site-packages\nvidia\cu13",
    "D:\Den\dencli\.venv\Lib\site-packages\nvidia\cu13",
    "C:\Users\james\AppData\Local\Programs\Python\Python314\Lib\site-packages\nvidia\cu13",
    "C:\Den\den-pytorch\Lib\site-packages\nvidia\cu13"
)
$pipCuda = $null
foreach ($root in $pipRoots) {
    $cand = Join-Path $root "bin\nvcc.exe"
    if (Test-Path $cand) { $pipCuda = $cand; break }
}
if (-not $pipCuda) {
    Write-Host "ERROR: pip CUDA 13.3 not found in any of:" -ForegroundColor Red
    $pipRoots | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host "Install: pip install nvidia-cuda-nvcc nvidia-cu13 nvidia-cuda-cccl" -ForegroundColor Yellow
    exit 1
}
$pipRoot = Split-Path (Split-Path $pipCuda -Parent) -Parent
Write-Host "NVCC: $pipCuda" -ForegroundColor Green

# CRITICAL: MSBuild's CUDA integration (CUDA 13.3.targets) resolves the toolkit via
# $(CUDA_PATH) — if it points at the system CUDA, pip nvcc is bypassed and the
# system nvcc (Linux-targeted nvcc.profile) breaks on MSVC. Force pip root.
$env:CUDA_PATH = $pipRoot
$env:CUDA_PATH_V13_3 = $pipRoot
Write-Host "CUDA_PATH set to pip root: $pipRoot" -ForegroundColor Green

# --- Ensure nvcc can find cl.exe: prepend MSVC Hostx64\x64 to PATH ---
$msvcBin = $null
if ($env:VCToolsInstallDir -and (Test-Path "$env:VCToolsInstallDir\bin\Hostx64\x64")) {
    $msvcBin = "$env:VCToolsInstallDir\bin\Hostx64\x64"
} else {
    $probe = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($probe) {
        $cand = Join-Path $probe.FullName "bin\Hostx64\x64"
        if (Test-Path $cand) { $msvcBin = $cand }
    }
}
if ($msvcBin) {
    $env:PATH = "$msvcBin;$env:PATH"
    Write-Host "MSVC cl on PATH: $msvcBin" -ForegroundColor Green
} else {
    Write-Host "WARNING: could not locate MSVC Hostx64\x64 cl.exe on PATH" -ForegroundColor Yellow
}

# --- Unconditional cleanup of build_win (stale CMAKE_CUDA_COMPILER cache kills configure) ---
if (Test-Path build_win) {
    Write-Host "CLEANING build_win (always, to avoid corrupt cache)..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force build_win
}
# Also sweep other stale build dirs from the corrupt-cache era
foreach ($stale in @("build_win_cu133", "build_final", "build_check_msvc")) {
    if (Test-Path $stale) {
        Write-Host "Removing stale build dir $stale..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $stale
    }
}

# --- Configure with FORCE pins so find_package(CUDAToolkit) resolves pip ---
$cmakeArgs = @(
    "-B", "build_win",
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DGGML_CUDA=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=120a",
    "-DCMAKE_TOOLCHAIN_FILE=cmake/msvc_toolchain.cmake",
    "-DCMAKE_CUDA_COMPILER=$pipCuda",
    "-DCUDAToolkit_ROOT=$pipRoot",
    "-DCUDAToolkit_NVCC_EXECUTABLE=$pipCuda",
    "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=OFF"
)

if ($CPU) {
    Write-Host "WARNING: CPU-only build requested" -ForegroundColor Yellow
    $cmakeArgs = $cmakeArgs | Where-Object { $_ -notmatch "^\-DGGML_CUDA=ON" }
}

Write-Host "CONFIGURE: cmake $($cmakeArgs -join ' ')" -ForegroundColor Gray
$result = & cmake @cmakeArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "CONFIGURE FAILED:" -ForegroundColor Red
    Write-Host ($result -join "`n")
    exit 1
}
Write-Host "CONFIGURE: OK" -ForegroundColor Green

# --- Configure-time gates (abort if CUDA silently disabled / wrong compiler) ---
$cacheCompiler = (Select-String -Path "build_win\CMakeCache.txt" -Pattern "^CMAKE_CUDA_COMPILER:FILEPATH=" | Select-Object -First 1).Line
$cacheVersion  = (Select-String -Path "build_win\CMakeCache.txt" -Pattern "^CUDAToolkit_VERSION" | Select-Object -First 1).Line
Write-Host "GATE CMAKE_CUDA_COMPILER: $cacheCompiler" -ForegroundColor Gray
Write-Host "GATE CUDAToolkit_VERSION:  $cacheVersion" -ForegroundColor Gray
if ($cacheCompiler -and $cacheCompiler -notmatch [regex]::Escape($pipRoot.Replace("\","/"))) {
    Write-Host "GATE FAIL: CMAKE_CUDA_COMPILER is not pip nvcc. Aborting." -ForegroundColor Red
    exit 1
}
if ($cacheVersion -and $cacheVersion -notmatch "13\.3") {
    Write-Host "GATE FAIL: CUDAToolkit_VERSION is not 13.3. Aborting." -ForegroundColor Red
    exit 1
}
Write-Host "GATES: OK" -ForegroundColor Green

# --- Build ---
Write-Host "BUILD: cmake --build build_win --config $Config -j $Jobs" -ForegroundColor Gray
$result = & cmake --build build_win --config $Config -j $Jobs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "BUILD FAILED:" -ForegroundColor Red
    Write-Host ($result -join "`n")
    exit 1
}
Write-Host "BUILD: OK" -ForegroundColor Green

# --- Report ---
$bin = "build_win\bin\$Config\llama-cli.exe"
if (Test-Path $bin) {
    $size = (Get-Item $bin).Length / 1MB
    Write-Host "BINARY: $bin ({0:F1} MB)" -f $size -ForegroundColor Green
} else {
    Write-Host "BINARY: not found (check build_win\bin\)" -ForegroundColor Yellow
    Get-ChildItem build_win\bin\$Config\*.exe 2>$null | ForEach-Object {
        Write-Host "  $($_.Name) ({0:F1} MB)" -f ($_.Length / 1MB)
    }
}
