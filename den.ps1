#Requires -Version 5.1
<#
.SYNOPSIS
  Den engine — one command. Setup, run, chat.

.DESCRIPTION
  den           -> start the model (serve). The obvious thing.
  den setup     -> build the engine if needed + install defaults.
  den chat      -> talk to the model in this terminal.
  den serve     -> start the local API/UI server (http://127.0.0.1:8080).
  den bench     -> quick speed test.
  den doctor    -> check everything is healthy.
  den models    -> list models it can find.
  den stop      -> stop the running server.
  den help      -> this text.

  Pick a model:  den [35|heretic|golden|9] ...
    den 9 chat        -> talk to the 9B model
    den golden chat   -> talk to the golden baseline 35B
    den heretic       -> serve the abliterated 35B (default)

  Overrides (env):  DEN_MODEL  DEN_MODELS  DEN_PORT  DEN_CTX  DEN_GPU_LAYERS
  Tune knobs:       DEN_NVFP4_KV_CACHE=0  DEN_GDN_FAST_EXP=1  (auto-set where needed)
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "serve",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ---- paths ----
$Root       = $PSScriptRoot
$Bin        = Join-Path $Root "build_ninja\bin"
$Server     = Join-Path $Bin "llama-server.exe"
$Cli        = Join-Path $Bin "llama-cli.exe"
$Bench      = Join-Path $Bin "llama-bench.exe"
$BuildScript= Join-Path $Root "build_den.ps1"
$ModelsDir  = if ($env:DEN_MODELS) { $env:DEN_MODELS } else { "I:\models" }
$ModelAlias = $null   # set when user runs "den golden chat" style

# ---- pretty output ----
function Step($m) { Write-Host "==> " -NoNewline -ForegroundColor Cyan;  Write-Host $m }
function Ok($m)   { Write-Host "  ok  " -NoNewline -ForegroundColor Green;  Write-Host $m }
function Warn($m) { Write-Host "  !   " -NoNewline -ForegroundColor Yellow; Write-Host $m }
function Fail($m) { Write-Host "  X   " -NoNewline -ForegroundColor Red;    Write-Host $m; exit 1 }

function Banner {
    Write-Host ""
    Write-Host "  Den engine" -ForegroundColor Magenta
    Write-Host ""
}

# ---- model map ----
$MODEL_FILES = @{
    "35"      = "Ornith-1.0-35B-Heretic-MTP-APEX-I-Mini.gguf"
    "heretic" = "Ornith-1.0-35B-Heretic-MTP-APEX-I-Mini.gguf"
    "golden"  = "ornith-1.0-35b-APEX-I-Mini-MTP-TemplateFix.gguf"
    "9"       = "Ornith-1.0-9B-heretic-MTP-Q4_K_M.gguf"
}
$MMPROJ_FILES = @{
    "35"      = "mmproj-Ornith-1.0-35B-Heretic-MTP-BF16.gguf"
    "heretic" = "mmproj-Ornith-1.0-35B-Heretic-MTP-BF16.gguf"
    "golden"  = "mmproj-Ornith-1.0-35B-Heretic-MTP-BF16.gguf"
    "9"       = "mmproj-Ornith-1.0-9B-heretic-MTP-BF16.gguf"
}

# ---- resolve model name + path ----
function Get-ModelPath {
    # env override wins
    if ($env:DEN_MODEL) {
        if (Test-Path $env:DEN_MODEL) { return $env:DEN_MODEL }
        Warn "DEN_MODEL set but not found: $env:DEN_MODEL"
    }
    # name from args: den 35 / den 9 / den golden / den heretic
    $name = "35"
    if ($ModelAlias) { $name = $ModelAlias }
    elseif ($Command -in $MODEL_FILES.Keys) { $name = $Command }
    $file = $MODEL_FILES[$name]
    $p = Join-Path $ModelsDir $file
    if (Test-Path $p) { return $p }
    # last resort: any *35B*.gguf
    $any = Get-ChildItem -Path $ModelsDir -Filter "*35B*.gguf" -File -ErrorAction SilentlyContinue |
           Select-Object -First 1 -ExpandProperty FullName
    if ($any) { Warn "preferred model '$file' not found; using $(Split-Path $any -Leaf)"; return $any }
    Fail "No model found in '$ModelsDir'. Put a .gguf there (or set DEN_MODEL)."
}

function Get-MMProjPath {
    $pick = if ($ModelAlias) { $ModelAlias } elseif ($Command -in $MODEL_FILES.Keys) { $Command } else { $null }
    if ($pick) {
        $file = $MMPROJ_FILES[$pick]
        $p = Join-Path $ModelsDir $file
        if (Test-Path $p) { return $p }
    }
    # generic: try 35B projector
    $p35 = Join-Path $ModelsDir $MMPROJ_FILES["35"]
    if (Test-Path $p35) { return $p35 }
    return $null
}

# ---- sanity checks ----
function Assert-Build {
    if (Test-Path $Server) { return }
    Warn "engine not built yet (no $Server)."
    Step "first run: building. This can take a while — grab a coffee."
    Invoke-Setup
    if (-not (Test-Path $Server)) { Fail "build finished but $Server still missing." }
}

function Assert-Model {
    $null = Get-ModelPath
}

# ---- setup ----
function Invoke-Setup {
    Banner
    Step "Setup: building the Den engine"
    if (-not (Test-Path $BuildScript)) { Fail "missing $BuildScript" }
    & pwsh -NoProfile -ExecutionPolicy Bypass -File $BuildScript -phase all
    if ($LASTEXITCODE -ne 0) { Fail "build failed (exit $LASTEXITCODE)." }
    Ok "build complete"

    Step "installing sane defaults"
    $cfgDir = Join-Path $env:APPDATA "llama.cpp"
    $cfg    = Join-Path $cfgDir "config.ini"
    if (-not (Test-Path $cfgDir)) { New-Item -ItemType Directory -Path $cfgDir -Force | Out-Null }
    $defaults = @"
; Den engine defaults — applied to every model run (llama.cpp preset system).
; Tune per-run with den.ps1 overrides; this file is safe to delete.
[*]
mmap = 1
flash-attn = on
"@
    Set-Content -Path $cfg -Value $defaults -Encoding UTF8
    Ok "wrote $cfg"

    Ok "setup done."
    Ok "next: run 'den' to start, or 'den chat' to talk."
}

# ---- serve ----
function Invoke-Serve {
    param([switch]$Safe)

    Assert-Build
    $model  = Get-ModelPath
    $mmproj = Get-MMProjPath
    $gpu    = if ($env:DEN_GPU_LAYERS) { $env:DEN_GPU_LAYERS } else { "99" }
    $ctx    = if ($env:DEN_CTX) { $env:DEN_CTX } else { "32768" }
    $port   = if ($env:DEN_PORT) { $env:DEN_PORT } else { "8080" }

    Banner
    Step "starting Den"
    Ok "model : $(Split-Path $model -Leaf)"
    if ($mmproj) { Ok "vision: $(Split-Path $mmproj -Leaf)" }
    Ok "server: http://127.0.0.1:$port"

    # fast path: NVFP4 KV off (KVarN does it better), KVarN6 KV cache
    $env:DEN_NVFP4_KV_CACHE = "0"

    $srv = @("-m", $model)
    if ($mmproj) { $srv += @("--mmproj", $mmproj) }
    $srv += @(
        "-ngl", $gpu
        "-c", $ctx
        "-fa", "on"
    )
    if (-not $Safe) { $srv += @("-ctk", "kvarn6", "-ctv", "kvarn6") }
    $srv += @(
        "--host", "127.0.0.1"
        "--port", $port
        "--alias", "den"
        "--jinja"
        "--reasoning-format", "deepseek"
        "--reasoning-budget", "2048"
        "--temp", "0.6"
        "--repeat-last-n", "4096"
    )

    Write-Host ""
    & $Server @srv
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Warn "server exited with code $code."
        Warn "if it complained about 'kvarn', rerun without the fast path: den safe"
    }
}

# ---- chat (interactive) ----
function Invoke-Chat {
    Assert-Build
    $model = Get-ModelPath
    $gpu   = if ($env:DEN_GPU_LAYERS) { $env:DEN_GPU_LAYERS } else { "99" }
    $ctx   = if ($env:DEN_CTX) { $env:DEN_CTX } else { "32768" }
    $env:DEN_NVFP4_KV_CACHE = "0"

    Banner
    Step "chatting with $(Split-Path $model -Leaf)"
    Write-Host ""
    & $Cli "-m" $model "-ngl" $gpu "-c" $ctx "-fa" "on" `
        "-ctk" "kvarn6" "-ctv" "kvarn6" `
        "--jinja" "--reasoning-format" "deepseek" `
        "--temp" "0.6" "-cnv"
    $code = $LASTEXITCODE
    if ($code -ne 0) { Warn "chat exited with code $code." }
}

# ---- bench ----
function Invoke-Bench {
    Assert-Build
    $model = Get-ModelPath
    $env:DEN_NVFP4_KV_CACHE = "0"
    Banner
    Step "speed test: $(Split-Path $model -Leaf)"
    Write-Host ""
    & $Bench "-m" $model "-ngl" "99" "-c" "2048" "-n" "128" "-fa" "on" "-ctk" "kvarn6" "-ctv" "kvarn6"
}

# ---- doctor ----
function Invoke-Doctor {
    Banner
    Step "checking"
    $good = $true

    if (Test-Path $Server) { Ok "engine    : built ($Server)" } else { $good = $false; Warn "engine    : NOT built — run 'den setup'" }
    $model = Get-ModelPath 2>$null
    if ($model) { Ok "model     : $(Split-Path $model -Leaf)" } else { $good = $false; Warn "model     : none found in $ModelsDir" }

    $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvsmi) {
        $vram = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        if ($vram) { Ok "GPU       : $( [math]::Round($vram/1024,1) ) GB VRAM" } else { Warn "GPU       : nvidia-smi found but no VRAM read" }
    } else { $good = $false; Warn "GPU       : nvidia-smi not found — CUDA may be missing" }

    $cfg = Join-Path $env:APPDATA "llama.cpp\config.ini"
    if (Test-Path $cfg) { Ok "defaults  : $cfg" } else { Warn "defaults  : none — run 'den setup'" }

    if ($good) { Ok "all good."; Ok "run 'den' to start." }
    else { Warn "some items need attention (see above)." }
}

# ---- models ----
function Invoke-Models {
    Banner
    Step "models in $ModelsDir"
    Get-ChildItem -Path $ModelsDir -Filter "*.gguf" -File -ErrorAction SilentlyContinue |
        Sort-Object Length -Descending |
        ForEach-Object { "  {0,8:N1} GB   {1}" -f ($_.Length/1GB), $_.Name }
    if (-not (Test-Path $ModelsDir)) { Warn "directory does not exist: $ModelsDir" }
}

# ---- stop ----
function Invoke-Stop {
    Step "stopping llama-server"
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Ok "done."
}

# ---- help ----
function Invoke-Help {
    Get-Help $PSCommandPath -Detailed
}

# ---- dispatch ----
# "den 9 chat" style: model alias then subcommand
if ($Command -in $MODEL_FILES.Keys -and $Rest.Count -gt 0) {
    $ModelAlias = $Command
    $Command = $Rest[0]
    $Rest = $Rest[1..($Rest.Count-1)]
}

switch ($Command.ToLowerInvariant()) {
    "setup"  { Invoke-Setup }
    "serve"  { Invoke-Serve }
    "s"      { Invoke-Serve }
    "chat"   { Invoke-Chat }
    "cli"    { Invoke-Chat }
    "talk"   { Invoke-Chat }
    "bench"  { Invoke-Bench }
    "speed"  { Invoke-Bench }
    "doctor" { Invoke-Doctor }
    "check"  { Invoke-Doctor }
    "models" { Invoke-Models }
    "list"   { Invoke-Models }
    "stop"   { Invoke-Stop }
    "help"   { Invoke-Help }
    "-h"     { Invoke-Help }
    "--help" { Invoke-Help }
    "safe"   { Invoke-Serve -Safe }
    { $_ -in $MODEL_FILES.Keys } { Invoke-Serve }
    default  {
        Warn "unknown command: '$Command'"
        Write-Host ""
        Invoke-Help
        exit 1
    }
}
