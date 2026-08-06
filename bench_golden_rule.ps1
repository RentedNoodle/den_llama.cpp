# bench_golden_rule.ps1 - Phase 2 golden-rule suite (run via bench_golden_rule.bat)
# ASCII-ONLY: may run under Windows PowerShell 5.1 (CP1252) - non-ASCII bytes kill parsing.
#
# References (same card, mainline, locked):
#   35B all-GPU (APEX Q3_K) = 168 tok/s   <- measured at -ngl 40 (NOT 99; 14.4GB+KV does not fit 16GB at ngl 99)
#   35B ncmoe-16 (AEON NVFP4) = 63
#   9B (Ornith NVFP4 native) = 113 | gemma-26B Q4_K_M = 161
#
# The staging tier (--den-stage) only fires when experts are host-resident = the
# AEON -ncmoe 16 cases.  Per-case timeout kills a hung llama-cli so the suite survives.
param([int]$n = 32, [int]$ctx = 32768, [int]$timeout = 240)
$ErrorActionPreference = 'Continue'
$bin   = 'I:\den_llama.cpp\build_ninja\bin\llama-cli.exe'
$mod   = 'I:\Models'
$logs  = 'I:\bench_den2'

if (-not (Test-Path $bin)) { Write-Host "MISSING BINARY: $bin - build first"; exit 1 }
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Get-VramUsed {
  try { return [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) 2>$null) } catch { return -1 }
}
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  for ($i=0; $i -lt 20; $i++) {
    $u = Get-VramUsed
    if ($u -lt 0) { break }
    if ($u -lt 2000) { break }
    Write-Host "VRAM busy ($u MiB) - waiting 15s..."; Start-Sleep 15
  }
  Write-Host "VRAM = $((Get-VramUsed)) MiB"
} else {
  Write-Host "[warn] nvidia-smi not on PATH - skipping VRAM gate"
}

function Run-Case {
  param([string]$name, [string]$model, [int]$ngl, [string]$extra, [string]$expect)
  $log = Join-Path $logs "$name.log"
  $cmd = "-m `"$model`" -ngl $ngl -c $ctx -ctk q8_0 -ctv q8_0 -t 16 -tb 16 " +
         "-p `"The capital of France is`" -n $n --no-display-prompt --no-conversation"
  if ($extra -and $extra.Trim() -ne '') { $cmd += ' ' + $extra.Trim() }
  Write-Host "`n>>> [$name] ngl=$ngl (expect >= $expect, timeout ${timeout}s)"

  $p = Start-Process -FilePath $bin -ArgumentList $cmd -WorkingDirectory (Split-Path $bin -Parent) `
       -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -NoNewWindow
  if (-not $p.WaitForExit($timeout * 1000)) {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Write-Host "[$name] TIMEOUT after ${timeout}s - killed"
    return
  }

  $lines = Get-Content $log -ErrorAction SilentlyContinue
  $eval  = ($lines | Select-String 'eval time = .*tokens per second' | Select-Object -Last 1).Line
  $ppe   = ($lines | Select-String 'prompt eval time' | Select-Object -Last 1).Line
  $probe = ($lines | Select-String 'Den expert-stage L3 probe' | Select-Object -Last 1).Line
  $fail  = $lines | Select-String 'error|CUDA error|out of memory|failed to load' | Select-Object -First 3
  Write-Host "[$name] $eval"
  if ($ppe)   { Write-Host "[$name]   PPE: $ppe" }
  if ($probe) { Write-Host "[$name]   $probe" }
  if ($fail)  { $fail | ForEach-Object { Write-Host "[$name]   !! $($_.Line)" } }
  Write-Host "[$name]   EXPECT >= $expect  EXIT=$($p.ExitCode)"
}

$models = @{
  '35b_allgpu'    = "$mod\ornith-1.0-35b-APEX-I-Mini-MTP.gguf"
  '35b_ncmoe16'   = "$mod\ornith-1.0-35b-APEX-I-Mini-MTP.gguf"
  '35b_stage'     = "$mod\ornith-1.0-35b-APEX-I-Mini-MTP.gguf"
  '35b_stageprobe'= "$mod\ornith-1.0-35b-APEX-I-Mini-MTP.gguf"
  '9b'            = "$mod\Ornith-1.0-9B-NVFP4-v2.gguf"
  'gemma26'       = "$mod\google-gemma-4-26B-A4B-it-Q4_K_M.gguf"
}

Write-Host "`n=== GOLDEN RULE SUITE (mainline rebuild, AVX-512, den-stage) ==="
Run-Case '35b_allgpu'    $models['35b_allgpu']    40  '-fa on -np 1 -ub 256 --reasoning on'    '168'
Run-Case '35b_ncmoe16'   $models['35b_ncmoe16']   99  '-ncmoe 16 --reasoning on'                '63'
Run-Case '35b_stage'     $models['35b_stage']     99  '-ncmoe 16 --den-stage --reasoning on'    '63'
Run-Case '35b_stageprobe' $models['35b_stageprobe'] 99 '-ncmoe 16 --den-stage-probe --reasoning on' '63'
Run-Case '9b'            $models['9b']            99  ''                                 '113'
Run-Case 'gemma26'       $models['gemma26']       99  ''                                 '161'
Write-Host "`n=== DONE. Compare 35b_stage vs 35b_ncmoe16 (staging must NOT regress). ==="
