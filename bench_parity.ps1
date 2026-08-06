# bench_parity.ps1 - non-interactive parity gate via llama-bench (golden rule)
# ASCII-ONLY (Windows PowerShell 5.1 reads .ps1 as CP1252 - non-ASCII kills parse).
#
# Measures generation tok/s (llama-bench "tg") at the reference configs:
#   35B all-GPU (APEX Q3_K)  = 168 | 35B ncmoe16 (AEON NVFP4) = 63
#   9B (Ornith NVFP4 native) = 113 | gemma-26B Q4_K_M = 161
# Also measures the staging tier (--den-stage) vs baseline ncmoe16 - the Phase 2 gate.
# Non-interactive: llama-bench prints pp/tg tables and exits. No REPL, no reasoning dump.
param([int]$n = 64, [int]$ctx = 32768, [int]$pp = 64, [int]$timeout = 400)
$ErrorActionPreference = 'Continue'
$bin  = 'I:\den_llama.cpp\build_ninja\bin\llama-bench.exe'
$mod  = 'I:\Models'
$logs = 'I:\bench_den2'

if (-not (Test-Path $bin)) { Write-Host "MISSING llama-bench: $bin - build first"; exit 1 }
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Run-Bench {
  param([string]$name, [string]$model, [int]$ngl, [int]$ncmoe, [string]$extra, [string]$expect)
  $log = Join-Path $logs "parity_$name.log"
  $cmd = "-m `"$model`" -ngl $ngl -ncmoe $ncmoe -t 16 -tb 16 -c $ctx -p $pp -n $n"
  if ($extra -and $extra.Trim() -ne '') { $cmd += ' ' + $extra.Trim() }
  Write-Host "`n>>> [$name] ngl=$ngl ncmoe=$ncmoe (expect tg >= $expect, timeout ${timeout}s)"
  $p = Start-Process -FilePath $bin -ArgumentList $cmd -WorkingDirectory (Split-Path $bin -Parent) `
       -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -NoNewWindow
  if (-not $p.WaitForExit($timeout * 1000)) {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Write-Host "[$name] TIMEOUT after ${timeout}s - killed"
    return
  }
  $lines = Get-Content $log -ErrorAction SilentlyContinue
  $tg = $lines | Select-String '^\|\s*tg\b' | Select-Object -Last 1
  $ppl = $lines | Select-String '^\|\s*pp\b' | Select-Object -Last 1
  if ($tg) { Write-Host "[$name]   $($tg.Line.Trim())" } else { Write-Host "[$name]   (no tg row - see $log)" }
  if ($ppl) { Write-Host "[$name]   $($ppl.Line.Trim())" }
  $fail = $lines | Select-String 'error|CUDA error|out of memory|failed to load|Assertion' | Select-Object -First 3
  if ($fail) { $fail | ForEach-Object { Write-Host "[$name]   !! $($_.Line)" } }
  Write-Host "[$name]   EXPECT tg >= $expect  EXIT=$($p.ExitCode)"
}

$models = @{
  '35b_allgpu'   = "$mod\ornith-1.0-35b-APEX-I-Mini-MTP.gguf"
  '35b_ncmoe16'  = "$mod\ornith-1.0-35b-APEX-I-Mini-MTP.gguf"
  '35b_stage'    = "$mod\ornith-1.0-35b-APEX-I-Mini-MTP.gguf"
  '9b'           = "$mod\Ornith-1.0-9B-NVFP4-v2.gguf"
  'gemma26'      = "$mod\google-gemma-4-26B-A4B-it-Q4_K_M.gguf"
}

Write-Host "`n=== PARITY GATE (llama-bench, non-interactive) ==="
Run-Bench '35b_allgpu'   $models['35b_allgpu']   40  0  ''                    '168'
Run-Bench '35b_ncmoe16'  $models['35b_ncmoe16']  99  16 ''                    '63'
Run-Bench '35b_stage'    $models['35b_stage']    99  16 '--den-stage'         '63'
Run-Bench '9b'           $models['9b']           99  0  ''                    '113'
Run-Bench 'gemma26'      $models['gemma26']      99  0  ''                    '161'
Write-Host "`n=== DONE. tg(35b_stage) vs tg(35b_ncmoe16): staging must NOT regress. ==="
