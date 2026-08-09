@echo off
setlocal EnableDelayedExpansion
title MULTI-MODEL NVFP4 KV ACCURACY GATE

REM ============================================================
REM MULTI-MODEL NVFP4 KV ACCURACY GATE
REM Tests KLD/cosine accuracy gate on ALL available models.
REM ============================================================
REM Dual-context in-process: F32 oracle vs NVFP4 KV candidate.
REM 500 tokens/model (1024 for 35B), 256-token F32 tail.
REM
REM MODELS:
REM   1. Ornith 9B NVFP4 v2   — dense SSM,   head_dim=256, 4 KV
REM   2. Gemma 4 12B Q4_0     — dense SWA,   head_dim=256, 8 KV
REM   3. Gemma 4 26B Q4_K_M   — MoE A4B,     may OOM @ ngl 99
REM   4. Qwen3 4B Q4_K_M      — standard MHA, baseline
REM   5. Ornith 35B APEX Mini — MoE,         KLD=0 @ 8192
REM
REM ENV: DEN_THRIFT_ATTENTION=1 (K8V8), DEN_NVFP4_KV_CACHE=1
REM ============================================================

set "PY=C:\Den\den-py314\Scripts\python.exe"
set "SCRIPT=I:\den_llama.cpp\tools\gate_accuracy_kv.py"

set DEN_THRIFT_ATTENTION=1
set DEN_NVFP4_KV_CACHE=1

cd /d I:\den_llama.cpp

echo.
echo  ################################################################
echo  #  MULTI-MODEL NVFP4 KV ACCURACY GATE
echo  #  F32 Oracle vs NVFP4 KV Candidate (KLD + Cosine)
echo  #  %DATE% %TIME%
echo  ################################################################
echo.

set PASS=0
set FAIL=0
set "FAIL_LIST="

REM ═══════════════════════════════════════════════════════════════════════
REM [1/5] Ornith 9B NVFP4 v2 — dense SSM, head_dim=256, 4 KV heads
REM ═══════════════════════════════════════════════════════════════════════
echo.
echo  ──────────────────────────────────────────────────────────────────
echo   [1/5] Ornith-1.0-9B-NVFP4-v2  (dense SSM, head_dim=256, 4 KV)
echo  ──────────────────────────────────────────────────────────────────
%PY% %SCRIPT% --model "I:\models\Ornith-1.0-9B-NVFP4-v2.gguf" --tokens 500 --ngl 99 --no-expert-stage
if !ERRORLEVEL! equ 0 (
    set /a PASS+=1
    echo   ^>^>^> [PASS] ornith-9b-nvfp4 ^<^<^<
) else (
    set /a FAIL+=1
    set "FAIL_LIST=!FAIL_LIST!ornith-9b-nvfp4 "
    echo   ^>^>^> [FAIL] ornith-9b-nvfp4 ^<^<^<
)

REM ═══════════════════════════════════════════════════════════════════════
REM [2/5] Gemma 4 12B Q4_0 — dense SWA, head_dim=256, 8 KV heads
REM ═══════════════════════════════════════════════════════════════════════
echo.
echo  ──────────────────────────────────────────────────────────────────
echo   [2/5] Gemma-4-12B-Q4_0  (dense SWA, head_dim=256, 8 KV)
echo  ──────────────────────────────────────────────────────────────────
%PY% %SCRIPT% --model "I:\models\gemma-4-12B-it-qat-q4_0-uncensored-heretic-Q4_0.gguf" --tokens 500 --ngl 99 --no-expert-stage
if !ERRORLEVEL! equ 0 (
    set /a PASS+=1
    echo   ^>^>^> [PASS] gemma4-12b-q4_0 ^<^<^<
) else (
    set /a FAIL+=1
    set "FAIL_LIST=!FAIL_LIST!gemma4-12b-q4_0 "
    echo   ^>^>^> [FAIL] gemma4-12b-q4_0 ^<^<^<
)

REM ═══════════════════════════════════════════════════════════════════════
REM [3/5] Gemma 4 26B Q4_K_M — MoE, may OOM at ngl 99
REM ═══════════════════════════════════════════════════════════════════════
echo.
echo  ──────────────────────────────────────────────────────────────────
echo   [3/5] Gemma-4-26B-Q4_K_M  (MoE A4B — may OOM with ngl 99)
echo  ──────────────────────────────────────────────────────────────────
%PY% %SCRIPT% --model "I:\models\google-gemma-4-26B-A4B-it-Q4_K_M.gguf" --tokens 500 --ngl 99
if !ERRORLEVEL! equ 0 (
    set /a PASS+=1
    echo   ^>^>^> [PASS] gemma4-26b-q4_k ^<^<^<
) else (
    set /a FAIL+=1
    set "FAIL_LIST=!FAIL_LIST!gemma4-26b-q4_k(OOM?) "
    echo   ^>^>^> [FAIL/OOM] gemma4-26b-q4_k ^<^<^<
    echo   If OOM: retry with --ngl 0 or smaller model.
)

REM ═══════════════════════════════════════════════════════════════════════
REM [4/5] Qwen3 4B Q4_K_M — standard attention, test baseline
REM ═══════════════════════════════════════════════════════════════════════
echo.
echo  ──────────────────────────────────────────────────────────────────
echo   [4/5] Qwen3-4B-Q4_K_M  (standard MHA, test-only baseline)
echo  ──────────────────────────────────────────────────────────────────
%PY% %SCRIPT% --model "I:\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf" --tokens 500 --ngl 99 --no-expert-stage
if !ERRORLEVEL! equ 0 (
    set /a PASS+=1
    echo   ^>^>^> [PASS] qwen3-4b-q4_k_m ^<^<^<
) else (
    set /a FAIL+=1
    set "FAIL_LIST=!FAIL_LIST!qwen3-4b-q4_k_m "
    echo   ^>^>^> [FAIL] qwen3-4b-q4_k_m ^<^<^<
)

REM ═══════════════════════════════════════════════════════════════════════
REM [5/5] Ornith 35B APEX Mini — MoE, 1024 tokens (fair comparison)
REM ═══════════════════════════════════════════════════════════════════════
echo.
echo  ──────────────────────────────────────────────────────────────────
echo   [5/5] Ornith-35B-APEX-Mini  (MoE, KLD=0 verified @ 8192 context)
echo  ──────────────────────────────────────────────────────────────────
%PY% %SCRIPT% --model "I:\models\ornith-1.0-35b-APEX-I-Mini-MTP.gguf" --tokens 1024 --ngl 99
if !ERRORLEVEL! equ 0 (
    set /a PASS+=1
    echo   ^>^>^> [PASS] ornith-35b-apex ^<^<^<
) else (
    set /a FAIL+=1
    set "FAIL_LIST=!FAIL_LIST!ornith-35b-apex "
    echo   ^>^>^> [FAIL] ornith-35b-apex ^<^<^<
)

REM ═══════════════════════════════════════════════════════════════════════
REM SUMMARY — comparison table
REM ═══════════════════════════════════════════════════════════════════════
set /a TOTAL=PASS+FAIL
echo.
echo  ################################################################
echo  #  MULTI-MODEL ACCURACY_KV SUMMARY
echo  ################################################################
echo  #
echo  #  MODEL                  TOKENS  VERDICT
echo  #  ---------------------  ------  -------
echo  #  ornith-9b-nvfp4        500     [check above]
echo  #  gemma4-12b-q4_0        500     [check above]
echo  #  gemma4-26b-q4_k        500     [check above]
echo  #  qwen3-4b-q4_k_m        500     [check above]
echo  #  ornith-35b-apex        1024    [check above]
echo  #
echo  #  Passed : !PASS!/!TOTAL!
echo  #  Failed : !FAIL!/!TOTAL!
if not "!FAIL_LIST!"=="" (
    echo  #
    echo  #  FAILED MODELS:!FAIL_LIST!
    echo  #
    echo  #  ACTIONS:
    echo  #    OOM ^> retry with --ngl 0 (CPU-only KV compare)
    echo  #    KLD ^> 0.001 ^> NVFP4 KV degrades on this arch
    echo  #    cos ^< 0.9995 ^> KV quantization loses direction
    echo  #    crash/segfault ^> different KV head config issue
)
echo  #
echo  #  Scroll up for per-model KLD/cosine/top-1 breakdown.
echo  #  TILE region = positions past 256-token F32 tail.
echo  ################################################################
echo.

pause
