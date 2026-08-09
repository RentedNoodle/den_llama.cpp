@echo off
REM ============================================================
REM CONTEXT SCALING — NVFP4 KV accuracy at 1k/2k/4k/8k/16k
REM
REM Tests if KLD=0/cos=1.0 HOLDS at increasing context sizes.
REM The 500-token gate passed (KLD=0) at ~600 context.
REM This runs the gate at 1k, 2k, 4k, 8k to find when
REM KLD starts accumulating. Reviewers' #1 concern.
REM
REM VRAM: F32 KV cache at 8k ~ 6.4 GB/context (9B est).
REM Two contexts = ~12.8 GB + model = tight on 16 GB.
REM If OOM: retry with 9B model or --ngl 0.
REM
REM Usage: gate_context_scaling.bat [model] [ctx_sizes]
REM Examples:
REM   gate_context_scaling.bat
REM   gate_context_scaling.bat I:\models\ornith-1.0-9b-NVFP4.gguf
REM   gate_context_scaling.bat I:\models\ornith-35b.gguf 1024,2048,4096
REM   gate_context_scaling.bat "" "1024,2048,4096,8192,16384"
REM ============================================================
setlocal
set "MODEL=I:\models\ornith-1.0-35b-APEX-I-Mini-MTP.gguf"
set "CTX_SIZES=1024,2048,4096,8192"
set "NGL=99"

if not "%~1"=="" set "MODEL=%~1"
if not "%~2"=="" set "CTX_SIZES=%~2"

echo ============================================================
echo CONTEXT SCALING TEST: NVFP4 KV Accuracy vs F32 Oracle
echo ============================================================
echo Model:        %MODEL%
echo Context sizes: %CTX_SIZES%
echo GPU layers:   %NGL%
echo.
echo VRAM NOTE: For 8192+ contexts on 16 GB, use 9B model or
echo --ngl 0. F32 KV cache at 8k = ~6.4 GB per context (9B).
echo Two contexts + model may OOM on 35B.
echo.
echo If OOM, run:
echo   gate_context_scaling.bat I:\models\ornith-1.0-9b-NVFP4.gguf 1024,2048,4096
echo ============================================================
echo.

cd /d I:\den_llama.cpp
C:\Den\den-py314\Scripts\python.exe tools\gate_accuracy_context_scaling.py --model "%MODEL%" --ctx-sizes "%CTX_SIZES%" --ngl %NGL%

echo.
echo Exit code: %ERRORLEVEL%
if %ERRORLEVEL% equ 0 (
    echo [PASS] NVFP4 KV passes all accuracy gates at all context sizes.
) else (
    echo [FAIL] One or more context sizes failed the accuracy gate. See table above.
)
pause
