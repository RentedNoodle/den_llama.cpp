# DEN BUILD SPEC — the toolchain recipe that must NEVER be lost again

> 2026-08-17: after the 08/09 "chased NVFP4, broke the build, lost the CMake config" incident,
> this file is the versioned, recoverable source of truth for reproducing the 191.55 t/s baseline.
> If a build ever breaks and this state seems "lost", THIS FILE + the git commits are the way back.

## Proven baseline (raw engine)
- **191.55 tg64** @ build `e7ba620ea` (per NEXT_BLOCKERS_2026-08-16; engine 180 baseline RESTORED 08/16, ref 08/09's 184.4)
- **CONFIRMED 2026-08-17 09:58 by Noodle's own bench: `tg64 = 190.79 ± 0.30` @ build d0bb6edf7** (llama-bench -m Ornith-1.0-35B-Heretic-MTP-APEX-I-Mini.gguf -ngl 99 -n 64; pp512 1300.06). Engine NOT regressed; the perceived regression = server-vs-engine gap (Blocker 1).
- KV NVFP4 ENABLED in this build (K8V8 ThriftAttention, 41 layers, F32 tail 256), L2-PERSIST 8MB budget.
- **Model baseline decision:** engine work is measured against the PRE-QUANTIZED proven pair (35B Heretic Q3_K + 9B Q4 worker) as-is. Our own quant = only when the .den/Den2P4 lane opens (SSM-firewall re-convert first), gated by cos>0.9999 + sentinels.

## Exact build recipe (audited 2026-08-17, live in I:\den_llama.cpp\build_ninja\CMakeCache.txt)
| Setting | Value |
|---|---|
| Generator | Ninja |
| CMAKE_BUILD_TYPE | Release |
| CMAKE_CUDA_ARCHITECTURES | `120a` (RTX 5070 Ti, GB203) |
| CMAKE_CUDA_COMPILER | `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe` |
| GGML_CUDA | ON |
| GGML_NATIVE | ON |
| CUDA_FLAGS | `-D_WINDOWS -Xcompiler=" /EHsc"` |
| Toolchain | MSVC 2022 Community (14.44.x, Hostx64/x64) · cmake 3.31.6-msvc6 · ninja (VS-bundled) |

## Reproduce
```powershell
cd I:\den_llama.cpp
cmake -B build_ninja -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES=120a `
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe"
cmake --build build_ninja --config Release --parallel 16
# verify (user runs benchmarks):
build_ninja\bin\llama-bench.exe -m I:\models\Ornith-1.0-35B-Heretic-MTP-APEX-I-Mini.gguf -ngl 99 -n 64
```
Expected: tg64 ≈ 191 (raw engine). `--cpu-moe` = pathological, never baseline.

## Rules that keep the body alive
- NVFP4 = capacity play, NOT decode-speed play (no 160+ on 16GB yet; APEX Nano 121 ceiling).
- Server overhead fixes (GPU sampling, double-buffered decode, tokenize off-path) = Blocker 1, in flight (bfcb0d3be shipped adaptive draft controller + GPU sampling).
- >160 t/s ceiling path = SPECULATION with larger M + **SSM-state checkpoint/rollback draft** (CATS disabled on recurrent GDN 35B).
- Priority order: expert-locality (llama.cpp #26563, 1.7-2.1x) THEN DSpark/DFlash block-drafting.
- Never delete/overwrite `build_ninja\CMakeCache.txt` without committing a new copy of THIS spec.

## Companion facts
- Bench model: `Ornith-1.0-35B-Heretic-MTP-APEX-I-Mini.gguf` (13.29 GB, hero).
- Q8 KV, batch/ubatch 64, `-ngl 99 -ncmoe 0` for the matrix (Milestone A: c256/Q8 164.08, c4096/Q8 159.44, c4096/KVarN6 162.23).
