# PROJECT DEN — LIVE ENGINE REFERENCE (`I:\den_llama.cpp`)

> The engine's own reference doc. A future AI can understand the live engine state from this alone. Compiled 2026-08-16. This is the FIRST in-repo Den doc — there was none before (state lived in commit history + probe logs).

## Git state (live, at time of writing)

- **Branch:** `rebase-clean` — ahead of `origin/rebase-clean` by **2 commits** (clean tree). HEAD `fef07ee7c` "Dual CE: route ≥1MB H2D through CE1 when DEN_DUAL_CE=1".
- **Remotes:** `origin` = `RentedNoodle/den_llama.cpp`, `upstream` = `ikawrakow/ik_llama.cpp`, `llama` = `ggml-org/llama.cpp`.
- **Rebase lineage (bottom→top):** llama/master (`370465eec` + repetition fix) → only-active-experts (`b266c371d`, ik #698) → Den speed stack (`54bed2620`) → 5× KVarN phases → Den HEAD commits.

## What's ported (the speed stack — all committed)

| Commit | Component |
|---|---|
| `d07bc13da` | KVarN Phase 1 — llama-core foundation |
| `309e91538` | KVarN Phase 2 — ggml core ops |
| `a95c1c542` | KVarN Phase 3 — CUDA kernels `kvarn.cu`/`kvarn-wht.cu`/fwht |
| `b8d19e33a` | KVarN Phase 4 — attention fusion `fattn-kvarn-dispatch.cu` (1079 lines, 53 MMA instances) |
| `1f6a934e6` | KVarN final wiring — KVARN_DOMAIN, create_memory, CLI, `context_params.kvarn` |
| `54bed2620` | NVFP4-KV (K8V8) + per-kv_head dequant + L2 persist + Windows timer |
| `765be6427` | L2 persist auto-wire (`den-l2-persist.cuh`), default ON |
| `050044f65` | Windows timer — `timeBeginPeriod(1)` + `HIGH_PRIORITY_CLASS` |
| `a503ce047` | per-kv_head FP32 dequant cache (`fattn-nvfp4-kv.cu/.cuh`) |
| `a43adc9f5` | MTP multi-layer head chaining (`qwen35moe.cpp`, `nextn_layer_offset`) |
| `dc4189aa6` | MUL_MAT_ID always offload-eligible (HEAD-2) |
| `fef07ee7c` | Dual CE — route ≥1MB H2D through CE1 stream (Blocker 4 fix) |
| `9250d58ea` | GDN fast-exp2 `DEN_GDN_FAST_EXP=1` (opt-in) |
| `32f88ee61`/`065dd475b` | kvarn streaming-store `__stcs` |
| `b266c371d` | only-active-experts (ik #698) |

Repetition fix landed in `370465eec` (penalty_repeat 1.10, dry 0.8; args in `common/common.h`/`common/arg.cpp`).

## Key components — file paths

- **KVarN TurboQuant KV:** `ggml/src/ggml-cuda/kvarn.cu`+`.cuh`, `kvarn-wht.cu`+`.cuh`, `fattn-kvarn-dispatch.cu`+`.cuh`, `fattn-kvarn-portable.cuh`, `fattn-kvarn-route-policy.h`, `fattn-kvarn-vec*.cuh`, `fattn-mma-kvarn*.cuh`, ~70 template instances in `template-instances/fattn-mma-kvarn-*.cu`. llama-layer: `src/llama-kvarn.cpp`+`.h`, `src/llama-kv-cache-kvarn.cpp`+`.h`. CLI: `--cache-type-k/-v kvarnN` (N∈2..8), SWA overrides `--cache-type-kvarn-swa-*`, parsed in `common/arg.cpp`.
- **NVFP4-KV (K8V8):** `ggml/src/ggml-cuda/fattn-nvfp4-kv.cu` (1133 lines)+`.cuh`, wired in `fattn.cu`+`ggml-cuda.cu`. (Weak vs KVarN — keep disabled via `DEN_NVFP4_KV_CACHE=0` when using KVarN.)
- **MTP:** `src/models/qwen35moe.cpp` (chaining), `src/models/qwen35.cpp`. CLI `--mtp`, `--spec-type draft-mtp` (`COMMON_SPECULATIVE_TYPE_DRAFT_MTP`).
- **Expert offload:** `src/llama-context.cpp:282,615,651` (`only_active_experts`), `src/llama-cparams.h`, CLI `--no-offload-only-active-experts`. MoE models in `src/models/*moe.cpp`.
- **L2 persist:** `ggml/src/ggml-cuda/den-l2-persist.cuh` (516 lines), wired in `ggml-cuda.cu`. `DEN_L2_PERSIST=0` off.
- **Windows timer:** `tools/cli/cli.cpp:41-46`. `DEN_HIGH_PRIORITY=0` off.

## Build

- **Canonical:** `build_den.ps1` (`-phase configure|ggml-cuda|all`). Ninja + `build_ninja`, toolchain `cmake/msvc_toolchain.cmake` (pip CUDA 13.3, VS2022), `-DCMAKE_CUDA_ARCHITECTURES=120a`, AVX-512 (VBMI/VNNI/BF16) ON. Binaries in `build_ninja\bin\` (`llama-server.exe`, `llama-cli.exe`, `llama-bench.exe`, `llama-quantize.exe`, `llama-perplexity.exe`, `llama.dll`, `ggml*.dll`).
- Speed test (KVarN6): `$env:DEN_NVFP4_KV_CACHE=0; llama-cli -m <35B.gguf> -ngl 99 -ctk kvarn6 -ctv kvarn6 -p "Hello" -n 100`.

## Models / launch

- No in-repo reference to models (launch scripts `start_dreya.bat`/`start_harness.bat` are EMPTY). Models on `I:\models\` (see `I:\PROJECT_DEN_LIVE_REFERENCE.md`). Harness launch scripts live in `I:\den_harness\`.

## Gaps / next

- The two unpushed commits (Dual CE + offload fix) are the live edge at HEAD. No in-repo Den design doc existed — this file is the first. Next engine work: native MTP head (1.3×), BTL-4 abliteration, pocCharlies .den conversion, KVarN gate verification.

*See also the master reference: `C:\Den\den-nvfp4-optimizations\docs\PROJECT_DEN_ULTIMATE_REFERENCE.md`.*
