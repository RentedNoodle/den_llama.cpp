# BUILD2 GOLDEN MASTER

Recovery/repro record for the production `build2` engine. `build2/` itself is gitignored
(binaries are not committed); this file + the source at the recorded commit are the source of
truth for rebuilding an identical `llama-server` on a fresh machine.

## Source of truth

| Field | Value |
|---|---|
| Commit (HEAD) | `0875eab41bf5a85189aa3a04bc8333ef86be3407` |
| Upstream base | `b10687` (`git describe`: `b10687-72-g0875eab41`) |
| Local branch | `master` |
| Publish remote | `den` -> `https://github.com/RentedNoodle/den_llama.cpp.git` |
| Publish branch | `main` (was `0875eab41` before this doc; fast-forward) |
| Note | `origin` is `ggml-org/llama.cpp` (read-only upstream) — never push there. |

## Toolchain (verified from `build2/CMakeCache.txt`)

| Component | Version / path |
|---|---|
| Generator | Ninja 1.13.0 (`C:/Den/den-py314/Scripts/ninja.exe`) |
| Build type | `Release` |
| CUDA toolkit | v13.3.33 (`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe`) |
| CUDA arch (effective) | `compute_120a` / `sm_120a` (Blackwell GB203, RTX 5070 Ti) |
| C/C++ compiler | MSVC 19.x, `14.44.35207` (VS 2022 Community) |
| Compression mode | `-compress-mode=size` (`GGML_CUDA_COMPRESSION_MODE=size`) |

## Build configuration (actual, from CMakeCache.txt)

Key cache values as-built:

```
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_GENERATOR:INTERNAL=Ninja
GGML_CUDA:BOOL=ON
GGML_CUDA_FA:BOOL=ON
GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF
GGML_CUDA_COMPRESSION_MODE:STRING=size
GGML_CUDA_GRAPHS:BOOL=ON
GGML_CUDA_NCCL:BOOL=ON
GGML_CUDA_PEER_MAX_BATCH_SIZE:STRING=256
GGML_NATIVE:BOOL=ON
GGML_LLAMAFILE:BOOL=ON
GGML_OPENMP:BOOL=ON
GGML_CPU_REPACK:BOOL=ON
GGML_ACCELERATE:BOOL=ON
GGML_BUILD_EXAMPLES:BOOL=OFF
GGML_BUILD_TESTS:BOOL=OFF
GGML_BACKEND_DL:BOOL=OFF
GGML_LTO:BOOL=OFF
```

> **DISCREPANCY / WARNING — verify before trusting.** `DEN.md` (and earlier notes) claim
> `-DGGML_CUDA_FA_ALL_QUANTS=ON`, but the *actual* `build2` was configured and compiled with it
> **OFF**: the option is `OFF` in `CMakeCache.txt`, and neither `build.ninja` nor
> `compile_commands.json` contains the `GGML_CUDA_FA_ALL_QUANTS` define. `ggml-cuda.dll`
> (built 2026-08-30) therefore has only the default FA kernel set (f16-f16 + q4_0-q4_0), which is
> why the ship flags use **q4_0** KV. If you rebuild with `ON`, you get more quantized-KV FA
> kernels but a **different** binary than this golden master. Rebuild with the value recorded above
> to match `build2` bit-for-bit in behaviour; only change it deliberately and re-run the gates.

## Exact configure command

From a clean checkout at `0875eab41`, on Windows with CUDA 13.3 + VS2022:

```powershell
cmake -S . -B build2 -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CUDA=ON `
  -DGGML_CUDA_FA_ALL_QUANTS=OFF `
  -DGGML_CUDA_GRAPHS=ON `
  -DGGML_CUDA_NCCL=ON `
  -DGGML_CUDA_COMPRESSION_MODE=size `
  -DGGML_NATIVE=ON `
  -DGGML_LLAMAFILE=ON `
  -DGGML_OPENMP=ON `
  -DGGML_CPU_REPACK=ON `
  -DGGML_BUILD_EXAMPLES=OFF `
  -DGGML_BUILD_TESTS=OFF
```

`GGML_CUDA_NCCL=ON` is the as-built cache value (harmless no-op on Windows single-GPU).
`GGML_NATIVE=ON` selects the host CPU feature set — on a different CPU this is the one value that
legitimately differs; set `-DGGML_NATIVE=OFF` for a portable binary.

## Build command

```powershell
cmake --build build2 --target llama-server --config Release -j 8
```

Do **not** run a second build concurrently; do not build while a live `llama-server` is up.
Project rule: one Ninja binary only (`C:/Den/den-py314/Scripts/ninja.exe`, v1.13) — mixing Ninja
versions resets `.ninja_log` and forces a full 424-target rebuild.

## Patch stack committed in this golden master (10 files)

Diff vs upstream `b10687` (source at `0875eab41` + these working-tree patches):

| File | What it does |
|---|---|
| `common/speculative.cpp` | MTP carryover-reset on a new prompt (local fix #27296): zero stale `pending_h`/`verify_h` for the sequence so a previous request's MTP carryover cannot poison the first catch-up row. Also an env-gated (`LLAMA_DUMP_MTP`) debug dump of the first single-token draft. |
| `common/arg.cpp` | New `--embeddings-nextn` flag (sets `embedding=true` + `embeddings_nextn`), env `LLAMA_ARG_EMBEDDINGS_NEXTN`. |
| `common/common.h` | New `common_params::embeddings_nextn` field. |
| `common/common.cpp` | Calls `llama_set_embeddings_nextn(lctx, true)` when the flag is set. |
| `include/llama.h` | Declares `llama_set_embeddings_nextn()` and `llama_get_embeddings_nextn_ith()`. |
| `src/llama-context.cpp` | NULL-safe `llama_get_embeddings_nextn_ith` wrapper (try/catch) + env-gated (`LLAMA_DUMP_MTP`) MTP tensor-dump for byte-compare vs a PyTorch reimpl. |
| `src/models/delta-net-base.cpp` | Gated-delta-net conv-state snapshot bound fix: `K = min(n_rs_seq, n_tok)+1` instead of full depth, so deeper slots survive and a rollback to a previous larger batch still reads them (fixes `test-recurrent-state-rollback`). |
| `tools/server/server-task.h` | Adds `std::vector<std::vector<float>> nextn` to `server_task_result_embd`. |
| `tools/server/server-task.cpp` | Emits `"nextn"` in the non-OAI-compat embeddings JSON when present. |
| `tools/server/server-context.cpp` | Captures per-token MTP head input hidden (`h_nextn`) into `res->nextn` when `embeddings_nextn` is enabled. |

Env-gated debug blocks (`LLAMA_DUMP_MTP`, writing to `C:\Den\dreya\mtp\dump\`) are intentionally
part of this commit because they were present in the source that produced `build2`; they are inert
unless the env var is set.

## Verification procedure (after rebuild)

1. **Load:** `build2\bin\llama-server.exe --version` then boot the PEAK command and confirm the
   model loads and MTP is active (~89 t/s on the 27B @ 262k):
   ```
   llama-server -m I:\models\Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP.gguf ^
     --spec-type draft-mtp --spec-draft-n-max 2 --ctx-size 262144 -fa on ^
     -ctk q8_0 -ctv q4_0 -ngl all -b 512 -ub 512 -np 1 ^
     --kv-stream-stage-mib 2048 -t 8 -tb 8 --cache-ram 8192 --reasoning-budget 256
   ```
2. **2-case needle spot:** run NIAH at two depths (e.g. 32k and 196k); expect **2/2**.
3. **Gates (no-regression):** NIAH 1/1, tool 3/3, t/s vs `master_bench.json`. A regression must be
   planned + justified before any promote.

## Fresh-machine restore sequence

```powershell
git clone https://github.com/RentedNoodle/den_llama.cpp.git
cd den_llama.cpp
git checkout 0875eab41bf5a85189aa3a04bc8333ef86be3407   # or: git checkout main (same tip after push)
# install: VS2022 + CUDA 13.3 + Ninja >= 1.13
cmake -S . -B build2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON `
  -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_NCCL=ON -DGGML_CUDA_COMPRESSION_MODE=size `
  -DGGML_NATIVE=ON -DGGML_LLAMAFILE=ON -DGGML_OPENMP=ON -DGGML_CPU_REPACK=ON `
  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF
cmake --build build2 --target llama-server --config Release -j 8
# model weights are separate artifacts (not in this repo) — fetch per their own license.
```

## Intentionally NOT committed

- `build2/` (gitignored; binaries ~50 MB+) — rebuilt from source, never shipped in git.
- Model files (`I:\models\*.gguf`) — separate artifacts, permission-only, never uploaded.
- MTP debug dumps (`C:\Den\dreya\mtp\dump\`) — transient debug output.
