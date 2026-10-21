<div align="center">

# den_llama.cpp

**A llama.cpp fork for long-context Qwen3.8-27B inference with native MTP speculative decoding on one 16 GB Blackwell GPU.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: C++](https://img.shields.io/badge/language-C%2B%2B-00599C.svg)](#build)
[![CUDA: Blackwell sm_120a](https://img.shields.io/badge/CUDA-Blackwell%20sm__120a-76B900.svg)](#build)
[![Status: build2 stable + experimental branches](https://img.shields.io/badge/status-build2%20stable%20%2B%20experimental%20branches-orange.svg)](#branches-tags-and-stability)

[Build](#build) &middot; [Run](#run) &middot; [Feature matrix](#feature-matrix) &middot; [Benchmarks](#benchmarks) &middot; [Repository layout](#repository-layout) &middot; [Credits](#credits)

</div>

---

> [!IMPORTANT]
> **Source only.** This repository ships no binaries, model weights, or releases. `build2` and `build3` are build directories recorded in the docs, not committed trees. The current stable line is `build2`; `build2.1` is next. `build3` is regressed and superseded and is documented only for history. The reference target is one NVIDIA Blackwell GPU (RTX 5070 Ti, GB203, sm_120a, 16 GB) running Qwen3.8-27B with an in-model multi-token-prediction (MTP) head.

den_llama.cpp is the public GGUF-oriented engine line associated with [Project Den](https://github.com/RentedNoodle/Project_Den). It is a llama.cpp fork with changes for a single-card, long-context deployment: MTP draft-and-verify, quantized KV codecs, layer-wise KV precision, adaptive KV streaming, and MoE expert placement. The private native runtime is a separate development line.

## Branches, tags, and stability

| Ref | Type | Stability | Contents |
|---|---|---|---|
| `abd589d4c` | commit | **Stable reference (`build2`)** | The pinned engine source recorded in [`BUILD2_GOLDEN.md`](BUILD2_GOLDEN.md). There is no `build2` tag; `build2` is this commit plus the recorded build configuration. |
| `main` | branch | Default | The stable engine source above plus documentation commits. The branch head moves as docs change; pin the commit for reproducible builds. |
| `build3` | branch / tag | **Regressed / superseded (historical)** | FA fix, NVFP4 KV, layer-wise split-KV, Smart Expert Reduction (SER), MoE `--fit` fix. Tag `build3-golden-20260924` points at `a22b5fea0`. Do not use as the current runtime. |
| `feat/q4-g64-kv` | branch | Experimental (based on historical `build3`) | `build3` plus the `GGML_TYPE_Q4_G64` KV codec and the KV headroom guard (`9cd4febbb`). |
| `den-legacy` | branch | Retired | Prior engine history (`a99ff8f9`). |
| `archive/stage1-nvfp4-abd589d4c` | branch | Archive | Pre-consolidation archive (`42bf457d`). |

The stable path is `abd589d4c`; `build2.1` is the next line. `build3` is regressed/superseded and `feat/q4-g64-kv` is experimental; neither is the recommended runtime. There are no GitHub releases or binary assets in this repository.

## Feature matrix

Status means the feature is present in the named ref and has a CLI surface in that ref. It does not mean every model or GPU combination is validated.

| Capability | Ref | Surface |
|---|---|---|
| MTP draft-and-verify | `main` | `--spec-type draft-mtp`, `--spec-draft-n-max` |
| Server MTP hidden-state output | `main` | `--embeddings-nextn` |
| NVFP4 KV cache | `build3` (historical) | `--cache-type-k nvfp4`, `--cache-type-v nvfp4` |
| Layer-wise split-KV | `build3` (historical) | `--kv-high-precision-layers N` |
| Smart Expert Reduction (SER) | `build3` (historical) | `-ser MIN,THRESH`, `--ser-top-p`, `--ser-ramp` |
| MoE expert placement | `main`, `build3` (historical) | `--fit`, `-ncmoe N` |
| Adaptive KV streaming | `main` | `--kv-stream-stage-mib N` |
| Q4_G64 KV cache | `feat/q4-g64-kv` (experimental) | `--cache-type-k q4_g64`, `--cache-type-v q4_g64` |
| KV headroom guard | `feat/q4-g64-kv` (experimental) | `--kv-headroom-warn`, `--kv-headroom-block`, `--kv-headroom-force` |
| NVFP4 weight MMA path | `main` | Experimental; see `ggml/src/ggml-cuda/mma.cuh` |

`build3` does **not** contain the Q4_G64 codec or the KV headroom guard. It is historical and is not the recommended runtime. The experimental `feat/q4-g64-kv` ref carries those flags.

## Build

Verified toolchain: Windows, CUDA 13.3, Visual Studio 2022, Ninja 1.13. The project declares CMake 3.14 through 3.28.

```powershell
git clone https://github.com/RentedNoodle/den_llama.cpp.git
cd den_llama.cpp
git checkout abd589d4c
```

### Reference build2 (exact source)

`build2` was configured with `GGML_CUDA_FA_ALL_QUANTS=OFF`. With that value the CUDA FlashAttention path rejects mixed K/V types (`ggml/src/ggml-cuda/fattn.cu`), so the reference server uses `q4_0` for both caches.

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

cmake --build build2 --target llama-server --config Release -j 8
```

`GGML_CUDA_NCCL=ON` is the recorded as-built value; it is a no-op on a single Windows GPU. `GGML_NATIVE=ON` selects the host CPU feature set; use `OFF` for a portable binary.

### Mixed quantized KV

To run `q8_0` K with `q4_0` V, or other mixed cache types, configure with `-DGGML_CUDA_FA_ALL_QUANTS=ON`. The resulting binary has the full quantized-KV FA kernel set and is not byte-identical to the recorded `build2`.

```powershell
cmake -S . -B build2 -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CUDA=ON `
  -DGGML_CUDA_FA_ALL_QUANTS=ON `
  -DGGML_CUDA_GRAPHS=ON `
  -DGGML_NATIVE=ON

cmake --build build2 --target llama-server --config Release -j 8
```

## Run

Download a model GGUF separately. The reference artifact is [`Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored`](https://huggingface.co/RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored); weights are not part of this repository.

```powershell
build2\bin\llama-server.exe `
  -m <path-to-model.gguf> `
  --spec-type draft-mtp --spec-draft-n-max 2 `
  --ctx-size 196608 -fa on -ctk q8_0 -ctv q4_0 -ngl all `
  -b 512 -ub 512 -np 1 --kv-stream-stage-mib 2048 `
  -t 8 -tb 8 --cache-ram 8192 --reasoning-budget 256
```

The command above requires the `GGML_CUDA_FA_ALL_QUANTS=ON` build because K and V use different types. For the exact `build2` binary, use `-ctk q4_0 -ctv q4_0`.

For depth-0.5 retrieval on the reference model, use `-b 2048 -ub 2048`. That is a model chunking artifact, not an engine requirement.

### Historical and experimental refs

`build3` is regressed and superseded. It is documented for availability and measurement history; it is
not the current runtime. `feat/q4-g64-kv` is experimental and is based on that historical line. Neither
ref is recommended for new deployments.

For reference, the historical `build3` surface includes `-ctk nvfp4 -ctv nvfp4`,
`--kv-high-precision-layers N`, `-ser MIN,THRESH`, and `-ncmoe N`. The experimental
`feat/q4-g64-kv` surface adds `-ctk q4_g64 -ctv q4_g64` and `--kv-headroom-warn/-block/-force`.

## Benchmarks

These are reported measurements from the reference machine. Raw logs are not published in this repository, and the repo does not contain a turnkey harness for every row. Conditions are included so the numbers are not read as universal.

| Measurement | Result | Conditions |
|---|---|---|
| Needle retrieval | 6/6 | `build2`; `v2.0` model; 15K-word haystack, depths 0.1-0.9, temp 0.0 |
| WikiText-2 perplexity | 6.1745 +/- 0.14889 | `build2`; 40x512 chunks; `-fa on`; default KV |
| Serve decode, 32K ctx | 82.73 t/s median (79.4-89.6) | `build2`; 5 prompts x 300 tok x 2 reps; MTP n-max 2; accept 0.6703 |
| Serve decode, 65K ctx | 64.16 t/s median (51.11-72.48) | Mixed-KV build (`FA_ALL_QUANTS=ON`); `-ctk q8_0 -ctv q4_0`; MTP n-max 2; accept 0.7675 |
| MoE 26B-A4B decode | 94-117 t/s | `build3` (historical); Q4_K_M; `-ncmoe 4`; full residency collapses at the VRAM ceiling |
| NVFP4 KV decode / KLD | 88.6 t/s / 0.0151 | `build3` (historical); matched `nvfp4` K/V; validated at or below 96K |

Context capacity is not recall. The 196K figure is a placement result on a 16 GB card, not a deep-recall measurement; the needle result above is the largest verified retrieval window recorded for this tree.

## Repository layout

| Path | What |
|---|---|
| `common/speculative.cpp` | MTP carryover reset and draft-and-verify plumbing |
| `common/arg.cpp` | CLI surface for MTP, KV, SER, MoE placement, and streaming |
| `include/llama.h`, `src/llama-context.cpp` | MTP hidden-state API and context wiring |
| `src/models/delta-net-base.cpp` | Gated-delta-net conv-state snapshot bound fix |
| `common/fit.cpp` | MoE-aware `--fit` margin cap |
| `ggml/src/ggml-cuda/mma.cuh` | Blackwell block-scaled MMA path |
| `ggml/src/ggml-cuda/mmq-config-blackwell.cuh` | Blackwell MMQ configuration |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu` | NVFP4 MMQ instance |
| `benchmarks/benchmark_kv_stream.py` | Adaptive KV streaming sweep driver |
| `BUILD2_GOLDEN.md` | Recorded build2 toolchain and configure values |

## Adaptive KV streaming

`--kv-stream-stage-mib N` keeps the authoritative KV tensors in pinned host memory while a bounded CUDA pool is shared by resident KV pages and a transfer ring. The runtime grows the resident split as budget allows and prefetches later layers while the current layer computes. This is the mechanism that makes a long context fit when weights leave too little VRAM for a complete KV cache.

This is research code. It is tuned for the reference CUDA configuration and one server slot; other models, KV combinations, parallel slots, and non-CUDA backends are not validated. See [`benchmarks/README.md`](benchmarks/README.md) for the sweep driver and its output format.

## Credits

Den builds on open work.

**Foundation:** [llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml, CUDA backend, base engine), [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (expert offload, speculative decode, MXFP4/NVFP4 lineage), [Qwen3.8](https://github.com/QwenLM/Qwen3.8) (model architecture).

**KV, quantization, and prediction:** [BeeLlama](https://github.com/Anbeeld/beellama.cpp), [Lobo](https://github.com/alectodescent/lobo-qwen38-16gb), [unsloth](https://github.com/unslothai/unsloth), [ISTA-DASLab](https://github.com/IST-DASLab) (GSQ + RCO), [adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming), [KVMem](https://github.com/1314521gjy/ninfer-fusion-kvmem).

**Speculative decode and MoE:** [gem16](https://github.com/Danmoreng/gem16), [Strata](https://github.com/Niko1221/Strata).

**Kernels and hardware:** [CUTLASS](https://github.com/NVIDIA/cutlass), [sass-king](https://github.com/florianmattana/sass-king).

## License

MIT. See [`LICENSE`](LICENSE). This fork builds on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
