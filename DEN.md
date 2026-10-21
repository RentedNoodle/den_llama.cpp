# den_llama.cpp - local-inference fork

Upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (`main` tracks a pinned upstream commit plus the patch stack below; rebase rather than fork-diverge).
Private development work is not linked from this repository. Retired engine history remains on the `den-legacy` branch. Den additions land here only if they don't break the gates.

## What this is

A llama.cpp build tuned for one job: **Qwen3.8-27B at 196K context with native MTP speculative decoding on a single 16 GB GPU** (RTX 5070 Ti, Blackwell GB203). The stable reference is `abd589d4c`; see `BUILD2_GOLDEN.md` for the recorded build.

Published results: 69-82 t/s with ship flags, needle 6/6. Model: [model card](https://huggingface.co/RentedNoodle/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored).

## Build (Windows, CUDA 13.3)

```powershell
cmake -S . -B build2 -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build2 --config Release -j 8
```

`build2/` is gitignored. The recorded stable build used `GGML_CUDA_FA_ALL_QUANTS=OFF` with matching `q4_0` K/V. `ON` enables mixed quantized KV pairs, for example `q8_0` K with `q4_0` V; see `BUILD2_GOLDEN.md`.

## Ship flags (RTX 5070 Ti 16 GB)

```powershell
llama-server -m <model>.gguf --ctx-size 196608 -fa on -ctk q4_0 -ctv q4_0 `
  -ngl 99 -b 1024 -ub 1024 -np 1 -t 8 --jinja --fit off `
  --reasoning-budget 256 --spec-type draft-mtp --spec-draft-n-max 2 `
  --spec-draft-backend-sampling
```

Non-negotiables found the hard way: `--fit off` (fit-on + MTP randomly drops to 9–20 t/s), `n-max 2` (3/4 slower despite higher acceptance), `q4_0` KV (`iq4_nl` has no fast kernel here), exactly one server per GPU.

## Gates (run before any merge)

`needle` (retrieval at depth), `toolcall` (name + args + must-not-fire + chains), `coherence` (multi-turn), and `speed` (MTP ladder plus temperature/KV A-Bs, medians of 3). Prompts and raw logs are not published in this repository.

## Patch stack

- kv-stream series (~47): adaptive-KV streaming merge + Windows/WDDM fixes, device-resident pool under UVM, bench automation
- seven-stack (5): GDN O(1) record/replay across 48 layers, HybridKV head-aware classifier (kvarn4 resident / q8 streamed), NIAH gates
- shippable-q4 lane: 196K q8_0/q4_0 resident configs
- GDN-O1→MTP feed (`gdn_replay_get_hidden`), HybridKV per-head policy + sink pin, PDL sync + q/k normalize fixes, DFlash NaN defensive fix
- FA_ALL_QUANTS build flag (quantized KV on hybrid arch)

Additional donor work is tracked on development branches and is not part of the stable reference.

## License

MIT. See `LICENSE`. Model weights are separate artifacts.
