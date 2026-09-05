# den_llama.cpp — peak local-inference fork

Upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (`master` here tracks a pinned upstream commit plus the patch stack below — rebase, don't fork-diverge).
Live work happens in [den_llama.cpp-work](https://github.com/RentedNoodle/den_llama.cpp-work); retired engine history lives in [the-den](https://github.com/RentedNoodle/the-den) and the `den-legacy` branch. Den additions land here only if they don't break the gates.

## What this is

A llama.cpp build tuned for one job: **Qwen3.8-27B at 196K context with native MTP speculative decoding on a single 16 GB GPU** (RTX 5070 Ti, Blackwell GB203). Pinned at `3231ee89` (2026-08-30).

Proven numbers (this file's gates, not promises): 59 t/s MTP2 (+50% over serial), 69–82 t/s with ship flags, needle 6/6, toolcall 8/8, livebench 12/12, coherence 4/4. Full results: [model card](https://huggingface.co/RentedNoodle/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored).

## Build (Windows, CUDA 13.3)

```powershell
cmake -S . -B build2 -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build2 --config Release -j 8
```

`build2/` is gitignored. `-DGGML_CUDA_FA_ALL_QUANTS=ON` is required for quantized KV cache on the Qwen3.5 hybrid arch (mainline prebuilts lack it and silently fall back to CPU).

## Ship flags (RTX 5070 Ti 16 GB)

```powershell
llama-server -m <model>.gguf --ctx-size 196608 -fa on -ctk q4_0 -ctv q4_0 `
  -ngl 99 -b 1024 -ub 1024 -np 1 -t 8 --jinja --fit off `
  --reasoning-budget 256 --spec-type draft-mtp --spec-draft-n-max 2 `
  --spec-draft-backend-sampling
```

Non-negotiables found the hard way: `--fit off` (fit-on + MTP randomly drops to 9–20 t/s), `n-max 2` (3/4 slower despite higher acceptance), `q4_0` KV (`iq4_nl` has no fast kernel here), exactly one server per GPU.

## Gates (run before any merge)

`needle` (retrieval @ depth) · `toolcall` (name + args + must-not-fire + chains) · `coherence` (multi-turn) · `speed` (MTP ladder + temp/KV A-Bs, medians-of-3, baseline re-measured at end). Prompts + raw logs ship per release (see model repo `evals/`).

## Patch stack (top of pinned upstream)

- GDN-O1→MTP feed (`gdn_replay_get_hidden`, O(1) GDN into MTP)
- HybridKV per-head residency, adaptive-KV streaming
- beellama reasoning-loop guard (force-close repetitive hidden-reasoning loops)
- Gated-DeltaNet PDL sync + q/k L2-normalize fixes

## License

Upstream GPL-adjacent terms apply (see upstream LICENSE); Den additions same terms. Model weights are separate artifacts with their own licenses.
