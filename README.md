# den_llama.cpp -- Den Engine

**Sovereign NVFP4 inference on consumer Blackwell.**
Single-engine super-fork: union of mainline llama.cpp + ik_llama.cpp + beellama.cpp.
OMMA.SF.16864 tensor cores. 108 proprietary commits ahead of upstream.

---

## Not a mirror

This fork adds **proprietary, silicon-specific features** absent from upstream llama.cpp:

| Capability | Upstream | den_llama.cpp |
|------------|:--------:|:-------------:|
| NVFP4 weight inference (OMMA.SF.16864) | -- | Direct OMMA path, E2M1+UE4M3 |
| NVFP4 KV cache (K8V8, lossless) | -- | KLD=0, cos=1.0 at 8K context |
| KVarN KV cache (variance-normalized) | -- | 2.72x coherent compression |
| Expert offloading (ncmoe) | -- | active set ~4.7 GB on 16 GB |
| Precision tail (256-token F32 ring) | -- | accuracy gate at tile region |
| Data-driven UE4M3 scale LUT | -- | cap 1.5, zero KLD drift |
| MTP K=2 spec decode | -- | full transformer draft head |
| Persistent kernel (multi-SM) | -- | warp-specialized OMMA, TDR-aware |
| ccache + -j8 | -- | 12 min full rebuild |
| Windows native build | -- | pip nvcc + Ninja + MSVC |
| In-process KLD accuracy gate | -- | ctypes, 5 hard metrics |
| Tri-vector gate (accuracy/speed/context) | -- | automated CI guard |

---

## Tri-vector gate

| Vector | Metric | Status |
|--------|--------|--------|
| **Accuracy** | NVFP4 KV KLD=0, cos=1.0 at 8192 context (7840 tile positions) | Verified at 1K/2K/4K/8K |
| **Speed** | 35B tg64 >= 179.57 tok/s | Golden rule CI guard |
| **Context** | 64K+ via Sparse-VMM | Core wired, KV allocation routing pending |

**Gate tooling:** `tools/gate_accuracy_kv.py` (dual-context ctypes), `tools/gate_accuracy_context_scaling.py`, `tools/coherence_gate.py`, `tools/regression_baseline.py`, `tools/repro_check.py`.

---

## Verified models

| Model | Format | Status |
|-------|--------|--------|
| Ornith 9B | NVFP4 GGUF | Coherent, OMMA path |
| Ornith 35B | NVFP4 GGUF | Coherent, expert offload |
| Ornith 35B | Q8_0 GGUF | Reference oracle |
| Gemma 4 12B | NVFP4-FP8 GGUF | PASS |
| Gemma 4 26B | NVFP4 safetensors | Pending .den convert |

---

## Quick build

```cmd
C:\Users\james\Desktop\build_now.bat
```

**Manual:**
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build --config Release -j8
```

Prerequisites: CUDA 13.3, sm_120a GPU (RTX 5070 Ti / GB203), Ninja, MSVC 2022.

See [docs/build.md](docs/build.md) for the upstream build guide (all backends).

---

## Key documents

| Document | Purpose |
|----------|---------|
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | NVFP4 KV accuracy methodology, gate definitions, regression infrastructure |
| [plans/jaunty-seeking-wren.md](plans/jaunty-seeking-wren.md) | Master optimization plan: 30+ silicon exploits, blocker matrix, .den roadmap |
| [CLAUDE.md](CLAUDE.md) | Project instructions for AI agents (points to AGENTS.md) |
| [AGENTS.md](AGENTS.md) | Contributor guidelines |

---

## Architecture

```
den_llama.cpp (this repo)
├── ggml/src/ggml-cuda/
│   ├── fattn-nvfp4-kv.cu/cuh          # NVFP4 KV cache: 4 quant kernels + fused attention
│   ├── mma.cuh                         # OMMA PTX instrinsics via mma.sync.kind::mxf4nvf4.4X
│   ├── mmq.cuh                         # Quantized matmul dispatch (NVFP4 -> OMMA)
│   ├── den-rt-expert-router.cu/cuh     # RT Core expert router (tiers 2+3)
│   ├── den_expert_stage.cu             # CPU L3-resident expert staging + Markov predictor
│   ├── sparse-vmm.cu/cuh               # Sparse VMM: cuMemAddressReserve + cuMemCreate + cuMemMap
│   └── topk-moe.cu                     # Warp-level top-K MoE gating
├── src/
│   ├── llama-context.cpp               # Auto-enable NVFP4, sparse VMM, memory hooks
│   ├── llama-graph.cpp                 # MoE FFN graph build
│   └── llama-model.cpp                 # Model loading
├── tools/
│   ├── gate_accuracy_kv.py             # Primary KLD accuracy gate
│   ├── coherence_gate.py               # 6-gate NVFP4 vs Q8_0 oracle
│   ├── regression_baseline.py          # Baseline DB + regression detection
│   └── repro_check.py                  # One-shot reproducibility check
└── .github/workflows/
    └── bench-guard.yml                 # CI: tg64 < 184 -> fail
```

---

## Silicon target

| Property | Value |
|----------|-------|
| **GPU** | RTX 5070 Ti, GB203-300-A1, 70 SMs, 16 GB GDDR7 |
| **CUDA** | 13.3.33, sm_120a |
| **Tensor cores** | OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X |
| **SMEM** | 99 KB/block |
| **Key ISA** | cp.async.bulk.tensor (TMA), Thread Block Clusters (max 8), DSMEM, no WGMMA/tcgen05/TMEM |

---

## .den Format

`.den` is the project's native weight format — a 160B NULLGLASS tile container replacing GGUF
as the primary storage and dispatch layer. Engineered specifically for Blackwell tensor cores.

**Tile layout (160 bytes):**
- 128B E2M1 nibbles (256 elements × 4 bits)
- 16B UE4M3 block scales (one per 16-element group)
- 4B tile RMS norm (float32)
- 2B dispatch byte + K-stride
- 2B holographic parent pointer (for differential updates)
- 8B reserved (CRC-8, generation-ID, precision tier)

**Why .den:**
- **OMMA-native.** Tiles decompress directly into tensor core register fragments.
  No GGUF block_nvfp4 repack. Zero CPU conversion at runtime.
- **Per-tile precision tiers.** Each tile carries its own format dispatch byte —
  F16, BF16, NVFP4, Q8_0, or skip — selected at quantize time per tensor sensitivity.
- **Differential updates.** Only changed tiles need re-downloading between model versions.
  Holographic parent pointer chains tiles across versions. 14 GB → 700 MB delta.
- **Universal object.** Future: compute graphs, KV state, LoRA adapters, modality descriptors
  in the same container. GGUF for weights; `.den` for everything else.

**Integration status:**
- `DONE` — Direct OMMA NULLGLASS path (loads 160B tiles into OMMA registers, zero PTX change)
- `DONE` — `.den` loader (`src/llama-den-loader.cpp`) with tensor inventory + slot mapping
- `TODO` — Model-loading detection (`.den` extension → route to `den_loader`, ~50 lines)
- `TODO` — Per-tensor precision tier dispatch (jashepp-style 3-tier: F16/Q8_0/NVFP4)
- `TODO` — Differential tile updates via holographic parent pointer + tile generation-ID

GGUF remains supported for compatibility. `.den` is the performance path.

---

## License

MIT (same as upstream llama.cpp)
