# den_llama.cpp — Neural engine beneath Project Den

**Blackwell-native inference and execution engine. NVFP4 OMMA. MoE offloading. Persistent state. The emerging Den Runtime.**

This is the computational substrate for [Project Den](https://github.com/RentedNoodle/Project_Den) — an open research project exploring persistent AI cognition on local hardware.

It started as a llama.cpp fork. It's becoming something else.

The immediate problem is practical: **how much persistent AI cognition fits in one consumer GPU?**

The longer problem is bigger: **can one runtime provide the computational body for persistent machine cognition across language, vision, audio, diffusion, and 3D?**

This repo is where that experiment meets silicon.

---

## What makes it different

Not a collection of CUDA patches around llama.cpp. A research engine for:

- Blackwell-native tensor execution (OMMA.SF.16864)
- NVFP4 inference — weights AND KV cache
- MoE expert residency and offloading
- GPU memory orchestration (sparse VMM, L2 persistence)
- Persistent neural state across sessions
- Native `.den` object format
- Deterministic accuracy verification
- Hardware-aware scheduling

Current implementation is heavily optimized for **RTX 5070 Ti / GB203 / sm_120a / 16 GB GDDR7**. That GPU is the laboratory, not just the deployment target.

---

## Architecture

```
                       Project Den
                            │
                     Cognitive Runtime
                            │
                   ┌────────▼────────┐
                   │  Den Runtime    │
                   │ tensors / state │
                   │ memory / graphs │
                   │ scheduling      │
                   └────────┬────────┘
                            │
                  ┌─────────▼─────────┐
                  │ den_llama.cpp     │
                  │ current host      │
                  └─────────┬─────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
           LLM             MoE         Future heads
             │              │        vision/audio/3D...
             └──────────────┼──────────────┘
                            │
                    Blackwell GPU
```

Long-term: separate reusable execution substrate from modality-specific heads.

---

## Tri-vector gate

| Vector | Metric | Status |
|--------|--------|--------|
| **Accuracy** | NVFP4 KV KLD=0, cos=1.0 vs F32 oracle | 1K–64K all CHECKMARK. 128K test running. |
| **Speed** | 35B tg64 ≥ 184 tok/s | Golden rule CI guard (bench-guard.yml) |
| **Context** | 256K target | Gated on attention sink + Sparse-VMM KV routing |

**Gate tooling:** `tools/gate_accuracy_kv.py`, `tools/gate_accuracy_context_scaling.py`, `tools/coherence_gate.py`, `tools/regression_baseline.py`, `tools/repro_check.py`.

---

## Capabilities vs upstream

| Capability | Upstream | den_llama.cpp |
|------------|:--------:|:-------------:|
| NVFP4 weight inference (OMMA.SF.16864) | -- | Direct OMMA path, E2M1+UE4M3 |
| NVFP4 KV cache (K8V8) | -- | KLD=0, cos=1.0 at 64K context |
| KVarN KV cache (variance-normalized) | -- | 2.72× coherent compression |
| Expert offloading (ncmoe) | -- | Active set ~4.7 GB on 16 GB |
| Precision tail (1024-token F32 ring) | -- | Accuracy gate at tile region |
| Data-driven UE4M3 scale LUT | -- | Cap 1.5, measured 104K blocks |
| MTP K=2 spec decode | -- | Full transformer draft head |
| ccache + -j8 | -- | 12 min full rebuild |
| Windows native build | -- | pip nvcc 13.3 + Ninja + MSVC |
| In-process KLD accuracy gate | -- | ctypes, 5 hard metrics |
| Tri-vector gate | -- | Automated CI guard |

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

---

## Silicon target

| Property | Value |
|----------|-------|
| **GPU** | RTX 5070 Ti, GB203-300-A1, 70 SMs, 16 GB GDDR7 |
| **CUDA** | 13.3.33, sm_120a |
| **Tensor cores** | OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X |
| **SMEM** | 99 KB/block |
| **Key ISA** | cp.async.bulk.tensor (TMA), Thread Block Clusters (max 8), DSMEM |
| **Confirmed** | Clusters (test_cluster_sm120.cu: 0xDEAD), TMA+mbarrier (quadbit production use) |

---

## .den Format

`.den` is Den's native object format. Today it's an optimized NVFP4 weight container. The design extends further.

**Tile layout (160 bytes):**
- 128B E2M1 nibbles (256 elements × 4 bits)
- 16B UE4M3 block scales (one per 16-element group)
- 4B tile RMS norm (float32)
- 2B dispatch byte + K-stride
- 2B holographic parent pointer (differential updates)
- 8B reserved (CRC-8, generation-ID, precision tier)

**Why .den:**
- **OMMA-native.** Tiles decompress directly into tensor core register fragments. No GGUF block_nvfp4 repack. Zero CPU conversion at runtime.
- **Per-tile precision tiers.** Each tile carries its own format dispatch byte — F16, BF16, NVFP4, Q8_0, or skip — selected at quantize time per tensor sensitivity.
- **Differential updates.** Only changed tiles re-download between model versions. Holographic parent pointer chains tiles across versions. 14 GB → 700 MB delta.
- **Universal object.** Future: compute graphs, KV state, LoRA adapters, modality descriptors in the same container. GGUF for weights; `.den` for everything else.

**Integration:**
- `DONE` — Direct OMMA NULLGLASS path (loads 160B tiles into OMMA registers)
- `DONE` — `.den` loader with tensor inventory + slot mapping
- `TODO` — Model-loading detection (`.den` extension → route to den_loader)
- `TODO` — Per-tensor precision tier dispatch (F16/Q8_0/NVFP4)
- `TODO` — Differential tile updates via holographic parent pointer

GGUF stays for compatibility. `.den` is the performance path.

---

## Key documents

| Document | Purpose |
|----------|---------|
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | NVFP4 KV accuracy methodology, context scaling results, regression infra |
| [AGENTS.md](AGENTS.md) | Contributor guidelines |

---

## Source map

```
den_llama.cpp
├── ggml/src/ggml-cuda/
│   ├── fattn-nvfp4-kv.cu/cuh          # NVFP4 KV cache: quant kernels + fused attention
│   ├── mma.cuh                         # OMMA PTX instrinsics (mma.sync.kind::mxf4nvf4.4X)
│   ├── mmq.cuh                         # Quantized matmul dispatch (NVFP4 → OMMA)
│   ├── den-rt-expert-router.cu/cuh     # RT Core expert router
│   ├── den_expert_stage.cu             # CPU L3-resident expert staging + Markov predictor
│   ├── sparse-vmm.cu/cuh               # Sparse VMM: cuMemAddressReserve + cuMemCreate + cuMemMap
│   └── topk-moe.cu                     # Warp-level top-K MoE gating
├── src/
│   ├── llama-context.cpp               # Auto-enable NVFP4, sparse VMM, memory hooks
│   ├── llama-graph.cpp                 # MoE FFN graph build
│   └── llama-model.cpp                 # Model loading
├── tools/
│   ├── gate_accuracy_kv.py             # Primary KLD accuracy gate
│   ├── gate_accuracy_context_scaling.py # Multi-context scaling
│   ├── coherence_gate.py               # 6-gate NVFP4 vs Q8_0 oracle
│   ├── regression_baseline.py          # Baseline DB + regression detection
│   ├── repro_check.py                  # One-shot reproducibility check
│   ├── test_needle_retrieval.py        # Needle-in-haystack functional test
│   ├── test_negative_gate.py           # Corruption rejection gate
│   ├── measure_vram_context.py         # VRAM compression metering
│   └── nsys_profile.bat               # Nsight Systems profiling
├── docs/
│   └── VERIFICATION.md                 # Full verification methodology
└── .github/workflows/
    └── bench-guard.yml                 # CI: tg64 < 184 → fail
```

---

## Relationship to Project Den

```
Project_Den
    │
    ├── cognition
    ├── memory
    ├── identity
    ├── agency
    ├── relationships
    └── multimodal systems
              │
              ▼
       den_llama.cpp  ← YOU ARE HERE
              │
              ├── tensors
              ├── kernels
              ├── KV state
              ├── MoE
              ├── memory
              └── GPU execution
```

Project Den asks the question. The engine builds the body. Eventually these boundaries move. That's intentional.

---

## Status legend

| Badge | Meaning |
|-------|---------|
| 🟢 **VERIFIED** | Implemented, built, tested, gate-passed |
| 🔵 **IMPLEMENTED** | Code exists, broader validation ongoing |
| 🟡 **EXPERIMENTAL** | Working research code, not production-stable |
| 🟣 **DESIGNED** | Architecture/spec exists, implementation pending |
| ⚪ **DEFERRED** | Not being built yet |

Research engine. Active development. Architecture deliberately ahead of implementation in several areas. Check verification docs before treating planned components as available.

---

## License

MIT (same as upstream llama.cpp)
