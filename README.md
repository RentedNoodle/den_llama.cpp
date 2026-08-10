# den_llama.cpp

### The execution substrate for Project Den

den_llama.cpp is the neural execution engine behind [Project Den](https://github.com/RentedNoodle/Project_Den), an open research project exploring persistent AI cognition on local hardware.

**Project Den is the cognitive architecture. This repo is the machine underneath it.**

It started as a llama.cpp fork. It's becoming a heterogeneous neural runtime. One that treats compute, memory, state, and model execution as a single schedulable system on constrained consumer hardware.

The immediate question is practical: **how much persistent AI cognition fits in one consumer GPU?**

---

## Demonstrated

What's built, tested, and verified:

| What | Detail |
|------|--------|
| **NVFP4 KV cache** | [KLD](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)=0, [cos](https://en.wikipedia.org/wiki/Cosine_similarity)=1.0 vs F32 reference through 64K context. 5 models pass. Hybrid K8V8 with 1024-token F32 precision tail. |
| **OMMA.SF.16864 MoE FFN** | Native Blackwell [tensor core](https://en.wikipedia.org/wiki/Tensor_core) path for [MoE](https://en.wikipedia.org/wiki/Mixture_of_experts) expert weights. Role-gated: OMMA for MoE FFN only, soft-GEMV for attention/GDN. |
| **MoE expert offloading** | ncmoe: active expert set ~4.7 GB on 16 GB card. 3-tier staging (static/selective/deferred). |
| **Sparse virtual memory** | [cuMemAddressReserve](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-memory-management) + cuMemCreate + cuMemMap. Growth hooks wired. |
| **Direct OMMA NULLGLASS path** | 160B tiles load byte-for-byte into OMMA B-fragment registers. Zero GGUF dequant overhead. |
| **Precision tail** | 1024-token sliding F32 ring buffer. Runtime-configurable via `DEN_NVFP4_KV_TAIL`. |
| **Accuracy gates** | In-process dual-context KLD/cosine measurement. 9 verification tools. Regression baseline DB. |
| **Thread Block Clusters** | Confirmed on sm_120a (test_cluster_sm120.cu: 0xDEAD). [TMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-access) + mbarrier confirmed. |
| **ccache + -j8** | 12 min full rebuild. CUDA_SEPARABLE_COMPILATION ON. |

## In development

| What | Status |
|------|--------|
| **Conditional [CUDA](https://en.wikipedia.org/wiki/CUDA) graphs for MoE** | Designed (91h estimate). Device-side expert dispatch without stream sync. |
| **Split-K soft-GEMV** | Designed. Target 80+ tok/s on GDN attention from 38.56 baseline. |
| **128-token attention sink** | KVSink audit found zero sink tokens. Adding 128-token F32 sink buffer + sink bias. |
| **Direct .den model loading** | Loader exists. Model-loading detection + precision tier dispatch TODO. |
| **128K context validation** | Running. 1K–64K all pass. |

## Deferred

These are architectural targets, not current capabilities:

- Multimodal heads (vision, audio, diffusion, 3D)
- Full Den Runtime abstraction
- Universal .den object format (tensors + graphs + state + pipelines)
- Differential .den model updates
- GPU System Processor (GSP) offload (no public SDK)

---

## Silicon exploits

This engine targets **one GPU** as a laboratory: RTX 5070 Ti / [GB203](https://en.wikipedia.org/wiki/Blackwell_(GPU_architecture)) / 70 SMs / 16 GB [GDDR7](https://en.wikipedia.org/wiki/GDDR7_SDRAM).

Exploits that go beyond standard llama.cpp CUDA:

| Exploit | Hardware | Status |
|---------|----------|--------|
| OMMA.SF.16864 role-gated dispatch | Tensor cores (280) | Running |
| Dual Copy Engine concurrent DMA | CE0 + CE1 | Infra ready |
| L2 cache persistence (cuMemAdvise) | 48 MB L2 | Infra ready |
| RT Core MoE expert routing | RT cores (70) | Tiers 2+3 ported |
| CPU L3 Claustrum orchestrator | 96 MB AMD [3D V-Cache](https://en.wikipedia.org/wiki/3D_V-Cache) | Running (0.8B model, 0 VRAM) |
| [PCIe 4.0](https://en.wikipedia.org/wiki/PCI_Express) atomics (fetch_add/CAS) | PCIe BAR | Moderate |
| TMU texture cache weight prefetch | TMUs (280) | Lab |
| Sparse VMM | GPU page tables | Core wired |

---

## Tri-vector gate

| Vector | Metric | Status |
|--------|--------|--------|
| **Accuracy** | NVFP4 KV matches F32 reference | 1K–64K CHECKMARK. 128K test running. |
| **Speed** | 35B tg64 ≥ 184 tok/s | Golden rule CI guard |
| **Context** | 256K target | Gated on attention sink + Sparse-VMM KV routing |

**Gate tooling:** `tools/gate_accuracy_kv.py`, `tools/gate_accuracy_context_scaling.py`, `tools/coherence_gate.py`, `tools/regression_baseline.py`, `tools/repro_check.py`.

---

## Capabilities vs upstream

| Capability | Upstream | Here |
|------------|:--------:|:----:|
| NVFP4 weight inference (OMMA.SF.16864) | -- | Direct OMMA path, role-gated |
| NVFP4 KV cache (K8V8 + precision tail) | -- | Matches F32 reference through 64K |
| MoE expert offloading (ncmoe) | -- | Active set ~4.7 GB on 16 GB |
| KVarN KV cache ([Hadamard](https://en.wikipedia.org/wiki/Hadamard_transform) pre-transform) | -- | 2.72× coherent compression |
| RT Core expert routing | -- | BVH nearest-neighbor, tiers 2+3 |
| Sparse virtual memory | -- | cuMemAddressReserve + cuMemCreate |
| MTP K=2 spec decode | -- | Full transformer draft head |
| Thread Block Clusters + DSMEM | -- | Confirmed sm_120a |
| In-process KLD accuracy gate | -- | ctypes dual-context, 9 tools |
| Windows native build (pip nvcc 13.3) | -- | Ninja + MSVC, ccache |

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
| **Tensor cores** | 280 (70 SM × 4), OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X |
| **SMEM** | 99 KB/block |
| **L2 cache** | 48 MB (702 KB/SM) |
| **Confirmed** | TMA (cp.async.bulk.tensor), Thread Block Clusters (max 8), DSMEM, barrier.cluster |

---

## .den Format

`.den` is the project's native weight format: a 160B NULLGLASS tile container that maps directly to OMMA register fragments. GGUF stays for compatibility; `.den` is the native path.

**Current (implemented):**
- 128B E2M1 nibbles + 16B UE4M3 block scales + metadata
- Direct OMMA NULLGLASS path: tiles load into registers with zero dequant
- `.den` file loader with tensor inventory + slot mapping

**In progress:**
- Model-loading autodetect (`.den` extension → route to den_loader)
- Per-tensor precision tier dispatch (F16/Q8_0/NVFP4 per tensor sensitivity)

**Future (designed, not built):**
- Differential tile updates via holographic parent pointer
- Universal object: compute graphs, KV state, adapters, modality descriptors

---

## Architecture

```
Project Den (cognitive thesis)
      │
      ▼
Den Runtime (execution substrate, emerging)
      │
      ▼
den_llama.cpp (current engine host. YOU ARE HERE)
      │
      ▼
Blackwell GPU (GB203 / sm_120a / 16 GB)
```

Long-term: llama.cpp becomes a compatibility backend. Den Runtime becomes the host. `.den` becomes the universal object format. Today, den_llama.cpp is where the rubber meets silicon.

---

## Relationship to Project Den

```
Project_Den
    │
    ├── cognition (memory, identity, agency, affect)
    ├── neural models (Cortex 35B, Claustrum 0.8B, Draft 2B)
    └── multimodal systems (designed, deferred)
              │
              ▼
       den_llama.cpp  ← YOU ARE HERE
              │
              ├── tensor execution (OMMA, soft-GEMV)
              ├── KV state (NVFP4 cache)
              ├── MoE residency (expert staging)
              ├── memory (sparse VMM, L2 persistence)
              └── GPU scheduling
```

Project Den asks the cognitive question. This repo builds the body. Eventually the boundary moves. That's intentional.

---

## Key documents

| Document | Purpose |
|----------|---------|
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | NVFP4 KV methodology, context scaling results, regression infra |
| [AGENTS.md](AGENTS.md) | Contributor guidelines |

---

## Source map

```
den_llama.cpp
├── ggml/src/ggml-cuda/
│   ├── fattn-nvfp4-kv.cu/cuh          # NVFP4 KV cache: quant kernels + fused attention
│   ├── mma.cuh                         # OMMA PTX intrinsics
│   ├── mmq.cuh                         # Quantized matmul dispatch (NVFP4 → OMMA)
│   ├── den-rt-expert-router.cu/cuh     # RT Core expert router (BVH, tiers 2+3)
│   ├── den_expert_stage.cu             # CPU L3 expert staging + Markov predictor
│   ├── sparse-vmm.cu/cuh               # cuMemAddressReserve + cuMemCreate + cuMemMap
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
│   └── measure_vram_context.py         # VRAM compression metering
├── docs/
│   └── VERIFICATION.md                 # Full verification methodology
└── .github/workflows/
    └── bench-guard.yml                 # CI: tg64 < 184 → fail
```

---

## Status legend

| Marker | Meaning |
|--------|---------|
| **[*]** VERIFIED | Implemented, built, tested, gate-passed |
| **[+]** IMPLEMENTED | Code exists, broader validation ongoing |
| **[~]** EXPERIMENTAL | Working research code, not production-stable |
| **[ ]** DESIGNED | Architecture/spec exists, implementation pending |
| **[-]** DEFERRED | Not being built yet |

Research engine. Active development. Architecture deliberately ahead of implementation in several areas. Check verification docs before treating planned components as available.

---

## Acknowledgments

**[llama.cpp](https://github.com/ggerganov/llama.cpp)**: The foundation. ggml, CUDA backend, quantized inference.

**[BeeLlama](https://github.com/Intelligent-Internet-Of-Engineers/beellama.cpp)**: KVarN KV cache, 1M context validation, precision tail research.

**[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)**: Expert offloading (ncmoe), speculative decode, NVFP4 inference patterns.

**[sass-king](https://github.com/florianmattana/sass-king)**: Blackwell SASS corpus, OMMA instruction verification.

**[quadbit](https://github.com/quadbit-org/quadbit)**: TMA + mbarrier sm_120a validation (BSD-3).

**[BlackweLLM](https://github.com/blackwellm-org/blackwellm)**: CUDA graph research for LLM inference (MIT).

---

## License

MIT (same as upstream llama.cpp)
