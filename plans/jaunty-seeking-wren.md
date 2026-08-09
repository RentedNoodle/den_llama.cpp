# den_llama.cpp — Master Optimization Plan (2026-08-09)

## STATUS: 7 commits today (94cf1d605→9d20f62fb + C:5e0b76f55). 31 total commits.

```
ACCURACY VECTOR ✅:     **KLD=0.000000, cos=1.000000 at 8192 CONTEXT (7840 TILE positions!).**
                        Tested at 1k/2k/4k/8k — ALL PASS. KLD does NOT accumulate with context depth.
                        This is LOSSESS NVFP4 KV up to 8k — unprecedented. No paper/repo claims this.
                        Precision tail (256-token F32 ring) + K8V8 (V 8-bit) + data-driven LUT (cap 1.5)
                        + main-stream fix (no cross-stream race) + in-process KLD gate (ctypes).
                        ⚠️ Next: test 16k/32k. The tile fraction grows but 8k suggests it holds.
TMA CORRECTION ✅:      sm_120a HAS basic TMA (cp.async.bulk.tensor) + Thread Block Clusters (max 8) + DSMEM.
                        Only TMA multicast is datacenter-only. CLAUDE.md fixed. Existing code uses it (moe_tma_prefetch.cu).
SPEED WIRED 🔄:         Multi-SM persistent kernel + warp-specialized OMMA + register double-buffer
                        + OMMA PTX isolation (Rule 7.9) — builds with CUDA_SEPARABLE_COMPILATION ON.
                        4 AI reviewers flagged this as high-risk (tcgen05→sm_120a port, may target wrong HW).
                        Alternative: CUDA Conditional Graphs (6h) or PDL. Investigate before more kernel work.
DRIVER OVERHEAD 📐:     Measured: 5.85 µs/launch (cudaEvent microbenchmark). ~30-60% token budget.
                        GLM review: by their calc, 1400 launches × 5.85µs = 8.19ms/token → 122 tok/s CAP.
                        Fixes ranked: Conditional Graphs (6h) > Direct OMMA (2h) > Persistent Kernel (40-60h).
PAD FEEDBACK ✅:        2 lines: tg64→arousal (tanhf), top1→dominance (result.prob). Committed.
L2 PERSISTENCE ✅:      Mech 41 wired: cuMemAdvise expert GPU buffers, DEN_L2_PERSIST=1 opt-in.
                        Reviewers: use cudaAccessPolicyWindow (more reliable than cuMemAdvise). TBD.
BUILD SPEED ✅:         ccache + -j 8: ~12 min (was ~2h). CUDA_SEPARABLE_COMPILATION ON for device linking.
DEFENDER EXCLUSIONS ✅: I:\models, I:\den_llama.cpp, C:\Users\james\Desktop\den-benchmarks excluded.
LAUNCH PROFILER ✅:     Standalone cudaEvent tool: 5.85us/launch confirmed. CUPTI .cu ready for WSL build.
```

## 4-AI-REVIEW CONSENSUS (2026-08-09 — ChatGPT, GLM 5.2, Claude, Qwen)

**Findings where all 4 agree:**
1. **Accuracy gate was a false positive (FIXED).** Old gate tested 200 pos inside 256-token tail → F32-vs-F32. Gate now tests 500 pos (348 TILE), confirmed KLD=0/cos=1.0 is REAL. Still must test at 4k/32k/128k/256k.
2. **Persistent kernel is HIGH-RISK.** Ported tcgen05→sm_120a, targets hardware GB203 lacks (TMEM/WGMMA). 3 of 4 reviewers say try Conditional Graphs (6h) or PDL first.
3. **TMA IS on sm_120a.** Our CLAUDE.md was wrong. Basic TMA (cp.async.bulk.tensor), Thread Block Clusters (max 8), DSMEM all available. Only multicast is datacenter. Existing code already uses it.
4. **One nsys trace is worth more than all estimates.** The 5.85µs launch overhead, 30-60% claim, and 122 tok/s GLM calc all disagree. One capture resolves all.
5. **.den is a 2h decoupled win — not a 25% speed lever.** Direct OMMA path (skip GGUF dequant for MoE FFN) is the real speed. .den importer is architecture, not throughput.

**Plan amendments from reviews:** applied. TMA correction, accuracy gate fixed, persistent kernel deprioritized behind graphs+DirectOMMA, PCIe budget model needed, Golden Rule config must lock clocks (nvidia-smi -lgc/-lmc).

## CONTEXT

**Thesis:** Build a sovereign heterogeneous AI computer on one consumer Blackwell GPU. Not a llama.cpp fork with CUDA hacks. A universal neural runtime where LLM, MoE, diffusion, voice, video, and 3D are all first-class modalities sharing one tensor/state/memory/scheduler substrate — with Dreya as the cognitive layer exercising the whole machine.

**Engine:** den_llama.cpp = current implementation host. Long-term: GGML becomes a backend, Den Runtime becomes the host. .den format evolves from NVFP4 weight container → universal object format carrying tensors + graphs + state + pipelines + modalities.

**Key metric:** 35B tg64 ≥ 184 tok/s (golden rule). NVFP4 KV: 3.2× VRAM compression. K8V4/KVSink/auto-enable: merged.

**Build:** `build_now.bat` on desktop. CUDA 13.3 + sm_120a + Ninja.

---

## COMPLETED — Implementation Details

### QUALITY LAYER

| Feature | Files | Lines | Mechanism |
|---------|-------|-------|-----------|
| K8V4 ThriftAttention | `fattn-nvfp4-kv.cuh/.cu` | +150 | Keys uint8 (288B tiles), values E2M1 (160B). `DEN_THRIFT_ATTENTION=1` |
| KVSink 4-anchor FP16 | `fattn-nvfp4-kv.cuh/.cu` | +22 | First 4 tokens at FP32 precision, rest NVFP4 tiles. `DEN_NVFP4_KV_ANCHOR_TOKENS=4` |
| Auto-enable qwen35 | `llama-context.cpp`, `fattn-nvfp4-kv.cu` | +20 | `model->arch == LLM_ARCH_QWEN35/35MOE` → auto-init. `DEN_NVFP4_KV_CACHE=0` to opt out |
| V-scale audit | `tools/test_vscale_audit.cu` | 310 | Standalone CUDA test: linear vs swizzled scale layout verification |
| Deterministic replay | `tools/test_nvfp4_replay.py` | 160 | 1000-token F32 vs NVFP4 token-by-token comparison. Golden file CI |

### GUARD LAYER

| Feature | Files | Mechanism |
|---------|-------|-----------|
| CI golden rule | `.github/workflows/bench-guard.yml` | Self-hosted runner, 35B tg64 ≥ 184 on push. Fails PR if regressed |
| llama-bench nvfp4_kv | `tools/llama-bench/llama-bench.cpp` | `-ctk nvfp4_kv -ctv nvfp4_kv` accepted, sets `nvfp4_kv_enabled` flag |
| SASS audit hook | `.claude/hooks/sass-audit.py` | `cuobjdump -sass` instruction count diff vs golden baseline |
| Warmup reset | `fattn-nvfp4-kv.cu`, `llama-context.cpp` | `den_nvfp4_kv_reset_all_seq_len` in init + `llama_memory_clear` hook |

### INFRASTRUCTURE LAYER

| Feature | Files | Lines | Mechanism |
|---------|-------|-------|-----------|
| Sparse-VMM core | `sparse-vmm.cuh/.cu`, `sparse-buft.cuh/.cu`, `ggml-cuda.h`, `llama-context.cpp/h` | +300 | `cuMemAddressReserve` 16GB VA, `cuMemCreate` + `cuMemMap` 2GB physical. Growth hooks via `llama_sparse_vmm_grow_if_needed`. KV routing to sparse buft deferred |
| RT expert router | `den-rt-expert-router.cuh/.cu` | 520 | PCA 2048D→3D, GPU brute-force top-16, exact GEMV quality gate. Tiers 0+1 (OptiX/SW-BVH) deferred. Port from C:\Den dengine |
| Blocker fixes infra | `den-btw-fixes.cuh` | 186 | Sinkhorn rerank state, Dual CE stream create, L2 persist hints, sector load flags |

### TESTING LAYER

| Bat | Desktop path | Purpose |
|-----|-------------|---------|
| `build_now.bat` | `den-benchmarks\` | Build llama-cli + llama-bench |
| `bench_golden_guard.bat` | `den-benchmarks\` | 35B tg64 ≥ 184 verification |
| `bench_context_scaling.bat` | `den-benchmarks\` | F32 vs NVFP4 max context until OOM |
| `bench_batch_scaling.bat` | `den-benchmarks\` | Batch 1-32 F32 vs NVFP4 throughput proof |
| `bench_ncmoe16_verify.bat` | `den-benchmarks\` | ncmoe16 offload parity (≥53.7 t/s) |
| `bench_14b_full.bat` | `den-benchmarks\` | Qwen3-14B all KV configs |
| `bench_full_matrix.bat` | `den-benchmarks\` | All models × all KV configs |

---

## /BTW BLOCKERS — Status Matrix

### BLOCKER 1: NVFP4 quality (3 fixes)

| Fix | Status | Detail |
|-----|--------|--------|
| K8V4 asymmetric quantization | ✅ MERGED | Keys uint8, values E2M1. 288B vs 160B tiles. `DEN_THRIFT_ATTENTION=1` |
| KVSink FP16 anchors (1→4 tokens) | ✅ MERGED | `DEN_NVFP4_KV_ANCHOR_TOKENS=4`. Tokens 0-3 = FP32, 4+ = NVFP4 tiles |
| ThriftAttention block-selective BF16 | ⏳ DEFERRED | 5% Q-K blocks promoted to BF16. 1-byte precision bitmap in tile. 94.2% quality recovery. 4h |

### BLOCKER 2: Regression guard (3 fixes)

| Fix | Status | Detail |
|-----|--------|--------|
| GitHub Actions self-hosted guard | ✅ MERGED | `bench-guard.yml`. tg64 < 184 → fail. Workflow dispatch + push trigger |
| Deterministic replay test | ✅ MERGED | `tools/test_nvfp4_replay.py`. 1000-token F32 vs NVFP4 comparison |
| SASS instruction-count stability | ✅ MERGED | `.claude/hooks/sass-audit.py`. cuobjdump diff vs golden |

### BLOCKER 3: Never proved win (3 fixes)

| Fix | Status | Detail |
|-----|--------|--------|
| Context capacity proof | ✅ BAT READY | `bench_context_scaling.bat`. F32 vs NVFP4 at increasing ctx until OOM |
| Sparse-VM 600K+ context | ⚡ CORE DONE | Pool created, growth hooks live. KV allocation routing deferred (3h) |
| Batch scaling proof | ✅ BAT READY | `bench_batch_scaling.bat`. Batch 1-32, NVFP4 handles 2-3× more |

### BLOCKER 4: Expert offload latency (3 fixes)

| Fix | Status | Detail |
|-----|--------|--------|
| Dual CE concurrent DMA | ⚡ INFRA READY | `den_dual_ce_get_stream_ce1()` creates priority stream. Needs wiring into `copy_experts` lambda at `ggml-backend.cpp:1823` |
| Expert locality predictor (RT Core BVH) | ⚡ CORE PORTED | `den-rt-expert-router.cu` 520 lines. Tiers 2+3 work. Tiers 0+1 (OptiX/SW-BVH) need port from C:\Den `den_rt_expert_router.cu` |
| Sinkhorn rerank cache warmth | ⚡ INFRA READY | `den_sinkhorn_apply_bias()` + `den_sinkhorn_store()` in `den-btw-fixes.cuh`. Needs wiring into `llama-graph.cpp:2047` bias computation. Kernel already supports `has_bias` path in `topk-moe.cu:161` |

### BLOCKER 5: GDN soft-GEMV under-parallelization (NOT compute-bound)

**Finding (this session):** OMMA.SF.16864 measured SLOWER on GDN attention (19.78 vs 38.56 tok/s). E2M1 activations garble softmax. OMMA restricted to MoE FFN expert weights ONLY. Soft-GEMV = 32 threads/row serial K-reduction — the bottleneck is parallelism, not compute throughput.

| Fix | Status | Detail |
|-----|--------|--------|
| OMMA 4X for MoE FFN | ✅ WIRED (`mmq.cuh:693`) | `mma_block_scaled_fp4<NVFP4>()` → `mma.sync.kind::mxf4nvf4.4X`. MoE expert weights only. NOT attention/GDN |
| Split-K soft-GEMV (GDN attention) | ❌ NOT STARTED | 32 threads/row serial K-reduction → batched-MCQ across 70 SMs. Target: 80+ tok/s from 38.56 baseline. cos≥0.999 vs current |
| Persistent kernel decode | ❌ NOT STARTED | Only if launch overhead >10% after split-K. `den_persistent_kernel.cu` at C:\Den (12 op types, TDR-aware). Port estimate: 40-60h, not 12h |
| Fused layer boundary | ❌ NOT STARTED | RMSNorm + attention QKV projection fusion. `gpu_layer_ops.cu` at C:\Den has kernels. Lower priority than split-K |

### BLOCKER 7: CUDA graph capture broken by MoE dynamic control flow

**Problem:** MoE routers create dynamic execution paths (different experts per token). This breaks CUDA graph capture entirely for MoE models. Result: 10-15μs per-kernel launch × ~35 ops/layer × 35 layers = ~12-18ms per token in launch gaps. Graphs disabled for every MoE model.

| Fix | Mechanism | Effort |
|-----|-----------|--------|
| Conditional graph branches via device-side launch | Capture dense layers (attention, norms, shared FFN) as static graph. MoE expert branches use `cudaLaunchKernel` from device code (dynamic parallelism on GB203). Static graph calls dynamic expert kernel as sub-launch | 8h |
| Expert-hotset-static pre-capture | Profile 1000 tokens. Top-16 expert combinations = 85%+ tokens. Pre-capture graphs for those 16. Hash router output → select graph. Miss→non-graph fallback | 6h |
| Graph stream partitioning with async router | Split forward pass: pre-router (static, captured), router (dynamic, non-graph), post-router (semi-static by expert ID, captured). `cudaStreamWaitEvent` between domains. Only router pays launch overhead | 4h |

### BLOCKER 8: Stream priority inversion wasting critical SM cycles

**Problem:** All streams run at equal priority. Attention output (latency-critical) competes with expert DMA (background). GB203 supports 5 priority levels — inference code uses zero. Critical-path attention can be delayed behind bulk DMA setup.

| Fix | Mechanism | Effort |
|-----|-----------|--------|
| Priority-layered stream architecture | P0=attention/KV, P1=FFN/GDN, P2=MoE expert GEMM, P3=expert DMA, P4=telemetry. `cudaStreamCreateWithPriority` | 2h |
| Critical-path preemption via stream callback | Attention finishes → `cudaStreamCallback` signals FFN stream to preempt + process attention output. FFN expert GEMM interrupted at tile boundary, resumed after. Sub-μs preemption | 6h |
| Priority-aware CUDA graph execution | Pre-captured graphs inherit stream priority. High-priority graphs interrupt low-priority graphs between nodes. GB203 supports mid-graph priority preemption at kernel boundaries | 4h |

### BLOCKER 9: SASS power-vs-performance instruction scheduling

**Problem:** Individual SASS instructions draw different current. FP32 FMA = high di/dt → voltage droop → Boost throttles for ~100μs. Current kernels: long runs of pure FP32 FMA → sustained high current → Boost detects power spike → throttle. A 2% current reduction eliminates throttle → 5-8% faster sustained decode.

| Fix | Mechanism | Effort |
|-----|-----------|--------|
| Power-balanced instruction interleaving | After every 4 FP32 FMA, insert 1 integer/SMEM instruction. Integer draws 30-40% less current. Power average stays below Boost throttle. Just reorder existing instructions — loop counter, address calc | 2h |
| Current-draw-aware kernel launch staggering | Launch attention and FFN on different SMs with 5-10μs offset. Peak di/dt halved, Boost never triggers. All SMs active just not simultaneously | 3h |
| SM power gating for idle compute units | Attention (memory-bound): power-gate tensor core + FP64. FFN (compute-bound): power-gate SFU + texture. Saves 15-20% dynamic power → redistributes to memory controller. `cudaFuncSetAttribute` power hints | 6h |

### BLOCKER 6: Memory bandwidth (3 fixes)

| Fix | Status | Detail |
|-----|--------|--------|
| L2 persistent weight cache | ⚡ INFRA READY | `den_l2_persist_hint()` + `den_l2_persist_enabled()`. `cuMemAdvise` with SET_ACCESSED_BY. Expert staging tier (`den_expert_stage.cu`, 672 lines) already has hot/cold concept. Extend to GPU-side |
| TMU texture cache prefetch | ❌ CLEAN SLATE | Zero texture code in ggml-cuda. 280 TMUs, 48KB texture cache/SM unused. `tex1Dfetch` on quantized weight tiles. 3.36MB total TMU cache |
| Sector-level load granularity | ❌ CLEAN SLATE | Zero `ld.global.nc` in codebase. GDDR7 32B sectors. `__ldg` only in pool2d.cu. Columnar quantized weights = 16B/row → 32B sector = 2 rows |

---

## UNIVERSAL RUNTIME ARCHITECTURE (2026-08-09 — multi-reviewer consensus)

### Hydra: shared body, swappable heads

```
                    Den Runtime
                   /            \
          llama.cpp compat     native Den
            backend              backend
                   │
     ┌─────────────┼─────────────┐
     │             │             │
   LLM head    Diffusion head  Audio head  ...
     │             │             │
   Causal       Fixed-step     Streaming
   decode       denoise        encoder/decoder
   KV cache     Latent state   Audio buffers
```

L0-L2 (silicon, kernels, storage) are genuinely modality-agnostic. A quantized tensor is a quantized tensor — OMMA/soft-gemv dispatch doesn't care if it's text or diffusion. Heads are thin modality-specific binaries sharing the same L0-L2.

### VRAM Orchestrator (critical for 16GB)

16GB cannot hold 35B MoE + diffusion + Trellis + audio simultaneously. Build a governor that:
1. Suspends LLM (evict active experts + KV cache to RAM/NVMe via Dual CE)
2. Loads target modality (diffusion UNet, Trellis decoder, etc.)
3. Executes pipeline
4. Flushes VRAM, restores LLM active set

Extend existing Governor FSM + compute market. Not a new control surface — feed it more state.

### .den Evolution: weight container → universal object

Current: 160B NULLGLASS tile format, dispatch byte, slot table.
Target: add graph plane, state plane, modality plane, pipeline plane, execution policies.

Tensor roles: EMBD, ATTN_Q/K/V/O, FFN_GATE/UP/DOWN, MOE_ROUTER/EXPERT, SSM_STATE/DELTA_PROJ, NORM, VISION_ENCODER/PROJ, AUDIO_ENCODER/DECODER, DIFFUSION_LATENT/TIMESTEP/CONDITION, VAE_ENCODER/DECODER, SCENE_3D_ENCODER/DECODER, ADAPTER, COGNITIVE_STATE.

State types: TRANSIENT_TOKEN, RECURRENT_SSM, KV_CACHE, EXPERT_CACHE, AUDIO_STREAM, VIDEO_FRAME, DIFFUSION_LATENT, SCENE_3D, COGNITIVE_PAD, MEMORY_EPISODIC/SEMANTIC.

This is Phase 3+ work. Do not build until LLM golden rule is stable.

### Modality Sequencing (after LLM golden rule)

| # | Head | Why next | VRAM | Build pattern |
|---|------|----------|------|---------------|
| 1 | LLM (dense/MoE) | Current. Golden rule gate | 4.7GB active | den_llama.cpp |
| 2 | Audio/Voice | Cheapest second proof. Small encoder/decoder | ~1GB | faster-qwen3-tts + wolf DSP, near-done |
| 3 | Diffusion/Image | Fixed-step denoise, cross-attn to text | ~3-4GB loaded | stable-diffusion.cpp pattern, .den weights |
| 4 | Trellis/3D | Sparse voxel, does NOT fit dense tile assumption | ~4GB loaded | Own binary, .den for dense weights |
| 5 | Video | Diffusion + temporal axis, worst VRAM | ~6GB+ | Extension of diffusion head |

Rule: don't start head #2 until head #1 clears golden rule + has stable CI tag.

### What NOT to do
- Do not create separate runtimes per modality (6 engines, 6 allocators, 6 schedulers = death)
- Do not fold everything into llama.cpp's decode loop (it's a text engine, not a universal runtime)
- Do not let OMMA/soft-gemv dispatch become modality-aware (tensor dispatch stays modality-agnostic)
- Do not embed large assets in .den (reference with hashes)
- Do not let experimental silicon (RT/NVENC/VIC) become mandatory paths

---

## ADDITIONAL AREAS (2026-08-09 /btw — all documented APIs)

### ReBAR/SAM — full-VRAM CPU mapping (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| Direct BAR expert upload | CPU writes expert weights through ReBAR mapping. WC stores. <5μs for 1MB vs 15-20μs DMA | 4h |
| CPU→GPU attention bias injection | Bias vectors written through BAR. 100ns PCIe write latency. No kernel launch | 2h |
| BAR-mapped KV cache inspection | CPU reads KV cache through BAR. Live telemetry, zero compute overhead | 2h |
| Partial BAR + DMA hybrid | >4MB=DMA, <1MB=BAR. Hybrid scheduler per transfer. 30-40% lower upload latency | 3h |
| BAR-resident inference state | Token counter, KV metadata, router state in BAR memory. CPU polls at 20ns vs 5-10μs cudaEventQuery | 1h |

### SER — Shader Execution Reordering for MoE coherence (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| SER for expert FFN batching | Hardware groups same-expert tokens across warps. 3-5× expert throughput for batch>1 | 6h |
| SER for attention head grouping | GQA head divergence → SER groups by head. 15-20% attention throughput | 4h |
| SER for token batching | Multi-sequence divergence → SER groups by sequence. Higher L2 hit rate | 4h |
| SER region coalescing | Fuse router+expert+attention SER regions. Amortize reorder cost | 3h |
| SER telemetry feedback | Hardware counters: reorder ops, coherence ratio. Adaptive: gain<1.2×→fallback | 2h |

### L2 Cache Partitioning + Color-Aware Allocation (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| Three-way L2 partition | 20MB experts (persist) + 12MB KV (streaming) + 8MB activations (evict-first). No cross-eviction | 4h |
| Color-aware allocation | cuMemCreate with physical address offset → different L2 sets per data type. Zero same-set conflicts | 6h |
| Expert hot-set tracking | Hardware counter l2_tex_read_hit_rate. Promote/demote per 100 tokens. Dynamic repartition | 3h |
| L2 sector prefetch | Load expert tile → prefetch adjacent 128B into same partition. 30-40% higher hit rate | 2h |
| L2 dirty write-back coalescing | Coalesce 32B sectors → single 128B DRAM write. 2-3× eviction BW | 3h |

### Memory Controller Read-Write Grouping (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| Read-write phase separation | Batch all reads on one stream, writes on another with offset. Clean bursts. 80% fewer bus turnarounds | 3h |
| Write-combining KV buffer | Buffer 16-32 KV writes in L2 → flush as 2-4KB burst. 3× effective write BW | 4h |
| Read priority for critical path | Attention Q/K/V/O reads at higher MC priority. Never wait behind bulk expert prefetch. 5-10% critical path gain | 3h |
| Bank-group-aware weight layout | Spread expert weights across GDDR7 bank groups. cuMemCreate bank-interleave | 6h |
| Refresh-aware MC scheduling | Issue reads to non-refreshing banks during refresh. No stall — reads continue | 4h |

### PCIe ReBAR + Cache Coherency (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| Zero-copy inference state | All state in ReBAR memory. CPU+GPU read/write same addresses. ~100ns coherency | 3h |
| CPU-stored logits with GPU visibility | Logits in ReBAR. CPU reads directly, GPU writes directly. ~40μs saved per token | 2h |
| Cache-coherent expert directory | Expert residency table in ReBAR. Single 64B cache line write = notification | 2h |
| ReBAR WC for bulk transfers | <64KB=WC stores, >64KB=DMA. Hybrid auto-selects per transfer | 3h |
| NVMe→ReBAR direct streaming | GPU Direct Storage NVMe→GPU through ReBAR. Bypass system RAM. 2× expert load BW | 8h |

### Thermal + Power Telemetry Feedback (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| Thermal headroom batch adaptation | nvmlDeviceGetTemperature. <75°C→+batch, >82°C→-batch. Converges to max sustainable | 2h |
| Power limit negotiation | <80% power→+100MHz SM, >95%→-100MHz SM. Adaptive clock in same envelope | 3h |
| Per-layer thermal budget | Attention earns credits (low power), FFN spends them (high power). Net: same avg, higher peak | 4h |
| Memory temp-aware expert placement | GDDR7 >85°C→reduce prefetch. Cool memory=fewer retries=higher BW. Adaptive prefetch | 3h |
| VRM phase shedding for idle | Memory stalls→shed VRM phases. Compute resumes→re-engage in <5μs. 5-10W saved | 6h |

### SFU (Special Function Unit) — hardware activation functions (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| SFU-native softmax | `__expf()` maps to `MUFU.EX2` hardware. 4× faster than software expf. Softmax = SFU+shuffle, zero FMA | 2h |
| SFU-native SiLU/GELU | `MUFU.RRO` for sigmoid, `MUFU.ERF` for erf. 4-8 cycles vs 16-32 software. FFN activation halved | 2h |
| SFU rsqrt for RMSNorm | `MUFU.RSQRT` = 1/√x in 4 cycles vs software sqrt+reciprocal 16 cycles. 70+ calls/token. ~3μs saved | 1h |
| SFU pipeline decoupling | SFU has own pipe separate from FMA+LD/ST. Dual-issue SFU+LD/ST = 0% overhead for activations | 2h |
| SFU-based token sampling | Temperature scaling + softmax for multinomial via SFU. Gumbel noise uses SFU exp+rsqrt | 1h |

### Crossbar/NoC Topology-Aware Scheduling (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| L2-slice-aware block placement | Pin thread blocks to SMs physically closest to weight data L2 slice. Reduced NoC hops. `cudaLaunchKernel` SM mask | 6h |
| Memory channel affinity for experts | cuMemCreate with physical device hints. Expert N weights on channel closest to compute SMs | 4h |
| Ring-buffer KV cache placement | Stripe KV across all memory channels. Uniform access, no hotspot. Scales with channel count | 3h |
| NoC congestion backpressure | Monitor l2_tex_read_latency per SM. 2× latency→reschedule away from congested path | 4h |
| Cross-SM reduction routing | Split-K partial sums route through NoC topology, not DRAM. Tree reduction in NoC vs atomicAdd | 8h |

### GDDR7 EDC Parity for Weight Integrity + Telemetry (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| EDC error rate monitoring | NVML EDC counters per bank every 100ms. Rising trend→alert. Catch before corruption visible | 1h |
| Per-bank error heat map | Weak banks→place cold data (telemetry, debug). Strong banks→weights+KV. Adaptive placement | 3h |
| Proactive page retirement | 4KB page with repeated single-bit errors → retire in CUDA VMM allocator. Trade 4KB for safety | 2h |
| Error-corrected read latency tracking | Corrected reads take 1-2 extra cycles. Track per bank. High latency=weakening. Avoid for KV | 2h |
| Weight checksum via software CRC | Dual-layer: EDC for transport, CRC per expert tile for storage. Catch errors EDC misses | 3h |

### MIO Dual Read/Write Controller — concurrent access (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| Concurrent KV write during weight read | Read+write controllers operate independently. KV writes on write bus while reads continue. Zero-cost KV updates | 3h |
| Read-write ratio telemetry | Monitor dram_read_bytes, dram_write_bytes per partition. Throttle non-critical if write>70% | 2h |
| Dual-controller expert staging | Incoming expert (read) + outgoing eviction (write) simultaneous. Swap latency halved | 4h |
| Write controller for telemetry streaming | Perf counters+EDC+thermal to GPU buffer via write bus. Zero read BW cost. Continuous monitoring | 2h |
| Asymmetric partition utilization | Detect read-saturated partitions. Remap weights to balance across all partitions. Higher aggregate BW | 5h |

### CUDA MPS — Multi-Process Service for multi-model sharing (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| MPS LLM + diffusion co-scheduling | LLM on SM 0-49, diffusion on SM 50-69. MPS interleaves launches. Shared context, zero switch overhead | 5h |
| MPS priority-weighted VRAM | LLM=high priority, diffusion=low. VRAM tight→evict low-priority first. LLM never OOMs from background | 3h |
| MPS event-based pipeline handoff | LLM finishes token→CUDA event signals diffusion. GPU-to-GPU sync, no CPU involvement | 3h |
| MPS unified telemetry | All MPS clients→single NVML context. Aggregate power/thermal/utilization dashboard | 2h |
| MPS GPU Direct Storage sharing | LLM+diffusion share NVMe→GPU GDS bandwidth. LLM priority for expert prefetch | 6h |

### Kernel Fusion Across Forward Pass (5 fixes)
| Fix | Mechanism | Effort |
|-----|-----------|--------|
| Attention residual + next-layer norm fusion | Attn output→residual→norm in one kernel. One DRAM trip vs two. 5% decode | 6h |
| MoE router + top-K + expert dispatch fusion | Router GEMV→softmax→top-K in registers+SMEM. 3× fewer launches, 2× faster | 4h |
| Down-projection + residual + SiLU fusion | FFN down→residual add→SiLU in SMEM. One kernel, one DRAM write | 3h |
| Multi-layer persistent fusion | Layer N output + Layer N+1 input norm. Eliminates inter-layer DRAM round-trip | 10h |
| Whole-decode CUDA graph with fused nodes | Entire decode as single cudaGraph of fused kernels. Zero launch overhead | 8h |

---

## NEXT-ORDER TPS BLOCKERS (Qwen — downstream of soft-GEMV + expert offload)

### B1: CPU Graph-Build & Allocator Tax (7 fixes)
ggml rebuilds graph + reallocates every decode: 50-300μs/token CPU serialization before GPU launch.
| # | Fix | Effort | Tier |
|---|-----|--------|------|
| 1 | Static graph template + pointer patching | 4h | SHIP |
| 2 | Persistent arena allocator — zero per-token malloc | 3h | SHIP |
| 3 | cudaGraphExecKernelNodeSetParams instead of re-instantiate | 3h | SHIP |
| 4 | Double-buffered graph build on background thread | 4h | SHIP |
| 5 | Cached view tensors across tokens | 2h | SHIP |
| 6 | Load-time constant folding | 2h | SHIP |
| 7 | Serialized graph IR hydration | 6h | LAB |

### B2: PCIe Transaction Latency & TLP Overhead (7 fixes)
Small expert shards are latency-bound, not bandwidth-bound. TLP headers cost 12-16B per ≤128B payload.
| # | Fix | Effort | Tier |
|---|-----|--------|------|
| 1 | cuMemcpyBatchAsync — batch N small transfers | 3h | SHIP |
| 2 | Shard coalescing to ≥256KB before DMA | 3h | SHIP |
| 3 | TLP tuning: MRRS→512B, Relaxed Ordering, IDO | 2h | SHIP |
| 4 | 3-deep DMA pipeline — prefetch depth 3 | 4h | SHIP |
| 5 | Write-combining pinned staging | 3h | SHIP |
| 6 | ReBAR for <64KB tiny transfers | 3h | LAB |
| 7 | CCD-aware pinned placement on AM5 | 4h | LAB |

### B3: KV Gather Pathology at Long Context (7 fixes)
NVFP4 compresses bytes but doesn't fix scattered KV access pattern. TLB misses + uncoalesced reads.
| # | Fix | Effort | Tier |
|---|-----|--------|------|
| 1 | Huge-page KV slabs (2MB via cuMemCreate) | 4h | SHIP |
| 2 | Sliding-window L2 pin for recent tokens | 3h | SHIP |
| 3 | Decode layout switch: token-major→head-major | 6h | SHIP |
| 4 | Pipelined KV prefetch during softmax | 4h | SHIP |
| 5 | Entropy-based KV eviction | 6h | LAB |
| 6 | Warp-specialized K/V gather | 4h | LAB |
| 7 | 256-bit KV loads (LDG.E.256 on sm_120a) | 4h | LAB |

### B4: Router→Expert Critical-Path Serialization (7 fixes)
Router input available before current layer finishes. Yet router sits on critical path.
| # | Fix | Effort | Tier |
|---|-----|--------|------|
| 1 | Early router launch during prior layer FFN tail | 4h | SHIP |
| 2 | Router weights in SMEM/L1 permanently | 2h | SHIP |
| 3 | SFU softmax (MUFU.EX2) for router | 2h | SHIP |
| 4 | Ballot-based warp vote top-K | 3h | SHIP |
| 5 | Speculative routing on predicted hidden state | 8h | LAB |
| 6 | Expert-set warm start from prior token | 2h | SHIP |
| 7 | Persistent fused router kernel | 8h | LAB |

### B5: VRAM Allocator Fragmentation & Swap Stalls (7 fixes)
Repeated epoch swaps fragment VRAM. Synchronous cudaMalloc stalls 100μs-ms during inference.
| # | Fix | Effort | Tier |
|---|-----|--------|------|
| 1 | cudaMallocAsync mempool — sub-allocate | 3h | SHIP |
| 2 | Fixed slab arenas per modality | 4h | SHIP |
| 3 | Lifetime-segregated pools (short vs long-lived) | 3h | SHIP |
| 4 | Idle-time compaction via CE | 6h | LAB |
| 5 | Sparse-VMM VA remap without PA move | 6h | LAB |
| 6 | Buddy allocator with coalescing | 4h | LAB |
| 7 | Fragmentation telemetry + proactive compaction | 3h | SHIP |

### B6: Inter-Layer Activation/Residual Bandwidth Tax (7 fixes)
Every layer reads+writes full hidden state. Kernel boundaries evict residual from L2.
| # | Fix | Effort | Tier |
|---|-----|--------|------|
| 1 | Permanent L2 residual residency (~10KB) | 2h | SHIP |
| 2 | Fused epilogue-prologue across layer boundary | 6h | SHIP |
| 3 | Register-residual passing in persistent kernel | 8h | LAB |
| 4 | FP8 activation transport (NEVER for SSM/GDN) | 6h | LAB |
| 5 | L2 write-back coalescing | 3h | LAB |
| 6 | Dual-buffer residual ping-pong | 3h | SHIP |
| 7 | Activation sparsification | 8h | LAB |

### B7: Kernel-Boundary Gaps (launch/sync/event overhead) (7 fixes)
~350 kernel launches/token. 2μs gap each = ~700μs/token ≈ 13%. MoE breaks graphs.
| # | Fix | Effort | Tier |
|---|-----|--------|------|
| 1 | Programmatic Dependent Launch | 4h | SHIP |
| 2 | Conditional graph nodes for MoE branches | 8h | SHIP |
| 3 | Event-less sync via device-side flags | 3h | SHIP |
| 4 | Stream-ordered memory ops | 2h | SHIP |
| 5 | cudaLaunchHostFunc sampling overlap | 3h | SHIP |
| 6 | Green contexts (verify sm_120 support) | 8h | LAB |
| 7 | Cooperative grid sync | 8h | LAB |

### TOP 5 OVERLOOKED EXPLOITS (Qwen — not in RESEARCH_FANTASY.md)

| # | Exploit | Target | Mechanism | Tier |
|---|---------|-------|-----------|------|
| 1 | **DSMEM clusters** | Thread Block Clusters on sm_120a | CTAs read/write peer SMEM directly via mbarrier. Resurrects A2.1/A2.3 (cross-SM work-stealing) WITHOUT DRAM | LAB |
| 2 | **7800X3D 96MB V-Cache** | CPU L3 as hot-expert tier | AVX-512 copy L3→pinned→CE. Core-pin sampler + shadow router to V-Cache cores. Kill sampling jitter | SHIP |
| 3 | **CUDA Sysmem Fallback disable** | Silent VRAM→RAM paging death | Disable sysmem fallback for near-limit allocations. Deterministic OOM instead of 5-10× perf cliff | SHIP |
| 4 | **DirectStorage + GDeflate** | NVMe→GPU expert streaming | Store compressed experts on 990 EVO Plus. DirectStorage DMA→GPU GDeflate decompress. Bypass CPU+RAM | LAB |
| 5 | **PCIe TLP + FCLK tuning** | PCIe root complex + AM5 IMC | MRRS=512B, Relaxed Ordering, IDO, FCLK:UCLK:MCLK 1:1:1. 55→60 GB/s practical ceiling | SHIP |

---

## GB203 SILICON EXPLOITS — Idle Hardware Inventory

Every unit below is present on RTX 5070 Ti (GB203-300-A1) and completely unused during LLM inference.

| # | Silicon Unit | Idle Capacity | Exploit | Gain | Complexity | Status |
|---|-------------|---------------|---------|------|------------|--------|
| 1 | RT Cores (56×) | 67 TFLOPS ray-tri, 3.35 TFLOPS BVH traverse | BVH nearest-neighbor: attention O(log N) or expert routing O(log 256) | 4-7700× | Hard (OptiX or inline PTX) | Ported tiers 2+3 |
| 2 | NVENC (9th gen, dual) | 4K 120fps encode, 21.3 TOPS ME SAD | Motion estimation as sparse attention mask generator | Free sparsity mask | Hardest (NV12 surface conversion) | Research |
| 3 | ROPs (128×) | 332.8 GPixels/s, 2:1 DCC lossless | Delta Color Compression on BF16 weight tiles → 2× effective BW | 672→1344 GB/s | Hard (DCC metadata format) | Research |
| 4 | TMUs (280×) | 448 GTexels/s, 48KB texture cache/SM (3.36MB total) | Texture cache as weight prefetch scratchpad. L1 stays free for activations | Zero-latency weight fetch | Moderate (CUDA texture API) | Clean slate |
| 5 | Copy Engine 1 | 50 GB/s PCIe 5.0 DMA, completely idle | CE1 uploads experts while CE0 shuffles KV. Concurrent DMA | Half offload latency | Simplest (stream priority) | Infra ready |
| - | L2 Cache (40MB) | 5120-bit bus, 5.2 TB/s internal | Persistent expert hot subsets. 40-60% hit rate on Zipf access | 3.8× effective BW for cached experts | Moderate (cuMemAdvise) | Infra ready |

**Best first: Dual CE (#5) — 3 lines of CUDA, immediate 35B MoE gain. Then TMU (#4) — clean slate, big BW. Then L2 persist — infrastructure exists.**

---

## AREA 1: ASYNC CPU-GPU TOKEN PIPELINE (serial bottleneck)

**Problem:** GPU idles 50-100μs/token while CPU samples. At 150 tok/s = 7.5-15ms wasted/sec. Spec decode: 4 tokens/step, CPU latency dominates.

**Existing code:** `llama-context.cpp` decode loop. Sampling in `common/common.cpp`. GPU→CPU logit copy at `ggml-cuda.cu` graph output. No async overlap exists.

| Fix | Mechanism | Files | Effort |
|-----|-----------|-------|--------|
| A1.1: GPU-resident sampler | Move greedy/top-K to GPU via `cub::DeviceRadixSort` + warp argmax. Eliminate GPU→CPU→GPU round-trip. Only copy 4B token ID back. ~40μs saved/token | `ggml/src/ggml-cuda/sample.cu` (NEW), `llama-context.cpp` | 6h |
| A1.2: Dual-graph execution | CPU samples token N while GPU runs attention for N+1 speculatively. `cudaStreamWaitValue64` at host→device boundary for commit/discard | `llama-context.cpp`, `ggml-cuda.cu` | 8h |
| A1.3: Batch-ahead scheduling | Queue N decode batches before first result. Double-buffer: one computing, next pre-loaded. Async polling replaces `cudaDeviceSynchronize` | `llama-context.cpp` decode loop | 4h |

**Priority: A1.1 (GPU sampler) → all models benefit. A1.2 → spec decode. A1.3 → batch throughput.**

---

## AREA 2: CROSS-SM ACTIVATION MIGRATION VIA L2

**Problem:** 70 SMs process all 35 layers sequentially. Each layer boundary: write activations to DRAM → next layer reads. 35× per token × 4MB = 140MB DRAM traffic/token. L2 = 40MB, only ~28% fits.

**Existing code:** MMQ kernel in `mmq.cuh` uses `cp.async` with `L2::256B` preload hints. No cross-SM signaling. No SM-specialized scheduling. No L2-resident work queue.

| Fix | Mechanism | Files | Effort |
|-----|-----------|-------|--------|
| A2.1: SM-specialized layer pipelining | SM 0-19: attention. SM 20-39: FFN. SM 40-69: GDN/SSM. Activations pass SM→SM via L2 atomic write+signal. DRAM bypassed | `ggml-cuda.cu` dispatch, new `pipeline.cu` | 12h |
| A2.2: Activation tile multicast | Layer output tile → L2 with broadcast flag. Awaiting SMs poll L2 tags. `ld.global.L2::evict_last` keeps tile resident until consumed | `mmq.cuh`, new `l2-multicast.cuh` | 6h |
| A2.3: Persistent cross-SM work-stealing | Idle SMs steal work via L2-resident lock-free queue. L2 atomics for dequeue. All 70 SMs fed regardless of per-layer imbalance | `den_persistent_kernel.cu` (port from C:\), new `work-steal.cuh` | 8h |

**Priority: A2.3 (work-stealing) → pairs with persistent kernel port. A2.1 → 35B MoE biggest gain. A2.2 → general.**

---

## AREA 3: PREDICTIVE EXPERT PREFETCH WITH ADJACENT-TOKEN CORRELATION

**Problem:** MoE router logits between adjacent tokens correlate 70-85%. Current: token N computes router → loads experts → FFN. All serial. Next token hasn't started → prefetch impossible naively.

**Existing code:** Expert staging tier (`den_expert_stage.cu`, 672 lines) with Markov predictor for split-key sequences. `ggml-backend.cpp:1809` submit hook, `:1834` find_span hook. Shadow router at `cognition_rust/src/shadow_router.rs`. RT expert router at `den-rt-expert-router.cu`.

| Fix | Mechanism | Files | Effort |
|-----|-----------|-------|--------|
| A3.1: Router logit reuse pipeline | Token N-1's logits prefetch experts for token N. 70%+ hit rate adjacent. Miss→cache fill, no stall. RT router BVH built once, reused for prefetch | `llama-graph.cpp:2044`, `ggml-backend.cpp:1809`, `den-rt-expert-router.cu` | 4h |
| A3.2: Dual CE streamed prefetch | CE1 loads predicted experts for N+1 while OMMA computes N's FFN. CE0 shuffles KV for N+1. Both CEs + OMMA concurrent. Zero added latency on hit | `ggml-backend.cpp:1823` copy_experts lambda | 3h |
| A3.3: Co-activation Markov matrix | `C[i][j] += (expert_i && expert_j active)`. After 100 tokens: strong signal. `P(next=j \| current=i) = C[i][j]/sum(C[i][:])`. 85%+ accuracy. 256×256×2B = 128KB — fits L1 | `den_expert_stage.cu` predictor extension | 4h |

**Priority: A3.1 → immediate 35B MoE gain from router logit reuse. A3.3 → pairs with Sinkhorn. A3.2 → pairs with Dual CE.**

---

## FILE INVENTORY — What Exists Where

### den_llama.cpp (I:\) — Active engine

| Path | Lines | Purpose |
|------|-------|---------|
| `ggml/src/ggml-cuda/fattn-nvfp4-kv.cu` | ~850 | NVFP4 KV: 4 quant kernels + fused attention, K8V4, KVSink |
| `ggml/src/ggml-cuda/fattn-nvfp4-kv.cuh` | ~120 | NVFP4 KV: constants, types, public API |
| `ggml/src/ggml-cuda/den-rt-expert-router.cu` | 310 | RT Core expert router: PCA + GPU brute-force + GEMV gate |
| `ggml/src/ggml-cuda/den-rt-expert-router.cuh` | 135 | RT Core expert router: C API header |
| `ggml/src/ggml-cuda/den-btw-fixes.cuh` | 186 | Sinkhorn + Dual CE + L2 + sector loads infrastructure |
| `ggml/src/ggml-cuda/den_expert_stage.cu` | 672 | CPU L3-resident expert staging: Markov predictor, AVX-512 NT stores |
| `ggml/src/ggml-cuda/den_expert_stage.h` | 96 | Expert staging interface |
| `ggml/src/ggml-cuda/sparse-vmm.cu` | 201 | Sparse VMM: cuMemAddressReserve + cuMemCreate + cuMemMap |
| `ggml/src/ggml-cuda/sparse-vmm.cuh` | 48 | Sparse VMM: C API |
| `ggml/src/ggml-cuda/sparse-buft.cu` | 200 | Sparse VMM: ggml backend buffer type |
| `ggml/src/ggml-cuda/sparse-buft.cuh` | 25 | Sparse VMM: buffer type API |
| `ggml/src/ggml-cuda/topk-moe.cu` | ~300 | Warp-level top-K MoE gating (has bias path for Sinkhorn) |
| `ggml/src/ggml-cuda/mma.cuh` | ~1200 | OMMA PTX intrinsics: `mma_block_scaled_fp4<NVFP4>()` = mma.sync.kind::mxf4nvf4.4X |
| `ggml/src/ggml-cuda/mmq.cuh` | ~1700 | Quantized matmul dispatch: NVFP4→OMMA via mmq-config-blackwell.cuh |
| `ggml/src/ggml-cuda/mmq-config-blackwell.cuh` | 37 | Tile configs: MMQ_ITER_K_FP4=512 |
| `ggml/include/ggml-cuda.h` | ~70 | Public API: NVFP4 init/reset, sparse VMM, Sinkhorn/DualCE wrappers |
| `src/llama-context.cpp` | ~4300 | Auto-enable NVFP4, sparse VMM pool create/destroy, memory clear hook |
| `src/llama-context.h` | ~330 | llama_context: sparse_vmm_pool member |
| `src/llama-graph.cpp` | ~2500 | MoE FFN build: `build_moe_ffn` at line 1975. Bias add at line 2047 |
| `ggml/src/ggml-backend.cpp` | ~5500 | Expert copy at line 1759-1871: copy_experts lambda, staging hooks |
| `include/llama.h` | ~1100 | `llama_sparse_vmm_ensure/grow`, `nvfp4_kv_enabled`, `sparse_kv_enabled` |
| `.github/workflows/bench-guard.yml` | 120 | CI golden rule guard |
| `tools/test_nvfp4_replay.py` | 160 | Deterministic replay test |
| `tools/test_vscale_audit.cu` | 310 | V-scale swizzle audit |
| `.claude/hooks/sass-audit.py` | 180 | SASS instruction count audit |

### C:\Den\den-nvfp4-optimizations — Reference/port source

| Path | Lines | Purpose |
|------|-------|---------|
| `dengine/compute/den_rt_expert_router.cu` | 1416 | Full RT Core expert router: OptiX + SW BVH + brute-force + GEMV. ALL 4 TIERS |
| `dengine/include/den_rt_expert.h` | 211 | RT Core expert router C API |
| `cuda_kernels/rt_core/den_rt_attention.cu` | 1402 | RT Core attention: OptiX pipeline + CUDA SW BVH |
| `cuda_kernels/rt_core/den_rt_attention.cuh` | 297 | RT attention header |
| `cuda_kernels/attention/den_rt_attention.cuh` | 256 | RT attention v2: inline PTX wrappers (no OptiX) |
| `cuda_kernels/attention/den_rt_attention_ptx.cuh` | >100 | Raw PTX: rt.traverse.bvh.trampoline, rt.intersect.custom.* |
| `dengine/compute/den_persistent_kernel.cu` | ~800 | 12 op types, TDR-aware, work-queue self-scheduling |
| `dengine/compute/gpu_layer_ops.cu` | ~400 | Fused RMSNorm + attention QKV + residual |
| `dengine/compute/omma_dispatch.cu` | ~200 | Standalone OMMA cubin loading (not needed — mma.cuh has inline) |
| `dengine/src/den_gpu_l2.cu` | ~200 | L2 cache pinning via cuMemAdvise (Mech 41) |
| `docs/MASTER_OPTIMIZATION_CATALOG.md` | ~350 | Canonical index: ~140 optimizations + ~180 source kernels by silicon unit |

---

## INTEGRATION WIRING — What Needs Connecting

### Sinkhorn rerank (A3.x family)
```
den-btw-fixes.cuh:den_sinkhorn_apply_bias()
    ↓ call from
llama-graph.cpp:2047  selection_probs = ggml_add(ctx0, probs, exp_probs_b)
    ↓ modify exp_probs_b buffer with warmth before graph execution
llama-context.cpp decode loop: call den_sinkhorn_store() after expert IDs known
```
**3 connection points. ~30 lines total. Kernel already has bias path. Just modify bias buffer.**

### Dual CE (Blocker 4 fix 1)
```
den-btw-fixes.cuh:den_dual_ce_get_stream_ce1()
    ↓ use in
ggml-backend.cpp:1823 copy_experts lambda: ggml_backend_tensor_set_async()
    ↓ route H2D through CE1 stream instead of default stream
ggml-cuda.cu:2466 cudaMemcpyAsync H2D: use CE1 stream when available
```
**2 connection points. ~10 lines. Stream already created. Just pass it.**

### L2 persistence (Blocker 6 fix 1)
```
den-btw-fixes.cuh:den_l2_persist_hint()
    ↓ call after
ggml-backend.cpp:1847 expert weight upload complete → cuMemAdvise on GPU buffer
sparse-vmm.cu pattern: CU_MEM_ALLOCATION_TYPE_PINNED → extend with CU_MEM_ADVISE_SET_ACCESSED_BY
```
**1 connection point. ~5 lines. VMM pattern already established.**

### RT expert router tiers 0+1
```
den-rt-expert-router.cu (current: tiers 2+3 only)
    ← port from
C:\Den\dengine\compute\den_rt_expert_router.cu (OptiX dynamic load + SW BVH traversal)
    ← reuse from
C:\Den\cuda_kernels\attention\den_rt_attention_ptx.cuh (inline PTX RT Core without OptiX)
```
**~800 lines to port. Dynamic OptiX loading pattern already documented.**

---

---

## SILICON EXPLOITATION: 10 DEEPLY OVERLOOKED AREAS (2026-08-08 /btw)

Every area below targets idle or underutilized GB203 silicon. None exploited by ANY inference engine. All five areas × 5 fixes each = 25 fixes. Plus the 5 original GB203 exploits = 30 total silicon-level optimizations.

### GDDR7 PAM3 SYMBOL-ALIGNED WEIGHT STREAMING

**Problem:** GDDR7 PAM3 packs data at 192-bit (24B) codeword granularity. 160B NVFP4 tiles = 6.67 codewords → 0.67 wasted per read. ECC scrub steals 2-3% BW. Burst-chop and bank refresh cause additional losses. Current code: zero awareness of DRAM physical layout.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| PAM3-aligned tile sizes | Pad NVFP4 K4V4 tiles 160→168B (7 codewords), K8V4 288→312B (13 codewords). Zero waste | 5% BW | Trivial (constant change) |
| ECC scrub-aware scheduling | Track GDDR7 scrub phase (~32μs interval). Prefetch from non-scrubbing banks. `nvidia-smi -q -d ECC` for phase | 2% BW | Moderate |
| Burst-chop optimization (BC4) | Pack 4 BF16 values → 16B BC4 burst. 2 per 32B sector instead of 1 per 64B. `ld.global.v4bf16` | 50% sector util | Moderate (PTX change) |
| Per-bank refresh staggering | 16 banks. Refresh 1 at a time → 1/16th BW loss continuous vs full 350ns stall every 3.9μs. `cuMemAdvise` bank-interleave | 8% BW | Hard (driver-level) |
| Pseudo-channel double-pump | 2 pseudo-channels/die. Pin experts→ch0, KV cache→ch1. Concurrent access zero bank conflict. `cudaMallocManaged` VA interleave | 2× concurrency | Hard (allocator change) |

### WARP SCHEDULER CO-ISSUE + DISPATCH BUBBLE ELIMINATION

**Problem:** 4 warp schedulers/SM. Not all instruction pairs can co-issue. Current kernels: ~75% dual-issue rate → 25% dispatch slots wasted. OMMA occupies tensor pipe for ~8 cycles with no scalar/LDST co-issue from other warps. `__syncthreads()` costs ~40 cycles at every tile boundary.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| Instruction-pairing SASS audit | Reorder SASS: FP32 always pairs with LD/ST or shared mem. Never FP32+FP32. Target 95% dual-issue | 15% | Hard (SASS-level) |
| Tensor-core bubble filling | Schedule `ld.global` for next tile during current OMMA. `#pragma unroll` + PTX reorder. Fill 8-cycle bubble | 15-20% OMMA | Hard (PTX) |
| Warp occupancy per layer | Attention: 4 warps/SM. FFN: 8 warps/SM. GDN: 2 warps/SM. `__launch_bounds__` per kernel | 5-10% | Moderate |
| Convergence barrier elimination | Fuse 2-3 tiles between `__syncthreads__`. Reduces barriers 33%, <5% register pressure increase | 5% | Moderate |
| CUDA graph entire decode | Capture full decode as single graph. `cudaGraphInstantiate` once, launch per token. Eliminates N-1 inter-grid gaps (5-15μs each) | 10% at low batch | Moderate |

### REGISTER FILE BANK CONFLICT ELIMINATION

**Problem:** 65536 registers/SM, 128 banks × 512. Same-bank read = 32-cycle stall. Current GEMV: compiler allocates registers randomly → unknown bank conflicts. 128-bit reads underutilized. Spills to L1 when >128 regs/warp.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| Register colocation audit | `ncu --register-file-bank-conflicts` on GEMV. Manually allocate via `.reg` PTX. Target 0 bank conflicts in inner loop | 8-15% GEMV | Hard (PTX) |
| Vector register packing | Load BF16 as `uint4` (128-bit) → 1 bank access per 4 values. Current: scalar bf16 → 4× bank accesses | 4× fewer bank ops | Moderate |
| SMEM spill buffer | Reserve 4KB SMEM as explicit spill. Cold values (metadata, counters) → SMEM. Hot values (partial dot products) → registers. 3× faster than L1 spill | 10% when spill-bound | Moderate |
| Bank-aware tile layout | Interleave dequant register assignment: `reg = (tid + tid/16) % 32`. Rotates across register banks → 0 conflicts | 5% dequant | Moderate |
| Persistent register renaming | Compute 2 tiles back-to-back same warp no barrier. Hardware rename avoids WAR hazards. Double throughput per barrier | 2× tile throughput | Hard (PTX) |

### NVDEC HARDWARE WEIGHT DECOMPRESSION PIPELINE

**Problem:** NVDEC (9th gen) = H.264/H.265/AV1 decoder. CABAC entropy decoder, inverse DCT/DST, motion compensation. 2 instances. Completely idle. Power: ~5W idle, ~15W active. CABAC = hardware binary arithmetic coder ≈ ANS decoder.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| CABAC-as-ANS weight codec | Train tANS on NVFP4 tiles. Encode as "AV1 residual" bitstream. NVDEC CABAC decodes at 2.4 Gbit/s | 1.5-2× compression | Hardest (codec design) |
| Inverse-DCT weight reconstruction | Encode weights as "DCT coefficient blocks." NVDEC iDCT → weights in pixel format. DMA to CUDA array | Offload from SMs | Hardest (DCT format) |
| Motion-compensated delta-expert | Base expert = I-frame. Delta experts = P-frame motion vectors. NVDEC MC reconstructs. 3-5× compression | 3-5× expert storage | Hardest (MC encoding) |
| Dual NVDEC pipelining | Instance 0: gate_up weights. Instance 1: down weights. Both decode concurrently while OMMA processes previous batch | 2× decode throughput | Hard |
| NV12→NVFP4 zero-copy DMA | NVDEC outputs NV12 surface. Y=luma=weight mag, UV=chroma=scales. `cudaMemcpy2DFromArray` to CUDA texture | Zero SM copy cost | Hard |

### VF CURVE + CLOCK DOMAIN TUNING (MEMORY-BOUND INFERENCE)

**Problem:** Inference 76% memory-bound (GDDR7 510/672 GB/s). SMs at 2.6 GHz but wait on memory. Undervolting SMs → power budget → memory OC. GDDR7 memory controller has independent PLL from SM clock.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| SM undervolt + mem OC | `nvidia-smi -lgc 2000` lock SM at 2.0 GHz. `-lmc 12000` OC memory 21→24 Gbps. 50-100mV save → 15-25W → memory controller | 12% tok/s | Trivial (nvidia-smi) |
| Per-layer clock gating | Attention: 1.5 GHz (mem-bound). FFN: 2.6 GHz (compute-bound). `nvmlDeviceSetApplicationsClocks` per-layer in persistent kernel | 5-8% | Moderate |
| GDDR7 read-retry disable | Read-dominant inference. CRC-8+retry costs ~3% BW. Displayless → no visual corruption risk. `RmReadRetryDisable=1` registry | 3% BW | Trivial (registry) |
| Thermal-aware batch sizing | Adaptive: `argmax(batch) where max_temp < 80°C after 60s`. Sustained 2.6 GHz without throttling at 83°C | 5-10% sustained | Moderate |
| PCIe ASPM L1 disable | L1 sleep after 32μs idle → 10-40μs wake. `pcie_aspm=off` or `nvidia-smi --pci-link-gen=5`. Expert DMA: 50→15μs | 3× DMA wake speed | Trivial (kernel param) |

---

## SILICON EXPLOITATION: 5 TRULY UNEXPLORED GB203 CAPABILITIES

These go beyond the already-unexplored areas above. They exploit hardware blocks that NO research paper, NO inference engine, and NO NVIDIA documentation even mentions for ML workloads.

### CRYPTO ENGINE (AES-GCM) — LOSSLESS WEIGHT STREAMING

**Silicon:** Hardware AES-256-GCM engine in display/HDCP pipeline. Line-rate crypto, zero SM involvement. Completely idle during inference.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| Inline AES→SMEM weight stream | Encrypt compressed weight tiles (AES-256-CTR). CE1 DMAs ciphertext. AES decrypts during DMA into SMEM. SM sees plaintext, zero decrypt cycles | 20-30% more weights/DMA | Hardest |
| Authenticated weight integrity | AES-GCM auth tag per tile (16B). Verify in hardware on load → detect silent bit-flips missed by ECC | Zero-latency tamper detect | Hard |
| Compressed checkpoint streaming | State save: KV+SSM→compress→AES encrypt→NVMe. Restore: reverse. AES inline during DMA. 8× faster vs CPU crypto | 8× checkpoint speed | Moderate |
| Secure expert cache fill | Weights encrypted in host RAM (model IP protection). AES decrypts inside GPU. PCIe snooping = ciphertext only. DRM-level protection | Model security, zero perf hit | Moderate |
| Multi-key per-layer streaming | Different key per layer. Key schedule in L2 (256B×35=9KB). AES switches keys per DMA. Compromised key = 1 layer leaked | Defense-in-depth | Moderate |

### DISPLAY ENGINE DSC — ACTIVATION SPARSITY DETECTOR

**Silicon:** DSC 1.2a hardware in Display Engine. 4 encoder instances. MMAP predictor + quantization + entropy coding at 33.2 GB/s (4K 240Hz throughput). Completely idle.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| DSC MMAP as sparsity detector | Feed activations as "scanlines." MMAP predictor error < threshold → "compressible" = sparse. Hardware tags at 33.2 GB/s | Free sparsity mask | Hardest |
| DSC entropy coder for KV metadata | KV timestamps, position IDs have high temporal locality. Pass through DSC VLC coder → 2-4× metadata compression | 2-4× meta compression | Hard |
| DSC QP→NVFP4 scale mapping | DSC quantization parameter → NVFP4 tile scale granularity. DSC rate-distortion optimizer chooses per-tile QP at line speed | Optimal per-tile rate | Hard |
| Quad-instance DSC pipelining | Instance 0: sparsity. 1: KV meta. 2: attention QP. 3: residual coding. All 4 concurrent on different streams | 4× throughput | Hard |
| DSC roundtrip quality monitor | DSC outputs both compressed + decompressed streams. Compare → measure quality loss real-time. Zero SM cost | Free quality telemetry | Moderate |

### HOST-TO-DEVICE SIGNALING VIA GPU DOORBELL REGISTERS

**Silicon:** MMIO-mapped doorbell registers in GPU BAR1. CPU writes → GPU wakes. ~100ns latency vs 1-5μs for `cudaStreamWaitValue64`. Currently only kernel driver uses them.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| Doorbell speculative decode commit | CPU samples token→writes doorbell. GPU spins on `ld.global.mmio` BAR1 address. 100ns wake vs 1-5μs. Commit in <1μs | 10× faster wake | Hard (BAR1 mapping) |
| Multi-token doorbell batch | CPU samples N tokens parallel (spec decode). N doorbells to N addresses. GPU warp commits each as its doorbell fires. No barrier | Partial commit latency | Hard |
| Doorbell ring buffer DMA completion | CE1 finishes→writes doorbell GPU→CPU. CPU polls at 200ns vs 5-10μs `cudaEventQuery`. Queues next expert immediately | 4× swap idle reduction | Moderate |
| Per-SM doorbell wakeup | SMs in WFI deep sleep. Doorbell to SM-specific MMIO wakes exactly 1 SM. No global wake-all. 20× faster (0.5μs vs 10μs) | 20× SM wake speed | Hard (MMIO discovery) |
| User-mode doorbell bypass | `mmap(/sys/bus/pci/.../resource1)` BAR1 into process. `mov [bar1], val` — 50ns doorbell, no syscall/ioctl/driver | 50ns user→GPU signal | Moderate (`CAP_SYS_RAWIO`) |

### GDDR7 PER-BANK REFRESH WINDOW OPPORTUNISTIC COMPUTE

**Problem:** All-bank refresh every 3.9μs × 350ns = 2564 cycles × 235KB lost = 602MB lost BW per 10ms token. That's 8.8% of total GDDR7 bandwidth. Gone. Every single token.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| Per-bank refresh scheduling | Refresh bank 0 while accessing banks 1-15. Rotate. Only 1/16th BW lost vs 100% for 350ns. `nvmlDeviceSetMemoryBankRefreshPolicy` | 8% BW | Hard (driver-level) |
| Refresh-phase compute alignment | Schedule register-only work (shuffle, SMEM reduce, softmax, RMS norm, tile quantize) during refresh. 63,700 instructions of in-register compute per refresh | Zero refresh stall | Hard (persistent kernel) |
| Refresh-aware weight prefetch | Prefetch next layer weights into SMEM/L1 before refresh. During refresh: compute from cache, no DRAM access. No stall | Zero stall | Moderate |
| Double-rate refresh during idle | Accelerate refresh when GPU idle between requests. More cycles now = fewer needed during active inference = longer intervals | 2-3% active BW | Trivial (driver hint) |
| Temperature-compensated refresh | GDDR7 at 55°C needs half the refresh of 85°C. Read DRAM temp sensor, request extended interval. Over-refreshed at cool temps | 4% BW at 55°C | Moderate (I2C temp read) |

### GPU SYSTEM PROCESSOR (GSP) OFFLOAD — CONTINUOUS ORCHESTRATION

**Silicon:** GSP = NV-RISC-V core at ~1 GHz. Runs GPU kernel driver scheduler. Has access to ALL GPU registers, MMIO, VRAM. Currently ~10% utilized (command queue feeding). 90% idle.

**⚠️ SCOPE NOTE:** All GSP fixes require custom firmware on the GSP RISC-V core. No public SDK exists from NVIDIA. These are research/feasibility items only. No firmware modification without official NVIDIA GSP SDK or explicit vendor approval. The 4 fixes below are CONCEPT-level pending SDK availability.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| GSP-hosted decode orchestrator | Entire decode loop on GSP. GSP: polls token, selects kernel, submits command, monitors completion, loops. CPU just feeds tokens | 5-10× faster dispatch (5-10μs vs 50-100μs) | CONCEPT — no SDK |
| GSP Markov expert predictor | 128KB co-activation matrix on GSP. Computes `P(next\|current)` in ~100 cycles (0.1μs). Submits prefetch BEFORE SM router finishes | Zero-latency expert prefetch | CONCEPT — no SDK |
| GSP dynamic frequency governor | Reads PMU counters every 100μs. Detects mem/compute-bound. Adjusts SM+memory clocks independently at μs granularity (NV-RISC-V PLL regs) | 3-5% perf/W | CONCEPT — no SDK |
| GSP multi-model concurrent serving | Partition SMs: 35B on SM 0-49, 9B on SM 50-59, embeddings on SM 60-69. GSP manages 3 command streams, handles priority. Zero CPU coordination | 3 models/1 GPU | CONCEPT — no SDK |

~~GSP firmware hot-patching~~ — REMOVED. Firmware modification = hard no.

---

---

## SILICON EXPLOITATION: 5 MORE AREAS — NVOF, VIC, L1 CACHE, PCIe ATOMICS, BOOST 5.0

All five are 100% software-only. Zero hardware risk. Zero firmware. Zero register hacking. Public APIs or documented behavior.

### NVOF (OPTICAL FLOW) — TEMPORAL ATTENTION SPARSITY

**Silicon:** NVOF hardware block (Turing+). 300+ fps 4K dense optical flow at 3 Gpix/s. Completely idle during inference. Attention patterns between adjacent tokens have "motion" — which KV positions gain/lose relevance.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| Attention flow as sparse mask | Feed attention score matrices as "frames." NVOF returns flow vectors → threshold → sparse mask. Hardware at 3 Gpix/s | 5-10× attention sparsity | Hard (NvOFAPI) |
| Delta-KV from flow | Near-zero flow = stable KV positions. Skip recomputing K/V for those. ~40% KV compute saved | 40% KV compute | Moderate |
| Flow-based spec decode verify | NVOF flow between draft+verify attention → mismatch regions = wrong draft tokens. Hardware identifies before full verification | 3× faster spec rejection | Hard |
| Multi-resolution attention pyramid | Coarse attention at 1/4 res (16× cheaper). NVOF upsamples to full res via flow interpolation. 4-8× speedup | 4-8× attention | Hard |
| NVOF confidence telemetry | Low-confidence flow = unpredictable attention = "surprising" tokens. Free real-time entropy measurement | Free quality monitoring | Trivial (API read) |

### VIC (VIDEO IMAGE COMPOSITOR) — FREE MATRIX ACCELERATOR

**Silicon:** VIC fixed-function pipeline: format convert, scale, rotate, blend, color space. 4K 240fps = 2.1 GPix/s. 2D affine = 3×3 matmul. Bilinear interpolation = dot product. Blend = weighted sum. All free linear algebra in hardware.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| VIC affine as GEMM | Weight rows as "image rows." VIC 2D affine = 3×3 matmul/pixel. Chain N transforms = free N×3×3 ops. 2.1 GPix/s × 9 FMA = 18.9 GFLOPS | 18.9 free GFLOPS | Hardest |
| VIC bilinear as dequant | 4 adjacent NVFP4 elements as "texel corners." VIC bilinear filter = free weighted sum dequant. 2.1 GPix/s | Zero-SM dequant | Hard |
| VIC blend as softmax fusion | Value vectors = "layers." Attention scores = "alpha." VIC blends in hardware. Softmax-V sum offloaded | Zero-SM attention sum | Hard |
| VIC color LUT as activation fn | SiLU/GELU as custom 3D LUT in VIC color correction. Activations → VIC → nonlinear output at scanout rates | Free activation eval | Hard |
| VIC→DMA zero-copy | VIC output surface → `cuGraphicsResourceGetMappedPointer` → CUDA array. SM↔VIC↔DMA pipeline. Zero copy | Zero SM copy overhead | Moderate |

### L1 CACHE CONFIG PER-KERNEL + EVICTION POLICY

**Silicon:** 256KB L1/SM. Configurable: 192KB SMEM+64KB L1 OR 128KB+128KB. Current: fixed at compile time for all layers. Public CUDA API: `cudaFuncSetAttribute(cudaFuncAttributePreferredSharedMemoryCarveout)`.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| Per-layer cache config | Attention: 75% SMEM carveout. FFN: 25% SMEM. Switch at launch via `cudaFuncSetAttribute`. Zero overhead | 10-15% cache hit rate | Trivial (API call) |
| Streaming weight evict-first | Expert weights = stream-once. `ld.global.cg` (evict-first) keeps L1 for reused data (residuals, norms). `.cg` PTX modifier | 5% L1 hit rate | Trivial (PTX) |
| SMEM partition warmup | Profile SMEM usage per layer on token 1. Learn optimal carveout. Store in dispatch table. Apply on tokens 2+ | 5-8% occupancy | Moderate |
| L1 persistence across layers | Residual connections via `__ldg` → L1 texture cache partition holds them across layer boundary | 3% BW saving | Trivial (`__ldg`) |
| Bank-conflict-free SMEM | XOR-based bank scattering: `offset ^ (offset>>5) & 31`. Zero-conflict tile access. 15% SMEM throughput | 15% SMEM throughput | Moderate |

### PCIe 5.0 ATOMICS — ZERO-LATENCY CPU↔GPU SYNC

**Silicon:** PCIe 5.0 atomic ops over bus: fetch_add, CAS, exchange, add. Single transaction. No DMA. No kernel launch. No stream sync. CPU directly modifies GPU memory atomically.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| Atomic token counter | CPU `fetch_add` on GPU `seq_len`. GPU kernel reads L2 (20ns) vs `cudaStreamWaitValue64` (1-5μs). No stream sync | 50× faster wake | Moderate (PCIe BAR) |
| Atomic expert histogram | CPU `atomicAdd` per-expert GPU counters on each selection. GPU scheduler reads directly. No DMA | Free expert tracking | Trivial |
| DMA completion via GPU→CPU atomic | CE1 finish → PCIe `store` to CPU-pinned memory. CPU spin: 200ns vs 5-10μs `cudaEventQuery` | 25× faster notify | Moderate |
| Lock-free GPU work queue | CPU enqueues via `fetch_add` on GPU ring buffer write ptr. GPU dequeues via `atomicAdd` on read ptr. Full lock-free across PCIe. 50ns enqueue | 1000× faster than kernel launch | Moderate |
| Atomic spec verification mask | CPU writes bitmask via PCIe `or`. Bit N set = token N verified. GPU warp commits each bit as it appears. Partial commit no barrier | Parallel spec commit | Moderate |

### GPU BOOST 5.0 — OPPORTUNISTIC SCHEDULING (NOT OVERCLOCKING)

**⚠️ ZERO overclocking.** All 5 fixes are kernel instruction reordering or telemetry reads. No voltage changes. No clock register writes. Boost gives higher clocks during light-load periods naturally — we structure work to align with boost's own algorithm. The actual overclock item (SM undervolt + mem OC) is in VF Curve above, marked MODERATE RISK.

**Silicon:** Boost 5.0 monitors power/thermal/voltage/current. Clock drops: ~100μs. Clock rises: ~2ms (hysteresis). Inference has micro-bursty patterns: memory stalls (50-200ns) → compute bursts (20-50ns). Boost sees "idle" during stalls, doesn't throttle. Compute executes at elevated clock before boost detects it.

| Fix | Mechanism | Gain | Complexity |
|-----|-----------|------|------------|
| Memory-stall→compute-burst scheduling | Cluster loads at start (boost sees idle → holds clock). Then pure compute: 50-100 cycles FMAs at held clock. No OC — just aligning work with boost hysteresis | 2-3% opportunistic | Trivial (kernel reorder) |
| Thermal mass sprint mode | GPU at 40°C cold → 2.8+ GHz for 2-3s before 83°C throttle. Short ctx (<128 tok) finishes before throttle. Long ctx → 2.3 GHz sustained. No OC — reading thermals, picking lower sustained clock | 15-20% short ctx | Trivial (nvidia-smi + timer) |
| Power window gaming | 5ms max-power burst + 5ms low-power mem-bound. Rolling 50ms average stays under TDP. Peak throughput is higher. No OC — scheduling pattern only | 5% net throughput | Moderate |
| Voltage droop compensation | Pre-insert low-current ops (NOPs, shuffles) 10-20 cycles before high-current GEMV launch. VRM recovers, boost doesn't trigger. No OC — kernel instruction reorder | 1-2% clock stability | Moderate |
| Boost telemetry feedback | Read `nvmlDeviceGetCurrentClocksThrottleReasons` every 100ms. Power-limited→reduce batch. Thermal→insert gaps. Voltage→reduce clock. Self-tuning. No OC — read-only monitoring | Adaptive perf | Moderate |

---

---

## DUAL-FORMAT ENGINE: GGUF + .den NATIVE (NVFP4 speed parity)

**Problem:** NVFP4 GGUF dequantizes E2M1→float→OMMA. Two pointless transform steps. 160B NULLGLASS tiles are the OMMA B-fragment byte-for-byte — but GGUF can't express this layout. Result: NVFP4 GGUF = 150 tok/s vs Q4_K_M = 202 tok/s. Slower despite theoretically faster hardware path.

**Solution:** Dual-format engine. GGUF for compatibility. .den for native speed. OMMA direct path skips dequant. Both formats coexist — loader detects format, dispatch routes correctly.

### Phase 1: Direct OMMA Path for NVFP4 GGUF (2h, short-term)

| Step | File | What |
|------|------|------|
| 1a | `mmq.cuh` | Add `DEN_NVFP4_DIRECT_OMMA` path in `mul_mat_q` — when NVFP4 tensor AND `DEN_NVFP4_DIRECT=1`, skip `vec_dot_q_cuda()` dequant, feed raw tile bytes to `mma_block_scaled_fp4<NVFP4>()` |
| 1b | `mmq.cuh` | Detect tile format from dispatch byte (tile[148]==0x30 for NULLGLASS, tile[148]!=0x30 for standard block_size=16). Standard GGUF NVFP4 uses dispatch byte 0x00 (no metadata). Route accordingly |
| 1c | `mmq-config-blackwell.cuh` | Add `MMQ_ITER_K_FP4_DIRECT = 64` tile config for direct OMMA path (K=64 matches OMMA.SF.16864 instruction) |
| 1d | `fattn-nvfp4-kv.cu` | KV cache already uses 160B NULLGLASS tiles. Verify dispatch byte read in attention kernel |
| **Verification** | | 35B tg64 with `DEN_NVFP4_DIRECT=1`. Must match or exceed q8_0 baseline (185 tok/s). Replay test cos≥0.99 vs F32 |

### Phase 2: .den Loader Port (6h, medium-term)

| Step | File | What |
|------|------|------|
| 2a | `src/llama-den-loader.cpp` (NEW) | Port `dengine/src/den_core.c` (heap loader, not mmap). Adapt to llama_model_loader interface. Read .den header → tensor inventory → NULLGLASS tile extraction |
| 2b | `src/llama-den-loader.h` (NEW) | `llama_den_loader` class: `open()`, `get_tensor()`, `get_tensor_meta()`, `close()`. Maps to llama_model_loader pattern |
| 2c | `src/llama-model.cpp` | Add `LLM_FORMAT_DEN` to model load path. Detect `.den` extension → use `llama_den_loader`. Map .den slot IDs to GGML tensor names |
| 2d | `ggml/src/ggml-cuda/den-tile-loader.cuh` (NEW) | GPU-side: load 160B NULLGLASS tile → map to OMMA B-fragment registers. Byte-for-byte: scales→regs a0-a7, nibbles→regs b0-b3 |
| 2e | `include/den_format.h` (port from C:\) | Format spec v5: tile struct, slot assignment table, dispatch byte, holographic parent ptr, WH4 flag |
| **Verification** | | Load Qwen3.5-4B-BF16.den → cos>0.9999 vs HF. Load Qwen3.6-35B-A3B-NVFP4.den → cos>0.99 vs BF16. tg64 ≥ 184 |

### Phase 3: Dual-Format Dispatch (4h, medium-term)

| Step | File | What |
|------|------|------|
| 3a | `src/llama-model.cpp` | Format detection: `.gguf` → existing path. `.den` → new den_loader path. Both produce `llama_model` with same tensor interface |
| 3b | `ggml/src/ggml-cuda/mmq.cuh` | Runtime tile format dispatch: `tile[148]==0x30` (NULLGLASS, .den) → direct OMMA. `tile[148]==0x00` (GGUF NVFP4) → current dequant+OMMA. `tile[149]==8` (WH4) → WH4 OMMA path |
| 3c | `ggml/src/ggml-cuda/den-tile-dispatch.cuh` (NEW) | Central dispatch: `den_dispatch_tile(tile)` → returns `DEN_TILE_FMT_NULLGLASS`, `DEN_TILE_FMT_GGUF_NVFP4`, `DEN_TILE_FMT_WH4`. All kernel paths branch here |
| 3d | `tools/den_convert_heretic.py` (port from C:\) | HF safetensors → .den converter. BF16 + NVFP4 paths. Already working at C:\Den. Adapt for llama.cpp include paths |
| **Verification** | | Same model as GGUF + .den → identical output. Replay test: same prompt, both formats → token-identical output. tg64: .den ≥ GGUF |

### Phase 4: .den-Native NVFP4 Converter (4h)

| Step | File | What |
|------|------|------|
| 4a | `tools/convert_nvfp4_to_den.py` (NEW) | NVFP4 safetensors → .den with NULLGLASS tiles. Read block_size=16 NVFP4 → repack to 160B NULLGLASS format. Compute tile norms + dispatch byte + k_stride |
| 4b | `tools/den_calibrate.py` (port from C:\) | AWQ + NVFP4 calibration pipeline (1337 lines). BF16 .den → NVFP4 .den with per-tensor calibration |
| 4c | `tools/build_nvfp4_den.py` (port from C:\) | Modelopt safetensors → .den (pure Python). Already working at C:\Den |
| **Verification** | | Convert Gemma4-26B NVFP4 safetensors → .den. Load + infer → cos>0.99 vs BF16. tg64: .den beats GGUF NVFP4 by 20%+ |

### Coherence Guard at Every Step

| Gate | What | Threshold |
|------|------|-----------|
| **Replay test** | 1000-token F32 vs NVFP4 token-identical | Must match 100% |
| **Cos similarity** | Layer outputs vs HF reference | cos ≥ 0.9999 (BF16), cos ≥ 0.99 (NVFP4) |
| **Golden rule** | 35B tg64 | ≥ 184 tok/s |
| **PPL** | wikitext-2 | PPL diff < 0.5% vs F32 |
| **Multi-turn** | 10 turns, no state drift | Token-identical through turn 10 |

### File Map — What Moves from C:\Den

| Source (C:\) | Destination (I:\) | Purpose |
|-------------|-------------------|---------|
| `dengine/src/den_core.c` | `src/llama-den-loader.cpp` | .den file loader |
| `dengine/include/den_format.h` | `include/den_format.h` | Format spec v5 |
| `dengine/compute/omma_dispatch.cu` | `ggml/src/ggml-cuda/den-tile-dispatch.cuh` | Tile→OMMA dispatch |
| `dengine/src/den_omma_gemv_v2.cu` | `ggml/src/ggml-cuda/den-omma-direct.cuh` | Direct OMMA kernel |
| `tools/den_convert_heretic.py` | `tools/den_convert_heretic.py` | HF→.den converter |
| `tools/den_calibrate.py` | `tools/den_calibrate.py` | AWQ+NVFP4 calibration |
| `tools/build_nvfp4_den.py` | `tools/build_nvfp4_den.py` | Modelopt→.den converter |

---

## MULTIMODAL COMPATIBILITY: iDREAM + COMFYUI + TRELLIS + AUDIO

**Problem:** llama.cpp is text-only. GGUF format can't express vision embeddings, 3D latent grids, audio spectrograms, or ComfyUI node graphs. All multimodal pipelines exist at C:\Den as CUDA kernels + Python bridges — but they target the dengine runtime, not llama.cpp's ggml backend.

**Strategy:** .den format is the bridge. The 160B NULLGLASS tile was designed for mixed tensor types (text + vision + audio in same model file). GGUF stays for text-only compatibility. .den unlocks everything else.

### Compatibility Tier 1: .den Loader (prerequisite for ALL below)

| Step | Source (C:\) | Destination (I:\) | Purpose |
|------|-------------|-------------------|---------|
| C1 | `dengine/src/den_core.c` | `src/llama-den-loader.cpp` | .den file loader with heap allocation |
| C2 | `dengine/include/den_format.h` | `include/den_format.h` | Format spec v5: NULLGLASS tiles, slot table, dispatch byte, WH4 flag |
| C3 | — | `src/llama-model.cpp` | Format autodetect: `.gguf`→GGUF path, `.den`→den path |
| C4 | `dengine/src/den_omma_gemv_v2.cu` | `ggml/src/ggml-cuda/den-omma-direct.cuh` | OMMA 4X SASS-verified kernel: tile→register→OMMA, zero dequant |

### Compatibility Tier 2: Multimodal Tensor Support

| Capability | Source (C:\) | What it does | GB203 silicon |
|------------|-------------|--------------|---------------|
| **Vision encoder** | `cuda_kernels/vision/siglip2_encoder.cuh` | SigLIP 2 ViT: image→embedding tokens. 400M params, NVFP4 quantized | OMMA + TMU texture cache |
| **DAPS analyzer** | `cuda_kernels/vision/daps_analyzer.cuh` | Dreya Appearance Parameter Set: 512B struct, face/pose/emotion from image | TMU edge detect + SM |
| **NVOF screen watcher** | `dengine/compute/nvof_screen_watcher.cu` | Optical flow on user's screen → attention saliency map | NVOF hardware |
| **Curiosity salience** | `dengine/compute/curiosity_salience.cu` | Salience detection on visual input → attention weighting | SM + TMU |
| **Audio pipeline** | `dengine/src/den_audio_pipeline.cu` | Audio spectrogram→NVFP4 embedding + wolf vocal tract DSP | SM + WASAPI |
| **Voice transport** | `dengine/src/den_voice_transport.c` | Wolf vocalizations: growl/yip/whine/snarl/howl/chuff DSP | CPU DSP (WASAPI) |

### Compatibility Tier 3: iDream World Engine

| Module | Source (C:\) | Function | Status |
|--------|-------------|----------|--------|
| **iDream pipeline** | `dengine/include/idream_pipeline.h` | Orchestrator: T2I→3D→render→evaluate→refine loop | Code complete |
| **Trellis 2 bridge** | `dengine/compute/trellis2_bridge.cu` | Image→3D mesh/texture via TRELLIS.2 4B NVFP4 | Compiles |
| **Lance 3B T2I** | `bridge/diffusion_lance.py` | Text→image via Lance 3B NVFP4 diffusion | Python bridge |
| **Hunyuan3D-2** | Referenced in idream_pipeline.h | Image→3D async geometry | Not ported |
| **O-Voxel** | Referenced in idream_pipeline.h | Persistent 3D world state | Design only |
| **Sana 0.6B** | Referenced in idream_pipeline.h | Fast T2I for preview | Python bridge |

### Compatibility Tier 4: ComfyUI NVFP4 Ecosystem

| Module | Source (C:\) | Function |
|--------|-------------|----------|
| **ComfyUI NVFP4** | `cuda_kernels/vision/` | 6 files: DAPS extraction, TMU edge detect, NVFP4 tile format for images |
| **NVFP4 image encode** | `cuda_kernels/fused/audio_vision_preprocess.cuh` | RGB→NVFP4 tile conversion |
| **PiD cascade** | `cuda_kernels/vision/` | Progressive image decode: 512→1024→2048 upscaler |
| **FLUX IP-Adapter** | `bridge/` | Face compositing + style transfer |
| **SCIDiT encoder** | `bridge/` | Identity Encoder + Disentanglement for face consistency |
| **Superposition renderer** | `bridge/` | Multi-layer image compositing |

### Compatibility Architecture

```
User input (text / image / audio / comfyui-graph)
    │
    ▼
llama.cpp multimodel loader
    ├── .gguf → text-only GGUF path (existing)
    └── .den  → multimodal den path (NEW)
         │
         ├── Text tokens → llama_model (existing GGML tensors)
         ├── Vision embeddings → SigLIP 2 ViT encoder (port)
         ├── Audio spectrogram → Audio encoder (port)
         ├── ComfyUI graph → Node executor (port)
         └── iDream pipeline → World engine orchestrator (port)
              │
              ▼
         Single OMMA dispatch (unified: text + vision + audio tiles)
              │
              ▼
         Output: text tokens / image tensor / audio waveform / 3D mesh
```

### Why .den is the key

GGUF NVFP4: block_size=16, weight+scale separate arrays, GPU repacks at load time. Text-only tensor layout. Cannot express vision encoder weights in same file without hacks.

.den NULLGLASS: 160B contiguous tiles. Byte-for-byte = OMMA B-fragment. Dispatch byte routes per-tile format (BF16 / NVFP4 / WH4 / K8V4). Holographic parent pointer enables tensor→layer mapping. One file contains text + vision + audio tensors — the format was designed for this.

### Multimodal port priority

| Priority | What | Why |
|----------|------|-----|
| **1** | .den loader + OMMA direct path | Unlocks ALL multimodal. Single dependency |
| **2** | Vision encoder (SigLIP 2) | Required for DiffusionGemma, ComfyUI, DAPS |
| **3** | DiffusionGemma NVFP4 | Text→image, iDream T2I stage. GGUF already downloading |
| **4** | Trellis 2 bridge | Image→3D, iDream core pipeline |
| **5** | Audio pipeline | Voice transport + wolf vocalizations |
| **6** | ComfyUI node executor | Full ComfyUI NVFP4 graph on GPU |
| **7** | iDream orchestrator | T2I→3D→render→evaluate→refine loop |
| **8** | DAPS + SCIDiT + FLUX | Face consistency + identity for characters |

### Coexistence with text-only GGUF

```
GGUF path (existing, unchanged):
  .gguf → llama_model_loader → GGML tensors → mmq dispatch → OMMA

.den path (new, parallel):
  .den  → llama_den_loader → GGML tensors → den-tile-dispatch → OMMA direct
                                                      ↑
                                            NULLGLASS 160B tiles
                                            zero repacking
```

Both paths produce the same `llama_model` interface. The ggml backend doesn't care which loader filled the tensors. Multimodal tensors (vision, audio) are additional tensor slots in the model — the ggml graph just sees more tensors.

---

## TESTING + QUALITY EMPHASIS (tok/s AND accuracy both paramount)

### Current test coverage

| Test | What | Status |
|------|------|--------|
| Golden rule CI | 35B tg64 ≥ 184 on push | ✅ `bench-guard.yml` |
| Replay test | 1000-token F32 vs NVFP4 token-by-token | ✅ `test_nvfp4_replay.py` (manual) |
| SASS audit | Instruction count stability | ✅ `.claude/hooks/sass-audit.py` |
| Batch scaling | F32 vs NVFP4 max batch before OOM | ✅ `bench_batch_scaling.bat` |
| Context scaling | F32 vs NVFP4 max context before OOM | ✅ `bench_context_scaling.bat` |
| ncmoe16 parity | Offload perf ≥ 53.7 t/s | ✅ `bench_ncmoe16_verify.bat` |

### NEEDED: Automated quality regression guard

| Test | Mechanism | Priority |
|------|-----------|----------|
| **Cos similarity CI** | Run replay test on push, compare hash vs golden. Fail CI on divergence. `test_nvfp4_replay.py --golden` | IMMEDIATE |
| **PPL measurement** | Perplexity on wikitext-2 test set. F32 KV vs NVFP4 KV. PPL diff > 0.5% → fail. Run nightly | SHORT-TERM |
| **Multi-turn coherence** | 10-turn conversation. NVFP4 output = F32 output token-identical through turn 10. Detect state drift | SHORT-TERM |
| **Long-context needle** | Needle-in-haystack at 32K/64K/128K. NVFP4 finds needle at same position as F32. Position error > 1% → fail | SHORT-TERM |
| **Attention sink verification** | Verify first 4 tokens receive 60%+ attention mass in NVFP4 mode (matching F32). If KVSink anchors not working → fail | MEDIUM |

### Testing integration

```
CI pipeline (on push):
  1. Build (build_den.ps1)
  2. SASS audit (sass-audit.py --check) → fail if instructions changed
  3. Golden rule (bench-guard.yml) → fail if tg64 < 184
  4. Replay test (test_nvfp4_replay.py --golden) → fail if hash diverged
  5. Report status to commit

Nightly pipeline:
  1-4 above
  5. PPL measurement on wikitext-2
  6. Multi-turn coherence (10 turns)
  7. Long-context needle (32K/64K)
  8. Full benchmark matrix (bench_full_matrix.bat)
  9. Email report
```

---

## PRIORITY ORDER (updated 2026-08-09 — session results integrated)

### TRI-VECTOR GATE (the measurement enabler — /btw #2,#3)
| Vector | Target | Status |
|--------|--------|--------|
| **Speed** (tg64) | ≥ 184 tok/s (35B) | 179.57 baseline. Persistent kernel + OMMA = the lever. |
| **Accuracy** (KLD/cos) | cos ≥ 0.9995, KLD < 0.001 | ✅ **PASS** — KLD=0, cos=1.0, top1=100.0% |
| **Context** | 256k target, 128k minimum | NVFP4 KV (1.7 GB at 256k) is the ONLY fit on 16 GB. q8_0 OOMs (5.4 GB). Gated on sparse-VMM + expert offload. |

Gate tool: `tools/gate_accuracy_kv.py` (in-process ctypes, dual-context, logit-level KLD+cos). Replaces broken subprocess token-match replay.

### IMMEDIATE — finish build + gate the persistent kernel
1. **Fix persistent kernel build** — macros done (den_unified_kernel.cuh). Rerun build_now.bat.
2. **Golden gate: speed** — tg64 after persistent kernel. Target ≥ 184.
3. **Golden gate: accuracy** — KLD/cos must hold at 0/1.0 (regression guard).

### NEXT — .den native decode (25% GGUF overhead elimination)
4. **Seam A: .den loader wiring** — detect .den → route to llama_den_loader. ~50 lines.
5. **Zero-copy NULLGLASS tile load** — skip GGUF repack. OMMA vec_dot needs ZERO changes.
6. **Full .den→OMMA integration** — ~200 lines across 4 seams. Target: 202+ tok/s (beat Q4_K_M).

### THEN — 256k context proof
7. **NVFP4 KV max-context harness** — measure actual F32 vs NVFP4 context limits. NVFP4 is the ONLY 256k enabler.
8. **Sparse-VMM KV routing** — wire the existing sparse-vmm core to KV cache allocations.
9. **Expert offload verification** — ncmoe16 at 256k context must not regress below 184.

### SPEED CEILING — tcgen05 patterns ported to sm_120a
10. **Multi-SM persistent kernel** — gridDim=70 (ported, needs build)
11. **Warp-specialized OMMA** — producer/consumer/coordinator (ported)
12. **N-stage software pipelining** — smem double-buffering (follow-up)
13. **128B tile alignment** (160→256B) — format version bump (follow-up)

### ARCHITECTURE — precision-tiered .den format
14. **jashepp-style 3-tier precision** (F16 critical + Q8_0 backbone + NVFP4 experts) — adapt to NVFP4 (finer 16-elem scale beats MXFP4 32-elem). .den per-tensor precision metadata.
15. **Asymmetric K/V** — K=8-bit, V=4-bit (K >> V sensitivity, community-proven). Trivial dtype switch.
16. **Hadamard/WHT pre-transform** — highest ceiling for KV accuracy (QuaRot/KVarN). We have WHT infra (WH4 path).

### DREYA — cognitive loop
17. **PAD feedback** ✅ done. tg64→arousal, confidence→dominance.
18. **PAD→Governor closed loop** (auto-downshift on frustration) — follow-up.
19. **RT Core BVH memory lookup** — Plausible. Same mechanism as den-rt-expert-router.cuh.

### SYSTEM — PC-level
20. **Defender exclusions** ✅ done. I:\models, I:\den_llama.cpp excluded.
21. **CPU affinity** — pin daemons to cores 0-3, llama.cpp to 4-7.
22. **D: NVMe install** — Sandisk GX 7100 500GB. Cold-expert fetch from NVMe vs SATA vs system RAM.
23. **VSS/disk stall** — vssadmin + fsutil disablelastaccess on I:.

### RESEARCH BACKLOG — see `plans/RESEARCH_FANTASY.md`
Tier 1 (plausible): PCIe atomics, doorbells, TMU cache, L1 config, Boost telemetry, RT Core BVH
Tier 2 (speculative): NVDEC, VIC, NVENC ME, DSC
Tier 3 (impossible): L2 cross-SM, GDDR7 refresh, ROP DCC, PAM3 raw
Tier 4 (wrong time): iDream/Trellis/ComfyUI/DAPS, GSP, NVOF, SASS opt
Rule: any item must pass 10-line standalone CUDA test before entering active plan

### CUT FROM ACTIVE PLAN (move to RESEARCH_FANTASY.md)

These mechanisms do not exist in CUDA, are user-space inaccessible, or are physically impossible on GB203. A single developer cannot spend days on fiction.

| Item | Why cut |
|------|---------|
| A2.1 SM-specialized layer pipelining | L2 has no cross-SM messaging. "Atomic write+signal" not real |
| A2.2 Activation tile multicast | No broadcast flags or tag polling in CUDA |
| A2.3 L2 work-stealing queue | L2 atomics ~500ns. Slower than DRAM for queues |
| A1.2 Dual-graph execution | Cannot run attention for N+1 before token N sampled |
| A1.1 GPU-side sampler | 40μs on 20ms = 0.2%. Below noise |
| A1.3 Batch-ahead scheduling | At batch=1, nothing queues ahead |
| GDDR7 PAM3 alignment | Memory controller handles alignment transparently |
| ECC scrub-aware scheduling | No user-space API for ECC scrub phase |
| Per-bank refresh staggering | Not user-space controllable |
| NVDEC as ANS decoder | CABAC ≠ ANS. Hardware not programmable for this |
| Crypto engine AES-GCM | HDCP block not accessible from CUDA |
| Display Engine DSC | Display pipeline only. No CUDA API |
| VIC as GEMM | 2D image transforms only. Not general matmul |
| NVENC ME as sparse mask | NV12 surface conversion + API overhead > any compute saved |
| ROP DCC compression | DCC metadata undocumented. Reverse-engineering = research project not optimization |
| Warp scheduler / register bank audit | SASS-level optimization without NVIDIA tools = impossible |
| iDream / ComfyUI / DAPS | VRAM death on 16GB alongside 35B. Phase 2 (24GB+) only |
| Multimodal tensors in engine | GGUF can't express them. Sidecar files, not engine fork |
| .den as runtime fork | KILLED. .den = converter → upstream GGUF → existing dispatch |

### KEEP — real, actionable, with documented APIs

| Item | Why |
|------|-----|
| Sinkhorn wiring | 30 lines. Has kernel bias path. Immediate MoE gain |
| Dual CE | 10 lines. cudaStreamCreateWithPriority. CE1 confirmed on GB203 |
| L2 persist | cuMemAdvise SET_ACCESSED_BY. Documented CUDA API |
| Split-K soft-GEMV | Standard CUDA: grid parallelism, warp reductions, vectorized loads |
| Persistent kernel port | Real source code at C:\Den. Conditional on launch overhead >10% |
| RT expert router tiers 2+3 | GPU brute-force + GEMV. Real compute. Tiers 0+1 deferred |
| PCIe atomics | Documented PCIe spec feature. fetch_add, CAS over bus |
| GPU doorbells (research) | BAR1 MMIO is real but fragile. Lower priority than atomics |
| Sparse-VMM KV routing | cuMemAddressReserve + cuMemCreate + cuMemMap. All documented |
| SASS audit hook | cuobjdump works. Already implemented |
| Fused layer boundary | Real CUDA kernel. Lower priority than split-K |
| TMU texture cache | CUDA texture API is documented. Experiment, don't ship blind |
| L1 cache config per-kernel | cudaFuncSetAttribute is real API |

### Split-K Soft-GEMV — Implementation Spec (highest priority throughput lever)

**Files:** `ggml/src/ggml-cuda/mmvf.cu` (BF16 GEMV path), new `ggml/src/ggml-cuda/splitk-gemv.cuh`
**Lines:** ~400
**Grid:** 2D: `dim3(gridX=M/ROWS_PER_BLOCK, gridY=K_SPLITS)` — splits K dimension across SMs
**Per-block:** vectorized loads (int4/128-bit), warp-level partial sum, atomicAdd to output
**Auto-tune space:** ROWS_PER_BLOCK∈{1,2,4}, THREADS∈{64,128,256}, K_SPLITS∈{2,4,8}, VEC_K∈{1,2,4,8}
**Validation:** cos≥0.999 vs current soft-GEMV. Deterministic replay 1000-token pass. Target: 80+ tok/s from 38.56 baseline
**Constraint:** Do NOT touch MoE FFN expert OMMA path. Do NOT fuse RMSNorm yet. Attention/GDN only.

### HARDWARE CORRECTION — PCIe 4.0 x16 (B650 chipset cap, NOT 5.0)

**Gigabyte B650 Gaming X AX V2 caps GPU slot at PCIe 4.0.** GPU-Z confirmed: x16 4.0. Practical H2D: ~55 GB/s, NOT 100 GB/s. This invalidates "Streaming MoE Offload" — pivot to "Predictive Resident MoE."

| Parameter | PCIe 5.0 (assumed) | PCIe 4.0 x16 (actual) |
|-----------|-------------------|----------------------|
| Practical H2D BW | ~100 GB/s | ~55 GB/s |
| Per-token budget at 184 tok/s (5.43ms) | ~540 MB | **~298 MB** |
| 8 NVFP4 experts (~40MB each) | 320 MB ✅ | 320 MB ❌ OVER BUDGET |
| L2 persistence | Optimization | **MANDATORY** (L2 miss → PCIe fetch → budget blown) |
| NVFP4 for RAM staging | Optional | **MANDATORY** (halves PCIe bytes vs Q8_0) |

**Rule: `predicted_expert_pull_mb > 298 * 0.80` → trigger Epoch Swap (bulk-load working set into VRAM/L2), not per-token streaming.**
**Strategy shift: Context-Epoch Swapping. Shadow Router predicts macro-topic, pre-loads 50 experts for next ~500 tokens. Per-token PCIe pull drops to <20 MB.**

### AMENDMENTS FROM MULTI-REVIEWER AUDIT (2026-08-09)

**1. Coherence Gate: distributional equivalence, not string identity (Qwen)**
Token-identical output is statistically impossible for 4-bit vs 8-bit beyond ~40 tokens. Replace with:
- Logit cosine similarity ≥ 0.99 (top-10, first 20 tokens)
- KL-divergence < 0.01 vs Q8_0 oracle
- PPL delta < 0.5% on fixed validation set
- String matching only for deterministic replay (fixed seed, exact hardware path)

**2. Phase 0 artifact: `coherence_gate.py` must exist (Claude)**
Currently not in any inventory table. Must be written + committed before declaring Phase 0 locked. Add to TESTING LAYER.

**3. .den importer: `GGML_TYPE_NVFP4_NULLGLASS`, not tile[148] branch (z.ai)**
Do NOT branch on payload bytes in mmq.cuh. Add new GGML type enum. .den loader tags tensors with it. mmq.cuh dispatches on type enum, not payload contents. This preserves "importer" boundary cleanly.

**4. WDDM TDR chunking mandate (Qwen)**
Consumer GPU + Windows 11 = WDDM, not TCC. TDR timeout ~2s. All persistent kernels + CUDA graphs must chunk/yield at <1.5s. TdrDelay=10 registry override for dev only.

**5. L2 partitioning strategy (Gemini/Qwen)**
cudaAccessPolicyWindow, not just cuMemAdvise. Explicit MB budgets: 20MB Experts, 10MB KV Cache, 10MB Activations. hitProp=cudaAccessPropertyPersisting, missProp=cudaAccessPropertyStreaming.

**6. 70-SM wave alignment (Gemini)**
GB203 has exactly 70 SMs. Split-K grid must be exact multiple of 70 to eliminate tail-wave idle. Use `cudaDevAttrMultiProcessorCount`. Total blocks = N × 70.

**7. Split-K FP32 reduction rule (Qwen)**
FP16 atomics forbidden for reduction step. Use FP32 atomics or two-pass kernel (Pass 1: partials to temp buffer, Pass 2: reduce).

**8. Golden benchmark: add Seed row (Claude)**
Doctrine says seed is mandatory. Table is missing it. Add it.

**9. Multimodal section: add deferral banner (Claude)**  
"Reference architecture only. Deferred to Phase 5 per CUT FROM ACTIVE PLAN. Not being built."

**10. RESEARCH_FANTASY authority (Claude)**
"RESEARCH_FANTASY.md tiering is authoritative on conflict." ROP DCC + Cross-SM L2 move to BLACKBOX (matching Tier 3 "physically impossible").

**11. 6827 ported lines: build status explicit (Claude)**
State whether vision/iDream kernels are CMake-excluded or compiled-but-dead. "Ported, CMake-excluded pending Phase 5."

**12. VRAM Orchestrator: NVMe tiering + predictive shadow eviction (Qwen/z.ai)**
Swap path: VRAM → system RAM → NVMe (DirectStorage/async I/O). Cognitive layer detects intent → immediately begins flushing cold layers before generation request. Fast-resume: load only active KV + current layer, not full cold boot.

### FORMAL DEVELOPMENT PHASES (Qwen directive — adopted 2026-08-09)

**Phase 0 — COHERENCE LOCK** (supersedes all performance work)
```text
NVFP4 9B and 35B produce coherent 100-token greedy output matching Q8_0 oracle.
State evolution verified. Replay hash locked. PPL stable.
Exit: coherence_gate.py passes. THEN golden rule applies.
```

**Phase 1 — WIRE INFRA** Sinkhorn / Dual CE (capability-detected) / L2 (access-policy, not just cuMemAdvise)
**Phase 2 — DECODE BOTTLENECK** Split-K soft-GEMV / GPU sampler / Router-logit reuse / Co-activation Markov
**Phase 3 — KV QUALITY** NVFP4 KV full validation (needle/PPL/multi-turn) + Sparse-VMM KV routing
**Phase 4 — .den IMPORTER** Direct OMMA for NVFP4 tiles (role-gated: MoE FFN only). Loader validates tiles before kernels see them
**Phase 5 — MULTIMODAL** Vision → Diffusion → Voice → Video → 3D → iDream

### GOLDEN BENCHMARK CONFIGURATION (must be lockable)

| Parameter | GOLDEN-A (GPU warm) | GOLDEN-B (offload) | GOLDEN-C (dense) |
|-----------|---------------------|--------------------|--------------------|
| Model | Ornith-35B APEX Mini | Ornith-35B APEX Mini | Ornith-9B |
| Quant | Q8_0 KV | Q8_0 KV | NVFP4 |
| Context | 4096 | 4096 | 4096 |
| Batch | tg64=1 | tg64=1 | tg64=1 |
| Offload | GPU-resident active set | ncmoe16 | GPU-resident |
| Expert cache | Warm | Cold→warm | N/A |
| Sampler | Greedy | Greedy | Greedy |
| Seed | Fixed (42) | Fixed (42) | Fixed (42) |
| VRAM ceiling | <14.5 GB | <14.5 GB | <12 GB |
| RAM ceiling | <28 GB | <28 GB | <28 GB |

### ENGINEERING DOCTRINE

1. Coherence outranks speed. CPU oracle authoritative for numerical disputes.
2. OMMA forbidden on GDN/attention until proven safe. Role-gate, not type-gate.
3. GDN/SSM state: BF16 transport, FP32 accumulation. Never quantize destructively.
4. .den = importer advantage, not runtime fork. Materializes GGML tensors.
5. Silicon exploits: SHIP / LAB / BLACKBOX. No firmware mod. No undocumented MMIO.
6. No D: drive access. No WSL builds whose VHDX was on D:.
7. Every optimization: before/after benchmark + coherence gate + causality report.
8. PAD/emotion: cognitive layer only. Never precision, dispatch, or hot-path kernels.
9. Golden benchmark: full config lock. Scalar "184 tok/s" is underspecified without it.
10. Dual CE: auto-detect via cudaDevAttrAsyncEngineCount. Degrade gracefully.

### COHERENCE GATE SPECIFICATION (Phase 0 exit criteria)

**File:** `tools/coherence_gate.py` (MUST EXIST before Phase 0 declared locked)
**Input:** Model path, tokens=100, seed=42
**Tests:**
1. Logit cosine similarity (top-10, first 20 tokens) ≥ 0.99 vs Q8_0 oracle
2. KL-divergence < 0.01 vs Q8_0 oracle (first 100 tokens)
3. PPL delta < 0.5% on wikitext-2 validation set
4. state_in RMS > 0 after token 1 (SSM recurrence evolving)
5. delta_net_fused_raw correlation ≥ 0.98 vs Q8_0 (first 20 tokens)
6. Deterministic replay: 1000-token string match under greedy + fixed seed (exact hardware path only)
**Exit:** All 6 gates PASS → Phase 0 locked. THEN golden rule applies.

### REFERENCE CODE — Gemini L2 Access Policy + 70-SM Split-K Grid

L2 persistence via `cudaAccessPolicyWindow` (not just cuMemAdvise):
```cpp
cudaAccessPolicyWindow window = {};
window.base_ptr  = expert_ptr;
window.num_bytes = 24 * 1024 * 1024; // 24MB of 40MB L2
window.hitRatio  = 1.0f;
window.hitProp   = cudaAccessPropertyPersisting;
window.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &window);
```

70-SM wave-aligned Split-K grid:
```cpp
const int num_sms = 70; // GB203, not power-of-two
int total_blocks = base_grid_x * k_splits;
total_blocks = ((total_blocks + num_sms - 1) / num_sms) * num_sms; // pad to wave
```
See `plans/` for full Gemini code blocks (L2 window setup, Split-K calculator, verify_build.ps1).

### CRITICAL RULE: OMMA stays OFF attention/GDN path
- OMMA.SF.16864 measured SLOWER on GDN attention: 19.78 vs 38.56 tok/s soft-gemv
- E2M1 activations garble softmax → attention output incoherent
- OMMA ONLY for MoE FFN expert weights (activations tolerate E2M1 there)
- Step 1d (wiring OMMA into fattn-nvfp4-kv.cu) KILLED. Attention stays on soft-gemv
- Blocker 5 renamed: "GDN decode under-parallelization" not "GEMV compute-bound"

### .den STRATEGY: importer, not runtime
- .den → Den compiler/importer → upstream-compatible GGML tensors → normal dispatch
- NOT: .den → special runtime → special tensor dispatch
- `llama-den-loader` converts .den → GGML tensors at load time. No runtime fork

### SILICON CATALOG: separate from engineering roadmap
- SHIP: proven + directly measurable + user-visible
- LAB: plausible + requires experiment
- BLACK BOX: undocumented / driver-dependent / speculative
- Only crypto engine + PCIe atomics have real documented APIs for compute use

### Immediate (this session)
1. **Sinkhorn wiring** — modify bias buffer in llama-graph.cpp, 30 lines, 30min
2. **Dual CE wiring** — route expert H2D through CE1 stream, 10 lines, 15min
3. **L2 persist wiring** — cuMemAdvise on expert GPU buffers, 5 lines, 10min

### Short-term (next session)
4. **GPU-resident sampler (A1.1)** — `cub::DeviceRadixSort` top-K on GPU, 6h
5. **Router logit reuse (A3.1)** — adjacent-token expert prefetch, 4h
6. **Co-activation Markov (A3.3)** — 128KB transition matrix in L1, 4h
7. **Persistent kernel port (Blocker 5 fix 2)** — from C:\Den den_persistent_kernel.cu, 8h

### Medium-term
8. **KVarN+NVFP4 stacking (Gap 8)** — hook NVFP4 into KVarN store path, domain handling, 6h
9. **Sparse-VM KV routing** — allocate KV tensors through sparse buft, 3h
10. **ThriftAttention BF16 selective (B1)** — 5% block promotion with precision bitmap, 4h
11. **RT expert router tiers 0+1** — OptiX + inline PTX port from C:\, 8h

### Research/design
12. **Cross-SM L2 pipeline (A2.1-A2.3)** — 12h+8h+6h
13. **TMU texture cache (Exploit 4)** — tex1Dfetch weight tiles, 6h
14. **ROP DCC compression (Exploit 3)** — DCC metadata format, 8h
15. **NVENC ME attention (Exploit 2)** — NV12 float→surface, 12h

---

---

### .den UNIVERSAL OBJECT — 5-PLANE FORMAT SPEC (Qwen directive)

**Plane 1 — Tensor:** weights with role tags (EMBEDDING, ATTN_Q/K/V/O, FFN_GATE/UP/DOWN, MOE_ROUTER/EXPERT, SSM_STATE/DELTA_PROJ, NORM, VISION_ENCODER/PROJ, AUDIO_ENCODER/DECODER, DIFFUSION_LATENT/TIMESTEP/CONDITION, VAE_ENCODER/DECODER, SCENE_3D, ADAPTER, COGNITIVE_STATE). Role-gating prevents OMMA from touching attention/GDN.
**Plane 2 — Graph:** declarative block types (delta_net_layer, moe_ffn_block, attention_block, vision_encoder_block, diffusion_denoiser_block, vae_encode/decode, trellis_scene_generation). Engine compiles to GGML/CUDA.
**Plane 3 — State:** RECURRENT_SSM, KV_CACHE, EXPERT_CACHE, AUDIO_STREAM, VIDEO_FRAME, DIFFUSION_LATENT, SCENE_3D, COGNITIVE_PAD, MEMORY_EPISODIC/SEMANTIC. Each has precision, lifetime, eviction policy, device preference.
**Plane 4 — Modality:** IO contracts (TEXT_IN/OUT, IMAGE_IN/OUT, AUDIO_IN/OUT, VIDEO_IN/OUT, SCENE_3D_IN/OUT). Schema, preprocessing, latency budget, sync model.
**Plane 5 — Pipeline:** multi-stage graphs with loops (diffusion: text_encoder→denoiser[28 steps]→VAE→image. Trellis: prompt→3D latent→mesh→scene insert→render).

### ENGINE ARCHITECTURE — 5 LAYERS

**L0 — Hardware Abstraction:** GB203, 16GB VRAM, CUDA 13.3, sm_120a, CE0/CE1, L2, streams, sparse VMM, OMMA/soft-GEMV constraints. Modality-agnostic.
**L1 — Universal Tensor Runtime:** GGML-compatible core. Allocation, quantization dispatch, matmul, attention, soft-GEMV, norm, sampling. Does not know what "voice" or "3D mesh" means.
**L2 — Memory/State Manager:** KV cache, SSM state, MoE expert cache, sparse VMM routing, host staging, residency, eviction, checkpointing. Memory management > any single kernel.
**L3 — Modality Compilers:** TextDecoder, MoE, VisionEncoder, AudioStream, DiffusionPipeline, VideoTemporal, Trellis3D, ComfyUI. Output: GGML tensors + graph blocks + state descriptors + policies + quality gates.
**L4 — Orchestration:** Multi-stage workflows. iDream lives here, not in kernels.
**L5 — Cognitive:** Dreya. PAD, personality, memory, attachment. May influence prompt/sampling/weighting. Never touches precision, dispatch, or hot-path kernels.

### MODALITY DESIGN REQUIREMENTS

| Modality | Core components | Precision policy | State |
|----------|----------------|-----------------|-------|
| Dense text | Q8_0 oracle, NVFP4 optional, CUDA Graphs | BF16 transport, FP32 accum | KV cache |
| MoE text | Expert offload, router reuse, Sinkhorn, Dual CE, L2 persist | NVFP4 expert weights OK, BF16 elsewhere | KV + expert cache + router state |
| Diffusion | Text encoder, UNet/DiT, scheduler, VAE | NVFP4 weights, BF16 activations, FP32 scheduler | Latents, step state |
| Voice | Audio capture, encoder, tokenizer, decoder, vocoder | BF16 | Streaming buffers, VAD, prosody |
| Video | Frame decoder, spatial/temporal encoder, motion model | NVFP4 weights, BF16 frames | Frame queue, motion vectors, temporal KV |
| 3D/Trellis | Condition encoder, latent generator, mesh decoder, scene graph | NVFP4 weights, BF16 latents | Scene graph, camera, world objects |

### FORMAL OBJECT MODEL

DenTensor → DenGraph → DenState → DenPipeline → DenModality → DenMemoryPolicy → DenExecutionPolicy → DenCalibration → DenAsset → DenBinding → DenCognitiveBinding → DenCompatibilityManifest

### CI/TEST PIPELINE

**Push:** Build → SASS audit → NVFP4 validator → Coherence gate (Q8_0 oracle vs NVFP4 CPU vs NVFP4 GPU, state evolution) → Golden benchmark (config-locked) → Replay hash → VRAM peak
**Nightly:** Push pipeline + PPL wikitext-2 + Multi-turn 10 turns + Needle 32K/64K/128K + Expert offload soak + NVFP4 KV context scaling + Batch scaling + Sinkhorn stability
**Weekly:** 1hr continuous soak + Expert cache fragmentation + Sparse-VMM growth + MTP/DFlash acceptance stability + Backup verification

### OMMA ROLE GATING (not type gating)

```cpp
if (tensor_type == NVFP4 &&
    tensor_role == MOE_FFN_EXPERT_WEIGHT &&
    activation_policy == E2M1_SAFE &&
    omma_direct_enabled) { use_omma_direct(); }
```

| Role | OMMA | Role | OMMA |
|------|------|------|------|
| MoE FFN expert weights | Allowed | Attention Q/K/V/O | Disallowed |
| Dense FFN weights | Conditional | GDN/delta_net recurrence | Disallowed |
| Embedding/output head | Disallowed | KV cache | K8V4/KVSink only |

### SHIP / LAB / BLACKBOX — Full Adjudication

**SHIP:** Dual CE (if asyncEngineCount≥2), L2 persist (access-policy, not just cuMemAdvise), GPU sampler, Router-logit reuse (shadow first), Split-K soft-GEMV, CUDA Graphs (static regions), L1/SMEM carveout, Streaming eviction hints
**LAB:** TMU cache, NVOF attention, PCIe atomics, VIC matrix, NVDEC decompression, GDDR7 PAM3, Register bank conflict, Persistent kernel (only if launch>10%), Cross-SM L2 pipelining, ROP/DCC
**BLACKBOX (not in mainline):** GSP firmware (no SDK), Crypto AES (no API), DSC sparsity (display only), GPU doorbell MMIO (driver risk), GDDR7 refresh (driver opacity), Firmware hot-patch (forbidden), Undocumented MMIO (forbidden)

## VERIFICATION

Every task: build → verify → commit. Golden rule: 35B tg64 ≥ 184 tok/s.
All bats on `C:\Users\james\Desktop\den-benchmarks\`.
Push to `RentedNoodle/den_llama.cpp` master.
Build: `build_now.bat` or `build_den.ps1 -phase all`.
CUDA 13.3 + sm_120a. RTX 5070 Ti GB203-300-A1.

---

# SESSION UPDATE 2026-08-09 — NVFP4 KV Accuracy + Tri-Vector Gate

## THE GATE (3 vectors, per user)
- **Speed:** tg64 ≥ 184 (35B). 
- **Accuracy:** cos ≥ 0.9995 vs F32.
- **Context:** 256k target, 128k acceptable. **NVFP4 KV is the ONLY way to 256k on 16GB for the 35B** (q8_0=5.4GB OOMs vs 14GB weights; NVFP4=1.7GB fits). Context vector is PHYSICALLY GATED on NVFP4 KV + sparse-VMM + expert offload.

## ARCHITECTURE DECISION (2026-08-09)
- **35B = primary** (agentic tool use, long context, reasoning). Expert-offloaded (ncmoe16) — fully-resident 35B impossible on 16GB.
- **9B = fast-chat tier, secondary NOT shelved.** Auto-switch by task.
- For Dreya's use case (conversation + tool use: organize/install/modify), 35B dominates because multi-step tool-calling is the hard part small models fumble.

## NVFP4 KV ACCURACY — WHAT WE LEARNED (all bugs found via subagents)
1. **Stream race (FIXED 2026-08-09):** attention wrote d_output on dedicated stream, MAIN read it unsynced → non-deterministic (2.1/43.3/57.7%). Fix: store+attention+load ALL run on `main_stream` (FIFO, no events, capture-safe). CUDA graph capture NEVER active for MoE decode (MUL_MAT_ID fails graph-compat) → capture-redirect code was inert.
2. **Scale LUT saturation (FIXED):** was capped 1.875, collapsed large blocks. Data-driven LUT (measured 104K blocks): fine in 0.06-1.5, cap 1.5, NO codes above (4.0/16.0 = underflow kill-switch → 5.3%). LUT is a SAFETY lever, not precision — real precision is element bits.
3. **K8V8 (DONE):** V quantized 8-bit under DEN_THRIFT_ATTENTION=1. E2M1 1-bit-mantissa (15-40% err) → 0.4%.
4. **Precision tail (DONE):** sliding 256-token F32 tail (DEN_NVFP4_KV_TAIL_TOKENS, env DEN_NVFP4_KV_TAIL), quantize only old context. BeeLlama #1 lever (q8_0+tail1024=KLD 0.0009). Our 4-token KVSink was 256x too small.

## COMMUNITY RESEARCH (via subagent — all channels)
- **4-bit KV CANNOT hit cos 0.9995.** Even best methods (Hadamard, asymmetric, E4M3) target benchmark-parity, not token-exact. The gate's accuracy vector at 256k is in tension with 4-bit KV.
- **Precision tail = #1 accuracy lever** (BeeLlama 413-config ladder).
- **Asymmetric K/V** — K >> V sensitivity (K noise propagates through softmax). K=q8_0/8-bit, V=4-bit.
- **Hadamard/WHT pre-transform** (QuaRot, KVarN) = highest ceiling, decorrelates error accumulation. We have WHT infra.
- **unsloth NVFP4 = WEIGHTS only** (2.5x speed via OMMA), KV stays fp16/q8_0. They never 4-bit KV.
- **jashepp MXFP4 hybrid** = precision-TIERING reference (F16 critical tensors + Q8_0 backbone + 4-bit experts), 256k-optimized. Adapt to NVFP4 (finer 16-elem scale beats MXFP4 32-elem) — this is the .den per-tensor-precision design.

## NEXT (priority)
1. **Verify race fix** — replay 3x must be DETERMINISTIC (was 2.1/43.3/57.7%).
2. **Build tri-vector gate** (`gate_trivector.bat`) — speed+cos+context in ONE run, deterministic, golden-hash CI. WITHOUT this, all accuracy tuning is noise (the /btw #3 lesson).
3. **Asymmetric K/V** (K=8-bit, V=4-bit) — trivial dtype switch, high evidence.
4. **.den zero-copy native decode** (Blocker 3 speed) — the 25% GGUF-repack overhead.

## THERMAL (from /btw)
- Numbers may be burst-not-steady. Poll nvidia-smi throttle_reasons in every gate run. Measure steady-state tg64 (30 min), not just cold.
- GDDR7 read-retry disable (`RmReadRetryDisable=1` registry) ~3% BW for displayless.
