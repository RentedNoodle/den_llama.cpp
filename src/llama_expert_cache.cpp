#include "llama_expert_cache.h"
#include "llama-impl.h"

// For NVFP4 quantization
#include "ggml-quants.h"
#include "ggml-common.h"

// AVX-512 pre-dequant kernels (Feature 4C)
extern "C" {
    int  ggml_avx512_dequant_iq2s_to_fp16(const void * iq2s_data, void * fp16_out, int n_values);
    int  ggml_avx512_dequant_nvfp4_to_fp16(const void * nvfp4_data, void * fp16_out, int n_values);
    int  ggml_avx512_expert_available(void);
    int  ggml_avx512_dequant_to_fp16(const void * src, void * dst, int n_values, int type);
}

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

// ============================================================================
// GPU operation callbacks — injected by the CUDA backend at init time.
// This keeps the cache policy layer pure C++ with no CUDA header dependency.
// ============================================================================

// Allocate size bytes on the default GPU device. Returns nullptr on failure.
static void* (*gpu_alloc_fn)(size_t size) = nullptr;

// Free a GPU pointer allocated by gpu_alloc_fn.
static void (*gpu_free_fn)(void* ptr) = nullptr;

// Asynchronously copy size bytes from src (host) to dst (device).
// Returns true on success. Caller must synchronize before accessing dst.
static bool (*gpu_copy_async_fn)(void* dst, const void* src, size_t size) = nullptr;

// Synchronize the default GPU stream.
static void (*gpu_sync_fn)(void) = nullptr;

// Get available GPU memory (free, total in bytes). Returns false if unavailable.
static bool (*gpu_mem_info_fn)(size_t* free, size_t* total) = nullptr;

void llama_expert_cache_set_gpu_ops(
    void* (*alloc_fn)(size_t),
    void  (*free_fn)(void*),
    bool  (*copy_async_fn)(void*, const void*, size_t),
    void  (*sync_fn)(void),
    bool  (*mem_info_fn)(size_t*, size_t*)) {

    gpu_alloc_fn     = alloc_fn;
    gpu_free_fn      = free_fn;
    gpu_copy_async_fn = copy_async_fn;
    gpu_sync_fn      = sync_fn;
    ::gpu_mem_info_fn = mem_info_fn;
}

// ============================================================================
// Cache policy constants
// ============================================================================

static constexpr float HEAT_DECAY      = 0.95f;
static constexpr float HEAT_BUMP       = 1.0f;
static constexpr float EVICT_THRESHOLD = 0.3f;
static constexpr float HOT_THRESHOLD   = 1.8f;

// ============================================================================
// Implementation
// ============================================================================

void llama_expert_cache_init(
    llama_expert_cache* cache,
    uint32_t num_experts,
    size_t expert_size_bytes,
    size_t vram_budget) {

    // Query available VRAM if no budget specified
    size_t vram_free = 0, vram_total = 0;
    if (vram_budget == 0 && gpu_mem_info_fn) {
        if (gpu_mem_info_fn(&vram_free, &vram_total)) {
            vram_budget = vram_free / 2; // Use at most 50% of free VRAM
            LLAMA_LOG_INFO("expert_cache: auto budget %zu MiB (50%% of %zu MiB free)\n",
                vram_budget / (1024*1024), vram_free / (1024*1024));
        }
    }
    if (vram_budget == 0) {
        vram_budget = 4ULL * 1024 * 1024 * 1024; // 4 GiB fallback
    }

    uint32_t max_slots = (uint32_t)(vram_budget / expert_size_bytes);
    if (max_slots < 4)  max_slots = 4;
    if (max_slots > num_experts) max_slots = num_experts;
    if (max_slots > 4096) max_slots = 4096;

    cache->max_slots         = max_slots;
    cache->num_experts       = num_experts;
    cache->global_tick       = 0;
    cache->expert_size_bytes = expert_size_bytes;
    cache->vram_budget       = (size_t)max_slots * expert_size_bytes;
    cache->gpu_ok            = false;
    cache->vram_pool         = nullptr;
    cache->slot_free         = nullptr; // allocated lazily
    cache->n_allocated       = 0;

    // Allocate entry tracking
    cache->entries.resize(num_experts);
    for (uint32_t i = 0; i < num_experts; i++) {
        auto& e = cache->entries[i];
        e.expert_id    = i;
        e.status       = EXPERT_COLD;
        e.heat_score   = 0.0f;
        e.last_used_tick = 0;
        e.vram_ptr     = nullptr;
        e.ram_ptr      = nullptr;
        e.size_bytes   = expert_size_bytes;
        // Dynamic Precision Expert Pool fields (Feature 4B)
        e.precision_tier = EXPERT_PRECISION_COLD;
        e.nvfp4_vram_ptr = nullptr;
        e.fp16_vram_ptr  = nullptr;
        e.fp16_size_bytes = 0;
        e.src_type       = GGML_TYPE_F16;  // default, caller overrides
        e.is_iq2_s       = false;
    }

    // Allocate VRAM pool
    if (gpu_alloc_fn) {
        cache->vram_pool = gpu_alloc_fn(cache->vram_budget);
        if (!cache->vram_pool) {
            LLAMA_LOG_WARN("expert_cache: gpu_alloc(%zu) failed, disabling\n",
                cache->vram_budget);
            cache->max_slots = 0;
            return;
        }
        cache->gpu_ok = true;

        // Track free slots via bitmap
        cache->slot_free = new bool[max_slots]();
        for (uint32_t i = 0; i < max_slots; i++) {
            cache->slot_free[i] = true; // all free initially
        }
    }

    LLAMA_LOG_INFO("expert_cache: %u slots, %zu MiB pool, %.1f MiB/expert%s\n",
        max_slots,
        cache->vram_budget / (1024*1024),
        expert_size_bytes / (1024.0 * 1024.0),
        cache->gpu_ok ? "" : " (CPU-only mode)");
}

void llama_expert_cache_init_gpu(llama_expert_cache* cache) {
    if (cache->gpu_ok || cache->max_slots == 0) return;
    if (!gpu_alloc_fn) return;

    cache->vram_pool = gpu_alloc_fn(cache->vram_budget);
    if (!cache->vram_pool) {
        LLAMA_LOG_WARN("expert_cache: deferred gpu_alloc(%zu) failed\n",
            cache->vram_budget);
        return;
    }

    cache->gpu_ok = true;
    cache->slot_free = new bool[cache->max_slots]();
    for (uint32_t i = 0; i < cache->max_slots; i++) {
        cache->slot_free[i] = true;
    }

    LLAMA_LOG_INFO("expert_cache: GPU pool allocated (%zu MiB, %u slots)\n",
        cache->vram_budget / (1024*1024), cache->max_slots);
}

void llama_expert_cache_free(llama_expert_cache* cache) {
    if (gpu_sync_fn) gpu_sync_fn();
    // Free split-resolution NVFP4 copies
    llama_expert_cache_free_nvfp4(cache);
    // Free FP16 VRAM copies (Feature 4B)
    if (gpu_free_fn) {
        for (auto& e : cache->entries) {
            if (e.fp16_vram_ptr) {
                gpu_free_fn(e.fp16_vram_ptr);
                e.fp16_vram_ptr = nullptr;
            }
        }
    }
    if (cache->vram_pool && gpu_free_fn) {
        gpu_free_fn(cache->vram_pool);
    }
    delete[] cache->slot_free;
    cache->entries.clear();
    cache->vram_pool = nullptr;
    cache->slot_free = nullptr;
    cache->gpu_ok    = false;
}

void llama_expert_cache_tick(llama_expert_cache* cache) {
    // Lazy GPU pool alloc on first tick (CUDA guaranteed initialized by now)
    if (!cache->gpu_ok && gpu_alloc_fn && cache->max_slots > 0) {
        llama_expert_cache_init_gpu(cache);
    }
    if (!cache->gpu_ok) {
        cache->global_tick++;
        return; // No GPU available, skip heat management
    }

    cache->global_tick++;

    for (auto& e : cache->entries) {
        if (e.status != EXPERT_COLD) {
            e.heat_score *= HEAT_DECAY;

            if (e.heat_score < EVICT_THRESHOLD) {
                e.status = EXPERT_COLD;
                // Keep VRAM slot allocated; will be reused on next eviction
            } else if (e.heat_score < HOT_THRESHOLD && e.status == EXPERT_HOT) {
                e.status = EXPERT_WARM;
            }
        }
    }

    // Dynamic Precision Expert Pool: promote/demote tiers (Feature 4B)
    llama_expert_cache_update_tiers(cache);
}

// ============================================================================
// Dynamic Precision Expert Pool (Feature 4B)
// ============================================================================

void llama_expert_cache_update_tiers(llama_expert_cache* cache) {
    if (!cache || !cache->gpu_ok || !gpu_alloc_fn || !gpu_copy_async_fn) return;

    for (auto& e : cache->entries) {
        // --- PROMOTE: WARM → HOT (heat > 2.5f) ---
        // Dequant NVFP4/IQ2_S → FP16 on CPU, upload FP16 to GPU
        if (e.heat_score > 2.5f && e.precision_tier == EXPERT_PRECISION_WARM) {
            if (e.fp16_vram_ptr) {
                // Already has FP16 copy, just update tier
                e.precision_tier = EXPERT_PRECISION_HOT;
                e.status = EXPERT_HOT;
                continue;
            }

            // Calculate FP16 size: n_values * sizeof(uint16_t)
            // For NVFP4: each block_nvfp4 holds 256 values, size_bytes / 160 = n_blocks
            // For IQ2_S: each block_iq2_s holds 256 values, size_bytes / 82 = n_blocks
            size_t n_values = 0;
            if (e.is_iq2_s) {
                // block_iq2_s: sizeof = 82 bytes per 256 values
                size_t n_blocks = e.size_bytes / sizeof(block_iq2_s);
                n_values = n_blocks * 256;
            } else if (e.nvfp4_vram_ptr || e.src_type == GGML_TYPE_NVFP4) {
                // block_nvfp4: sizeof = 160 bytes per 256 values
                size_t n_blocks = e.size_bytes / sizeof(block_nvfp4);
                n_values = n_blocks * 256;
            } else {
                // Default: assume FP16 source, size_bytes = n_values * sizeof(uint16_t)
                n_values = e.size_bytes / sizeof(uint16_t);
            }

            size_t fp16_size = n_values * sizeof(uint16_t);
            if (fp16_size == 0) continue;

            // Allocate FP16 VRAM buffer
            void* fp16_gpu = gpu_alloc_fn(fp16_size);
            if (!fp16_gpu) {
                LLAMA_LOG_WARN("expert_cache: FP16 gpu_alloc(%zu) failed for expert %u\n",
                    fp16_size, e.expert_id);
                continue;
            }

            // Dequant on CPU, then upload FP16 to GPU
            bool promoted = false;
            if (e.ram_ptr) {
                // Dequant CPU source → FP16 on CPU, then upload
                size_t fp16_cpu_bytes = n_values * sizeof(uint16_t);
                void* fp16_cpu = nullptr;

                #if defined(_WIN32)
                    fp16_cpu = _aligned_malloc(fp16_cpu_bytes, 64);
                #else
                    fp16_cpu = aligned_alloc(64, fp16_cpu_bytes);
                #endif

                if (fp16_cpu) {
                    int rc = -1;
                    if (ggml_avx512_expert_available()) {
                        // Use AVX-512 accelerated dequant
                        ggml_avx512_dequant_to_fp16(e.ram_ptr, fp16_cpu, (int)n_values, (int)e.src_type);
                        rc = 0;
                    } else {
                        // Fallback: dequant to FP32, then convert to FP16
                        std::vector<float> fp32_buf(n_values);
                        if (e.is_iq2_s) {
                            dequantize_row_iq2_s((const block_iq2_s*)e.ram_ptr, fp32_buf.data(), (int64_t)n_values);
                        } else if (e.src_type == GGML_TYPE_NVFP4) {
                            dequantize_row_nvfp4((const block_nvfp4*)e.ram_ptr, fp32_buf.data(), (int64_t)n_values);
                        } else if (e.src_type == GGML_TYPE_F16) {
                            // Already FP16 in CPU — just copy
                            memcpy(fp16_cpu, e.ram_ptr, fp16_cpu_bytes);
                            rc = 0;
                        }
                        if (rc != 0) {
                            // FP32 → FP16 conversion (scalar fallback)
                            uint16_t* f16 = (uint16_t*)fp16_cpu;
                            for (size_t j = 0; j < n_values; j++) {
                                // Round-to-nearest-even FP32→FP16
                                uint32_t bits;
                                memcpy(&bits, &fp32_buf[j], sizeof(float));
                                uint32_t sign = (bits >> 16) & 0x8000;
                                int32_t exp   = ((int32_t)(bits >> 23) & 0xFF) - 127 + 15;
                                uint32_t mant = (bits & 0x7FFFFF) >> 13;
                                if (exp <= 0) { exp = 0; mant = 0; }
                                else if (exp >= 31) { exp = 31; mant = 0; }
                                f16[j] = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant & 0x3FF));
                            }
                            rc = 0;
                        }
                    }

                    if (rc == 0) {
                        // DMA FP16 to GPU
                        if (gpu_copy_async_fn(fp16_gpu, fp16_cpu, fp16_cpu_bytes)) {
                            e.fp16_vram_ptr  = fp16_gpu;
                            e.fp16_size_bytes = fp16_cpu_bytes;
                            e.precision_tier  = EXPERT_PRECISION_HOT;
                            e.status          = EXPERT_HOT;
                            promoted = true;
                            LLAMA_LOG_INFO("expert_cache: promoted expert %u to HOT (FP16, %zu bytes)\n",
                                e.expert_id, fp16_cpu_bytes);
                        }
                    }

                    #if defined(_WIN32)
                        _aligned_free(fp16_cpu);
                    #else
                        free(fp16_cpu);
                    #endif
                }
            } else if (e.nvfp4_vram_ptr) {
                // NVFP4 data only on GPU — DMA back, dequant, re-upload as FP16
                size_t nvfp4_size = e.size_bytes;
                void* nvfp4_cpu = nullptr;
                #if defined(_WIN32)
                    nvfp4_cpu = _aligned_malloc(nvfp4_size, 64);
                #else
                    nvfp4_cpu = aligned_alloc(64, nvfp4_size);
                #endif
                if (nvfp4_cpu) {
                    // Download NVFP4 from GPU
                    if (gpu_copy_async_fn(nvfp4_cpu, e.nvfp4_vram_ptr, nvfp4_size)) {
                        if (gpu_sync_fn) gpu_sync_fn();

                        // Dequant NVFP4 → FP16 on CPU
                        size_t fp16_cpu_bytes = n_values * sizeof(uint16_t);
                        void* fp16_cpu = nullptr;
                        #if defined(_WIN32)
                            fp16_cpu = _aligned_malloc(fp16_cpu_bytes, 64);
                        #else
                            fp16_cpu = aligned_alloc(64, fp16_cpu_bytes);
                        #endif
                        if (fp16_cpu) {
                            if (ggml_avx512_expert_available()) {
                                ggml_avx512_dequant_to_fp16(nvfp4_cpu, fp16_cpu, (int)n_values, GGML_TYPE_NVFP4);
                            } else {
                                std::vector<float> fp32_buf(n_values);
                                dequantize_row_nvfp4((const block_nvfp4*)nvfp4_cpu, fp32_buf.data(), (int64_t)n_values);
                                uint16_t* f16 = (uint16_t*)fp16_cpu;
                                for (size_t j = 0; j < n_values; j++) {
                                    uint32_t bits;
                                    memcpy(&bits, &fp32_buf[j], sizeof(float));
                                    uint32_t sign = (bits >> 16) & 0x8000;
                                    int32_t exp   = ((int32_t)(bits >> 23) & 0xFF) - 127 + 15;
                                    uint32_t mant = (bits & 0x7FFFFF) >> 13;
                                    if (exp <= 0) { exp = 0; mant = 0; }
                                    else if (exp >= 31) { exp = 31; mant = 0; }
                                    f16[j] = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant & 0x3FF));
                                }
                            }

                            if (gpu_copy_async_fn(fp16_gpu, fp16_cpu, fp16_cpu_bytes)) {
                                e.fp16_vram_ptr  = fp16_gpu;
                                e.fp16_size_bytes = fp16_cpu_bytes;
                                e.precision_tier  = EXPERT_PRECISION_HOT;
                                e.status          = EXPERT_HOT;
                                promoted = true;
                                LLAMA_LOG_INFO("expert_cache: promoted expert %u to HOT (NVFP4→FP16 download, %zu bytes)\n",
                                    e.expert_id, fp16_cpu_bytes);
                            }
                            #if defined(_WIN32)
                                _aligned_free(fp16_cpu);
                            #else
                                free(fp16_cpu);
                            #endif
                        }
                    }
                    #if defined(_WIN32)
                        _aligned_free(nvfp4_cpu);
                    #else
                        free(nvfp4_cpu);
                    #endif
                }
            }

            if (!promoted && fp16_gpu) {
                gpu_free_fn(fp16_gpu);
            }
        }

        // --- DEMOTE: HOT → WARM (heat < 1.0f) ---
        // Free FP16 copy, revert to NVFP4
        else if (e.heat_score < 1.0f && e.precision_tier == EXPERT_PRECISION_HOT) {
            if (e.fp16_vram_ptr && gpu_free_fn) {
                gpu_free_fn(e.fp16_vram_ptr);
                e.fp16_vram_ptr  = nullptr;
                e.fp16_size_bytes = 0;
                LLAMA_LOG_INFO("expert_cache: demoted expert %u to WARM (freed FP16)\n", e.expert_id);
            }
            e.precision_tier = EXPERT_PRECISION_WARM;
            e.status         = EXPERT_WARM;
        }
    }
}

// Find the coldest loaded entry that can be evicted
static int find_evict_candidate(llama_expert_cache* cache) {
    float coldest = 1e10f;
    int   idx     = -1;
    for (size_t i = 0; i < cache->entries.size(); i++) {
        auto& e = cache->entries[i];
        if (e.status == EXPERT_COLD || e.vram_ptr == nullptr) continue;
        if (e.status == EXPERT_HOT) continue; // Never evict HOT
        if (e.heat_score < coldest) {
            coldest = e.heat_score;
            idx     = (int)i;
        }
    }
    return idx;
}

int llama_expert_cache_prefetch(
    llama_expert_cache* cache,
    const uint32_t*     expert_ids,
    uint32_t            count) {

    // Update stats counters
    cache->n_prefetch_calls++;
    cache->n_expert_requests += count;

    if (!cache->gpu_ok || cache->max_slots == 0 || !gpu_copy_async_fn) return 0;

    // Phase 1: Assign VRAM slots to cold experts, pack into staging buffer
    // Coalesced: ONE big PCIe DMA instead of N small ones
    struct cold_xfer {
        void *      dst;
        const void *src;
        size_t      sz;
        uint32_t    expert_id;
        uint32_t    slot_idx;
    };

    cold_xfer cold[32]; // max 32 cold experts per prefetch
    int n_cold = 0;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t id = expert_ids[i];
        if (id >= cache->num_experts) continue;
        auto& e = cache->entries[id];

        if (e.status != EXPERT_COLD) {
            e.heat_score += HEAT_BUMP;
            e.last_used_tick = cache->global_tick;
            if (e.heat_score > HOT_THRESHOLD) e.status = EXPERT_HOT;
            continue;
        }
        if (!e.ram_ptr || e.size_bytes == 0 || n_cold >= 32) continue;

        // Find VRAM slot: free pool first, then evict coldest
        void* slot = nullptr;
        uint32_t slot_idx = ~0u;

        for (uint32_t s = 0; s < cache->max_slots; s++) {
            if (cache->slot_free[s]) { slot_idx = s; break; }
        }

        if (slot_idx == ~0u) {
            int evict = find_evict_candidate(cache);
            if (evict < 0) continue;
            slot = cache->entries[evict].vram_ptr;
            for (uint32_t s = 0; s < cache->max_slots; s++) {
                uint8_t* base = (uint8_t*)cache->vram_pool;
                if (slot >= base + s * cache->expert_size_bytes &&
                    slot <  base + (s + 1) * cache->expert_size_bytes) {
                    slot_idx = s; break;
                }
            }
            cache->entries[evict].status   = EXPERT_COLD;
            cache->entries[evict].vram_ptr = nullptr;
            cache->entries[evict].heat_score = 0.0f;
        } else {
            slot = (uint8_t*)cache->vram_pool + (size_t)slot_idx * cache->expert_size_bytes;
        }

        cold[n_cold].dst       = slot;
        cold[n_cold].src       = e.ram_ptr;
        cold[n_cold].sz        = e.size_bytes;
        cold[n_cold].expert_id = id;
        cold[n_cold].slot_idx  = slot_idx;
        n_cold++;
    }

    if (n_cold == 0) return 0;

    // Phase 2: Coalesced DMA — pack all cold expert weights into staging buffer,
    // issue ONE PCIe transfer, then GPU-side scatter to VRAM slots.
    //
    // CPU Pre-Dequant Expert Streaming (Feature 4C):
    //   For IQ2_S cold experts, pre-dequant to FP16 on CPU using AVX-512
    //   BEFORE DMA. The GPU receives FP16 weights directly — no GPU-side
    //   dequant needed for the initial load. This increases PCIe size ~20x
    //   but eliminates GPU dequant overhead entirely.

    // First pass: build the staging buffer with optional pre-dequant
    // Track which experts were pre-dequantized so we can use correct sizes
    struct xfer_info {
        uint32_t expert_id;
        uint32_t slot_idx;
        void*    src_data;    // CPU source for DMA (may be pre-dequant FP16)
        size_t   src_size;    // size in bytes (FP16=larger, IQ2_S=smaller)
        void*    fp16_staging;// temporary FP16 buffer (null if no pre-dequant)
    };
    xfer_info xfer[32];
    int n_xfer = 0;

    size_t total_bytes = 0;
    for (int i = 0; i < n_cold; i++) {
        uint32_t eid = cold[i].expert_id;
        auto& entry = cache->entries[eid];

        xfer[n_xfer].expert_id   = eid;
        xfer[n_xfer].slot_idx    = cold[i].slot_idx;
        xfer[n_xfer].fp16_staging = nullptr;

        // Feature 4C: Pre-dequant IQ2_S → FP16 on CPU for cold experts
        if (entry.is_iq2_s && entry.ram_ptr && ggml_avx512_expert_available()) {
            // Calculate FP16 size: n_blocks * 256 * sizeof(uint16_t)
            size_t n_blocks = entry.size_bytes / sizeof(block_iq2_s);
            size_t n_values = n_blocks * 256;
            size_t fp16_bytes = n_values * sizeof(uint16_t);

            void* fp16_cpu = nullptr;
            #if defined(_WIN32)
                fp16_cpu = _aligned_malloc(fp16_bytes, 64);
            #else
                fp16_cpu = aligned_alloc(64, fp16_bytes);
            #endif

            if (fp16_cpu) {
                // AVX-512 dequant: IQ2_S → FP16
                ggml_avx512_dequant_to_fp16(
                    entry.ram_ptr,
                    fp16_cpu,
                    (int)n_values,
                    GGML_TYPE_IQ2_S);

                xfer[n_xfer].src_data    = fp16_cpu;
                xfer[n_xfer].src_size    = fp16_bytes;
                xfer[n_xfer].fp16_staging = fp16_cpu;
                total_bytes += fp16_bytes;
                n_xfer++;
                continue;
            }
        }

        // Default path: use raw source data as-is
        xfer[n_xfer].src_data = (void*)cold[i].src;
        xfer[n_xfer].src_size = cold[i].sz;
        total_bytes += cold[i].sz;
        n_xfer++;
    }

    if (n_xfer == 0) return 0;

    // Phase 2a: Pack transfers into contiguous staging buffer
    std::vector<uint8_t> staging(total_bytes);
    size_t offset = 0;
    for (int i = 0; i < n_xfer; i++) {
        memcpy(staging.data() + offset, xfer[i].src_data, xfer[i].src_size);
        offset += xfer[i].src_size;
    }

    // Phase 2b: Allocate GPU staging buffer
    void* gpu_staging = gpu_alloc_fn(total_bytes);
    if (!gpu_staging) {
        LLAMA_LOG_WARN("expert_cache: coalesced gpu_alloc(%zu) failed, fallback to per-expert\n", total_bytes);
        // Fallback to per-expert DMA
        for (int i = 0; i < n_xfer; i++) {
            void* dst = (uint8_t*)cache->vram_pool + (size_t)xfer[i].slot_idx * cache->expert_size_bytes;
            if (gpu_copy_async_fn(dst, xfer[i].src_data, xfer[i].src_size)) {
                uint32_t eid = xfer[i].expert_id;
                cache->slot_free[xfer[i].slot_idx] = false;
                cache->entries[eid].vram_ptr    = dst;
                cache->entries[eid].status      = EXPERT_WARM;
                cache->entries[eid].heat_score  = 1.0f;
                cache->entries[eid].last_used_tick = cache->global_tick;
                cache->entries[eid].precision_tier = EXPERT_PRECISION_WARM;
                cache->n_allocated++;
            }
            // Free FP16 staging if pre-dequant was used
            if (xfer[i].fp16_staging) {
                #if defined(_WIN32)
                    _aligned_free(xfer[i].fp16_staging);
                #else
                    free(xfer[i].fp16_staging);
                #endif
            }
        }
        if (gpu_sync_fn) gpu_sync_fn();
        cache->n_dma_transfers += n_xfer;
        return n_xfer;
    }

    // Phase 2c: ONE PCIe transfer: CPU staging → GPU staging
    if (!gpu_copy_async_fn(gpu_staging, staging.data(), total_bytes)) {
        LLAMA_LOG_WARN("expert_cache: coalesced DMA (%zu bytes) failed, fallback\n", total_bytes);
        gpu_free_fn(gpu_staging);
        // Free FP16 staging allocations
        for (int i = 0; i < n_xfer; i++) {
            if (xfer[i].fp16_staging) {
                #if defined(_WIN32)
                    _aligned_free(xfer[i].fp16_staging);
                #else
                    free(xfer[i].fp16_staging);
                #endif
            }
        }
        return 0;
    }

    // Phase 2d: GPU-side scatter: staging → VRAM slots (GPU-to-GPU, ~100 GB/s, no PCIe)
    offset = 0;
    for (int i = 0; i < n_xfer; i++) {
        void* dst = (uint8_t*)cache->vram_pool + (size_t)xfer[i].slot_idx * cache->expert_size_bytes;
        void* src_chunk = (uint8_t*)gpu_staging + offset;
        gpu_copy_async_fn(dst, src_chunk, xfer[i].src_size);
        offset += xfer[i].src_size;
    }

    // Mark entries as loaded (optimistically, before DMA completes)
    // Update precision tier to WARM
    for (int i = 0; i < n_xfer; i++) {
        uint32_t eid = xfer[i].expert_id;
        cache->slot_free[xfer[i].slot_idx] = false;
        cache->entries[eid].vram_ptr    = (uint8_t*)cache->vram_pool + (size_t)xfer[i].slot_idx * cache->expert_size_bytes;
        cache->entries[eid].status      = EXPERT_WARM;
        cache->entries[eid].heat_score  = 1.0f;
        cache->entries[eid].last_used_tick = cache->global_tick;
        cache->entries[eid].precision_tier = EXPERT_PRECISION_WARM;
        cache->n_allocated++;
    }

    // Free FP16 staging buffers
    for (int i = 0; i < n_xfer; i++) {
        if (xfer[i].fp16_staging) {
            #if defined(_WIN32)
                _aligned_free(xfer[i].fp16_staging);
            #else
                free(xfer[i].fp16_staging);
            #endif
        }
    }

    cache->n_dma_transfers += n_xfer;
    return n_xfer;
}

void llama_expert_cache_sync(llama_expert_cache* cache) {
    if (cache && cache->gpu_ok && gpu_sync_fn) {
        gpu_sync_fn();
    }
}

void llama_expert_cache_use(llama_expert_cache* cache, uint32_t expert_id) {
    if (expert_id >= cache->num_experts) return;
    auto& e = cache->entries[expert_id];
    if (e.status != EXPERT_COLD) {
        e.heat_score += HEAT_BUMP;
        e.last_used_tick = cache->global_tick;
        if (e.heat_score > HOT_THRESHOLD) {
            e.status = EXPERT_HOT;
            // Auto-promote to HOT precision tier on next update_tiers call
        }
    }
}

void* llama_expert_cache_get_vram_ptr(llama_expert_cache* cache, uint32_t expert_id) {
    if (expert_id >= cache->num_experts) return nullptr;
    auto& e = cache->entries[expert_id];
    // Feature 4B: return FP16 copy for HOT experts (highest quality)
    if (e.precision_tier == EXPERT_PRECISION_HOT && e.fp16_vram_ptr) {
        return e.fp16_vram_ptr;
    }
    return e.vram_ptr;
}

void llama_expert_cache_finalize_nvfp4(llama_expert_cache* cache) {
    if (!cache) return;
    cache->nvfp4_copy_ok = false;
    for (size_t i = 0; i < cache->nvfp4_vram_copies.size(); i++) {
        if (cache->nvfp4_vram_copies[i].up || cache->nvfp4_vram_copies[i].down) {
            cache->nvfp4_copy_ok = true;
            break;
        }
    }
    LLAMA_LOG_INFO("nvfp4_copy: finalize, %zu layers, %s\n",
        cache->nvfp4_vram_copies.size(),
        cache->nvfp4_copy_ok ? "OK" : "NO COPIES");
}

const llama_expert_cache::nvfp4_layer_copy * llama_expert_cache_get_nvfp4(
    llama_expert_cache* cache, int il) {
    if (!cache || !cache->nvfp4_copy_ok) return nullptr;
    if (il < 0 || (size_t)il >= cache->nvfp4_vram_copies.size()) return nullptr;
    return &cache->nvfp4_vram_copies[il];
}

void llama_expert_cache_free_nvfp4(llama_expert_cache* cache) {
    if (!cache) return;
    if (gpu_sync_fn) gpu_sync_fn();
    for (size_t i = 0; i < cache->nvfp4_vram_copies.size(); i++) {
        auto & c = cache->nvfp4_vram_copies[i];
        if (c.up && gpu_free_fn)   gpu_free_fn(c.up);
        if (c.gate && gpu_free_fn) gpu_free_fn(c.gate);
        if (c.down && gpu_free_fn) gpu_free_fn(c.down);
    }
    cache->nvfp4_vram_copies.clear();
    cache->nvfp4_copy_ok = false;
}

// Dequantize source data of any supported type to FP32.
static void dequantize_tensor_to_fp32(
    const void * src, float * fp32_out, int64_t count, enum ggml_type type) {

    switch (type) {
        case GGML_TYPE_F32: {
            memcpy(fp32_out, src, count * sizeof(float));
            return;
        }
        case GGML_TYPE_F16: {
            const uint16_t * fp16 = (const uint16_t *)src;
            for (int64_t i = 0; i < count; i++) {
                uint16_t h = fp16[i];
                uint32_t bits = (uint32_t)(h & 0x3FF) << 13;  // mantissa
                uint32_t e = (h >> 10) & 0x1F;
                uint32_t s = (h >> 15) & 1;
                if (e == 0) {
                    // subnormal / zero
                    bits = (s << 31) | (127 - 15) << 23 | bits;
                } else {
                    bits = (s << 31) | ((e - 15 + 127) << 23) | bits;
                }
                memcpy(&fp32_out[i], &bits, sizeof(float));
            }
            return;
        }
        case GGML_TYPE_BF16: {
            const uint16_t * bf16 = (const uint16_t *)src;
            for (int64_t i = 0; i < count; i++) {
                uint32_t bits = (uint32_t)bf16[i] << 16;
                memcpy(&fp32_out[i], &bits, sizeof(float));
            }
            return;
        }
        default: {
            // For quantized types: use ggml internal dequant if available
            // via quantize_row_nvfp4_ref which accepts float* — we need to
            // dequantize the source first. Since ggml-quants.h provides
            // dequantize_row_* functions, we use them if available.
            // For unsupported types: warn and zero-fill.
            LLAMA_LOG_WARN("nvfp4_copy: unsupported dequant type %d, zero-filling %lld elements\n",
                (int)type, (long long)count);
            memset(fp32_out, 0, count * sizeof(float));
            return;
        }
    }
}

// Quantize a matrix of FP32 data to NVFP4 blocks.
// src_data must have src_elem_count * n_expert float elements.
static void quantize_fp32_to_nvfp4(
    const float * src_data,
    size_t elem_per_expert,
    uint32_t n_expert,
    block_nvfp4 * dst) {

    size_t blocks_per_expert = (elem_per_expert + 255) / 256;
    for (uint32_t e = 0; e < n_expert; e++) {
        quantize_row_nvfp4_ref(
            src_data + e * elem_per_expert,
            dst + e * blocks_per_expert,
            (int64_t)elem_per_expert);
    }
}

// Helper: stage a single tensor's data: dequantize to FP32 → quantize to NVFP4.
static std::vector<uint8_t> stage_tensor_to_nvfp4(
    const void * src_data,
    enum ggml_type src_type,
    size_t elem_per_expert,
    uint32_t n_expert) {

    constexpr size_t BLOCK_SIZE = sizeof(block_nvfp4);
    size_t nvfp4_bytes = ((elem_per_expert + 255) / 256) * BLOCK_SIZE * n_expert;
    std::vector<uint8_t> staging(nvfp4_bytes);

    if (!src_data || elem_per_expert == 0 || n_expert == 0) {
        return staging;
    }

    // If already FP32, quantize directly
    if (src_type == GGML_TYPE_F32) {
        quantize_fp32_to_nvfp4(
            (const float *)src_data, elem_per_expert, n_expert,
            (block_nvfp4 *)staging.data());
        return staging;
    }

    // Dequantize to FP32, then quantize to NVFP4
    size_t total_elems = elem_per_expert * n_expert;
    std::vector<float> fp32_buf(total_elems);
    dequantize_tensor_to_fp32(src_data, fp32_buf.data(), (int64_t)total_elems, src_type);

    quantize_fp32_to_nvfp4(
        fp32_buf.data(), elem_per_expert, n_expert,
        (block_nvfp4 *)staging.data());

    return staging;
}

bool llama_expert_cache_alloc_nvfp4_copy(
    llama_expert_cache* cache,
    const void* up_exps,
    enum ggml_type up_type,
    const void* gate_exps,
    enum ggml_type gate_type,
    const void* down_exps,
    enum ggml_type down_type,
    uint32_t n_ff_up,    // ne[0] of up tensor
    uint32_t n_ff,       // true intermediate dim (ne[1] of down_exps)
    uint32_t n_embd,     // hidden dimension
    uint32_t n_expert,
    int il) {

    if (!cache || !gpu_alloc_fn || !gpu_copy_async_fn) return false;

    constexpr size_t BLOCK_SIZE = sizeof(block_nvfp4);

    // Element counts per expert
    size_t up_elements   = (size_t)n_ff_up * n_embd;
    size_t gate_elements = gate_exps ? (size_t)n_ff * n_embd : 0;
    size_t down_elements = (size_t)n_ff * n_embd;

    // Stage (dequantize + quantize) each tensor on CPU
    auto staging_up   = stage_tensor_to_nvfp4(up_exps,   up_type,   up_elements,   n_expert);
    auto staging_gate = stage_tensor_to_nvfp4(gate_exps, gate_type, gate_elements, gate_exps ? n_expert : 0);
    auto staging_down = stage_tensor_to_nvfp4(down_exps, down_type, down_elements, n_expert);

    size_t up_bytes   = staging_up.size();
    size_t gate_bytes = staging_gate.size();
    size_t down_bytes = staging_down.size();

    // Allocate GPU buffers
    void * gpu_up   = up_bytes   ? gpu_alloc_fn(up_bytes)   : nullptr;
    void * gpu_gate = gate_bytes ? gpu_alloc_fn(gate_bytes) : nullptr;
    void * gpu_down = down_bytes ? gpu_alloc_fn(down_bytes) : nullptr;

    if ((up_bytes && !gpu_up) || (gate_bytes && !gpu_gate) || (down_bytes && !gpu_down)) {
        LLAMA_LOG_WARN("nvfp4_copy: gpu_alloc failed for layer %d\n", il);
        if (gpu_up)   gpu_free_fn(gpu_up);
        if (gpu_gate) gpu_free_fn(gpu_gate);
        if (gpu_down) gpu_free_fn(gpu_down);
        return false;
    }

    // Upload async
    bool ok = true;
    if (up_bytes)   ok = ok && gpu_copy_async_fn(gpu_up,   staging_up.data(),   up_bytes);
    if (gate_bytes) ok = ok && gpu_copy_async_fn(gpu_gate, staging_gate.data(), gate_bytes);
    if (down_bytes) ok = ok && gpu_copy_async_fn(gpu_down, staging_down.data(), down_bytes);

    if (!ok) {
        LLAMA_LOG_WARN("nvfp4_copy: gpu_copy_async failed for layer %d\n", il);
        if (gpu_up)   gpu_free_fn(gpu_up);
        if (gpu_gate) gpu_free_fn(gpu_gate);
        if (gpu_down) gpu_free_fn(gpu_down);
        return false;
    }

    // Ensure layers vector is sized correctly
    if ((size_t)il >= cache->nvfp4_vram_copies.size()) {
        cache->nvfp4_vram_copies.resize(il + 1);
    }

    // Free previous copies if any
    auto & prev = cache->nvfp4_vram_copies[il];
    if (gpu_sync_fn) gpu_sync_fn();
    if (prev.up && gpu_free_fn)   gpu_free_fn(prev.up);
    if (prev.gate && gpu_free_fn) gpu_free_fn(prev.gate);
    if (prev.down && gpu_free_fn) gpu_free_fn(prev.down);

    prev.up     = gpu_up;
    prev.gate   = gpu_gate;
    prev.down   = gpu_down;
    prev.up_bytes    = up_bytes;
    prev.gate_bytes  = gate_bytes;
    prev.down_bytes  = down_bytes;

    LLAMA_LOG_INFO("nvfp4_copy: layer %d, %u experts, up=%zu MiB gate=%zu MiB down=%zu MiB GPU\n",
        il, n_expert,
        up_bytes / (1024 * 1024),
        gate_bytes / (1024 * 1024),
        down_bytes / (1024 * 1024));

    return true;
}

void llama_expert_cache_register_ram_ptr(
    llama_expert_cache* cache,
    uint32_t            expert_id,
    const void*         ram_ptr,
    size_t              size_bytes) {
    if (expert_id >= cache->num_experts) return;
    cache->entries[expert_id].ram_ptr = ram_ptr;
    if (size_bytes > 0) {
        cache->entries[expert_id].size_bytes = size_bytes;
    }
}

// Register CPU source with type info (Feature 4B/4C)
void llama_expert_cache_register_ram_ptr_type(
    llama_expert_cache* cache,
    uint32_t            expert_id,
    const void*         ram_ptr,
    size_t              size_bytes,
    enum ggml_type      src_type) {
    if (expert_id >= cache->num_experts) return;
    auto& e = cache->entries[expert_id];
    e.ram_ptr    = ram_ptr;
    e.src_type   = src_type;
    e.is_iq2_s   = (src_type == GGML_TYPE_IQ2_S);
    if (size_bytes > 0) {
        e.size_bytes = size_bytes;
    }
}

// ============================================================================
// CPU-side MoE router — zero-sync expert prefetch
// ============================================================================

void llama_expert_cpu_router_init(
    struct llama_expert_cpu_router * router,
    uint32_t n_layer,
    uint32_t n_expert,
    uint32_t n_embd) {

    router->n_layer  = n_layer;
    router->n_expert = n_expert;
    router->n_embd   = n_embd;

    size_t n_floats = (size_t)n_layer * (size_t)n_expert * (size_t)n_embd;
    router->weights  = new float[n_floats]();

    LLAMA_LOG_INFO("expert_cpu_router: %u layers x %u experts x %u embd = %zu MiB\n",
        n_layer, n_expert, n_embd, (n_floats * sizeof(float)) / (1024 * 1024));
}

void llama_expert_cpu_router_free(
    struct llama_expert_cpu_router * router) {

    if (router && router->weights) {
        delete[] router->weights;
        router->weights = nullptr;
    }
}

uint32_t llama_expert_cpu_router_predict(
    struct llama_expert_cpu_router * router,
    const float * hidden_state,
    uint32_t n_layers_to_check,
    uint32_t n_expert_used,
    uint32_t * out_experts,
    uint32_t max_experts) {

    if (!router || !router->weights || !hidden_state || !out_experts || max_experts == 0)
        return 0;

    const uint32_t n_layer = std::min(n_layers_to_check, router->n_layer);
    const uint32_t n_expert = router->n_expert;
    const uint32_t n_embd   = router->n_embd;
    const uint32_t k = std::min(n_expert_used, n_expert);

    if (n_layer == 0 || n_expert == 0 || n_embd == 0 || k == 0)
        return 0;

    uint32_t n_unique = 0;

    // Batched matmul: compute all layers' router logits in ONE pass
    //   weights reshaped: [n_layer * n_expert, n_embd]
    //   hidden_state:     [1, n_embd]
    //   logits:           [1, n_layer * n_expert]
    //
    // This is ~5x faster than per-layer loops because it exploits:
    //   - Sequential memory access (no strided expert reads)
    //   - CPU cache line utilization (n_embd consecutive elements)
    //   - Compiler auto-vectorization (reduction loop)
    const size_t n_total = (size_t)n_layer * n_expert;
    std::vector<float> logits_all(n_total);

    // Single batched matrix-vector product
    // hidden_state is reused across all experts and layers; the weight
    // matrix is laid out as [n_total, n_embd] in row-major order.
    const float * W = router->weights;  // [n_layer, n_expert, n_embd] = [n_total, n_embd]
    for (size_t i = 0; i < n_total; i++) {
        float sum = 0.0f;
        const float * w_row = W + i * (size_t)n_embd;
        #pragma GCC ivdep  // tell compiler no aliasing
        for (uint32_t d = 0; d < n_embd; d++) {
            sum += hidden_state[d] * w_row[d];
        }
        logits_all[i] = sum;
    }

    // Per-layer softmax + top-k
    std::vector<float> logits(n_expert);
    std::vector<std::pair<float, uint32_t>> scored(n_expert);

    for (uint32_t l = 0; l < n_layer; l++) {
        const float * layer_logits = logits_all.data() + (size_t)l * n_expert;

        // Copy to working buffer (logits_all is read-only)
        memcpy(logits.data(), layer_logits, n_expert * sizeof(float));
        for (uint32_t e = 0; e < n_expert; e++) {
            scored[e] = std::pair<float, uint32_t>(logits[e], e);
        }

        // Softmax: subtract max for numerical stability
        float max_logit = logits[0];
        for (uint32_t e = 1; e < n_expert; e++) {
            if (logits[e] > max_logit) max_logit = logits[e];
        }

        float sum_exp = 0.0f;
        for (uint32_t e = 0; e < n_expert; e++) {
            logits[e] = expf(logits[e] - max_logit);
            sum_exp += logits[e];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        for (uint32_t e = 0; e < n_expert; e++) {
            logits[e] *= inv_sum;
            scored[e].first = logits[e];
        }

        // Top-k via partial sort (largest first)
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
            [](const std::pair<float, uint32_t> & a,
               const std::pair<float, uint32_t> & b) {
                return a.first > b.first;
            });

        // Union across layers
        for (uint32_t i = 0; i < k; i++) {
            uint32_t eid = scored[i].second;
            bool found = false;
            for (uint32_t j = 0; j < n_unique; j++) {
                if (out_experts[j] == eid) { found = true; break; }
            }
            if (!found && n_unique < max_experts) {
                out_experts[n_unique++] = eid;
            }
        }
    }

    return n_unique;
}
