#pragma once

#include <cstdint>
#include <vector>

#include "ggml.h"

// Expert status in VRAM cache
enum llama_expert_status {
    EXPERT_COLD = 0,  // Not in VRAM
    EXPERT_WARM = 1,  // In VRAM, recently loaded
    EXPERT_HOT  = 2,  // In VRAM, frequently used
};

// Precision tier for Dynamic Precision Expert Pool (Feature 4B)
enum expert_precision_tier {
    EXPERT_PRECISION_COLD = 0,  // On CPU, IQ2_S format
    EXPERT_PRECISION_WARM = 1,  // In VRAM, NVFP4 format (permanent copy)
    EXPERT_PRECISION_HOT  = 2,  // In VRAM, FP16 format (highest quality)
};

// Single expert entry in the cache
struct llama_expert_cache_entry {
    uint32_t    expert_id;
    llama_expert_status status;
    float       heat_score;
    uint64_t    last_used_tick;
    void *      vram_ptr;   // GPU buffer (current active copy)
    const void *ram_ptr;    // CPU source data
    size_t      size_bytes; // Expert weight size (per-expert in current format)

    // Dynamic Precision Expert Pool fields (Feature 4B)
    expert_precision_tier precision_tier;  // Current precision tier
    void *      nvfp4_vram_ptr;  // NVFP4 copy (always resident for warm/hot)
    void *      fp16_vram_ptr;   // FP16 copy (only for hot experts)
    size_t      fp16_size_bytes; // Size of FP16 copy in bytes
    enum ggml_type src_type;      // Quant type of ram_ptr (GGML_TYPE_IQ2_S, etc.)
    bool        is_iq2_s;         // True if ram_ptr is IQ2_S format
};

// The expert cache (VRAM-pinned pool)
struct llama_expert_cache {
    uint32_t    max_slots;          // Max experts in VRAM at once
    uint32_t    num_experts;        // Total experts
    uint64_t    global_tick;        // Monotonic tick counter
    size_t      expert_size_bytes;  // Bytes per expert weight set
    size_t      vram_budget;        // Total VRAM allocated for cache

    std::vector<llama_expert_cache_entry> entries;

    // Benchmark stats counters (incremented by prefetch)
    uint64_t n_prefetch_calls    = 0;  // total number of llama_expert_cache_prefetch() calls
    uint64_t n_dma_transfers     = 0;  // total experts that required PCIe DMA
    uint64_t n_expert_requests   = 0;  // total expert IDs requested across all calls

    // GPU state (set by init)
    bool        gpu_ok;             // GPU allocation succeeded
    void *      vram_pool;          // Pre-allocated VRAM pool
    bool *      slot_free;          // Per-slot free bitmap
    uint32_t    n_allocated;        // Number of slots currently in use

    // Split-resolution: permanent NVFP4 copy of all expert weights
    // Per-layer, per-tensor GPU pointers, allocated once during init, never evicted.
    struct nvfp4_layer_copy {
        void * up;      // GPU pointer to NVFP4 up_gate_exps (or up_exps) data
        void * gate;    // GPU pointer to NVFP4 gate_exps data (null if fused)
        void * down;    // GPU pointer to NVFP4 down_exps data
        size_t up_bytes;    // Total bytes for up tensor
        size_t gate_bytes;  // Total bytes for gate tensor (0 if fused)
        size_t down_bytes;  // Total bytes for down tensor
    };
    std::vector<nvfp4_layer_copy> nvfp4_vram_copies;  // One per layer
    bool                          nvfp4_copy_ok;       // True if all copies allocated
};

// ============================================================================
// GPU callback injection — called by the CUDA backend during init.
// This keeps the cache policy layer pure C++ with no CUDA dependency.
// ============================================================================

void llama_expert_cache_set_gpu_ops(
    void* (*alloc_fn)(size_t),
    void  (*free_fn)(void*),
    bool  (*copy_async_fn)(void*, const void*, size_t),
    void  (*sync_fn)(void),
    bool  (*mem_info_fn)(size_t*, size_t*));

// ============================================================================
// Cache API
// ============================================================================

// Deferred GPU init — call after CUDA backends are ready.
// Allocates VRAM pool if llama_expert_cache_init was called before CUDA was up.
void llama_expert_cache_init_gpu(llama_expert_cache * cache);

// Initialize cache. vram_budget=0 for auto (50% of free VRAM).
void llama_expert_cache_init(
    llama_expert_cache * cache,
    uint32_t num_experts,
    size_t expert_size_bytes,
    size_t vram_budget);

// Free all GPU resources
void llama_expert_cache_free(llama_expert_cache * cache);

// Tick before each token — decay heat, demote cold entries
void llama_expert_cache_tick(llama_expert_cache * cache);

// Ensure expert_ids are in VRAM. Returns count that needed DMA.
int llama_expert_cache_prefetch(
    llama_expert_cache * cache,
    const uint32_t * expert_ids,
    uint32_t count);

// Mark expert as used (bump heat, refresh LRU)
void llama_expert_cache_use(llama_expert_cache * cache, uint32_t expert_id);

// Synchronize pending expert DMAs (call before expert-dependent graph compute)
void llama_expert_cache_sync(llama_expert_cache * cache);

// Get VRAM pointer for an expert (nullptr if not resident)
void * llama_expert_cache_get_vram_ptr(llama_expert_cache * cache, uint32_t expert_id);

// Dynamic Precision Expert Pool: promote/demote experts based on access heat.
// Hot experts (>2.5 heat) get FP16 VRAM copy (dequant NVFP4→FP16).
// Cooling experts (<1.0 heat) revert to NVFP4 (free FP16 copy).
void llama_expert_cache_update_tiers(llama_expert_cache * cache);

// ============================================================================
// Split-resolution: permanent NVFP4 copy of ALL expert weights in VRAM
// ============================================================================
// Allocate a permanent NVFP4-compressed copy of ALL expert weight data in GPU
// VRAM for a single layer. This copy is never evicted — it is used for the GPU
// path of split-resolution expert compute (bottom-6 experts by router weight).
//
// up_exps/gate_exps/down_exps are the FP32/FP16 source tensors (n_expert in
// the ne[2] dimension).  gate_exps may be null if up_gate_exps is used instead
// (fused up+gate).  n_ff is the intermediate (ffn) dimension, n_embd the
// hidden dimension.
//
// The NVFP4 copy is stored in cache->nvfp4_vram_copies[il] with separate GPU
// allocations per tensor (up, gate/up_gate, down).
// Returns true on success.
bool llama_expert_cache_alloc_nvfp4_copy(
    struct llama_expert_cache * cache,
    const void * up_exps,         // [n_ff_up * n_embd * n_expert] source weight data (fused up_gate if gate_exps==null)
    enum ggml_type up_type,       // data type of up_exps
    const void * gate_exps,       // [n_ff * n_embd * n_expert] source (or null if fused)
    enum ggml_type gate_type,     // data type of gate_exps
    const void * down_exps,       // [n_embd * n_ff * n_expert] source
    enum ggml_type down_type,     // data type of down_exps
    uint32_t n_ff_up,             // ne[0] of up tensor (n_ff for separate, 2*n_ff for fused)
    uint32_t n_ff,                // true intermediate dimension (ne[1] of down_exps)
    uint32_t n_embd,              // hidden dimension
    uint32_t n_expert,            // total experts
    int il);                      // layer index

// Get per-tensor NVFP4 pointers for a given layer.
// Returns nullptr for a given tensor if not allocated or not applicable.
const struct llama_expert_cache::nvfp4_layer_copy * llama_expert_cache_get_nvfp4(
    struct llama_expert_cache * cache, int il);

// Mark NVFP4 copies as ready (call after all layers' copies are allocated).
// Sets nvfp4_copy_ok to true if at least one layer has a copy.
void llama_expert_cache_finalize_nvfp4(struct llama_expert_cache * cache);

// Free all NVFP4 copies
void llama_expert_cache_free_nvfp4(struct llama_expert_cache * cache);

// Register a CPU-side source pointer for an expert's weight data
void llama_expert_cache_register_ram_ptr(
    llama_expert_cache * cache,
    uint32_t expert_id,
    const void * ram_ptr,
    size_t size_bytes);

// Register CPU source with quant type info (Feature 4B/4C).
// Sets is_iq2_s flag for IQ2_S experts, enabling pre-dequant path.
void llama_expert_cache_register_ram_ptr_type(
    llama_expert_cache * cache,
    uint32_t expert_id,
    const void * ram_ptr,
    size_t size_bytes,
    enum ggml_type src_type);

// ============================================================================
// CPU-side MoE router for zero-sync expert prefetch
// ============================================================================
// Eliminates the ~500µs GPU sync by computing routing on CPU using
// the last token's hidden state and per-layer router weights.
// After softmax, takes top-k expert IDs per layer and unions across layers.

struct llama_expert_cpu_router {
    float * weights;        // [n_layer, n_expert, n_embd] — F32 router weights on CPU
    uint32_t n_layer;
    uint32_t n_expert;
    uint32_t n_embd;
};

// Initialize CPU router — allocates F32 buffer for router weights.
// Caller must fill weights after init (one float per weight).
void llama_expert_cpu_router_init(
    struct llama_expert_cpu_router * router,
    uint32_t n_layer,
    uint32_t n_expert,
    uint32_t n_embd);

// Free CPU router weights buffer
void llama_expert_cpu_router_free(
    struct llama_expert_cpu_router * router);

// ============================================================================
// Pinned-memory ring buffer for expert weight streaming + hidden-state routing
// ============================================================================
// Allocates a GPU-visible pinned host buffer (cudaHostAllocMapped) so both CPU
// and GPU can access the data directly — no explicit cudaMemcpy for the hot path.
// Ring semantics: oldest entries are silently overwritten when full.
// Thread-safe via internal mutex.
//
// Use llama_ring_buffer_get_device_ptr() to obtain a GPU pointer for CUDA kernels.

// Allocate ring buffer.  capacity = max items, item_size = bytes per item.
// Returns opaque handle (ring_buffer *), nullptr on failure.
extern "C" void * llama_ring_buffer_alloc(int capacity, int item_size);

// Free ring buffer (pinned memory + struct).
extern "C" void llama_ring_buffer_free(void * buf);

// Push |data| (item_size bytes) into the ring.  Overwrites oldest if full.
// Returns true on success.
extern "C" bool llama_ring_buffer_push(void * buf, const void * data);

// Read the most recently pushed item into |out| (must hold item_size bytes).
// Returns 0 on success, -1 if buffer is empty.
extern "C" int llama_ring_buffer_read_last(void * buf, void * out);

// Get a GPU-accessible (device) pointer to item at logical |index|.
// index = 0 → oldest, index = -1 → most recent.
// Returns nullptr if buffer is empty or index is out of range.
extern "C" void * llama_ring_buffer_get_device_ptr(void * buf, int index);

// Query current item count / capacity.
extern "C" int llama_ring_buffer_count(void * buf);
extern "C" int llama_ring_buffer_capacity(void * buf);

// Run router on CPU: hidden_state × weights^T → softmax → top-k expert IDs
// Returns number of unique expert IDs written to out_experts (0 if none).
// out_experts buffer should hold at least n_layers_to_check * n_expert_used entries.
uint32_t llama_expert_cpu_router_predict(
    struct llama_expert_cpu_router * router,
    const float * hidden_state,     // [n_embd]
    uint32_t n_layers_to_check,     // how many layers to route (typically n_layer)
    uint32_t n_expert_used,         // top-k experts to select per layer
    uint32_t * out_experts,         // output buffer for unique expert IDs
    uint32_t max_experts);          // max entries in out_experts
