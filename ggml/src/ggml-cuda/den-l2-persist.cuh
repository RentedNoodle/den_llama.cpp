// den-l2-persist.cuh — L2 Cache Persistence for MoE Expert Weights
//
// Ported from dengine's Mech 41/42 (den_gpu_l2.cu + den_gpu_l2.h) to ggml-cuda.
// Pins critical inference tensors and MoE expert weight slabs in L2 cache
// to prevent L2 thrashing when expert weights stream through on each decode step.
//
// Two-layer L2 strategy (from dengine Mech 41):
//   1. cuMemAdvise (CUDA Driver API):
//      - SET_READ_MOSTLY       — driver keeps read-only copies in L2
//      - SET_ACCESSED_BY       — hint for frequent device access
//      - SET_PREFERRED_LOCATION — keep memory on this GPU
//   2. cudaAccessPolicyWindow (CUDA Runtime API):
//      - cudaDeviceSetLimit(PersistingL2CacheSize)  — reserve L2 capacity
//      - cudaStreamSetAttribute(accessPolicyWindow)  — per-stream persistence
//
// GB203-300-A1 (RTX 5070 Ti) specifics:
//   - 48 MB total L2 cache
//   - 5120-bit memory bus
//   - Max persisting L2 reservation: 36 MB (75% of 48 MB)
//   - Default budget: 8 MB (leaves 40 MB for transient expert streams)
//
// Environment:
//   DEN_L2_PERSIST=1  — enable L2 persistence (REQUIRED; default: off)
//   DEN_L2_PERSIST_MB — MB to reserve for L2 (default: 8, max: 36)
//
// Usage:
//   #include "den-l2-persist.cuh"
//   if (den_l2_persist_enabled()) {
//       den_l2_persist_init();
//       den_l2_persist_hint(weight_ptr, weight_bytes, 1); // priority=1 → critical
//   }
//
// Author: Project Den (ported from dengine/den_gpu_l2.cu Mech 41/42)

#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

// GB203-300-A1: 48 MB L2. Reserve up to 75% (36 MB) for persistence.
#define DEN_L2_GB203_TOTAL_MB     48
#define DEN_L2_MAX_BUDGET_MB      36   // 75% of 48 MB
#define DEN_L2_DEFAULT_BUDGET_MB   8   // leaves 40 MB for transient expert streams

// Maximum number of separate allocations we track for diagnostics
#define DEN_L2_MAX_TRACKED        128

// Maximum number of unique experts in the L2 hot cache
#define DEN_L2_EXPERT_CACHE_MAX    32

// ── cuMemAdvise enum fallback values (stable across CUDA versions) ──────
// These are from the CUDA driver API cuda.h. If the toolkit headers are
// older and don't define them, we provide the stable raw values.
#ifndef CU_MEM_ADVISE_SET_READ_MOSTLY
#define CU_MEM_ADVISE_SET_READ_MOSTLY        ((CUmem_advise)1)
#endif
#ifndef CU_MEM_ADVISE_SET_PREFERRED_LOCATION
#define CU_MEM_ADVISE_SET_PREFERRED_LOCATION ((CUmem_advise)2)
#endif
#ifndef CU_MEM_ADVISE_SET_ACCESSED_BY
#define CU_MEM_ADVISE_SET_ACCESSED_BY        ((CUmem_advise)3)
#endif

#ifndef CU_MEM_LOCATION_TYPE_DEVICE
#define CU_MEM_LOCATION_TYPE_DEVICE          ((CUmemLocationType)1)
#endif

// cuMemAdvise takes CUmemLocation by VALUE (struct) starting in CUDA 12.0.
// Pre-12.0 toolsets pass the CUdevice int directly.
// For CUDA 13.3+ (the primary build target on sm_120a), the struct form is required.

// ═════════════════════════════════════════════════════════════════════════════
// Internal state
// ═════════════════════════════════════════════════════════════════════════════

typedef struct {
    void   *d_ptr;
    size_t  bytes;
    int     priority;   // 0=normal, 1=critical (embedding/norm/first-layer)
    char    label[48];
} den_l2_tracked_entry_t;

typedef struct {
    int  initialized;
    int  device_id;

    // Capability flags (determined at init)
    int  cap_cu_mem_advise;
    int  cap_access_policy;

    // L2 reservation
    int     budget_mb;
    size_t  reserved_bytes;

    // Tracked allocations
    den_l2_tracked_entry_t tracked[DEN_L2_MAX_TRACKED];
    int  n_tracked;
    size_t total_tracked_bytes;

    // Access policy window
    int  window_active;

    // Expert cache (Mech 42)
    int  expert_ids[DEN_L2_EXPERT_CACHE_MAX];
    int  n_experts_cached;
    int  n_expert_slabs_pinned;

    // Hit/miss stats
    long long hits;
    long long misses;
} den_l2_state_t;

// Single global state — header-only, each including TU gets its own copy.
// This is safe because init is idempotent and the CUDA device context is
// per-process anyway. Include from at most one .cu file for clean semantics.
static den_l2_state_t g_den_l2;

// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════

static inline int den_l2_env_int(const char * name, int def) {
    const char * v = getenv(name);
    if (!v || !*v) return def;
    char * e = NULL;
    long l = strtol(v, &e, 10);
    return (e == v || l < 0) ? def : (int)l;
}

// ── cuMemAdvise wrapper — handles CUDA 12+ vs pre-12 API difference ─────
// CUDA 12.0+ requires CUmemLocation struct; pre-12.0 passes CUdevice int.
// We try the struct form first (the probe in init determines which works).
static inline int den_l2_do_mem_advise(CUdeviceptr dptr, size_t bytes,
                                        CUmem_advise advice, int device_id) {
#if CUDART_VERSION >= 12000
    // CUDA 12.0+: cuMemAdvise takes CUmemLocation by value
    CUmemLocation loc;
    loc.type = CU_MEM_LOCATION_TYPE_DEVICE;
    loc.id   = device_id;
    CUresult r = cuMemAdvise(dptr, bytes, advice, loc);
    return (r == CUDA_SUCCESS) ? 0 : (int)r;
#else
    // Pre-12.0: cuMemAdvise takes CUdevice directly
    CUresult r = cuMemAdvise(dptr, bytes, advice, (CUdevice)device_id);
    return (r == CUDA_SUCCESS) ? 0 : (int)r;
#endif
}

// ═════════════════════════════════════════════════════════════════════════════
// Public API
// ═════════════════════════════════════════════════════════════════════════════

// ── den_l2_persist_enabled — check DEN_L2_PERSIST env var ──────────────────
// Returns 1 if the user has opted into L2 persistence.
// Must be called before any hint/pin calls; init checks this internally.
static inline int den_l2_persist_enabled(void) {
    static int checked = 0, enabled = 0;
    if (!checked) {
        const char * env = getenv("DEN_L2_PERSIST");
        enabled = (env && env[0] == '1');
        checked = 1;
    }
    return enabled;
}

// ── den_l2_persist_init — probe L2 persistence support and reserve ──────
// Must be called after cudaSetDevice() exactly once.
// Returns 0 on success, -1 if unsupported (non-fatal — inference continues).
// On success, reserves DEN_L2_PERSIST_MB (default 8) MB of L2.
static inline int den_l2_persist_init(void) {
    if (g_den_l2.initialized) return 0;
    if (!den_l2_persist_enabled()) return -1;

    g_den_l2.budget_mb = den_l2_env_int("DEN_L2_PERSIST_MB", DEN_L2_DEFAULT_BUDGET_MB);
    if (g_den_l2.budget_mb < 1)  g_den_l2.budget_mb = 1;
    if (g_den_l2.budget_mb > DEN_L2_MAX_BUDGET_MB)
        g_den_l2.budget_mb = DEN_L2_MAX_BUDGET_MB;

    cudaError_t ce = cudaGetDevice(&g_den_l2.device_id);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[L2-PERSIST] cudaGetDevice failed: %s\n", cudaGetErrorString(ce));
        return -1;
    }

    // ── Query L2 cache size ───────────────────────────────────────────
    int l2_total = 0;
    cudaDeviceGetAttribute(&l2_total, cudaDevAttrL2CacheSize, g_den_l2.device_id);
    if (l2_total <= 0) {
        fprintf(stderr, "[L2-PERSIST] L2 cache not available on device %d\n",
                g_den_l2.device_id);
        return -1;
    }

    // ── Probe cuMemAdvise availability ─────────────────────────────────
    // Ported from dengine: pass an invalid pointer. If driver returns
    // CUDA_ERROR_INVALID_VALUE, the API function exists and is wired.
    // If CUDA_ERROR_NOT_SUPPORTED or CUDA_ERROR_INVALID_HANDLE, it's not.
    {
#if CUDART_VERSION >= 12000
        CUmemLocation loc;
        loc.type = CU_MEM_LOCATION_TYPE_DEVICE;
        loc.id   = g_den_l2.device_id;
        CUresult cu = cuMemAdvise(
            (CUdeviceptr)1, 0, CU_MEM_ADVISE_SET_READ_MOSTLY, loc);
#else
        CUresult cu = cuMemAdvise(
            (CUdeviceptr)1, 0, CU_MEM_ADVISE_SET_READ_MOSTLY,
            (CUdevice)g_den_l2.device_id);
#endif
        g_den_l2.cap_cu_mem_advise = (cu == CUDA_ERROR_INVALID_VALUE);
        if (cu == CUDA_ERROR_NOT_SUPPORTED || cu == CUDA_ERROR_INVALID_HANDLE)
            g_den_l2.cap_cu_mem_advise = 0;
    }

    // ── Reserve L2 persisting cache ───────────────────────────────────
    size_t reserve = (size_t)g_den_l2.budget_mb * 1024 * 1024;
    size_t max_r   = (size_t)l2_total * 3 / 4;  // 75% max
    if (reserve > max_r) {
        reserve = max_r;
        g_den_l2.budget_mb = (int)(reserve / (1024 * 1024));
    }

    ce = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserve);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[L2-PERSIST] cudaDeviceSetLimit(PersistingL2, %zu MB): %s\n",
                reserve / (1024 * 1024), cudaGetErrorString(ce));
        cudaGetLastError();  // clear the error
        g_den_l2.cap_access_policy = 0;
    } else {
        size_t granted = 0;
        cudaDeviceGetLimit(&granted, cudaLimitPersistingL2CacheSize);
        g_den_l2.reserved_bytes = granted;
        g_den_l2.cap_access_policy = (granted > 0);
    }

    g_den_l2.initialized = 1;

    fprintf(stderr, "[L2-PERSIST] init: device=%d L2=%dMB budget=%dMB granted=%zuMB "
            "cuMemAdvise=%s accessPolicy=%s\n",
            g_den_l2.device_id, l2_total, g_den_l2.budget_mb,
            g_den_l2.reserved_bytes / (1024 * 1024),
            g_den_l2.cap_cu_mem_advise ? "OK" : "N/A",
            g_den_l2.cap_access_policy  ? "OK" : "N/A");

    return 0;
}

// ── den_l2_persist_hint — advise L2 persistence for a memory region ──────
// Applies three cuMemAdvise hints (port from dengine gpu_l2_pin_buffer):
//   1. SET_READ_MOSTLY        — keep read-only copies in L2
//   2. SET_ACCESSED_BY        — hint for frequent device access
//   3. SET_PREFERRED_LOCATION — keep memory on this GPU
//
// Parameters:
//   ptr      — GPU device pointer
//   bytes    — size in bytes
//   priority — 0 = normal (workspace), 1 = critical (embedding/weights)
//
// Also sets up the access policy window to cover the full range of
// high-priority buffers for ST_RESIDENT L2 policy.
//
// Returns 0 on success, -1 on error (non-fatal).
static inline int den_l2_persist_hint(void * ptr, size_t bytes, int priority) {
    if (!g_den_l2.initialized && den_l2_persist_init() != 0) return -1;
    if (!ptr || bytes == 0) return -1;

    // ── Track for diagnostics ─────────────────────────────────────────
    if (g_den_l2.n_tracked < DEN_L2_MAX_TRACKED) {
        int idx = g_den_l2.n_tracked;
        g_den_l2.tracked[idx].d_ptr    = ptr;
        g_den_l2.tracked[idx].bytes    = bytes;
        g_den_l2.tracked[idx].priority = priority;
        snprintf(g_den_l2.tracked[idx].label,
                 sizeof(g_den_l2.tracked[idx].label),
                 priority ? "weight" : "buffer");
        g_den_l2.n_tracked++;
        g_den_l2.total_tracked_bytes += bytes;
    }

    // ── Layer 1: cuMemAdvise hints ────────────────────────────────────
    if (g_den_l2.cap_cu_mem_advise) {
        CUdeviceptr dptr = (CUdeviceptr)ptr;
        int dev_id = g_den_l2.device_id;

        // SET_READ_MOSTLY: weights (priority=1) are read-only — keep in L2
        if (priority > 0) {
            den_l2_do_mem_advise(dptr, bytes,
                                 CU_MEM_ADVISE_SET_READ_MOSTLY, dev_id);
        }

        // SET_ACCESSED_BY: this device will read this memory frequently
        den_l2_do_mem_advise(dptr, bytes,
                             CU_MEM_ADVISE_SET_ACCESSED_BY, dev_id);

        // SET_PREFERRED_LOCATION: keep on this device
        den_l2_do_mem_advise(dptr, bytes,
                             CU_MEM_ADVISE_SET_PREFERRED_LOCATION, dev_id);
    }

    // ── Layer 2: access policy window (unified across all pinned) ─────
    // Build a window that covers all high-priority tracked regions.
    // Uses ST_RESIDENT policy: hits → persist, misses → streaming.
    if (g_den_l2.cap_access_policy && priority > 0) {
        // Find min/max across all tracked high-priority regions
        void * min_ptr = ptr;
        void * max_ptr = (uint8_t *)ptr + bytes;

        for (int i = 0; i < g_den_l2.n_tracked; i++) {
            if (g_den_l2.tracked[i].priority <= 0) continue;
            void * p     = g_den_l2.tracked[i].d_ptr;
            void * p_end = (uint8_t *)p + g_den_l2.tracked[i].bytes;
            if (p < min_ptr)     min_ptr = p;
            if (p_end > max_ptr) max_ptr = p_end;
        }

        size_t window_bytes = (uint8_t *)max_ptr - (uint8_t *)min_ptr;
        if (window_bytes > g_den_l2.reserved_bytes && g_den_l2.reserved_bytes > 0) {
            window_bytes = g_den_l2.reserved_bytes;
        }

        if (window_bytes > 0) {
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.base_ptr  = min_ptr;
            attr.accessPolicyWindow.num_bytes = window_bytes;
            attr.accessPolicyWindow.hitRatio  = 1.0f;
            // ST_RESIDENT: hits persist in L2, misses are streaming
            attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

            cudaError_t e = cudaStreamSetAttribute(
                (cudaStream_t)0,  // default stream
                cudaStreamAttributeAccessPolicyWindow,
                &attr);
            if (e == cudaSuccess) {
                g_den_l2.window_active = 1;
            }
        }
    }

    return 0;
}

// ── den_l2_persist_pin_experts — pin top-K expert weight slabs ──────────
// Ported from dengine gpu_l2_expert_pin (Mech 42).
// Pins gate+up+down weight slabs for each expert in L2 via cuMemAdvise.
// Each slab gets SET_READ_MOSTLY + SET_ACCESSED_BY hints.
//
// Parameters:
//   expert_ptrs  — array of GPU device pointers (expects n*3 entries:
//                  [gate0, up0, down0, gate1, up1, down1, ...])
//   n_experts    — number of experts
//   expert_bytes — size in bytes of a single expert weight slab
//                  (gate/up/down share the same size)
//
// Returns number of slabs successfully pinned, -1 on error.
static inline int den_l2_persist_pin_experts(void ** expert_ptrs,
                                              int n_experts,
                                              size_t expert_bytes) {
    if (!g_den_l2.initialized && den_l2_persist_init() != 0) return -1;
    if (!expert_ptrs || n_experts <= 0 || expert_bytes == 0) return -1;
    if (!g_den_l2.cap_cu_mem_advise) {
        fprintf(stderr, "[L2-PERSIST] cuMemAdvise not available, "
                "expert cache disabled\n");
        return -1;
    }

    int pinned = 0;
    int dev_id = g_den_l2.device_id;

    for (int e = 0; e < n_experts; e++) {
        // Three slabs per expert: gate, up, down
        void * slabs[3] = {
            expert_ptrs[e * 3 + 0],  // gate
            expert_ptrs[e * 3 + 1],  // up
            expert_ptrs[e * 3 + 2],  // down
        };

        // Track expert ID in cache set
        if (g_den_l2.n_experts_cached < DEN_L2_EXPERT_CACHE_MAX) {
            int already = 0;
            for (int j = 0; j < g_den_l2.n_experts_cached; j++) {
                if (g_den_l2.expert_ids[j] == e) { already = 1; break; }
            }
            if (!already) {
                g_den_l2.expert_ids[g_den_l2.n_experts_cached++] = e;
            }
        }

        for (int s = 0; s < 3; s++) {
            if (!slabs[s]) continue;
            CUdeviceptr dptr = (CUdeviceptr)slabs[s];

            // SET_READ_MOSTLY: read-only weight slabs
            int r = den_l2_do_mem_advise(dptr, expert_bytes,
                                         CU_MEM_ADVISE_SET_READ_MOSTLY, dev_id);
            if (r != 0) continue;

            // SET_ACCESSED_BY: frequent reads from this device
            r = den_l2_do_mem_advise(dptr, expert_bytes,
                                     CU_MEM_ADVISE_SET_ACCESSED_BY, dev_id);
            if (r != 0) continue;

            pinned++;
        }
    }

    g_den_l2.n_expert_slabs_pinned += pinned;

    fprintf(stderr, "[L2-PERSIST] experts pinned: %d slabs (%d experts, "
            "%zu MB/slab, %zu MB total)\n",
            pinned, n_experts,
            expert_bytes / (1024 * 1024),
            (size_t)pinned * expert_bytes / (1024 * 1024));

    return pinned;
}

// ── den_l2_persist_is_cached — check if expert is in L2 hot cache ──────
static inline int den_l2_persist_is_cached(int expert_id) {
    for (int i = 0; i < g_den_l2.n_experts_cached; i++) {
        if (g_den_l2.expert_ids[i] == expert_id) return 1;
    }
    return 0;
}

// ── den_l2_persist_record_hit / _miss — update cache statistics ────────
static inline void den_l2_persist_record_hit(void) {
    g_den_l2.hits++;
}
static inline void den_l2_persist_record_miss(void) {
    g_den_l2.misses++;
}

// ── den_l2_persist_stats — read hit/miss counters ──────────────────────
static inline void den_l2_persist_stats(int * out_hits, int * out_misses) {
    if (out_hits)   *out_hits   = (int)g_den_l2.hits;
    if (out_misses) *out_misses = (int)g_den_l2.misses;
}

// ── den_l2_persist_reset — clear pins and release L2 reservation ──────
// Clears access policy window, resets tracked state.
// Does NOT free GPU memory — only removes L2 hints.
static inline void den_l2_persist_reset(void) {
    if (!g_den_l2.initialized) return;

    // Clear access policy window
    if (g_den_l2.cap_access_policy && g_den_l2.window_active) {
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr  = NULL;
        attr.accessPolicyWindow.num_bytes = 0;
        attr.accessPolicyWindow.hitRatio  = 0.0f;
        attr.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
        attr.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;
        cudaStreamSetAttribute((cudaStream_t)0,
                               cudaStreamAttributeAccessPolicyWindow,
                               &attr);

        cudaCtxResetPersistingL2Cache();
        g_den_l2.window_active = 0;
    }

    // Release L2 reservation
    if (g_den_l2.reserved_bytes > 0) {
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
    }

    // Reset all state
    memset(&g_den_l2, 0, sizeof(g_den_l2));
    fprintf(stderr, "[L2-PERSIST] reset: all pins cleared, L2 released\n");
}

// ── den_l2_persist_diag — write diagnostic string ──────────────────────
// Returns buf (for chaining in printf/fprintf).
static inline const char * den_l2_persist_diag(char * buf, size_t buf_size) {
    if (!buf || !buf_size) return NULL;

    if (!g_den_l2.initialized) {
        snprintf(buf, buf_size, "L2 Persist: NOT INITIALIZED");
        return buf;
    }

    long long total = g_den_l2.hits + g_den_l2.misses;
    float hit_rate = (total > 0) ? (100.0f * (float)g_den_l2.hits / (float)total) : 0.0f;

    int n = snprintf(buf, buf_size,
        "L2 Persist: %d tracked (%zu MB), %d experts cached, "
        "%d slabs, hits=%lld misses=%lld rate=%.1f%%, "
        "budget=%d MB, cuMemAdvise=%s, accessPolicy=%s, "
        "GB203 L2=48 MB",
        g_den_l2.n_tracked,
        g_den_l2.total_tracked_bytes / (1024 * 1024),
        g_den_l2.n_experts_cached,
        g_den_l2.n_expert_slabs_pinned,
        g_den_l2.hits, g_den_l2.misses, hit_rate,
        g_den_l2.budget_mb,
        g_den_l2.cap_cu_mem_advise ? "OK" : "N/A",
        g_den_l2.cap_access_policy  ? "OK" : "N/A");
    if (n < 0 || (size_t)n >= buf_size) buf[buf_size - 1] = '\0';
    return buf;
}

#ifdef __cplusplus
}
#endif
