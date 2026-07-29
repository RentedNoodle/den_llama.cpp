// l2_cache.cu — L2 cache pinning implementation
// Ported from dengine Mech 41/42 (den_gpu_l2.cu)
//
// CUDA 13.3+ cuMemAdvise API for L2 persistence.
// Pinned tensors benefit from reduced DRAM bandwidth via L2 residency.

#include "l2_cache.cuh"
#include <cstdio>
#include <cstdlib>

#define L2_CACHE_MAX_TENSORS  64
#define L2_CACHE_MAX_EXPERTS  256

static bool  g_l2_initialized    = false;
static int   g_l2_reserve_mb     = 0;

// ── Initialization ──────────────────────────────────────────────────────────

cudaError_t l2_cache_init(int reserve_mb) {
    if (g_l2_initialized) return cudaSuccess;

    if (reserve_mb <= 0) {
        // Auto-select: 8 MB default, up to 75% of 48 MB L2 = 36 MB
        reserve_mb = 8;
    }

    // Clamp: max 36 MB on GB203 (48 MB L2, 75% reservable)
    if (reserve_mb > 36) reserve_mb = 36;
    if (reserve_mb < 1)  reserve_mb = 1;

    size_t reserve_bytes = (size_t)reserve_mb * 1024 * 1024;

    cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserve_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "l2_cache_init: cudaDeviceSetLimit(%zu) failed: %s\n",
                reserve_bytes, cudaGetErrorString(err));
        return err;
    }

    g_l2_reserve_mb = reserve_mb;
    g_l2_initialized = true;

    fprintf(stderr, "l2_cache: reserved %d MB L2 for persistence\n", reserve_mb);
    return cudaSuccess;
}

// ── Tensor Pinning ──────────────────────────────────────────────────────────

cudaError_t l2_cache_pin_tensor(void* d_ptr, size_t bytes, bool read_only) {
    if (!g_l2_initialized || !d_ptr || bytes == 0) return cudaSuccess;

    CUmemLocation loc;
    loc.type = CU_MEM_LOCATION_TYPE_DEVICE;
    loc.id = 0;  // device 0

    CUresult cu_err;

    if (read_only) {
        // Hint: keep read-only data (weights) preferentially in L2
        cu_err = cuMemAdvise((CUdeviceptr)d_ptr, bytes,
                             CU_MEM_ADVISE_SET_READ_MOSTLY, loc);
        if (cu_err != CUDA_SUCCESS) return cudaErrorUnknown;
    }

    // Hint: this device accesses this memory frequently
    cu_err = cuMemAdvise((CUdeviceptr)d_ptr, bytes,
                         CU_MEM_ADVISE_SET_ACCESSED_BY, loc);
    if (cu_err != CUDA_SUCCESS) return cudaErrorUnknown;

    return cudaSuccess;
}

// ── Expert Weight Slab Pinning ──────────────────────────────────────────────

cudaError_t l2_cache_pin_experts(void** expert_ptrs, size_t* expert_bytes, int n_experts) {
    if (!g_l2_initialized || !expert_ptrs || n_experts <= 0) return cudaSuccess;

    int pinned = 0;
    for (int i = 0; i < n_experts && i < L2_CACHE_MAX_EXPERTS; i++) {
        if (!expert_ptrs[i] || expert_bytes[i] == 0) continue;

        // Expert weights are read-only — use SET_READ_MOSTLY + SET_ACCESSED_BY
        cudaError_t err = l2_cache_pin_tensor(expert_ptrs[i], expert_bytes[i], true);
        if (err == cudaSuccess) pinned++;
    }

    if (pinned > 0) {
        fprintf(stderr, "l2_cache: pinned %d expert weight slabs to L2\n", pinned);
    }
    return cudaSuccess;
}

// ── Query ───────────────────────────────────────────────────────────────────

bool l2_cache_is_initialized(void) {
    return g_l2_initialized;
}

int l2_cache_get_reserve_mb(void) {
    return g_l2_reserve_mb;
}

// ── Shutdown ────────────────────────────────────────────────────────────────

void l2_cache_shutdown(void) {
    if (!g_l2_initialized) return;

    // Clear persisting L2 — any future accesses will be normal cache misses
    cudaCtxResetPersistingL2Cache();

    g_l2_initialized = false;
    g_l2_reserve_mb = 0;
}
