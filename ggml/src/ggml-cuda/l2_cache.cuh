// l2_cache.cuh — L2 cache pinning for critical tensors and MoE expert weights
// Ported from dengine Mech 41/42 (den_gpu_l2.cu)
//
// Uses CUDA 13.3+ cuMemAdvise API to pin GPU buffers in L2 cache.
// Mechanism 41: Critical tensor pinning (embedding, output norm)
// Mechanism 42: Expert weight slab pinning (hot experts stay in L2)
//
#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize L2 cache subsystem. Reserves `reserve_mb` MB of L2 for persistence.
// Call once during CUDA backend initialization.
// Returns cudaSuccess on success.
cudaError_t l2_cache_init(int reserve_mb);

// Pin a GPU buffer to L2 cache.
// d_ptr: device pointer
// bytes: size in bytes
// read_only: true for weights (SET_READ_MOSTLY), false for workspace
cudaError_t l2_cache_pin_tensor(void* d_ptr, size_t bytes, bool read_only);

// Pin multiple expert weight slabs to L2 cache.
// expert_ptrs: array of device pointers to expert weight tensors
// expert_bytes: array of sizes in bytes
// n_experts: number of expert weight tensors
cudaError_t l2_cache_pin_experts(void** expert_ptrs, size_t* expert_bytes, int n_experts);

// Check if L2 cache subsystem is initialized.
bool l2_cache_is_initialized(void);

// Get current L2 reservation size in MB.
int l2_cache_get_reserve_mb(void);

// Release all L2 cache reservations.
void l2_cache_shutdown(void);

#ifdef __cplusplus
}
#endif
