// sparse-buft.cuh — Sparse VMM ggml backend buffer type
//
// Wraps den_sparse_vmm as a ggml_backend_buffer_type so KV cache tensors
// can be allocated into the sparse virtual address space.

#pragma once

#include "ggml-backend.h"
#include "sparse-vmm.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// Get the buffer type for a given sparse VMM pool.
// Returns NULL if VMM is unsupported on the current device.
ggml_backend_buffer_type_t ggml_backend_cuda_sparse_vmm_buffer_type(den_sparse_vmm_t pool);

// Check if VMM is supported on the current CUDA device.
bool ggml_backend_cuda_sparse_vmm_supported(void);

#ifdef __cplusplus
}
#endif
