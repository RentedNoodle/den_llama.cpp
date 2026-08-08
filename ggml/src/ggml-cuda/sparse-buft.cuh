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

// NOTE: Public API declarations are in ggml/include/ggml-cuda.h with GGML_BACKEND_API.
// This header is internal — only the implementation includes it.
// See ggml-cuda.h for ggml_backend_cuda_sparse_vmm_buffer_type() and friends.

#ifdef __cplusplus
}
#endif
