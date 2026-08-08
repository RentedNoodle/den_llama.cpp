#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// NVFP4 KV Cache: initialize the per-layer GPU tile buffers.
// Must be called after model load (when n_kv_heads + head_dim are known).
// Reads DEN_NVFP4_KV_CACHE env var (default: enabled).
GGML_BACKEND_API void ggml_backend_cuda_nvfp4_kv_init(
    int n_attn_layers, int n_kv_heads, int head_dim, int max_seq);

// Reset all NVFP4 KV cache sequence lengths to 0.
// Must be called after warmup or cache clear.
GGML_BACKEND_API void ggml_backend_cuda_nvfp4_kv_reset_all(void);

// Sparse Virtual Memory Manager for KV cache
// Enables 600K+ context by decoupling VA footprint from physical commit.
// Opaque pool handle
typedef struct den_sparse_vmm_pool * ggml_sparse_vmm_t;

GGML_BACKEND_API bool ggml_backend_cuda_sparse_vmm_supported(void);
GGML_BACKEND_API ggml_sparse_vmm_t ggml_backend_cuda_sparse_vmm_create(size_t reserve_bytes, size_t initial_bytes);
GGML_BACKEND_API int  ggml_backend_cuda_sparse_vmm_ensure(ggml_sparse_vmm_t pool, size_t required_bytes);
GGML_BACKEND_API void ggml_backend_cuda_sparse_vmm_destroy(ggml_sparse_vmm_t pool);
GGML_BACKEND_API void * ggml_backend_cuda_sparse_vmm_ptr(ggml_sparse_vmm_t pool);
GGML_BACKEND_API size_t ggml_backend_cuda_sparse_vmm_committed(ggml_sparse_vmm_t pool);
GGML_BACKEND_API size_t ggml_backend_cuda_sparse_vmm_reserved(ggml_sparse_vmm_t pool);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_sparse_vmm_buffer_type(ggml_sparse_vmm_t pool);

#ifdef  __cplusplus
}
#endif
