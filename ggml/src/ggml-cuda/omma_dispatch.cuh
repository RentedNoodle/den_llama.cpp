// omma_dispatch.cuh — OMMA.SF.16864 dispatch for NVFP4 matmul on sm_120a
//
// Loads the proven OMMA cubin at runtime via CUDA Driver API and dispatches
// NVFP4 GEMV through the tensor core OMMA instruction (3-5x faster than
// SW dequant + cuBLAS).
//
// Usage:
//   1. Call ggml_cuda_omma_init() once during backend init
//   2. For NVFP4 GEMV, call ggml_cuda_omma_gemv() instead of the normal path
//   3. Falls back gracefully if sm < 12.0 or cubin not found
//
// Requires: CUDA Driver API (libcuda.so / nvcuda.dll)
//           sm_120a or higher (Blackwell GB203/B200/B300)
//           Compiled cubin at a known path (omma_gemv_proven.cubin)

#pragma once

#include "ggml.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize OMMA: check cc >= 12, load cubin, extract kernel function.
// Returns true if OMMA is available and ready for dispatch.
// Safe to call multiple times — subsequent calls are no-ops.
bool ggml_cuda_omma_init(void);

// Returns true if OMMA cubin was loaded successfully and is ready.
bool ggml_cuda_omma_available(void);

// Dispatch NVFP4 GEMV through OMMA tensor core cubin.
// Returns true if the kernel was launched successfully.
// On failure (wrong GPU, no cubin, launch error), returns false — caller
// should fall back to the normal SW dequant + cuBLAS path.
//
// Parameters:
//   src0   - NVFP4 weight tensor [ne0=K, ne1=N] (columns K, rows N)
//   src1   - FP32 input tensor [ne0=K, ne1=1] (single column vector)
//   dst    - FP32 output tensor [ne0=N, ne1=1] (result vector)
//   stream - CUDA stream for kernel launch
bool ggml_cuda_omma_gemv(const struct ggml_tensor * src0,
                          const struct ggml_tensor * src1,
                          struct ggml_tensor * dst,
                          cudaStream_t stream);

// Dispatch NVFP4 GEMM (multi-column matmul) through OMMA tensor core cubin.
// Launches one GEMV per column — each column computes y[*,col] = weight * x[*,col].
// For batch sizes up to 32 columns. Returns true on successful launch.
//
// Parameters:
//   src0   - NVFP4 weight tensor [ne0=K, ne1=N] (columns K, rows N)
//   src1   - FP32 input tensor [ne0=K, ne1=M] (M column vectors)
//   dst    - FP32 output tensor [ne0=N, ne1=M] (M result vectors)
//   stream - CUDA stream for kernel launch
bool ggml_cuda_omma_gemm(const struct ggml_tensor * src0,
                          const struct ggml_tensor * src1,
                          struct ggml_tensor * dst,
                          cudaStream_t stream);

#ifdef __cplusplus
}
#endif
