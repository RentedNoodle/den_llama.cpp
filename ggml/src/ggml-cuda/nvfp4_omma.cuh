// nvfp4_omma.cuh — NVFP4 OMMA dispatch header for ik_llama.cpp
//
// DECLARATIONS ONLY — no implementation (PTX isolation per dengine Rule 7.9).
// The actual OMMA PTX lives in nvfp4_omma.cu, compiled as standalone cubin.
// Cubin is loaded at runtime via cuModuleLoadData().
//
// Compile cubin: nvcc -arch=compute_120a -code=sm_120a --cubin -o nvfp4_omma.cubin nvfp4_omma.cu

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Load the NVFP4 OMMA cubin and cache the kernel function handle.
// Call once during CUDA backend initialization.
// Returns cudaSuccess on success, or a cudaError_t on failure.
cudaError_t nvfp4_omma_init(void);

// Dispatch NVFP4 GEMV: y = tiles @ x
//   d_tiles: device pointer to tile data [N*tpr][160 bytes]
//   d_x:     device pointer to fp16 input activations [K]
//   d_y:     device pointer to float output [N]
//   N:       number of output rows
//   K:       number of input columns
//   tpr:     tiles per row (K / 256, rounded up)
//   stream:  CUDA stream
cudaError_t nvfp4_omma_gemv(
    const uint8_t* d_tiles,
    const uint16_t* d_x,
    float* d_y,
    int N, int K, int tpr,
    cudaStream_t stream);

// Utility: convert float32 activations to fp16 for OMMA kernel input
void nvfp4_f32_to_f16(const float* src, half* dst, int n, cudaStream_t stream);

// Forward declaration for mmvq dispatch integration
struct mmvq_args;
void mul_mat_vec_nvfp4_cuda(const mmvq_args& args, cudaStream_t stream);

// ── Governor dispatch integration ─────────────────────────────────────────
// Returns true if the OMMA cubin is loaded and the proven tile kernel is
// available. Call before den_omma_launch_gemv to decide path.
bool den_omma_cubin_ready(void);

// Launch one NVFP4 GEMV through the OMMA.SF.16864 cubin:
//   y = tiles @ x   (160B NULLGLASS blocks, fp32 activations, fp32 output)
// Returns cudaSuccess on success. Caller must fall back to the software path
// on any error return.
cudaError_t den_omma_launch_gemv(
    const uint8_t* weights, const float* act, float* dst,
    int N, int K, cudaStream_t stream);

// Launch one NVFP4 GEMV through the OMMA.SF.16864 cubin using the STANDARD
// NVFP4 NULLGLASS nibble-decode kernel (nvfp4_gemv_std_kernel). Produces the
// same result as the coherent soft-gemv / CPU dequant path, on tensor cores.
// Returns cudaSuccess on success. Caller must fall back on any error.
cudaError_t den_omma_launch_gemv_std(
    const uint8_t* weights, const float* act, float* dst,
    int N, int K, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
