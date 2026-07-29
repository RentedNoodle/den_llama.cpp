// nvfp4_omma.cuh — NVFP4 OMMA dispatch header for ik_llama.cpp
//
// DECLARATIONS ONLY — no implementation (PTX isolation per dengine Rule 7.9).
// The actual OMMA PTX lives in nvfp4_omma.cu, compiled as standalone cubin.
// Cubin is loaded at runtime via cuModuleLoadData().
//
// Compile cubin: nvcc -arch=compute_120a -code=sm_120a --cubin -o nvfp4_omma.cubin nvfp4_omma.cu

#pragma once
#include <cuda_runtime.h>
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

#ifdef __cplusplus
}
#endif
