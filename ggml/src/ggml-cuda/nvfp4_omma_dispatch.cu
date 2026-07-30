// nvfp4_omma_dispatch.cu — Host-side cubin loader and NVFP4 GEMV dispatcher
//
// Loads nvfp4_omma.cubin at runtime, converts GGML block_nvfp4 → 160B tile format,
// launches OMMA.SF.16864 kernel for single-token decode GEMV.
//
// Part of the NVFP4 OMMA port: dengine → ik_llama.cpp

#include "nvfp4_omma.cuh"
#include "common.cuh"
#include "mmvq-args.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ── Global cubin state ──────────────────────────────────────────────────
static CUmodule   g_nvfp4_module   = nullptr;
static CUfunction g_nvfp4_func       = nullptr;  // original (tile-based)
static CUfunction g_nvfp4_fused_func  = nullptr;  // fused (block-based, Innovation #1)
static CUfunction g_nvfp4_wh4_func    = nullptr;  // fused WH4 (WHT inline, Innovation #5)
static CUfunction g_nvfp4_probe_func  = nullptr;  // debug probe (pointer validation)
static CUfunction g_nvfp4_hardcoded_func = nullptr;  // hardcoded OMMA (no mem reads)
static CUfunction g_nvfp4_fullprobe_func = nullptr; // full probe (all tiles, no OMMA)
static CUfunction g_nvfp4_mt_func    = nullptr; // multi-thread OMMA test
static CUfunction g_nvfp4_batch_func  = nullptr;
static bool       g_nvfp4_loaded      = false;
static uint8_t*   g_tile_buffer       = nullptr;  // fallback tile buffer
static size_t     g_tile_buf_size     = 0;

// ── Load cubin from file or embedded data ───────────────────────────────
cudaError_t nvfp4_omma_init(void) {
    if (g_nvfp4_loaded) return cudaSuccess;

    // Try loading from cubin file alongside the binary
    // Falls back to embedded cubin if file not found
    CUresult cu_err;
    const char* cubin_paths[] = {
        "nvfp4_omma.cubin",
        "ggml-cuda/nvfp4_omma.cubin",
        "../ggml/src/ggml-cuda/nvfp4_omma.cubin",
        nullptr
    };

    FILE* f = nullptr;
    for (int i = 0; cubin_paths[i]; i++) {
        f = fopen(cubin_paths[i], "rb");
        if (f) break;
    }

    if (!f) {
        fprintf(stderr, "nvfp4_omma_init: could not find nvfp4_omma.cubin\n");
        return cudaErrorFileNotFound;
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t* cubin_data = (uint8_t*)malloc(size);
    if (!cubin_data) { fclose(f); return cudaErrorMemoryAllocation; }
    if (fread(cubin_data, 1, size, f) != size) { fclose(f); free(cubin_data); return cudaErrorInvalidValue; }
    fclose(f);

    cu_err = cuModuleLoadData(&g_nvfp4_module, cubin_data);
    free(cubin_data);

    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "nvfp4_omma_init: cuModuleLoadData failed: %d\n", cu_err);
        return cudaErrorUnknown;
    }

    cu_err = cuModuleGetFunction(&g_nvfp4_func, g_nvfp4_module, "nvfp4_gemv_kernel");
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "nvfp4_omma_init: cuModuleGetFunction(gemv) failed: %d\n", cu_err);
        return cudaErrorUnknown;
    }

    cu_err = cuModuleGetFunction(&g_nvfp4_batch_func, g_nvfp4_module, "nvfp4_gemv_batch_kernel");
    if (cu_err != CUDA_SUCCESS) {
        g_nvfp4_batch_func = nullptr;
    }

    // Load fused kernel (Innovation #1 — block_nvfp4 → OMMA directly)
    cu_err = cuModuleGetFunction(&g_nvfp4_fused_func, g_nvfp4_module, "nvfp4_gemv_fused_kernel");
    if (cu_err != CUDA_SUCCESS) {
        g_nvfp4_fused_func = nullptr;
    }

    // Load WH4 fused kernel (Innovation #5 — WHT inline + OMMA)
    cu_err = cuModuleGetFunction(&g_nvfp4_wh4_func, g_nvfp4_module, "nvfp4_gemv_fused_wh4_kernel");
    if (cu_err != CUDA_SUCCESS) {
        g_nvfp4_wh4_func = nullptr;
    }

    // Load probe kernel (debug — pointer validation)
    cu_err = cuModuleGetFunction(&g_nvfp4_probe_func, g_nvfp4_module, "nvfp4_probe_kernel");
    if (cu_err != CUDA_SUCCESS) g_nvfp4_probe_func = nullptr;

    // Load hardcoded OMMA kernel (debug — isolates PTX from memory)
    cu_err = cuModuleGetFunction(&g_nvfp4_hardcoded_func, g_nvfp4_module, "nvfp4_omma_hardcoded_kernel");
    if (cu_err != CUDA_SUCCESS) g_nvfp4_hardcoded_func = nullptr;

    // Load full probe kernel (debug — same access patterns, no OMMA)
    cu_err = cuModuleGetFunction(&g_nvfp4_fullprobe_func, g_nvfp4_module, "nvfp4_probe_full_kernel");
    if (cu_err != CUDA_SUCCESS) g_nvfp4_fullprobe_func = nullptr;

    // Load multi-thread OMMA test kernel
    cu_err = cuModuleGetFunction(&g_nvfp4_mt_func, g_nvfp4_module, "nvfp4_omma_mt_kernel");
    if (cu_err != CUDA_SUCCESS) g_nvfp4_mt_func = nullptr;

    g_nvfp4_loaded = true;
    return cudaSuccess;
}

// ── Convert GGML block_nvfp4 → 160B tile on GPU ─────────────────────────
//
// GGML block_nvfp4 layout:  d[0:1](fp16), scales[2:17], qs[18:145]    (146 bytes)
// 160B NULLGLASS tile:      scales[0:15],  qs[16:143],  norm[144:147]  (160 bytes)
//
// Conversion: d → float at [144], scales shifted to [0], qs shifted to [16]
__global__ void convert_blocks_to_tiles_kernel(
    const uint8_t* __restrict__ blocks,  // [N * tpr] block_nvfp4 structs
    uint8_t*       __restrict__ tiles,    // [N * tpr] 160B tiles
    int num_blocks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;

    const uint8_t* blk = blocks + idx * 146;  // sizeof(block_nvfp4) = 146
    uint8_t*       til = tiles + idx * 160;

    // Copy scales: blk[2:17] → til[0:15]
    for (int i = 0; i < 16; i += 4) {
        *(uint32_t*)(til + i) = *(const uint32_t*)(blk + 2 + i);
    }

    // Copy nibbles: blk[18:145] → til[16:143]
    for (int i = 0; i < 128; i += 4) {
        *(uint32_t*)(til + 16 + i) = *(const uint32_t*)(blk + 18 + i);
    }

    // Convert d (fp16) to float at til[144:147]
    uint16_t d_bits;
    memcpy(&d_bits, blk, 2);
    float norm = __half2float(*(const __half*)&d_bits);
    memcpy(til + 144, &norm, 4);

    // Zero reserved bytes
    *(uint32_t*)(til + 148) = 0;
    *(uint32_t*)(til + 152) = 0;
    *(uint32_t*)(til + 156) = 0;
}

// ── GEMV dispatch (called from mmvq.cu) ─────────────────────────────────
//
// Matches the mmvq function signature pattern: (mmvq_args, cudaStream_t)
void mul_mat_vec_nvfp4_cuda(const mmvq_args& args, cudaStream_t stream) {
    // Ensure cubin is loaded
    cudaError_t err = nvfp4_omma_init();
    if (err != cudaSuccess) {
        cudaMemsetAsync(args.dst, 0, args.nrows_x * sizeof(float), stream);
        return;
    }

    int N = args.nrows_x;
    int K = args.ncols_x;
    int tpr = (K + 255) / 256;

    const uint8_t* blocks = (const uint8_t*)args.vx_u;
    void* d_blocks_ptr = const_cast<uint8_t*>(blocks);
    void* d_x_ptr = const_cast<void*>(static_cast<const void*>(args.vy));
    void* d_y_ptr = static_cast<void*>(args.dst);

    // Dispatch chain: WH4 fused → regular fused → tile fallback
    CUresult cu_err = CUDA_SUCCESS;
    // Final: WH4 fused with noinline OMMA wrapper
    if (g_nvfp4_wh4_func) {
        void* kargs[] = { &d_blocks_ptr, &d_x_ptr, &d_y_ptr, &N, &K, &tpr };
        cu_err = cuLaunchKernel(g_nvfp4_wh4_func, N, 1, 1, 32, 1, 1, 0, stream, kargs, nullptr);
    } else if (g_nvfp4_fused_func) {
        void* kargs[] = { &d_blocks_ptr, &d_x_ptr, &d_y_ptr, &N, &K, &tpr };
        cu_err = cuLaunchKernel(g_nvfp4_fused_func, N, 1, 1, 32, 1, 1, 0, stream, kargs, nullptr);
    } else {
        cudaMemsetAsync(args.dst, 0, args.nrows_x * sizeof(float), stream);
        return;
    }
    if (cu_err != CUDA_SUCCESS) {
        const char* es; cuGetErrorString(cu_err, &es);
        fprintf(stderr, "nvfp4_omma: launch: %s\n", es);
        cudaMemsetAsync(args.dst, 0, args.nrows_x * sizeof(float), stream);
    }
}
