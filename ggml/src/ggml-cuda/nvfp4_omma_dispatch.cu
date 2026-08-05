// nvfp4_omma_dispatch.cu — Host-side cubin loader and NVFP4 GEMV dispatcher
//
// Loads nvfp4_omma.cubin at runtime, converts GGML block_nvfp4 → 160B tile format,
// launches OMMA.SF.16864 kernel for single-token decode GEMV.
//
// Part of the NVFP4 OMMA port: dengine → ik_llama.cpp

#include "nvfp4_omma.cuh"
#include "common.cuh"
#include "mmvq-args.h"
#include "nvfp4_omma_cubin_data.h"  // embedded cubin (generated at build time)
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ── Global cubin state ──────────────────────────────────────────────────
static CUmodule   g_nvfp4_module   = nullptr;
static CUfunction g_nvfp4_func       = nullptr;  // original (tile-based)
static CUfunction g_nvfp4_std_func   = nullptr;  // standard NULLGLASS nibble-decode GEMV
static CUfunction g_nvfp4_fused_func  = nullptr;  // fused (block-based, Innovation #1)
static CUfunction g_nvfp4_wh4_func    = nullptr;  // fused WH4 (WHT inline, Innovation #5)
static CUfunction g_nvfp4_probe_func  = nullptr;  // debug probe (pointer validation)
static CUfunction g_nvfp4_hardcoded_func = nullptr;  // hardcoded OMMA (no mem reads)
static CUfunction g_nvfp4_fullprobe_func = nullptr; // full probe (all tiles, no OMMA)
static CUfunction g_nvfp4_mt_func    = nullptr; // multi-thread OMMA test
static CUfunction g_nvfp4_smem_func  = nullptr; // SMEM-staged fused (no register pressure)
static CUfunction g_nvfp4_batch_func  = nullptr;
static bool       g_nvfp4_loaded      = false;
static uint8_t*   g_tile_buffer       = nullptr;  // fallback tile buffer
static size_t     g_tile_buf_size     = 0;

// Persistent fp16 activation staging buffer (M==1 GEMV input conversion).
// Allocated once and grown on demand — avoids a cudaMalloc per dispatch.
static half*      g_x_f16             = nullptr;
static size_t     g_x_f16_cap         = 0;

// ── Load cubin from embedded data or file ───────────────────────────────
cudaError_t nvfp4_omma_init(void) {
    if (g_nvfp4_loaded) return cudaSuccess;

    CUresult cu_err;
    const uint8_t* cubin_data = nullptr;
    size_t size = 0;

    // 1. Try the embedded cubin first (robust — no CWD dependence).
    if (nvfp4_omma_cubin_len > 0) {
        cubin_data = nvfp4_omma_cubin;
        size = nvfp4_omma_cubin_len;
    }

    // 2. Fall back to reading the cubin file from disk.
    uint8_t* heap_data = nullptr;
    if (!cubin_data) {
        const char* cubin_paths[] = {
            "nvfp4_omma.cubin",
            "bin/nvfp4_omma.cubin",
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
            fprintf(stderr, "nvfp4_omma_init: no embedded cubin and could not find nvfp4_omma.cubin\n");
            return cudaErrorFileNotFound;
        }

        fseek(f, 0, SEEK_END);
        size = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);

        heap_data = (uint8_t*)malloc(size);
        if (!heap_data) { fclose(f); return cudaErrorMemoryAllocation; }
        if (fread(heap_data, 1, size, f) != size) { fclose(f); free(heap_data); return cudaErrorInvalidValue; }
        fclose(f);
        cubin_data = heap_data;
    }

    cu_err = cuModuleLoadData(&g_nvfp4_module, cubin_data);
    if (heap_data) free(heap_data);

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

    // Load standard NULLGLASS nibble-decode GEMV (standard NVFP4 block path)
    cu_err = cuModuleGetFunction(&g_nvfp4_std_func, g_nvfp4_module, "nvfp4_gemv_std_kernel");
    if (cu_err != CUDA_SUCCESS) {
        g_nvfp4_std_func = nullptr;
        fprintf(stderr, "nvfp4_omma_init: cuModuleGetFunction(gemv_std) failed: %d\n", cu_err);
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

    // Load SMEM-staged fused kernel (definitive fix for register conflict)
    cu_err = cuModuleGetFunction(&g_nvfp4_smem_func, g_nvfp4_module, "nvfp4_gemv_smem_fused_kernel");
    if (cu_err != CUDA_SUCCESS) g_nvfp4_smem_func = nullptr;

    g_nvfp4_loaded = true;
    fprintf(stderr, "nvfp4_omma_init: OMMA.SF.16864 cubin loaded (%u bytes, %s)\n",
            (unsigned)size, size == nvfp4_omma_cubin_len ? "embedded" : "file");
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

// ── Float32 → Float16 conversion kernel ──────────────────────────────────
__global__ void nvfp4_f32_to_f16_kernel(const float* __restrict__ src,
                                         half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void nvfp4_f32_to_f16(const float* src, half* dst, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    nvfp4_f32_to_f16_kernel<<<blocks, 256, 0, stream>>>(src, dst, n);
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

    // Dispatch chain: proven 160B NULLGLASS tile kernel → fused fallbacks
    CUresult cu_err = CUDA_SUCCESS;
    // nvfp4_gemv_kernel — PRIMARY. Reads the fork's 160B block_nvfp4 layout
    // directly (d4 UE4M3 scales @0-15, E2M1 nibbles @16-143, dispatch byte
    // @148, per-tile norm @152:155). SASS-proven on sm_120a.
    if (g_nvfp4_func) {
        void* kargs[] = { &d_blocks_ptr, &d_x_ptr, &d_y_ptr, &N, &K, &tpr };
        cu_err = cuLaunchKernel(g_nvfp4_func, N, 1, 1, 32, 1, 1, 0, stream, kargs, nullptr);
    } else if (g_nvfp4_smem_func) {
        void* kargs[] = { &d_blocks_ptr, &d_x_ptr, &d_y_ptr, &N, &K, &tpr };
        cu_err = cuLaunchKernel(g_nvfp4_smem_func, N, 1, 1, 32, 1, 1, 0, stream, kargs, nullptr);
    } else if (g_nvfp4_mt_func) {
        void* kargs[] = { &d_y_ptr, &N };
        cu_err = cuLaunchKernel(g_nvfp4_mt_func, N, 1, 1, 32, 1, 1, 0, stream, kargs, nullptr);
    } else if (g_nvfp4_wh4_func) {
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

// ── Governor dispatch entry points ───────────────────────────────────────
// These route the active NVFP4 GEMV dispatch (den_governor_dispatch) through
// the proven OMMA.SF.16864 cubin, with the software in-library OMMA as
// fallback. Input activations are fp32 (converted to fp16 for the kernel).

bool den_omma_cubin_ready(void) {
    if (nvfp4_omma_init() != cudaSuccess) return false;
    return g_nvfp4_func != nullptr;
}

// Launch one NVFP4 GEMV: y = tiles @ x. N rows, K cols, fp32 activations.
cudaError_t den_omma_launch_gemv(
    const uint8_t* weights, const float* act, float* dst,
    int N, int K, cudaStream_t stream)
{
    if (nvfp4_omma_init() != cudaSuccess || g_nvfp4_func == nullptr) {
        return cudaErrorUnknown;
    }
    int tpr = (K + 255) / 256;

    static int den_omma_launch_count = 0;
    if (den_omma_launch_count < 3) {
        fprintf(stderr, "DEN_OMMA: cubin GEMV N=%d K=%d tpr=%d\n", N, K, tpr);
        den_omma_launch_count++;
    }

    // Stage fp16 activations in the persistent buffer (grow on demand).
    const size_t need = (size_t)K;
    if (need > g_x_f16_cap) {
        if (g_x_f16) cudaFree(g_x_f16);
        g_x_f16_cap = need;
        if (cudaMalloc(&g_x_f16, g_x_f16_cap * sizeof(half)) != cudaSuccess) {
            g_x_f16_cap = 0;
            return cudaErrorMemoryAllocation;
        }
    }
    nvfp4_f32_to_f16(act, g_x_f16, K, stream);

    const uint8_t* blocks = weights;
    void* kargs[] = { (void*)&blocks, (void*)&g_x_f16, (void*)&dst, &N, &K, &tpr };
    CUresult cu_err = cuLaunchKernel(g_nvfp4_func, (unsigned)N, 1, 1, 32, 1, 1, 0, stream, kargs, nullptr);
    if (cu_err != CUDA_SUCCESS) {
        const char* es; cuGetErrorString(cu_err, &es);
        fprintf(stderr, "nvfp4_omma: governor launch failed: %s\n", es);
        return cudaErrorLaunchFailure;
    }
    return cudaSuccess;
}

// ── Standard NVFP4 NULLGLASS GEMV (GDN/hybrid attn + output path) ────────────
// Same launch shape as den_omma_launch_gemv, but uses nvfp4_gemv_std_kernel:
// the corrected NULLGLASS nibble decode that reproduces the coherent
// soft-gemv / CPU dequant result on the OMMA.SF.16864 tensor core.
// fp32 activations are staged to fp16 (mxf4nvf4 requires an E2M1 B operand).
// Global scale is NOT applied here — the folded per-tile norm [152:155] is
// read in-kernel, exactly matching the proven-coherent soft-gemv path.
cudaError_t den_omma_launch_gemv_std(
    const uint8_t* weights, const float* act, float* dst,
    int N, int K, cudaStream_t stream)
{
    if (nvfp4_omma_init() != cudaSuccess || g_nvfp4_std_func == nullptr) {
        return cudaErrorUnknown;
    }
    int tpr = (K + 255) / 256;

    // Stage fp16 activations in the persistent buffer (grow on demand).
    const size_t need = (size_t)K;
    if (need > g_x_f16_cap) {
        if (g_x_f16) cudaFree(g_x_f16);
        g_x_f16_cap = need;
        if (cudaMalloc(&g_x_f16, g_x_f16_cap * sizeof(half)) != cudaSuccess) {
            g_x_f16_cap = 0;
            return cudaErrorMemoryAllocation;
        }
    }
    nvfp4_f32_to_f16(act, g_x_f16, K, stream);

    const uint8_t* blocks = weights;
    void* kargs[] = { (void*)&blocks, (void*)&g_x_f16, (void*)&dst, &N, &K, &tpr };
    CUresult cu_err = cuLaunchKernel(g_nvfp4_std_func, (unsigned)N, 1, 1, 32, 1, 1, 0, stream, kargs, nullptr);
    if (cu_err != CUDA_SUCCESS) {
        const char* es; cuGetErrorString(cu_err, &es);
        fprintf(stderr, "nvfp4_omma: standard-launch failed: %s\n", es);
        return cudaErrorLaunchFailure;
    }
    return cudaSuccess;
}
