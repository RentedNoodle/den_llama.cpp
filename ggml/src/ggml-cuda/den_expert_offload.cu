// den_expert_offload.cu — Async H2D copy engine + VRAM expert LRU cache pool
//
// Blocker 3 (expert-offload bandwidth): the per-layer selective upload of MoE
// expert weights from the host (CPU/CUDA_Host) buffer to the GPU copy tensor is
// the throughput wall for -ot exps=CPU (35B caps at ~0.8 t/s). The upload runs
// on the single compute stream and is fully exposed between a layer's attention
// and its expert compute.
//
// This TU provides:
//   1. A per-device NON-BLOCKING copy stream for the async expert H2D prefetch.
//      Issued from the scheduler right after a layer's expert compute is queued,
//      it overlaps with the GPU executing that layer + the next layer's attention.
//   2. A per-device D2D stream used to materialize cached experts into the graph's
//      copy tensor (input_cpy) at copy-inputs time. input_cpy aliases across
//      layers (ggml-alloc liveness reuse), so cached bytes live in THIS dedicated
//      pool — never aliased by other tensors.
//   3. A VRAM byte pool with a block free-list so LRU eviction can reclaim slots.
//
// Host API (extern "C", called from ggml/src/ggml-backend.cpp):
//   den_expert_offload_init(device)
//   den_expert_h2d_async(device, dst, src, bytes)   // copy stream
//   den_expert_d2d_async(device, dst, src, bytes)   // d2d stream
//   den_expert_offload_wait(device)                 // host-sync copy stream
//   den_expert_d2d_wait(device)                     // host-sync d2d stream
//   den_expert_cache_alloc(device, bytes) -> offset
//   den_expert_cache_free(device, offset, bytes)
//   den_expert_cache_base(device)
//   den_expert_cache_capacity(device) / used
//   den_expert_offload_shutdown(device)
//
// GB203 (RTX 5070 Ti) dual DMA copy engines run these streams independently of
// the SM compute stream. sm_120a. CUDA 13.3.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define DEN_EXP_MAX_DEVICES 16
#define DEN_EXP_DEFAULT_CACHE_BYTES (4ULL * 1024 * 1024 * 1024)
#define DEN_EXP_ALIGN 256

struct DenExpBlock { int64_t off; size_t size; };

struct DenExpertDevice {
    cudaStream_t copy_stream = nullptr;   // H2D prefetch (non-blocking)
    cudaStream_t d2d_stream  = nullptr;   // cache->input_cpy materialization
    void *       cache_base  = nullptr;
    size_t       cache_capacity = 0;
    std::vector<DenExpBlock> free_blocks;
    int          initialized = 0;
};

static DenExpertDevice g_den_exp_dev[DEN_EXP_MAX_DEVICES];

static DenExpertDevice * den_exp_dev(int device) {
    if (device < 0 || device >= DEN_EXP_MAX_DEVICES) return nullptr;
    return &g_den_exp_dev[device];
}

extern "C" {

int den_expert_offload_init(int device) {
    DenExpertDevice * d = den_exp_dev(device);
    if (!d) return -1;
    if (d->initialized) return 0;

    if (cudaSetDevice(device) != cudaSuccess) return -1;
    if (cudaStreamCreateWithFlags(&d->copy_stream, cudaStreamNonBlocking) != cudaSuccess) return -1;
    if (cudaStreamCreateWithFlags(&d->d2d_stream,  cudaStreamNonBlocking) != cudaSuccess) return -1;

    d->initialized = 1;
    return 0;
}

int den_expert_h2d_async(int device, void * dst, const void * src, size_t bytes) {
    DenExpertDevice * d = den_exp_dev(device);
    if (!d || !d->initialized) {
        if (den_expert_offload_init(device) != 0) return -1;
        d = den_exp_dev(device);
    }
    if (bytes == 0 || dst == nullptr || src == nullptr) return 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, d->copy_stream);
    if (err != cudaSuccess) return -1;
    return 0;
}

int den_expert_d2d_async(int device, void * dst, const void * src, size_t bytes) {
    DenExpertDevice * d = den_exp_dev(device);
    if (!d || !d->initialized) {
        if (den_expert_offload_init(device) != 0) return -1;
        d = den_exp_dev(device);
    }
    if (bytes == 0 || dst == nullptr || src == nullptr) return 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, d->d2d_stream);
    if (err != cudaSuccess) return -1;
    return 0;
}

void den_expert_offload_wait(int device) {
    DenExpertDevice * d = den_exp_dev(device);
    if (!d || !d->initialized || !d->copy_stream) return;
    cudaSetDevice(device);
    cudaStreamSynchronize(d->copy_stream);
}

void den_expert_d2d_wait(int device) {
    DenExpertDevice * d = den_exp_dev(device);
    if (!d || !d->initialized || !d->d2d_stream) return;
    cudaSetDevice(device);
    cudaStreamSynchronize(d->d2d_stream);
}

// Allocate `bytes` (aligned) from the pool. Returns pool offset or -1.
int64_t den_expert_cache_alloc(int device, size_t bytes) {
    DenExpertDevice * d = den_exp_dev(device);
    if (!d || !d->initialized) {
        if (den_expert_offload_init(device) != 0) return -1;
        d = den_exp_dev(device);
    }
    if (d->cache_base == nullptr) {
        d->cache_capacity = DEN_EXP_DEFAULT_CACHE_BYTES;
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) return -1;
        err = cudaMalloc(&d->cache_base, d->cache_capacity);
        if (err != cudaSuccess) { d->cache_base = nullptr; d->cache_capacity = 0; return -1; }
        d->free_blocks.clear();
        d->free_blocks.push_back({0, d->cache_capacity});
    }

    size_t need = (bytes + DEN_EXP_ALIGN - 1) & ~(DEN_EXP_ALIGN - 1);

    // first-fit
    for (size_t i = 0; i < d->free_blocks.size(); ++i) {
        DenExpBlock & b = d->free_blocks[i];
        if (b.size >= need) {
            int64_t off = b.off;
            b.off  += (int64_t)need;
            b.size -= need;
            if (b.size == 0) {
                d->free_blocks.erase(d->free_blocks.begin() + i);
            }
            return off;
        }
    }
    return -1;
}

void den_expert_cache_free(int device, int64_t offset, size_t bytes) {
    DenExpertDevice * d = den_exp_dev(device);
    if (!d || !d->initialized || offset < 0) return;
    size_t size = (bytes + DEN_EXP_ALIGN - 1) & ~(DEN_EXP_ALIGN - 1);
    // coalesce with adjacent free blocks
    bool merged = false;
    for (auto & b : d->free_blocks) {
        if (b.off + (int64_t)b.size == offset) {
            b.size += size;
            merged = true;
            break;
        }
        if (offset + (int64_t)size == b.off) {
            b.off  = offset;
            b.size += size;
            merged = true;
            break;
        }
    }
    if (!merged) {
        d->free_blocks.push_back({offset, size});
    }
}

void * den_expert_cache_base(int device) {
    DenExpertDevice * d = den_exp_dev(device);
    if (!d) return nullptr;
    return d->cache_base;
}

size_t den_expert_cache_capacity(int device) {
    DenExpertDevice * d = den_exp_dev(device);
    if (!d) return 0;
    return d->cache_capacity;
}

void den_expert_offload_shutdown(int device) {
    DenExpertDevice * d = den_exp_dev(device);
    if (!d || !d->initialized) return;
    if (d->copy_stream) { cudaStreamDestroy(d->copy_stream); d->copy_stream = nullptr; }
    if (d->d2d_stream)  { cudaStreamDestroy(d->d2d_stream);  d->d2d_stream  = nullptr; }
    if (d->cache_base)  { cudaFree(d->cache_base); d->cache_base = nullptr; }
    d->cache_capacity = 0;
    d->free_blocks.clear();
    d->initialized = 0;
}

} // extern "C"
