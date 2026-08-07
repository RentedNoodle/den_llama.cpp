// sparse-buft.cu — Sparse VMM buffer type implementation
//
// Provides a ggml_backend_buffer_type backed by den_sparse_vmm.
// Tensors allocated into this buffer get device pointers into the
// sparse virtual address region. Physical pages are committed lazily
// via den_sparse_vmm_ensure().

#include "sparse-buft.cuh"
#include "common.cuh"

#include <cstdio>
#include <cstdlib>

// ═══════════════════════════════════════════════════════
// Buffer context (tied to one pool)
// ═══════════════════════════════════════════════════════

struct ggml_backend_cuda_sparse_vmm_context {
    den_sparse_vmm_t pool;
};

static const char * sparse_vmm_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "CUDA_SparseVMM";
}

static ggml_backend_buffer_t sparse_vmm_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = (ggml_backend_cuda_sparse_vmm_context *)buft->context;
    if (!ctx || !ctx->pool) return nullptr;

    // Ensure physical capacity before handing out the pointer
    int ret = den_sparse_vmm_ensure(ctx->pool, size);
    if (ret != 0) {
        fprintf(stderr, "SparseVMM buft: failed to ensure %zu bytes\n", size);
        return nullptr;
    }

    void * base = den_sparse_vmm_ptr(ctx->pool);
    if (!base) return nullptr;

    // Create buffer that "owns" the VA range [0, size) in the pool.
    // The pool manages the actual physical memory; the buffer is just a view.
    ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, size);
    return buf; // TODO: replace with a proper device-side buffer
}

static void sparse_vmm_buft_free_buffer(ggml_backend_buffer_type_t buft, ggml_backend_buffer_t buf) {
    GGML_UNUSED(buft);
    // Buffer is a view — pool manages physical memory separately.
    // Just release the buffer wrapper.
    if (buf) {
        free(buf->context);
        free(buf);
    }
}

static size_t sparse_vmm_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128; // standard CUDA alignment
}

static size_t sparse_vmm_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_cuda_sparse_vmm_context *)buft->context;
    return ctx && ctx->pool ? den_sparse_vmm_reserved(ctx->pool) : 0;
}

static bool sparse_vmm_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false; // device memory
}

static size_t sparse_vmm_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    // Return padded size for this tensor
    size_t size = ggml_nbytes(tensor);
    return GGML_PAD(size, 128);
}

// ═══════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════

bool ggml_backend_cuda_sparse_vmm_supported(void) {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    auto & info = ggml_cuda_info();
    return info.devices[device].vmm_granularity > 0;
}

ggml_backend_buffer_type_t ggml_backend_cuda_sparse_vmm_buffer_type(den_sparse_vmm_t pool) {
    if (!pool) return nullptr;
    if (!ggml_backend_cuda_sparse_vmm_supported()) return nullptr;

    auto * ctx = new ggml_backend_cuda_sparse_vmm_context;
    ctx->pool = pool;

    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type;
    memset(buft, 0, sizeof(*buft));
    buft->context       = ctx;
    buft->name          = sparse_vmm_buft_name;
    buft->alloc_buffer  = sparse_vmm_buft_alloc_buffer;
    buft->free_buffer   = sparse_vmm_buft_free_buffer;
    buft->get_alignment = sparse_vmm_buft_get_alignment;
    buft->get_max_size  = sparse_vmm_buft_get_max_size;
    buft->is_host       = sparse_vmm_buft_is_host;
    buft->get_alloc_size= sparse_vmm_buft_get_alloc_size;

    return buft;
}
