// sparse-buft.cu — Sparse VMM buffer type (ggml_backend_buffer_type backend)
//
// Wraps den_sparse_vmm as a ggml backend buffer type following the
// ggml_backend_buffer_type_i interface pattern. Tensors allocated into
// this buffer get device pointers in the sparse VA region.

#include "sparse-buft.cuh"
#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

struct sparse_buft_context {
    den_sparse_vmm_t pool;
};

// ═══════════════════════════════════════════════════════
// Buffer (ggml_backend_buffer_i)
// ═══════════════════════════════════════════════════════

struct sparse_buf_context {
    den_sparse_vmm_t pool;
    void * base;
    size_t size;
};

static void sparse_vmm_buf_free(ggml_backend_buffer_t buf) {
    auto * ctx = (sparse_buf_context *)buf->context;
    delete ctx;
    delete buf;
}

static void * sparse_vmm_buf_get_base(ggml_backend_buffer_t buf) {
    return ((sparse_buf_context *)buf->context)->base;
}

static void sparse_vmm_buf_memset_tensor(ggml_backend_buffer_t buf, ggml_tensor * tensor,
                                          uint8_t value, size_t offset, size_t size) {
    auto * ctx = (sparse_buf_context *)buf->context;
    CUDA_CHECK(cudaMemset((char *)ctx->base + offset, value, size));
}

static void sparse_vmm_buf_set_tensor(ggml_backend_buffer_t buf, ggml_tensor * tensor,
                                       const void * data, size_t offset, size_t size) {
    auto * ctx = (sparse_buf_context *)buf->context;
    CUDA_CHECK(cudaMemcpy((char *)ctx->base + offset, data, size, cudaMemcpyHostToDevice));
}

static void sparse_vmm_buf_get_tensor(ggml_backend_buffer_t buf, const ggml_tensor * tensor,
                                       void * data, size_t offset, size_t size) {
    auto * ctx = (sparse_buf_context *)buf->context;
    CUDA_CHECK(cudaMemcpy(data, (char *)ctx->base + offset, size, cudaMemcpyDeviceToHost));
}

static const ggml_backend_buffer_i sparse_vmm_buffer_interface = {
    /* .free_buffer  = */ sparse_vmm_buf_free,
    /* .get_base     = */ sparse_vmm_buf_get_base,
    /* .init_tensor  = */ nullptr,
    /* .memset_tensor= */ sparse_vmm_buf_memset_tensor,
    /* .set_tensor    = */ sparse_vmm_buf_set_tensor,
    /* .get_tensor    = */ sparse_vmm_buf_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .supports_op   = */ nullptr,
    /* .usage         = */ nullptr,
};

// ═══════════════════════════════════════════════════════
// Buffer type (ggml_backend_buffer_type_i)
// ═══════════════════════════════════════════════════════

static const char * sparse_vmm_buft_name(ggml_backend_buffer_type_t buft) {
    return "CUDA_SparseVMM";
}

static ggml_backend_buffer_t sparse_vmm_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = (sparse_buft_context *)buft->context;
    if (!ctx || !ctx->pool) return nullptr;

    if (den_sparse_vmm_ensure(ctx->pool, size) != 0) {
        fprintf(stderr, "SparseVMM: ensure(%zu) failed\n", size);
        return nullptr;
    }

    void * base = den_sparse_vmm_ptr(ctx->pool);
    if (!base) return nullptr;

    auto * buf = new ggml_backend_buffer;
    auto * buf_ctx = new sparse_buf_context;
    buf_ctx->pool = ctx->pool;
    buf_ctx->base = base;
    buf_ctx->size = size;

    buf->iface   = sparse_vmm_buffer_interface;
    buf->context = buf_ctx;
    buf->size    = size;
    buf->usage   = size;

    return buf;
}

static size_t sparse_vmm_buft_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static size_t sparse_vmm_buft_max_size(ggml_backend_buffer_type_t buft) {
    auto * ctx = (sparse_buft_context *)buft->context;
    return den_sparse_vmm_reserved(ctx->pool);
}

static bool sparse_vmm_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static size_t sparse_vmm_buft_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return GGML_PAD(ggml_nbytes(tensor), 128);
}

static const ggml_backend_buffer_type_i sparse_vmm_buft_interface = {
    /* .get_name       = */ sparse_vmm_buft_name,
    /* .alloc_buffer   = */ sparse_vmm_buft_alloc,
    /* .get_alignment  = */ sparse_vmm_buft_alignment,
    /* .get_max_size   = */ sparse_vmm_buft_max_size,
    /* .get_alloc_size = */ sparse_vmm_buft_alloc_size,
    /* .is_host        = */ sparse_vmm_buft_is_host,
};

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

    auto * ctx = new sparse_buft_context;
    ctx->pool = pool;

    auto * buft = new ggml_backend_buffer_type;
    buft->iface   = sparse_vmm_buft_interface;
    buft->device  = nullptr; // no specific device binding
    buft->context = ctx;

    return buft;
}
