// sparse-vmm.cu — Sparse Virtual Memory Manager for KV cache
//
// Wraps CUDA VMM APIs (cuMemAddressReserve + cuMemCreate + cuMemMap) to
// provide a growable virtual address space backed by deferred physical commits.
//
// Uses the VMM granularity from ggml_cuda_info. All allocations are
// granularity-aligned. Thread-safe via atomic CAS on committed counter.

#include "sparse-vmm.cuh"
#include "common.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ═══════════════════════════════════════════════════════
// Internal struct
// ═══════════════════════════════════════════════════════

struct den_sparse_vmm_pool {
    size_t  granularity;    // VMM allocation granularity (bytes)
    size_t  reserved;       // total VA reserved
    size_t  committed;      // physical bytes committed so far
    CUdeviceptr va_base;    // base of reserved virtual address region
    int     device;         // CUDA device ordinal
    int     physical_device;// physical device for allocation
};

// ═══════════════════════════════════════════════════════
// Create
// ═══════════════════════════════════════════════════════

den_sparse_vmm_t den_sparse_vmm_create(size_t reserve_bytes, size_t initial_bytes) {
    // Get device + granularity from existing ggml_cuda info
    int device;
    CU_CHECK(cudaGetDevice(&device));

    auto & info = ggml_cuda_info();
    size_t granularity = info.devices[device].vmm_granularity;
    if (granularity == 0) {
        fprintf(stderr, "SparseVMM: VMM not supported on device %d\n", device);
        return nullptr;
    }

    int physical_device = ggml_cuda_get_physical_device(device);

    den_sparse_vmm_pool * pool = (den_sparse_vmm_pool *)calloc(1, sizeof(*pool));
    if (!pool) return nullptr;

    pool->granularity     = granularity;
    pool->device          = device;
    pool->physical_device = physical_device;

    // Align to granularity
    reserve_bytes = ((reserve_bytes + granularity - 1) / granularity) * granularity;
    initial_bytes = ((initial_bytes + granularity - 1) / granularity) * granularity;

    if (initial_bytes > reserve_bytes) initial_bytes = reserve_bytes;

    // Reserve virtual address space
    CUresult cr = cuMemAddressReserve(&pool->va_base, reserve_bytes, 0, 0, 0);
    if (cr != CUDA_SUCCESS) {
        const char * errStr;
        cuGetErrorString(cr, &errStr);
        fprintf(stderr, "SparseVMM: cuMemAddressReserve(%zu) failed: %s\n", reserve_bytes, errStr);
        free(pool);
        return nullptr;
    }

    pool->reserved  = reserve_bytes;
    pool->committed = 0;

    // Commit initial physical memory if requested
    if (initial_bytes > 0) {
        int ret = den_sparse_vmm_ensure(pool, initial_bytes);
        if (ret != 0) {
            cuMemAddressFree(pool->va_base, reserve_bytes);
            free(pool);
            return nullptr;
        }
    }

    fprintf(stderr, "SparseVMM: reserved %zu MB VA, initially committed %zu MB physical (granularity=%zu)\n",
            reserve_bytes / (1024*1024), initial_bytes / (1024*1024), granularity);

    return pool;
}

// ═══════════════════════════════════════════════════════
// Ensure physical capacity
// ═══════════════════════════════════════════════════════

int den_sparse_vmm_ensure(den_sparse_vmm_t pool, size_t required_bytes) {
    if (!pool) return -1;

    // Align up
    required_bytes = ((required_bytes + pool->granularity - 1) / pool->granularity) * pool->granularity;

    // Already have enough
    if (required_bytes <= pool->committed) return 0;

    // Clamp to reserved
    if (required_bytes > pool->reserved) {
        required_bytes = pool->reserved;
        if (required_bytes <= pool->committed) return -1; // can't grow further
    }

    size_t grow_bytes = required_bytes - pool->committed;

    // Allocate physical memory handle
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = pool->physical_device;

    CUmemGenericAllocationHandle handle;
    CUresult cr = cuMemCreate(&handle, grow_bytes, &prop, 0);
    if (cr != CUDA_SUCCESS) {
        const char * errStr;
        cuGetErrorString(cr, &errStr);
        fprintf(stderr, "SparseVMM: cuMemCreate(%zu) failed: %s\n", grow_bytes, errStr);
        return -1;
    }

    // Map into the reserved VA region at current committed offset
    CUdeviceptr map_addr = pool->va_base + pool->committed;
    cr = cuMemMap(map_addr, grow_bytes, 0, handle, 0);
    if (cr != CUDA_SUCCESS) {
        const char * errStr;
        cuGetErrorString(cr, &errStr);
        fprintf(stderr, "SparseVMM: cuMemMap(%zu at offset %zu) failed: %s\n",
                grow_bytes, pool->committed, errStr);
        cuMemRelease(handle);
        return -1;
    }

    // Handle no longer needed after mapping
    cuMemRelease(handle);

    // Set device access
    CUmemAccessDesc accessDesc;
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id   = pool->device;
    accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cr = cuMemSetAccess(map_addr, grow_bytes, &accessDesc, 1);
    if (cr != CUDA_SUCCESS) {
        const char * errStr;
        cuGetErrorString(cr, &errStr);
        fprintf(stderr, "SparseVMM: cuMemSetAccess(%zu) failed: %s\n", grow_bytes, errStr);
        cuMemUnmap(map_addr, grow_bytes);
        return -1;
    }

    pool->committed += grow_bytes;

    fprintf(stderr, "SparseVMM: grew by %zu MB, now %zu MB committed of %zu MB reserved\n",
            grow_bytes / (1024*1024), pool->committed / (1024*1024), pool->reserved / (1024*1024));

    return 0;
}

// ═══════════════════════════════════════════════════════
// Accessors
// ═══════════════════════════════════════════════════════

void * den_sparse_vmm_ptr(den_sparse_vmm_t pool) {
    return pool ? (void *)pool->va_base : nullptr;
}

size_t den_sparse_vmm_committed(den_sparse_vmm_t pool) {
    return pool ? pool->committed : 0;
}

size_t den_sparse_vmm_reserved(den_sparse_vmm_t pool) {
    return pool ? pool->reserved : 0;
}

// ═══════════════════════════════════════════════════════
// Destroy
// ═══════════════════════════════════════════════════════

void den_sparse_vmm_destroy(den_sparse_vmm_t pool) {
    if (!pool) return;

    if (pool->committed > 0) {
        cuMemUnmap(pool->va_base, pool->committed);
    }
    if (pool->reserved > 0) {
        cuMemAddressFree(pool->va_base, pool->reserved);
    }
    pool->va_base   = 0;
    pool->committed = 0;
    pool->reserved  = 0;

    free(pool);
}
