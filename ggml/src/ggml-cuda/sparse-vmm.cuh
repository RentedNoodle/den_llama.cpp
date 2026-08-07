// sparse-vmm.cuh — Sparse Virtual Memory Manager for KV cache
//
// Reserves a large contiguous virtual address region via cuMemAddressReserve
// but only commits physical pages (cuMemCreate + cuMemMap) as needed.
// Enables 600K+ context on 16 GB by decoupling VA footprint from physical commit.
//
// Hardware: sm_120a consumer GB203 — no demand paging, explicit commit required.
// Application must call ensure_capacity() before accessing unmapped pages.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle
typedef struct den_sparse_vmm_pool * den_sparse_vmm_t;

// Create a sparse VMM pool.
//   reserve_bytes: virtual address space to reserve (e.g., 16 GB for KV)
//   initial_bytes: physical memory to commit immediately (e.g., 2 GB for 32K ctx)
//   Returns NULL on failure.
den_sparse_vmm_t den_sparse_vmm_create(size_t reserve_bytes, size_t initial_bytes);

// Ensure at least `required_bytes` of physical memory is committed.
// Grows the pool by committing additional pages as needed.
// Returns 0 on success, -1 on OOM.
int  den_sparse_vmm_ensure(den_sparse_vmm_t pool, size_t required_bytes);

// Get base pointer of the virtual address region.
void * den_sparse_vmm_ptr(den_sparse_vmm_t pool);

// Get currently committed physical size.
size_t den_sparse_vmm_committed(den_sparse_vmm_t pool);

// Get total reserved virtual size.
size_t den_sparse_vmm_reserved(den_sparse_vmm_t pool);

// Destroy pool, release all physical memory, free virtual address space.
void den_sparse_vmm_destroy(den_sparse_vmm_t pool);

#ifdef __cplusplus
}
#endif
