// persistent_kernel.cuh — Work-queue persistent kernel infrastructure
// Ported from dengine (den_persistent_kernel.h, den_persistent_work.h)
//
// Single-launch kernel processes a work queue instead of launching per-op.
// Eliminates ~120 kernel launches per token (~1ms driver overhead at 172 tok/s).
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define PK_MAX_ITEMS  1024   // enough for 32 layers x ~20 ops
#define PK_TDR_MS     1800   // 1.8s before Windows TDR fires

// Op codes — subset of GGML ops mapped to work items
enum pk_op_t {
    PK_OP_SHUTDOWN  = 0,
    PK_OP_IDLE      = 1,
    PK_OP_RMS_NORM  = 2,
    PK_OP_GEMV_BF16 = 3,
    PK_OP_GEMV_NVFP4= 4,     // OMMA-accelerated NVFP4 GEMV
    PK_OP_MUL_MAT   = 5,     // general matmul (routes to cuBLAS if not specialized)
    PK_OP_ADD       = 6,
    PK_OP_MUL       = 7,
    PK_OP_SILU      = 8,
    PK_OP_ROPE      = 9,
    PK_OP_ATTN      = 10,
    PK_OP_EMBED     = 11,
    PK_OP_LM_HEAD   = 12,
};

// Work item: describes one operation for the persistent kernel
typedef struct {
    uint32_t op;         // pk_op_t
    uint32_t flags;      // reserved
    uint64_t in_ptr;     // input tensor data pointer
    uint64_t out_ptr;    // output tensor data pointer
    uint64_t weight_ptr; // weight tensor data pointer (for GEMV/MUL_MAT)
    uint32_t N, K;       // dimensions
} pk_work_item_t;

// Work queue: ring buffer with atomic head/tail
typedef struct {
    pk_work_item_t items[PK_MAX_ITEMS];
    uint32_t head;       // device pops (atomic add)
    uint32_t tail;       // host pushes
    uint32_t done;       // device increments per item completed
    uint32_t shutdown;   // host sets to 1 to signal kernel exit
    uint64_t tdr_threshold; // clock64 cycles before TDR checkpoint
} pk_work_queue_t;

// TDR checkpoint for graceful kernel restart
#define PK_TDR_MAGIC  0xDEADBEEF
typedef struct {
    uint32_t magic;
    uint32_t items_completed;
    uint32_t padding[2];
} pk_tdr_checkpoint_t;

// Initialize persistent kernel subsystem.
// Returns cudaSuccess on success.
cudaError_t pk_init(int reserve_sm_count);

// Build work queue from ggml_cgraph and copy to device.
// Returns number of work items enqueued.
int pk_build_queue(void* cgraph, pk_work_queue_t* h_queue);

// Launch the persistent kernel (call once, then update queue per token).
// grid_dim: number of SMs to occupy (0 = auto)
cudaError_t pk_launch(pk_work_queue_t* d_queue, int grid_dim, cudaStream_t stream);

// Update work queue for next token (host pushes new items).
// Must be followed by cudaMemcpy H2D of the queue.
void pk_update_queue(pk_work_queue_t* h_queue);

// Shutdown the persistent kernel gracefully.
cudaError_t pk_shutdown(pk_work_queue_t* d_queue, cudaStream_t stream);

// Check if persistent kernel is active.
bool pk_is_active(void);

#ifdef __cplusplus
}
#endif
