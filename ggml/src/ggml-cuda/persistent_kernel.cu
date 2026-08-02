// persistent_kernel.cu — Single-launch work-queue kernel with TDR throttling
// Ported from dengine den_persistent_kernel.cu
#include "persistent_kernel.cuh"
#include <cstdio>

static bool            g_pk_active     = false;
static pk_work_queue_t g_h_queue;       // host-side work queue
static int             g_pk_reserve_sm = 0;

// ── Persistent Kernel ────────────────────────────────────────────────────────
// Launched once, spins on work queue, processes items until shutdown.
// Single warp per block (32 threads) — enough for GEMV, RMSNorm, elementwise ops.
__global__ void persistent_decode_kernel(pk_work_queue_t* __restrict__ queue) {
    uint64_t t_start = clock64();

    while (1) {
        // Pop work item
        uint32_t idx = atomicAdd(&queue->head, 1u);
        if (idx >= queue->tail) {
            // Queue empty — check for shutdown
            if (queue->shutdown) break;
            __threadfence_block();
            continue;
        }

        pk_work_item_t item = queue->items[idx % PK_MAX_ITEMS];

        // Dispatch by op type
        switch (item.op) {
            case PK_OP_SHUTDOWN:
                queue->shutdown = 1;
                atomicAdd(&queue->done, 1u);
                return;

            case PK_OP_IDLE:
                atomicAdd(&queue->done, 1u);
                break;

            case PK_OP_RMS_NORM: {
                // Inline RMSNorm: out = in * rsqrt(mean(in^2) + eps)
                float* in  = (float*)(uintptr_t)item.in_ptr;
                float* out = (float*)(uintptr_t)item.out_ptr;
                int N = item.N;
                // Single-warp cooperative RMSNorm
                int lane = threadIdx.x;
                float sum_sq = 0;
                for (int i = lane; i < N; i += 32) {
                    float v = in[i];
                    sum_sq += v * v;
                }
                // Warp reduction
                for (int off = 16; off > 0; off /= 2)
                    sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
                float inv = rsqrtf(sum_sq / (float)N + 1e-6f);
                for (int i = lane; i < N; i += 32)
                    out[i] = in[i] * inv;
                atomicAdd(&queue->done, 1u);
                break;
            }

            default:
                atomicAdd(&queue->done, 1u);
                break;
        }

        // TDR self-throttle: checkpoint every ~1.8s
        if (clock64() - t_start > queue->tdr_threshold) {
            __threadfence_system();
            return;  // Host will relaunch from checkpoint
        }
    }
}

// ── Host API ─────────────────────────────────────────────────────────────────

cudaError_t pk_init(int reserve_sm_count) {
    if (g_pk_active) return cudaSuccess;
    memset(&g_h_queue, 0, sizeof(g_h_queue));
    g_h_queue.tdr_threshold = 2700000000ULL;  // 1.8s at ~1.5 GHz
    g_pk_reserve_sm = reserve_sm_count > 0 ? reserve_sm_count : 10;
    g_pk_active = true;
    return cudaSuccess;
}

int pk_build_queue(void* cgraph, pk_work_queue_t* h_queue) {
    // Stub: builds work items from ggml_cgraph
    // Full implementation maps GGML ops to PK ops
    (void)cgraph;
    h_queue->tail = 0;  // No items yet — filled by caller
    h_queue->head = 0;
    h_queue->done = 0;
    h_queue->shutdown = 0;
    return 0;
}

cudaError_t pk_launch(pk_work_queue_t* d_queue, int grid_dim, cudaStream_t stream) {
    if (!g_pk_active) return cudaErrorNotReady;
    if (grid_dim <= 0) grid_dim = g_pk_reserve_sm;

    // Launch N blocks, each with 1 warp (32 threads)
    persistent_decode_kernel<<<grid_dim, 32, 0, stream>>>(d_queue);
    return cudaGetLastError();
}

void pk_update_queue(pk_work_queue_t* h_queue) {
    // Reset queue for next token
    h_queue->head = 0;
    h_queue->done = 0;
    h_queue->tail = 0;
}

cudaError_t pk_shutdown(pk_work_queue_t* d_queue, cudaStream_t stream) {
    if (!g_pk_active) return cudaSuccess;

    // Push shutdown item
    uint32_t tail = g_h_queue.tail;
    g_h_queue.items[tail % PK_MAX_ITEMS].op = PK_OP_SHUTDOWN;
    g_h_queue.tail = tail + 1;

    cudaMemcpyAsync(d_queue, &g_h_queue, sizeof(pk_work_queue_t),
                    cudaMemcpyHostToDevice, stream);

    cudaError_t err = cudaStreamSynchronize(stream);
    g_pk_active = false;
    return err;
}

bool pk_is_active(void) {
    return g_pk_active;
}
