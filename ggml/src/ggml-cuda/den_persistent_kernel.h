// den_persistent_kernel.h — Persistent inference kernel API
// Single-launch work-queue architecture with device-side TDR self-throttling.
// All long-running kernels use %clock64 inline PTX to yield before Windows TDR.
// No host-side polling. No nvidia-smi. No PCIe traffic for timeout checks.

#pragma once
#include <stdint.h>
#include <cuda/std/atomic>

#ifdef __cplusplus
extern "C" {
#endif

// ── TDR Self-Throttling ───────────────────────────────────────────────────
// Read %clock64 (~1.5 GHz on GB203) at the top of each work-item loop.
// After enough cycles or items, checkpoint to global memory and return.
// Host detects the checkpoint, reads resume position, and relaunches.

// Default TDR timeout in milliseconds (1800ms = 1.8s, 200ms safety margin)
#define TDR_SAFE_MS_DEFAULT  1800

// Cycles at 1.5 GHz for the default 1800ms timeout
#define TDR_SAFE_CYCLES_1P5GHZ  2700000000ULL

typedef struct {
    uint32_t magic;                  // 0xDEADBEEF = valid checkpoint
    uint32_t last_token_processed;   // last completed token ID
    uint32_t last_layer_processed;   // last completed layer
    uint32_t items_completed;        // total work items completed
} tdr_checkpoint_t;

#define TDR_CHECKPOINT_MAGIC  0xDEADBEEF

// ── Work Queue ABI (shared between host and device) ───────────────────────

#define PK_WORK_SHUTDOWN   0
#define PK_WORK_IDLE       1
#define PK_WORK_EMBED      2
#define PK_WORK_RMS_NORM   3
#define PK_WORK_GEMV_BF16  4
#define PK_WORK_GEMV_FP4   5
#define PK_WORK_GDN_SSM    6
#define PK_WORK_ATTN       7
#define PK_WORK_MOE_ROUTE  8
#define PK_WORK_MOE_EXPERT 9
#define PK_WORK_LM_HEAD    10
#define PK_WORK_TOKEN_OUT  11
#define PK_WORK_SILU       12
#define PK_WORK_MUL        13
#define PK_WORK_ADD        14
#define PK_WORK_ROPE       15
#define PK_WORK_GEMV_FP4_BATCH 16

#define PK_MAX_WORK_ITEMS 4096
#define PK_MAX_HD     128    // max head dimension for register V array (512B, may spill)
#define PK_MAX_SEQ    2048   // max sequence length (KV cache stride per head)

// Per-layer configuration passed to kernel via constant memory
typedef struct {
    int H, V, L, max_seq;
    int nh, nkv, hd, nr;
    float eps, rope_theta;
    int full_attn_interval;  // 0 = GDN only, N = attention every N layers
    int arch;                // 0=GDN, 1=attn, 2=MoE
    int nvh;                 // GDN SSM heads (default 16)
    int kd;                  // GDN SSM key dimension
    int vd;                  // GDN SSM value dimension
} pk_model_config_t;

typedef struct {
    uint32_t type;               // PK_WORK_*
    uint32_t token_id;
    uint32_t layer;
    uint32_t flags;
    uint32_t expert_ids[8];
    float    expert_weights[8];
    uint64_t in_ptr;
    uint64_t out_ptr;
    uint64_t weight_ptr;
    uint64_t norm_ptr;
    uint32_t N, K;
    float    eps;
} pk_work_item_t;

typedef struct {
    pk_work_item_t items[PK_MAX_WORK_ITEMS];
    cuda::std::atomic<uint32_t> head;      // next item to consume (device advances)
    cuda::std::atomic<uint32_t> tail;      // next slot to write (host advances)
    cuda::std::atomic<uint32_t> done;      // items completed (device increments)
    uint32_t                    shutdown;  // host sets to 1 to terminate
} pk_work_queue_t;

// ── Device kernel declarations (launched from host API) ───────────────────

#ifdef __cplusplus
} // extern "C" — must end before __global__ decls (CUDA forbids C linkage on kernels)
#endif

#ifdef __CUDACC__
// Persistent forward pass: processes work items until queue is drained + shutdown
__global__ void pk_forward_pass(
    pk_work_queue_t* queue,
    const __nv_bfloat16* __restrict__ embedding,
    const __nv_bfloat16* __restrict__ all_weights,
    const int* __restrict__ tensor_offsets,
    const int* __restrict__ tensor_dims,
    const float* __restrict__ norm_weights,
    float* __restrict__ hidden_states,
    float* __restrict__ logits,
    float* __restrict__ scratch,
    float* __restrict__ gdn_state,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    int* __restrict__ d_seq_lens,
    pk_model_config_t cfg,
    int H, int V, int L, int batch_size,
    tdr_checkpoint_t* checkpoint);

__global__ void pk_bf16_gemv(
    pk_work_queue_t* queue,
    const __nv_bfloat16* weights,
    float* activations,
    tdr_checkpoint_t* checkpoint);

#endif // __CUDACC__

#ifdef __cplusplus
extern "C" {  // reopen C linkage for host API functions
#endif

// ── Host API ──────────────────────────────────────────────────────────────

// Initialize the persistent kernel work queue (cudaHostAlloc mapped, zero-fill)
int pk_init(void);

// Get the global work queue pointer (NULL if not initialized)
pk_work_queue_t* pk_get_queue(void);

// Enqueue a single work item (host→device copy, synchronous per-item)
int pk_enqueue(int type, int token, int layer,
               uint64_t in_ptr, uint64_t out_ptr,
               uint64_t weight_ptr, uint64_t norm_ptr,
               int N, int K, float eps);

// Block until at least `expected` items are marked done
int pk_wait_done(int expected);

// Set shutdown flag and free queue memory
void pk_shutdown(void);

// ── TDR-aware launch helpers ──────────────────────────────────────────────

// Read DEN_TDR_SAFE_MS from environment (falls back to TDR_SAFE_MS_DEFAULT)
int tdr_get_safe_ms(void);

// Convert milliseconds to cycles at ~1.5 GHz (GB203 nominal)
static inline unsigned long long tdr_ms_to_cycles(int ms) {
    return (unsigned long long)ms * 1500000ULL;  // 1.5 GHz = 1.5 cycles/ns
}

// Allocate device-side checkpoint buffer (call once, reuse across relaunches)
// Returns pointer on success, NULL on failure.
tdr_checkpoint_t* tdr_checkpoint_alloc(void);

// Free checkpoint buffer
void tdr_checkpoint_free(tdr_checkpoint_t* cp);

// Launch pk_forward_pass with TDR self-throttling.
// Automatically:
//   1. Checks for a valid checkpoint and resumes from it
//   2. Launches the kernel with checkpoint pointer
//   3. After kernel returns, re-launches if checkpoint indicates more work
//   4. Loops until the work queue is fully drained or shutdown is signalled
// Returns 0 on success, -1 on error.
int pk_forward_with_tdr(
    pk_work_queue_t* queue,
    const void* embedding,
    const void* all_weights,
    const int* tensor_offsets,
    const int* tensor_dims,
    const float* norm_weights,
    float* hidden_states,
    float* logits,
    float* scratch,
    float* gdn_state,
    float* k_cache,
    float* v_cache,
    int* d_seq_lens,
    pk_model_config_t cfg,
    tdr_checkpoint_t* checkpoint,
    int H, int V, int L, int batch_size);

// ── Work Queue Builder (host-side, forward pass integration) ──────────────

// Build work items for one complete forward pass of a transformer model.
// Enqueues: Embed -> layers (RMSNorm->GEMVs->SiLU->Mul->Add->
//   MoE/Attention/GDN->) -> LM Head.
// Uses gpu_weight_ptrs[] indexed by layer+sub-slot.
//
// Returns number of items enqueued, or -1 on error.
inline int pk_build_forward_work(
    pk_work_queue_t* queue,
    const void** d_weights,
    const void** d_tiles,
    const uint32_t* tensor_slot,
    const int* tensor_N,
    const int* tensor_K,
    int n_tensors,
    int token_id,
    const void* d_embedding,
    float* d_hidden,
    float* d_scratch,
    float* d_logits,
    float* d_gdn_state,
    float* d_k_cache,
    float* d_v_cache,
    int* d_seq_lens,
    int H, int V, int L,
    float eps, float theta,
    int arch, int fai,
    int seq_pos,
    int nh, int nkv, int hd, int nr,
    int nvh, int kd, int vd);

// Run one token through the persistent kernel: builds work queue,
// launches kernel, waits for completion, reads logits.
// Returns 0 on success, -1 on error.
int pk_forward_token(
    pk_work_queue_t* queue,
    const void** d_weights,
    const void** d_tiles,
    const uint32_t* tensor_slot,
    const int* tensor_N,
    const int* tensor_K,
    int n_tensors,
    int token_id,
    const void* d_embedding,
    float* d_hidden,
    float* d_scratch,
    float* d_logits,
    float* d_gdn_state,
    float* d_k_cache,
    float* d_v_cache,
    int* d_seq_lens,
    int H, int V, int L,
    float eps, float theta,
    int arch, int fai,
    int seq_pos,
    int nh, int nkv, int hd, int nr,
    int nvh, int kd, int vd,
    tdr_checkpoint_t* checkpoint);

#ifdef __cplusplus
}
#endif