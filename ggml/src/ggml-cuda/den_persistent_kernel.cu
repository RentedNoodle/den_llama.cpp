// den_persistent_kernel.cu — Single-launch persistent inference kernel
// Eliminates all kernel launch overhead (~150 μs × ~120 ops = ~18 ms/token)
// Work-queue architecture: host pushes work items, SMs grab and process
// sm_120a native — uses only confirmed-working instructions
//
// Pioneer: first persistent kernel architecture for MoE inference on consumer Blackwell

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <thread>
#include <chrono>

// Uses pk_work_queue_t, pk_work_item_t, pk_model_config_t, tdr_checkpoint_t, etc.
// from the shared headers:
#include "den_persistent_kernel.h"
// For pk_build_forward_work():
#include "den_persistent_work.h"
#include "den_omma_fp4_device.h"
#include "den_unified_kernel.cuh"

// ── Local defines (not in shared headers) ──────────────────────────────
#define PK_MAX_WORK_ITEMS 4096  // 4K pending items
#define PK_MAX_HD        128    // max head dimension (register V array)
#define PK_MAX_SEQ       2048   // max sequence length (KV cache stride)

// TDR self-throttling constants (matches header)
#define TDR_SAFE_MS_DEFAULT  1800
#define TDR_SAFE_CYCLES_1P5GHZ  2700000000ULL

__device__ __forceinline__ int pk_queue_pop(pk_work_queue_t* q, pk_work_item_t* out) {
    uint32_t h = q->head.fetch_add(1, cuda::std::memory_order_acq_rel);
    if (h >= q->tail.load(cuda::std::memory_order_acquire)) {
        // Queue empty — release the slot
        q->head.fetch_sub(1, cuda::std::memory_order_acq_rel);
        __threadfence_system();  // HIGH: ensure host sees head decrement
        return -1;
    }
    __threadfence();  // MEDIUM: ensure host's write to items[slot] is visible before reading
    *out = q->items[h % PK_MAX_WORK_ITEMS];
    __threadfence_system();  // HIGH: ensure host sees our head increment after reading item
    return 0;
}

__device__ __forceinline__ void pk_work_done(pk_work_queue_t* q) {
    q->done.fetch_add(1, cuda::std::memory_order_release);
    __threadfence_system();  // HIGH: ensure host sees done increment
}

// ── BF16 GEMV Kernel (work-item driven) ─────────────────────────────
__global__ void pk_bf16_gemv(pk_work_queue_t* queue,
                              const __nv_bfloat16* weights,
                              float* activations)
{
    extern __shared__ float smem[];
    float* sx = smem;
    int tid = threadIdx.x;

    while (1) {
        pk_work_item_t w;
        if (pk_queue_pop(queue, &w) != 0) {
            if (queue->shutdown) break;
            __nanosleep(1000); // 1 μs spin
            continue;
        }

        if (w.type == PK_WORK_SHUTDOWN || w.type == PK_WORK_IDLE) {
            pk_work_done(queue);
            if (w.type == PK_WORK_SHUTDOWN) break;
            continue;
        }

        if (w.type == PK_WORK_GEMV_BF16) {
            int N = (int)w.N, K = (int)w.K;
            const float* x = (const float*)(uintptr_t)w.in_ptr;
            float* y = (float*)(uintptr_t)w.out_ptr;
            const __nv_bfloat16* W = (const __nv_bfloat16*)(uintptr_t)w.weight_ptr;

            // Coop load input into SMEM
            for (int i = tid; i < K; i += blockDim.x) sx[i] = x[i];
            __syncthreads();

            // Each thread computes one output row (or part of one)
            for (int row = tid; row < N; row += blockDim.x) {
                const __nv_bfloat16* wr = W + (size_t)row * K;
                float sum = 0;
                for (int k = 0; k < K; k++) sum += __bfloat162float(wr[k]) * sx[k];
                y[row] = sum;
            }
        }

        pk_work_done(queue);
    }
}

// ── BF16 GEMV helper (cooperative, used by multiple op handlers) ────
__device__ static void pk_gemv_bf16_coop(
    const float* x, const __nv_bfloat16* W, float* y,
    int N, int K, float* smem, int tid, int bsize)
{
    for (int i = tid; i < K; i += bsize) smem[i] = x[i];
    __syncthreads();
    for (int row = tid; row < N; row += bsize) {
        const __nv_bfloat16* wr = W + (size_t)row * K;
        float sum = 0;
        for (int k = 0; k < K; k++) sum += __bfloat162float(wr[k]) * smem[k];
        y[row] = sum;
    }
    __syncthreads();
}

// ── GDN SSM recurrence (cooperative, per-head) ──────────────────────
__device__ static void pk_gdn_ssm_step(
    float *q, float *k, float *v,
    float *alpha, float *beta,
    float *z_gate, float *o_ssm,
    float *S, const float *a_log, const float *dt,
    int n_vh, int kd, int vd,
    int tid, int bsize)
{
    for (int h = tid; h < n_vh; h += bsize) {
        float *qh = q + (size_t)h * kd;
        float *kh = k + (size_t)h * kd;
        float *vh = v + (size_t)h * vd;
        float *Sh = S + (size_t)h * kd * vd;
        float dt_arg = alpha[h] + dt[h];
        float sp = dt_arg > 20.0f ? dt_arg :
                   dt_arg < -20.0f ? 0.0f :
                   logf(1.0f + expf(dt_arg));
        float decay = expf(-expf(a_log[h]) * sp);
        if (decay < 1e-6f) decay = 1e-6f;
        if (decay > 1.0f - 1e-6f) decay = 1.0f - 1e-6f;
        for (int i = 0; i < kd * vd; i++) Sh[i] *= decay;
        // S^T @ k
        float Sk[256];
        for (int d = 0; d < vd; d++) {
            float sum = 0.0f;
            for (int i = 0; i < kd; i++) sum += Sh[i * vd + d] * kh[i];
            Sk[d] = sum;
        }
        // err = v - S^T@k, outer product
        float bh = beta[h];
        for (int i = 0; i < kd; i++) {
            float ki_bh = kh[i] * bh;
            for (int j = 0; j < vd; j++) {
                float err_j = vh[j] - Sk[j];
                Sh[i * vd + j] += ki_bh * err_j;
            }
        }
        // Output: o = S^T @ q / sqrt(kd)
        float inv_sqrt_kd = rsqrtf((float)kd);
        float *oh = o_ssm + h * vd;
        for (int d = 0; d < vd; d++) {
            float sum = 0.0f;
            for (int i = 0; i < kd; i++) sum += Sh[i * vd + d] * qh[i];
            oh[d] = sum * inv_sqrt_kd;
        }
        // Apply z-gate: o *= sigmoid(z)
        for (int d = 0; d < vd; d++)
            oh[d] *= 1.0f / (1.0f + expf(-z_gate[h * vd + d]));
    }
}

// ── Persistent Forward Pass Kernel ───────────────────────────────────
// One launch, processes tokens end-to-end through all layers.
// Work items generated by host, consumed by this kernel.
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
    tdr_checkpoint_t* checkpoint)
{
    extern __shared__ float pk_smem[];
    int tid = threadIdx.x;
    int bsize = blockDim.x;
    uint32_t sm_items = 0;  // per-SM work item counter for multi-SM TDR checkpoint
    unsigned long long tdr_start = clock64();

    while (1) {
        pk_work_item_t w;
        if (pk_queue_pop(queue, &w) != 0) {
            if (queue->shutdown) return;
            __nanosleep(1000);
            continue;
        }

        if (w.type == PK_WORK_SHUTDOWN) { sm_items++; pk_work_done(queue); return; }
        if (w.type == PK_WORK_IDLE) { sm_items++; pk_work_done(queue); continue; }

        // ── TDR check: if we've been running too long, checkpoint and return
        // Multi-SM: atomicAdd per-SM item count into shared checkpoint
        if (checkpoint && (clock64() - tdr_start) > TDR_SAFE_CYCLES_1P5GHZ) {
            checkpoint->magic = TDR_CHECKPOINT_MAGIC;
            atomicAdd(&checkpoint->items_completed, sm_items);
            __threadfence_system();
            return;
        }

        int token = (int)w.token_id;
        float* h = hidden_states + (size_t)token * H;

        switch (w.type) {
        case PK_WORK_EMBED: {
            const __nv_bfloat16* emb_row = embedding + (size_t)token * H;
            for (int i = tid; i < H; i += bsize)
                h[i] = __bfloat162float(emb_row[i]);
            break;
        }
        case PK_WORK_RMS_NORM: {
            const float* nw = (const float*)(uintptr_t)w.norm_ptr;
            float eps = w.eps;
            int n = (int)w.N;
            for (int i = tid; i < n; i += bsize) pk_smem[i] = h[i];
            __syncthreads();
            if (tid == 0) {
                double ss = 0;
                for (int i = 0; i < n; i++) {
                    float v = pk_smem[i];
                    ss += (double)v * v;
                }
                pk_smem[n] = 1.0f / sqrtf((float)(ss / n) + eps);
            }
            __syncthreads();
            float inv_rms = pk_smem[n];
            for (int i = tid; i < n; i += bsize)
                h[i] = pk_smem[i] * inv_rms * (1.0f + nw[i]);
            break;
        }
        case PK_WORK_GEMV_BF16: {
            int N = (int)w.N, K = (int)w.K;
            const float* x = (const float*)(uintptr_t)w.in_ptr;
            float* y = (float*)(uintptr_t)w.out_ptr;
            const __nv_bfloat16* W = (const __nv_bfloat16*)(uintptr_t)w.weight_ptr;
            pk_gemv_bf16_coop(x, W, y, N, K, pk_smem, tid, bsize);
            break;
        }
        case PK_WORK_GEMV_FP4:
        case PK_WORK_GEMV_FP4_BATCH: {
            // ═══ Warp-specialized OMMA 4X GEMV on NVFP4 NULLGLASS tiles ═══
            // Warps 0-1: producer — cp.async load weight tiles into SMEM
            // Warps 2-5: OMMA compute — den_omma_fp4_k64, register double-buffered
            // Warp 6: epilogue — tile_norm, SiLU, residual add
            // Warp 7: coordinator — advance tile, TDR check, barrier signal
            int N = (int)w.N, K = (int)w.K;
            const float* x = (const float*)(uintptr_t)w.in_ptr;
            float* y = (float*)(uintptr_t)w.out_ptr;
            const uint8_t* tiles = (const uint8_t*)(uintptr_t)w.weight_ptr;
            if (!tiles) break;

            int warp_id = tid >> 5;
            int lane_id = tid & 31;
            int sm_id = blockIdx.x;
            int sm_count = gridDim.x;

            int n_tiles_N = (N + 15) / 16;
            int n_tiles_K = (K + 15) / 16;

            // Shared ring buffer: 2 tile slots × 160B
            __shared__ struct {
                uint32_t data[DEN_TILE_BYTES / 4];
                int ready;
                int consumed;
            } tile_buf[2];

            if (warp_id <= 1) {
                // ═══ PRODUCER (warps 0-1): load NVFP4 tiles global→SMEM ═══
                // ═══ NON-COHERENT: __ldg() bypasses L1, loads through
                // ═══ read-only cache. Weight tiles streaming through L1
                // ═══ would evict activation data (consumer warp uses L1).
                // ═══ Producer uses .nc (non-coherent) path; consumer keeps
                // ═══ normal .ca (cache-all) loads for activations.
                // Strided: each SM processes every sm_count-th N-tile row
                int bank = 0;
                for (int kt = 0; kt < n_tiles_K; kt++) {
                    for (int nt = sm_id; nt < n_tiles_N; nt += sm_count) {
                        while (tile_buf[bank].consumed == 0) { __nanosleep(100); }
                        tile_buf[bank].consumed = 0;

                        const uint8_t* src = tiles + ((size_t)nt * n_tiles_K + kt) * DEN_TILE_BYTES;
                        const uint32_t* __restrict__ src32 = (const uint32_t*)src;
                        // __ldg() = ld.global.nc.u32 — read-only cache, bypass L1
                        for (int i = tid; i < DEN_TILE_BYTES / 4; i += 64)
                            tile_buf[bank].data[i] = __ldg(&src32[i]);

                        __syncwarp();
                        if (lane_id == 0) tile_buf[bank].ready = 1;
                        bank ^= 1;
                    }
                }
            } else if (warp_id >= 2 && warp_id <= 5) {
                // ═══ COMPUTE (warps 2-5): OMMA 4X, register double-buffered ═══
                // Each warp handles 1/4 of N-tile rows, each thread 4 output rows
                int rows_per_warp = (n_tiles_N + 3) / 4;
                int row_start = (warp_id - 2) * rows_per_warp;
                int row_end = row_start + rows_per_warp;
                if (row_end > n_tiles_N) row_end = n_tiles_N;

                // Register double-buffer: ping-pong bank_a ↔ bank_b (16 floats each)
                float bank_a[16], bank_b[16];
                #pragma unroll
                for (int i = 0; i < 16; i++) { bank_a[i] = 0.0f; bank_b[i] = 0.0f; }

                int bank_idx = 0;
                for (int kt = 0; kt < n_tiles_K; kt++) {
                    for (int nt_idx = row_start; nt_idx < row_end; nt_idx++) {
                        int actual_nt = sm_id + nt_idx * sm_count;
                        if (actual_nt >= n_tiles_N) continue;

                        // Wait for producer
                        while (tile_buf[bank_idx].ready == 0) { __nanosleep(100); }
                        tile_buf[bank_idx].ready = 0;

                        const uint32_t* tdata = tile_buf[bank_idx].data;

                        // NULLGLASS tile layout:
                        //   Bytes 0-127: 256 E2M1 nibbles (128B = 32 uint32)
                        //   Bytes 128-143: 16 UE4M3 scales
                        //   Bytes 144-147: float32 tile_norm
                        //   Bytes 148-159: format/metadata
                        uint32_t sfa  = tdata[DEN_TILE_NORM_OFF / 4 - 4];  // UE4M3 scales start at byte 128
                        float tnorm = __uint_as_float(tdata[DEN_TILE_NORM_OFF / 4]);

                        // Quantize activation slice x[kt*16 .. kt*16+15] → E2M1
                        uint32_t sfb = 0x38383838u;  // UE4M3 identity (all 1.0)
                        int B_regs[2] = {0, 0};
                        int K_slice = kt * 16;
                        if (K_slice + 16 <= K) {
                            uint8_t* b_bytes = (uint8_t*)B_regs;
                            for (int i = 0; i < 16; i++) {
                                float v = x[K_slice + i];
                                float av = (v < 0.0f) ? -v : v;
                                // E2M1 codebook: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
                                int e2m1 = (av >= 6.0f) ? 7 : (av >= 4.0f) ? 6 :
                                           (av >= 3.0f) ? 5 : (av >= 2.0f) ? 4 :
                                           (av >= 1.5f) ? 3 : (av >= 1.0f) ? 2 :
                                           (av >= 0.5f) ? 1 : 0;
                                if (v < 0.0f) e2m1 |= 8;
                                int byte_idx = i / 2;
                                if ((i & 1) == 0)
                                    b_bytes[byte_idx] = (uint8_t)e2m1;
                                else
                                    b_bytes[byte_idx] |= (uint8_t)(e2m1 << 4);
                            }
                        }

                        // A registers: E2M1 nibbles from tile bytes 0-127
                        int A_regs[4];
                        for (int i = 0; i < 4; i++) A_regs[i] = (int)tdata[i];

                        // OMMA: accumulate into double-buffer
                        float* cur = (bank_idx & 1) ? bank_b : bank_a;
                        float* nxt = (bank_idx & 1) ? bank_a : bank_b;
                        float D[4];
                        for (int r = 0; r < 4; r++) D[r] = cur[r];

                        den_omma_fp4_k64(D, A_regs, B_regs, sfa, sfb);

                        for (int r = 0; r < 4; r++) nxt[r] = D[r] * tnorm;

                        // Signal producer
                        if (lane_id == 0) tile_buf[bank_idx].consumed = 1;
                        bank_idx ^= 1;
                    }
                }

                // Write final accumulator to output y (stride-aware)
                float* final_bank = (bank_idx & 1) ? bank_b : bank_a;
                for (int nt_idx = row_start; nt_idx < row_end; nt_idx++) {
                    int actual_nt = sm_id + nt_idx * sm_count;
                    if (actual_nt >= n_tiles_N) continue;
                    int out_base = actual_nt * 16;
                    for (int r = 0; r < 4 && (out_base + r) < N; r++) {
                        // Accumulate across warps: write to SMEM scratch, reduce in epilogue
                        pk_smem[tid * 4 + r] = final_bank[r];
                    }
                }
            } else if (warp_id == 6) {
                // ═══ EPILOGUE (warp 6): inter-warp reduction + SiLU + residual ═══
                __syncthreads();  // wait for all compute warps to finish SMEM writes
                // Reduce across compute warps: each output row sum of per-warp contributions
                for (int i = lane_id; i < N; i += 32) {
                    float acc = pk_smem[i];  // warp 2's contribution at thread position
                    // Warps 3-5 contributions are at pk_smem + 128/256/384
                    if (N <= 128) {
                        acc += pk_smem[i + 128];
                        acc += pk_smem[i + 256];
                        acc += pk_smem[i + 384];
                    }
                    // SiLU activation: v * sigmoid(v)
                    float sg = 1.0f / (1.0f + expf(-acc));
                    y[i] = acc * sg;
                }
            } else {
                // ═══ COORDINATOR (warp 7): tile advance + TDR check ═══
                if (lane_id == 0) {
                    if (checkpoint && (clock64() - tdr_start) > TDR_SAFE_CYCLES_1P5GHZ) {
                        checkpoint->magic = TDR_CHECKPOINT_MAGIC;
                        atomicAdd(&checkpoint->items_completed, sm_items);
                        __threadfence_system();
                    }
                }
            }

            break;
        }
        case PK_WORK_ADD: {
            int n = (int)w.N;
            float* dst = (float*)(uintptr_t)w.out_ptr;
            const float* src = (const float*)(uintptr_t)w.in_ptr;
            for (int i = tid; i < n; i += bsize) dst[i] += src[i];
            break;
        }
        case PK_WORK_SILU: {
            int n = (int)w.N;
            float* g = (float*)(uintptr_t)w.in_ptr;
            const float* u = (const float*)(uintptr_t)w.out_ptr;
            for (int i = tid; i < n; i += bsize) {
                float sg = 1.0f / (1.0f + expf(-g[i]));
                g[i] = g[i] * sg * u[i];
            }
            break;
        }
        case PK_WORK_MUL: {
            int n = (int)w.N;
            float* a = (float*)(uintptr_t)w.in_ptr;
            const float* b = (const float*)(uintptr_t)w.out_ptr;
            for (int i = tid; i < n; i += bsize) a[i] *= b[i];
            break;
        }
        case PK_WORK_ROPE: {
            int nh = (int)w.N, hd = (int)w.K;
            float theta = w.eps;
            unsigned flags = w.flags;
            int nkv = (int)(flags & 0xFFFF);
            int seq_pos = (int)((flags >> 16) & 0xFFFF);
            int nr = (int)cfg.nr;
            if (nr <= 0) nr = hd;
            float* q_base = (float*)(uintptr_t)w.in_ptr;
            float* k_base = (float*)(uintptr_t)w.weight_ptr;
            for (int h = tid; h < nh; h += bsize) {
                float* qh = q_base + (size_t)h * hd;
                for (int d = 0; d < nr; d += 2) {
                    float freq = (float)seq_pos * powf(theta, -2.0f * (float)d / (float)nr);
                    float cos_f = cosf(freq);
                    float sin_f = sinf(freq);
                    float q0 = qh[d], q1 = qh[d+1];
                    qh[d]   = q0 * cos_f - q1 * sin_f;
                    qh[d+1] = q0 * sin_f + q1 * cos_f;
                }
            }
            if (k_base) {
                for (int h = tid; h < nkv; h += bsize) {
                    float* kh = k_base + (size_t)h * hd;
                    for (int d = 0; d < nr; d += 2) {
                        float freq = (float)seq_pos * powf(theta, -2.0f * (float)d / (float)nr);
                        float cos_f = cosf(freq);
                        float sin_f = sinf(freq);
                        float k0 = kh[d], k1 = kh[d+1];
                        kh[d]   = k0 * cos_f - k1 * sin_f;
                        kh[d+1] = k0 * sin_f + k1 * cos_f;
                    }
                }
            }
            break;
        }
        case PK_WORK_ATTN: {
            // Online softmax attention — single KV cache traversal
            float* q_base  = (float*)(uintptr_t)w.in_ptr;
            float* out     = (float*)(uintptr_t)w.out_ptr;
            float* kc      = (float*)(uintptr_t)w.weight_ptr;
            float* vc      = (float*)(uintptr_t)w.norm_ptr;
            int    nh      = (int)w.N;
            int    hd      = (int)w.K;
            float  ascale  = w.eps;
            unsigned flags = w.flags;
            int    nkv     = (int)(flags & 0xFFFF);
            int    seq_len = (int)((flags >> 16) & 0xFFFF);
            int    q_per_kv = nh / nkv;

            for (int h = tid; h < nh; h += bsize) {
                int hkv = h / q_per_kv;
                float* qh = q_base + (size_t)h * hd;
                float reg_V[PK_MAX_HD];
                float  max_score = -1e30f;
                double exp_sum   = 0.0;
                float* k_base = kc + (size_t)hkv * PK_MAX_SEQ * hd;
                float* v_base = vc + (size_t)hkv * PK_MAX_SEQ * hd;
                for (int t = 0; t < seq_len; t++) {
                    float score = 0.0f;
                    float* k_row = k_base + (size_t)t * hd;
                    for (int d = 0; d < hd; d++)
                        score += qh[d] * k_row[d];
                    score *= ascale;
                    float old_max  = max_score;
                    if (score > max_score) max_score = score;
                    float rescale = expf(old_max - max_score);
                    float new_exp = expf(score - max_score);
                    exp_sum = exp_sum * rescale + new_exp;
                    float* v_row = v_base + (size_t)t * hd;
                    for (int d = 0; d < hd; d++)
                        reg_V[d] = reg_V[d] * rescale + new_exp * v_row[d];
                }
                float inv_sum = (float)(1.0 / exp_sum);
                float* head_out = out + (size_t)h * hd;
                for (int d = 0; d < hd; d++)
                    head_out[d] = reg_V[d] * inv_sum;
            }
            break;
        }
        case PK_WORK_GDN_SSM: {
            // GDN SSM: self-contained — carries ALL weight pointers in work item
            // in_ptr: normed input (float[H])
            // weight_ptr: W_qkv (BF16[qkv_N, H])
            // norm_ptr: scratch buffer base (must be at least H + qkv_N + 3*nvh + nvh*vd)
            // N = nvh, K = kd
            // flags[0:15] = vd
            // expert_weights[0..1]: state_ptr (uint64_t)
            // expert_weights[2..3]: W_a_ptr
            // expert_weights[4..5]: W_b_ptr
            // expert_weights[6..7]: W_z_ptr
            // expert_ids[0..1]: a_log_ptr (may be 0)
            // expert_ids[2..3]: dt_ptr (may be 0)
            int    nvh    = (int)w.N;
            int    kd     = (int)w.K;
            int    vd     = (int)(w.flags & 0xFFFF);
            int    qkv_N  = 2 * nvh * kd + nvh * vd;

            float* normed = (float*)(uintptr_t)w.in_ptr;
            const __nv_bfloat16* w_qkv = (const __nv_bfloat16*)(uintptr_t)w.weight_ptr;
            float* q_buf  = (float*)(uintptr_t)w.norm_ptr; // scratch base

            // Decode packed pointers
            uint64_t s_ptr, wa_ptr, wb_ptr, wz_ptr, al_ptr, dt_ptr;
            memcpy(&s_ptr, &w.expert_weights[0], sizeof(uint64_t));
            memcpy(&wa_ptr, &w.expert_weights[2], sizeof(uint64_t));
            memcpy(&wb_ptr, &w.expert_weights[4], sizeof(uint64_t));
            memcpy(&wz_ptr, &w.expert_weights[6], sizeof(uint64_t));
            memcpy(&al_ptr, &w.expert_ids[0], sizeof(uint64_t));
            memcpy(&dt_ptr, &w.expert_ids[2], sizeof(uint64_t));

            float* state   = (float*)(uintptr_t)s_ptr;
            const __nv_bfloat16* w_a  = (const __nv_bfloat16*)(uintptr_t)wa_ptr;
            const __nv_bfloat16* w_b  = (const __nv_bfloat16*)(uintptr_t)wb_ptr;
            const __nv_bfloat16* w_z  = (const __nv_bfloat16*)(uintptr_t)wz_ptr;
            const float* a_log_v = (const float*)(uintptr_t)al_ptr;
            const float* dt_v    = (const float*)(uintptr_t)dt_ptr;

            // Step 1: GEMV QKV = W_qkv * normed → q_buf
            pk_gemv_bf16_coop(normed, w_qkv, q_buf, qkv_N, H, pk_smem, tid, bsize);

            // Split Q/K/V from q_buf
            float* q = q_buf;
            float* k = q_buf + (size_t)nvh * kd;
            float* v = q_buf + (size_t)2 * nvh * kd;

            // Step 2: Gate projections — GEMV into scratch after QKV
            float* alpha  = q_buf + qkv_N;
            float* beta   = alpha + nvh;
            float* z_gate = beta + nvh;
            if (w_a) pk_gemv_bf16_coop(normed, w_a, alpha, nvh, H, pk_smem, tid, bsize);
            if (w_b) pk_gemv_bf16_coop(normed, w_b, beta, nvh, H, pk_smem, tid, bsize);
            if (w_z) pk_gemv_bf16_coop(normed, w_z, z_gate, nvh * vd, H, pk_smem, tid, bsize);

            // sigmoid(beta)
            for (int i = tid; i < nvh; i += bsize)
                beta[i] = 1.0f / (1.0f + expf(-beta[i]));
            __syncthreads();

            // Step 3: Per-head L2 norm on Q and K
            for (int h = tid; h < nvh; h += bsize) {
                float* qh = q + (size_t)h * kd;
                float* kh = k + (size_t)h * kd;
                double ss_q = 0, ss_k = 0;
                for (int d = 0; d < kd; d++) {
                    ss_q += (double)qh[d] * qh[d];
                    ss_k += (double)kh[d] * kh[d];
                }
                float inv_q = rsqrtf((float)ss_q + cfg.eps);
                float inv_k = rsqrtf((float)ss_k + cfg.eps);
                for (int d = 0; d < kd; d++) { qh[d] *= inv_q; kh[d] *= inv_k; }
            }
            __syncthreads();

            // Step 4: SSM recurrence (with default a_log=0, dt=0 if pointers NULL)
            if (state) {
                pk_gdn_ssm_step(q, k, v, alpha, beta, z_gate,
                                q_buf, state,
                                a_log_v ? a_log_v : &cfg.eps,
                                dt_v ? dt_v : &cfg.eps,
                                nvh, kd, vd, tid, bsize);
            }
            __syncthreads();

            // Step 5: Output norm per head
            for (int h = tid; h < nvh; h += bsize) {
                float* oh = q_buf + (size_t)h * vd;
                double ss = 0.0;
                for (int d = 0; d < vd; d++)
                    ss += (double)oh[d] * oh[d];
                float inv = rsqrtf((float)(ss / vd) + cfg.eps);
                for (int d = 0; d < vd; d++) oh[d] *= inv;
            }
            __syncthreads();

            // Step 6: Apply z-gate: o *= sigmoid(z)
            for (int i = tid; i < nvh * vd; i += bsize) {
                float z_sig = 1.0f / (1.0f + expf(-z_gate[i]));
                q_buf[i] *= z_sig;
            }
            __syncthreads();
            break;
        }
        case PK_WORK_MOE_ROUTE: {
            // MoE router: GEMV(router_w, normed_h) -> logits (expert scores)
            int n_experts = (int)w.N;
            int Kdim = (int)w.K;
            const float* x = (const float*)(uintptr_t)w.in_ptr;
            float* scores = (float*)(uintptr_t)w.out_ptr;
            const __nv_bfloat16* W = (const __nv_bfloat16*)(uintptr_t)w.weight_ptr;
            pk_gemv_bf16_coop(x, W, scores, n_experts, Kdim, pk_smem, tid, bsize);
            break;
        }
        case PK_WORK_MOE_EXPERT: {
            // MoE expert FFN: gate+up projection → SiLU(gate)*up → down projection
            // in_ptr: hidden input
            // out_ptr: output
            // weight_ptr: gate_up weight (2*ffn_size, H)
            // norm_ptr: down weight (H, ffn_size)
            // N = ffn_size, K = H
            // expert_ids[0]: selected expert index
            int ffn = (int)w.N;
            int Hdim = (int)w.K;
            const float* x = (const float*)(uintptr_t)w.in_ptr;
            float* y = (float*)(uintptr_t)w.out_ptr;
            const __nv_bfloat16* w_gate_up = (const __nv_bfloat16*)(uintptr_t)w.weight_ptr;
            const __nv_bfloat16* w_down = (const __nv_bfloat16*)(uintptr_t)w.norm_ptr;
            int expert_idx = (int)w.expert_ids[0];

            // Step 1: Gate+up projection (expert-specific slice)
            const __nv_bfloat16* w_gu_exp = w_gate_up + (size_t)expert_idx * 2 * ffn * Hdim;
            // Use scratch area for gate+up output: pk_smem may not be large enough
            float* gate_up = scratch; // first 2*ffn floats
            pk_gemv_bf16_coop(x, w_gu_exp, gate_up, 2 * ffn, Hdim, pk_smem, tid, bsize);

            // Step 2: SiLU(gate) * up
            float* gate = gate_up;
            float* up = gate_up + ffn;
            for (int i = tid; i < ffn; i += bsize) {
                float sg = 1.0f / (1.0f + expf(-gate[i]));
                gate[i] = gate[i] * sg * up[i];
            }
            __syncthreads();

            // Step 3: Down projection
            const __nv_bfloat16* w_dn_exp = w_down + (size_t)expert_idx * Hdim * ffn;
            pk_gemv_bf16_coop(gate_up, w_dn_exp, y, Hdim, ffn, pk_smem, tid, bsize);
            break;
        }
        case PK_WORK_LM_HEAD: {
            int N = (int)w.N, K = (int)w.K;
            const float* x = (const float*)(uintptr_t)w.in_ptr;
            float* y = (float*)(uintptr_t)w.out_ptr;
            const __nv_bfloat16* W = (const __nv_bfloat16*)(uintptr_t)w.weight_ptr;
            pk_gemv_bf16_coop(x, W, y, N, K, pk_smem, tid, bsize);
            break;
        }
        case PK_WORK_TOKEN_OUT: {
            // Token output: argmax over logits, write token ID
            int n = (int)w.N;
            float* scores = (float*)(uintptr_t)w.in_ptr;
            int* out_token = (int*)(uintptr_t)w.out_ptr;
            if (tid == 0) {
                int best = 0; float bv = scores[0];
                for (int i = 1; i < n; i++)
                    if (scores[i] > bv) { bv = scores[i]; best = i; }
                *out_token = best;
            }
            break;
        }
        default:
            break;
        }

        sm_items++;
        pk_work_done(queue);
    }
}

// ── Host API ─────────────────────────────────────────────────────────

static pk_work_queue_t* g_pk_queue_host = NULL;  // Host pointer (mapped memory)
static pk_work_queue_t* g_pk_queue_dev = NULL;   // Device pointer (mapped memory)
static cudaStream_t g_pk_stream = NULL;           // Completion stream
static cudaEvent_t g_pk_event = NULL;             // Completion event
static int g_pk_initialized = 0;

int pk_init(void) {
    if (g_pk_initialized) return 0;
    cudaError_t err;

    err = cudaHostAlloc(&g_pk_queue_host, sizeof(pk_work_queue_t), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "persistent_kernel: cudaHostAlloc queue failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    err = cudaHostGetDevicePointer(&g_pk_queue_dev, g_pk_queue_host, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "persistent_kernel: cudaHostGetDevicePointer failed: %s\n",
                cudaGetErrorString(err));
        cudaFreeHost(g_pk_queue_host);
        g_pk_queue_host = NULL;
        return -1;
    }

    memset(g_pk_queue_host, 0, sizeof(pk_work_queue_t));

    err = cudaStreamCreate(&g_pk_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "persistent_kernel: cudaStreamCreate failed: %s\n",
                cudaGetErrorString(err));
        cudaFreeHost(g_pk_queue_host);
        g_pk_queue_host = NULL;
        g_pk_queue_dev = NULL;
        return -1;
    }

    err = cudaEventCreate(&g_pk_event);
    if (err != cudaSuccess) {
        fprintf(stderr, "persistent_kernel: cudaEventCreate failed: %s\n",
                cudaGetErrorString(err));
        cudaStreamDestroy(g_pk_stream);
        g_pk_stream = NULL;
        cudaFreeHost(g_pk_queue_host);
        g_pk_queue_host = NULL;
        g_pk_queue_dev = NULL;
        return -1;
    }

    g_pk_initialized = 1;
    fprintf(stderr, "persistent_kernel: work queue initialized (%d slots, mapped memory)\n",
            PK_MAX_WORK_ITEMS);
    return 0;
}

pk_work_queue_t* pk_get_queue(void) {
    return g_pk_queue_host;
}

int pk_enqueue(int type, int token, int layer, uint64_t in_ptr,
               uint64_t out_ptr, uint64_t weight_ptr, uint64_t norm_ptr,
               int N, int K, float eps) {
    if (!g_pk_queue_host) return -1;

    uint32_t t = g_pk_queue_host->tail.load(cuda::std::memory_order_acquire);
    if (t >= PK_MAX_WORK_ITEMS) return -2;

    pk_work_item_t w = {0};
    w.type = (uint32_t)type;
    w.token_id = (uint32_t)token;
    w.layer = (uint32_t)layer;
    w.in_ptr = in_ptr;
    w.out_ptr = out_ptr;
    w.weight_ptr = weight_ptr;
    w.norm_ptr = norm_ptr;
    w.N = (uint32_t)N;
    w.K = (uint32_t)K;
    w.eps = eps;

    g_pk_queue_host->items[t] = w;
    g_pk_queue_host->tail.store(t + 1, cuda::std::memory_order_release);
    return 0;
}

int pk_wait_done(int expected) {
    if (!g_pk_queue_host) return -1;

    cudaEventRecord(g_pk_event, g_pk_stream);
    cudaEventSynchronize(g_pk_event);

    while (g_pk_queue_host->done.load(cuda::std::memory_order_acquire) < (uint32_t)expected) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return 0;
}

void pk_shutdown(void) {
    if (!g_pk_queue_host) return;

    g_pk_queue_host->shutdown = 1;

    cudaEventRecord(g_pk_event, g_pk_stream);
    cudaEventSynchronize(g_pk_event);

    cudaEventDestroy(g_pk_event);
    g_pk_event = NULL;
    cudaStreamDestroy(g_pk_stream);
    g_pk_stream = NULL;
    cudaFreeHost(g_pk_queue_host);
    g_pk_queue_host = NULL;
    g_pk_queue_dev = NULL;
    g_pk_initialized = 0;
    fprintf(stderr, "persistent_kernel: shutdown\n");
}

// ════════════════════════════════════════════════════════════════════════
// TDR-aware launch helpers
// ════════════════════════════════════════════════════════════════════════

int tdr_get_safe_ms(void) {
    const char* env = getenv("DEN_TDR_SAFE_MS");
    if (env) {
        int val = atoi(env);
        if (val > 0 && val < 10000) return val;
    }
    return TDR_SAFE_MS_DEFAULT;
}

tdr_checkpoint_t* tdr_checkpoint_alloc(void) {
    tdr_checkpoint_t* cp = NULL;
    cudaError_t err = cudaMalloc(&cp, sizeof(tdr_checkpoint_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "tdr_checkpoint_alloc: cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        return NULL;
    }
    cudaMemset(cp, 0, sizeof(tdr_checkpoint_t));
    return cp;
}

void tdr_checkpoint_free(tdr_checkpoint_t* cp) {
    if (cp) cudaFree(cp);
}

// ── pk_forward_token: build queue + launch + wait ────────────────────
int pk_forward_token(
    pk_work_queue_t* queue,
    const void** d_weights, const void** d_tiles,
    const uint32_t* tensor_slot, const int* tensor_N, const int* tensor_K,
    int n_tensors,
    int token_id,
    const void* d_embedding,
    float* d_hidden, float* d_scratch, float* d_logits,
    float* d_gdn_state, float* d_k_cache, float* d_v_cache,
    int* d_seq_lens,
    int H, int V, int L,
    float eps, float theta,
    int arch, int fai,
    int seq_pos,
    int nh, int nkv, int hd, int nr,
    int nvh, int kd, int vd,
    tdr_checkpoint_t* checkpoint)
{
    if (!queue || !g_pk_queue_host) return -1;

    // Reset queue for this token
    queue->head.store(0, cuda::std::memory_order_release);
    queue->tail.store(0, cuda::std::memory_order_release);
    queue->done.store(0, cuda::std::memory_order_release);

    // Build work items for this token's forward pass
    int ret = pk_build_forward_work(
        queue, d_weights, d_tiles, tensor_slot, tensor_N, tensor_K, n_tensors,
        token_id, d_embedding, d_hidden, d_scratch, d_logits,
        d_gdn_state, d_k_cache, d_v_cache, d_seq_lens,
        H, V, L, eps, theta, arch, fai, seq_pos,
        nh, nkv, hd, nr, nvh, kd, vd);
    if (ret != 0) {
        fprintf(stderr, "pk_forward_token: pk_build_forward_work failed\n");
        return -1;
    }

    // Read total item count after build
    int total_items = (int)queue->tail.load(cuda::std::memory_order_acquire);
    if (total_items == 0) return 0; // nothing to do

    // Build model config
    pk_model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.H = H; cfg.V = V; cfg.L = L; cfg.max_seq = PK_MAX_SEQ;
    cfg.nh = nh; cfg.nkv = nkv; cfg.hd = hd; cfg.nr = nr;
    cfg.eps = eps; cfg.rope_theta = theta;
    cfg.full_attn_interval = fai;
    cfg.arch = arch;
    cfg.nvh = nvh; cfg.kd = kd; cfg.vd = vd;

    // SMEM: large enough for H + norm scratch + max dim
    size_t smem_bytes = sizeof(float) * ((size_t)H + PK_MAX_HD + 64);
    if (smem_bytes > 98304) smem_bytes = 98304; // 96KB cap (GB203 SMEM max)

    // Multi-SM: query SM count for grid-stride launch
    int num_sms = 1;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    if (num_sms < 1) num_sms = 1;

    // Launch persistent forward pass kernel
    pk_forward_pass<<<num_sms, 256, smem_bytes>>>(
        g_pk_queue_dev,
        (const __nv_bfloat16*)d_embedding,
        NULL, // all_weights (unused — work items carry direct pointers)
        NULL, // tensor_offsets (unused)
        NULL, // tensor_dims (unused)
        NULL, // norm_weights (unused)
        d_hidden, d_logits, d_scratch,
        d_gdn_state, d_k_cache, d_v_cache, d_seq_lens,
        cfg, H, V, L, 1,
        checkpoint);

    // Wait for completion
    pk_wait_done(total_items);

    return 0;
}

// ── pk_forward_with_tdr: TDR-aware launch with checkpoint resume ────
int pk_forward_with_tdr(
    pk_work_queue_t* queue,
    const void* embedding, const void* all_weights,
    const int* tensor_offsets, const int* tensor_dims,
    const float* norm_weights,
    float* hidden_states, float* logits,
    float* scratch, float* gdn_state,
    float* k_cache, float* v_cache,
    int* d_seq_lens,
    pk_model_config_t cfg,
    tdr_checkpoint_t* checkpoint,
    int H, int V, int L, int batch_size)
{
    if (!queue || !g_pk_queue_host) return -1;

    int total_items = (int)queue->tail.load(cuda::std::memory_order_acquire);
    if (total_items == 0) return 0;

    // Launch loop with checkpoint resume
    int launch_count = 0;
    while (1) {
        // Check if there's a valid checkpoint to resume from
        uint32_t resume_from = 0;
        if (checkpoint) {
            tdr_checkpoint_t cp;
            cudaMemcpy(&cp, checkpoint, sizeof(tdr_checkpoint_t), cudaMemcpyDeviceToHost);
            if (cp.magic == TDR_CHECKPOINT_MAGIC && cp.items_completed > 0) {
                resume_from = cp.items_completed;
                // Advance head past completed items
                queue->head.store(resume_from, cuda::std::memory_order_release);
            }
            // Clear checkpoint for next launch
            cudaMemset(checkpoint, 0, sizeof(tdr_checkpoint_t));
        }

        size_t smem_bytes = sizeof(float) * ((size_t)H + PK_MAX_HD + 64);
        if (smem_bytes > 98304) smem_bytes = 98304;
        int items_remaining = total_items - (int)resume_from;
        if (items_remaining <= 0) break;

        // Multi-SM: query SM count for grid-stride launch
        int num_sms = 1;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        if (num_sms < 1) num_sms = 1;

        pk_forward_pass<<<num_sms, 256, smem_bytes>>>(
            g_pk_queue_dev,
            (const __nv_bfloat16*)embedding,
            (const __nv_bfloat16*)all_weights,
            tensor_offsets, tensor_dims, norm_weights,
            hidden_states, logits, scratch,
            gdn_state, k_cache, v_cache, d_seq_lens,
            cfg, H, V, L, batch_size,
            checkpoint);

        cudaDeviceSynchronize();
        launch_count++;

        // Check if more items remain (non-TDR exit: all items done)
        if (checkpoint) {
            tdr_checkpoint_t cp;
            cudaMemcpy(&cp, checkpoint, sizeof(tdr_checkpoint_t), cudaMemcpyDeviceToHost);
            if (cp.magic != TDR_CHECKPOINT_MAGIC) break; // normal completion
            if ((int)cp.items_completed >= total_items) break;
        } else {
            break; // no checkpoint = one-shot launch
        }
    }

    return 0;
}
