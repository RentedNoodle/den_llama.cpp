#include "common.cuh"
#include "escha-linear.cuh"
#include "escha-qwen38-sm120.cuh"
#include "mmid.cuh"

#include <cuda.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>

// Fused decode + routed matmul for Escha ESCHAM experts.
//
//   y = T128(T128(x * rin) @ decode(code)) * rout
//
// Weights decode independently -- weight[p] = lut[sum_j bit(payload, dep[p][j]) << j],
// no trellis state -- so the decode is a plain gather and the whole thing is one big
// embarrassingly parallel reduction.
//
// Split into three kernels because at batch 1 there is very little natural
// parallelism: 8 slots x OC/128 column groups is 32 blocks for gate/up, which leaves
// most of the GPU idle. Slicing the IC reduction across blocks and summing the
// partials afterwards is what fills it. The partials are summed in a fixed order
// rather than with atomics so results stay bit-reproducible run to run.
//
// There are two matmul kernels, picked by batch size:
//
//   escha_matmul_partial  one row per block, reduction sliced. Right when rows are
//                         scarce, but every row decodes the weights again.
//   escha_matmul_tiled    ESCHA_ROWS rows of ONE expert per block, so a decoded
//                         weight is reused across all of them. Needs the rows
//                         grouped by expert first, which is what mm_ids_helper
//                         already does for mul_mat_id.
//
// Both write the same partial layout and share escha_finalize.

#define ESCHA_TILE     16   // decode tile is 16x16
#define ESCHA_NT      128   // threads per block, one per output column
#define ESCHA_GROUPS  (ESCHA_NT/ESCHA_TILE)
#define ESCHA_MAX_W    24   // uint32 words per payload, 48 int16 at K=3
#define ESCHA_TARGET 4096  // block count we try to reach by slicing the reduction
#define ESCHA_ROWS     16   // rows of one expert per block on the batched path

struct escha_cubin_state {
    std::mutex mutex;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction had_in = nullptr;
    CUfunction mma_k2 = nullptr;
    CUfunction mma_k3 = nullptr;
    bool attempted = false;
};

static escha_cubin_state & escha_cubin() {
    static escha_cubin_state state;
    return state;
}

static bool escha_cubin_enabled() {
    const char * path = std::getenv("DEN_ESCHA_CUBIN_PATH");
    return path != nullptr && path[0] != '\0';
}

static bool escha_cubin_init() {
    if (!escha_cubin_enabled()) {
        return false;
    }
    auto & state = escha_cubin();
    std::lock_guard<std::mutex> lock(state.mutex);
    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current == nullptr) {
        return false;
    }
    if (state.module != nullptr && state.context == current) {
        return true;
    }
    if (state.attempted) {
        return false;
    }
    state.attempted = true;
    const char * path = std::getenv("DEN_ESCHA_CUBIN_PATH");
    const char * had_name = "_ZN12escha_escham20escham_had_in_kernelEP6__halfPKS0_S3_PKfi";
    const char * k2_name = "_ZN12escha_escham22escham_mma_gemv_kernelILi2ELi2EEEvPfPK6__halfPKtiiiii";
    const char * k3_name = "_ZN12escha_escham22escham_mma_gemv_kernelILi2ELi3EEEvPfPK6__halfPKtiiiii";
    CUmodule module = nullptr;
    if (cuModuleLoad(&module, path) != CUDA_SUCCESS ||
            cuModuleGetFunction(&state.had_in, module, had_name) != CUDA_SUCCESS ||
            cuModuleGetFunction(&state.mma_k2, module, k2_name) != CUDA_SUCCESS ||
            cuModuleGetFunction(&state.mma_k3, module, k3_name) != CUDA_SUCCESS) {
        GGML_LOG_WARN("ESCHA dense driver module unavailable: %s; using portable CUDA path\n", path);
        if (module != nullptr) {
            cuModuleUnload(module);
        }
        state.had_in = nullptr;
        state.mma_k2 = nullptr;
        state.mma_k3 = nullptr;
        return false;
    }
    state.context = current;
    state.module = module;
    GGML_LOG_INFO("ESCHA dense driver module enabled: %s\n", path);
    return true;
}

static __global__ void escha_f32_to_f16(
        const float * __restrict__ src, half * __restrict__ dst,
        const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

// Escha codebook A, the one this checkpoint uses (its config leaves "codebook" unset,
// which eschamoe.py defaults to cbA / codebook_id 1). It is computed, not stored -- the
// same QTIP-family trick as 3INST but with its own multiplier and no addend. Recovered
// from escham_reconstruct_kernel<1, K> and checked against all 65536 entries.
// The fp16 add must stay in fp16 to match the table bit for bit.
static __device__ __forceinline__ float escha_codebook(uint32_t idx) {
    const uint32_t x = ((idx*0xcbac1fedu) & 0x8fff8fffu) ^ 0x3b603b60u;
    return __half2float(__hadd(__ushort_as_half((unsigned short) x),
                               __ushort_as_half((unsigned short) (x >> 16))));
}

// in-place normalized Sylvester-Hadamard over each block of 128
static __device__ __forceinline__ void escha_hadamard_128(float * v, int n, int tid, int nt) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int idx = tid; idx < (n/128)*64; idx += nt) {
            const int blk = idx / 64;
            const int j   = idx % 64;
            const int i   = (j / len)*(2*len) + (j % len);

            float * b = v + blk*128 + i;
            const float a0 = b[0];
            const float a1 = b[len];

            b[0]   = a0 + a1;
            b[len] = a0 - a1;
        }
        __syncthreads();
    }

    const float scale = rsqrtf(128.0f);
    for (int i = tid; i < n; i += nt) {
        v[i] *= scale;
    }
    __syncthreads();
}

// u[row] = T128(x[row] * rin[expert]) -- hoisted out of the matmul so the column
// blocks do not each redo it
static __global__ void escha_rotate_in(
        const half    * __restrict__ rin,
        const float   * __restrict__ x,
        const int32_t * __restrict__ ids,
        float         * __restrict__ u,
        const int IC, const int n_x, const int n_ids,
        const int64_t nb_x1, const int64_t nb_x2,
        const int64_t nb_i0, const int64_t nb_i1) {
    extern __shared__ float s_u[];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int it  = row / n_ids;
    const int is  = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);

    const half  * rin_e = rin + (int64_t) e*IC;
    const float * x_row = (const float *)((const char *) x + (int64_t)(is % n_x)*nb_x1 + it*nb_x2);

    for (int i = tid; i < IC; i += blockDim.x) {
        s_u[i] = x_row[i]*__half2float(rin_e[i]);
    }
    __syncthreads();

    escha_hadamard_128(s_u, IC, tid, blockDim.x);

    float * dst = u + (int64_t) row*IC;
    for (int i = tid; i < IC; i += blockDim.x) {
        dst[i] = s_u[i];
    }
}

// partial[slice][row][c] = sum over this slice's input tiles of u . decode(code)
template <int K>
static __global__ void escha_matmul_partial(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        const int32_t * __restrict__ ids,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_ids, const int n_rows, const int n_slices,
        const int64_t nb_i0, const int64_t nb_i1) {
    extern __shared__ char s_raw[];

    // dep is stored transposed and packed two entries per word: [b/2][r][cc]. that makes
    // the 16 threads of a group read 16 consecutive words, which is conflict-free -- the
    // natural [p][b] layout puts them 32 bytes apart and costs an 8-way bank conflict
    uint32_t * s_dep = (uint32_t *) s_raw;                 // [8][16][16]
    uint32_t * s_pay = s_dep + 8*256;                      // [ESCHA_GROUPS][ESCHA_MAX_W]

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid   = threadIdx.x;
    const int row   = blockIdx.x;
    const int ocb   = blockIdx.y;
    const int slice = blockIdx.z;

    const int it = row / n_ids;
    const int is = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);
    const int16_t * code_e = code + (int64_t) e*nit*nct*(16*K);

    for (int j = tid; j < 8*256; j += ESCHA_NT) {
        const int b2 = j / 256;
        const int p  = j % 256;
        // the dep data is int16 bytes regardless of the declared type (I32 loads)
        const int16_t * dep_i16 = (const int16_t *) dep;
        s_dep[j] = (uint32_t) (uint16_t) dep_i16[p*16 + 2*b2]
                 | ((uint32_t) (uint16_t) dep_i16[p*16 + 2*b2 + 1] << 16);
    }
    __syncthreads();

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = ocb*ESCHA_GROUPS + grp;

    const int per   = nit/n_slices;
    const int ti0   = slice*per;
    const float * u_row = u + (int64_t) row*IC;

    uint32_t * pay = s_pay + grp*ESCHA_MAX_W;
    float sum = 0.0f;

    for (int ti = ti0; ti < ti0 + per; ++ti) {
        const uint32_t * src = (const uint32_t *)(code_e + (int64_t)(ti*nct + tj)*(16*K));
        for (int w = cc; w < n_wd; w += ESCHA_TILE) {
            pay[w] = src[w];
        }
        __syncwarp();

        const float * uu = u_row + ti*ESCHA_TILE;

        #pragma unroll 4
        for (int r = 0; r < ESCHA_TILE; ++r) {
            const uint32_t * d = s_dep + r*ESCHA_TILE + cc;

            uint32_t idx = 0;
            #pragma unroll
            for (int b2 = 0; b2 < 8; ++b2) {
                const uint32_t dd = d[b2*256];
                const int d0 = dd & 0xffff;
                const int d1 = dd >> 16;

                idx |= ((pay[d0 >> 5] >> (d0 & 31)) & 1u) << (2*b2);
                idx |= ((pay[d1 >> 5] >> (d1 & 31)) & 1u) << (2*b2 + 1);
            }

            sum += uu[r]*escha_codebook(idx);
        }
        __syncwarp();
    }

    partial[((int64_t) slice*n_rows + row)*OC + ocb*ESCHA_NT + tid] = sum;
}

// one work item per block of the tiled kernel: up to ESCHA_ROWS compact rows of expert e.
// order does not matter -- items write disjoint output rows -- so an atomic counter is enough
static __global__ void escha_build_work(
        const int32_t * __restrict__ bounds,
        int4          * __restrict__ work,
        int32_t       * __restrict__ n_work,
        const int n_expert) {
    for (int e = blockIdx.x*blockDim.x + threadIdx.x; e < n_expert; e += gridDim.x*blockDim.x) {
        const int lo = bounds[e];
        const int hi = bounds[e + 1];
        for (int s = lo; s < hi; s += ESCHA_ROWS) {
            work[atomicAdd(n_work, 1)] = make_int4(e, s, min(ESCHA_ROWS, hi - s), 0);
        }
    }
}

// partial[row][c] = u[row] . decode(code), for R rows sharing one expert
template <int K, int R>
static __global__ void escha_matmul_tiled(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        const int32_t * __restrict__ ids_dst,
        const int4    * __restrict__ work,
        const int32_t * __restrict__ n_work,
        float         * __restrict__ partial,
        const int IC, const int OC) {
    extern __shared__ char s_raw[];

    uint32_t * s_dep = (uint32_t *) s_raw;                             // [8][16][16]
    uint32_t * s_pay = s_dep + 8*256;                                  // [ESCHA_GROUPS][ESCHA_MAX_W]
    float    * s_u   = (float *)(s_pay + ESCHA_GROUPS*ESCHA_MAX_W);    // [R][16]

    // the grid is sized to an upper bound, so the tail blocks have nothing to do.
    // uniform across the block, so the syncs below are still safe
    if (blockIdx.x >= (unsigned) *n_work) {
        return;
    }

    const int4 w = work[blockIdx.x];
    const int e     = w.x;
    const int start = w.y;
    const int nrow  = w.z;

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid = threadIdx.x;

    for (int j = tid; j < 8*256; j += ESCHA_NT) {
        const int b2 = j / 256;
        const int p  = j % 256;
        s_dep[j] = (uint32_t) (uint16_t) dep[p*16 + 2*b2]
                 | ((uint32_t) (uint16_t) dep[p*16 + 2*b2 + 1] << 16);
    }

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = blockIdx.y*ESCHA_GROUPS + grp;

    const int16_t * code_e = code + (int64_t) e*nit*nct*(16*K);
    uint32_t * pay = s_pay + grp*ESCHA_MAX_W;

    float acc[R];
#pragma unroll
    for (int m = 0; m < R; ++m) {
        acc[m] = 0.0f;
    }
    __syncthreads();

    for (int ti = 0; ti < nit; ++ti) {
        // the whole block cooperates on one 16-wide slice of u per row, then every
        // thread reads all of it -- 16 threads of a group hit the same address, so
        // the reads broadcast instead of conflicting
        for (int j = tid; j < R*ESCHA_TILE; j += ESCHA_NT) {
            const int m = j / ESCHA_TILE;
            const int r = j % ESCHA_TILE;
            s_u[j] = m < nrow ? u[(int64_t) ids_dst[start + m]*IC + ti*ESCHA_TILE + r] : 0.0f;
        }

        const uint32_t * src = (const uint32_t *)(code_e + (int64_t)(ti*nct + tj)*(16*K));
        for (int wd = cc; wd < n_wd; wd += ESCHA_TILE) {
            pay[wd] = src[wd];
        }
        __syncthreads();

#pragma unroll 4
        for (int r = 0; r < ESCHA_TILE; ++r) {
            const uint32_t * d = s_dep + r*ESCHA_TILE + cc;

            uint32_t idx = 0;
#pragma unroll
            for (int b2 = 0; b2 < 8; ++b2) {
                const uint32_t dd = d[b2*256];
                const int d0 = dd & 0xffff;
                const int d1 = dd >> 16;

                idx |= ((pay[d0 >> 5] >> (d0 & 31)) & 1u) << (2*b2);
                idx |= ((pay[d1 >> 5] >> (d1 & 31)) & 1u) << (2*b2 + 1);
            }

            const float wv = escha_codebook(idx);
#pragma unroll
            for (int m = 0; m < R; ++m) {
                acc[m] += s_u[m*ESCHA_TILE + r]*wv;
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < nrow; ++m) {
        partial[(int64_t) ids_dst[start + m]*OC + blockIdx.y*ESCHA_NT + tid] = acc[m];
    }
}

// sum the slices, rotate the 128-column group, scale by rout
static __global__ void escha_finalize(
        const half    * __restrict__ rout,
        const int32_t * __restrict__ ids,
        const float   * __restrict__ partial,
        float         * __restrict__ dst,
        const int OC, const int n_ids, const int n_rows, const int n_slices,
        const int64_t nb_i0, const int64_t nb_i1,
        const int64_t nb_d1, const int64_t nb_d2) {
    __shared__ float s_acc[ESCHA_NT];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int ocb = blockIdx.y;

    const int it = row / n_ids;
    const int is = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);
    const int c = ocb*ESCHA_NT + tid;

    float sum = 0.0f;
    for (int s = 0; s < n_slices; ++s) {
        sum += partial[((int64_t) s*n_rows + row)*OC + c];
    }
    s_acc[tid] = sum;
    __syncthreads();

    escha_hadamard_128(s_acc, ESCHA_NT, tid, ESCHA_NT);

    float * dst_row = (float *)((char *) dst + is*nb_d1 + it*nb_d2);
    dst_row[c] = s_acc[tid]*__half2float(rout[(int64_t) e*OC + c]);
}

// Dense fold variants share the decode and Hadamard math above but deliberately
// have no ids parameter. Expert zero is part of the tensor contract, not data
// supplied by a graph input.
static __global__ void escha_rotate_dense(
        const half * __restrict__ rin,
        const float * __restrict__ s_in,
        const float * __restrict__ x,
        float * __restrict__ u,
        const int IC, const int64_t nb_x1) {
    __shared__ float s_u[ESCHA_NT];
    const int tid = threadIdx.x;
    const int tile = blockIdx.x;
    const int row = blockIdx.y;
    const int offset = tile*ESCHA_NT;
    const half * rin_e = rin;
    const float * x_row = (const float *) ((const char *) x + (int64_t) row*nb_x1);
    s_u[tid] = x_row[offset + tid]*s_in[offset + tid]*__half2float(rin_e[offset + tid]);
    __syncthreads();
    escha_hadamard_128(s_u, ESCHA_NT, tid, ESCHA_NT);
    u[(int64_t) row*IC + offset + tid] = s_u[tid];
}

static __device__ __forceinline__ int32_t escha_dep_at(
        const void * dep, const bool dep_i32, const int index) {
    return dep_i32 ? ((const int32_t *) dep)[index] : ((const int16_t *) dep)[index];
}

template <int K>
static __global__ void escha_matmul_dense_partial(
        const int16_t * __restrict__ code,
        const void    * __restrict__ dep,
        const bool dep_i32,
        const float   * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
    extern __shared__ char s_raw[];
    uint32_t * s_dep = (uint32_t *) s_raw;
    uint32_t * s_pay = s_dep + 8*256;
    const int nit = IC/ESCHA_TILE;
    const int nct = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int ocb = blockIdx.y;
    const int slice = blockIdx.z;

    for (int j = tid; j < 8*256; j += ESCHA_NT) {
        const int b2 = j/256;
        const int p = j%256;
        s_dep[j] = (uint32_t) (uint16_t) escha_dep_at(dep, dep_i32, p*16 + 2*b2)
                 | ((uint32_t) (uint16_t) escha_dep_at(dep, dep_i32, p*16 + 2*b2 + 1) << 16);
    }
    __syncthreads();

    const int grp = tid/ESCHA_TILE;
    const int cc = tid%ESCHA_TILE;
    const int tj = ocb*ESCHA_GROUPS + grp;
    const int per = nit/n_slices;
    const int ti0 = slice*per;
    const float * u_row = u + (int64_t) row*IC;
    uint32_t * pay = s_pay + grp*ESCHA_MAX_W;
    float sum = 0.0f;

    for (int ti = ti0; ti < ti0 + per; ++ti) {
        const uint32_t * src = (const uint32_t *)(code + (int64_t)(ti*nct + tj)*(16*K));
        for (int w = cc; w < n_wd; w += ESCHA_TILE) {
            pay[w] = src[w];
        }
        __syncwarp();
        const float * uu = u_row + ti*ESCHA_TILE;
        #pragma unroll 4
        for (int r = 0; r < ESCHA_TILE; ++r) {
            const uint32_t * d = s_dep + r*ESCHA_TILE + cc;
            uint32_t idx = 0;
            #pragma unroll
            for (int b2 = 0; b2 < 8; ++b2) {
                const uint32_t dd = d[b2*256];
                idx |= ((pay[(dd & 0xffff) >> 5] >> ((dd & 0xffff) & 31)) & 1u) << (2*b2);
                idx |= ((pay[(dd >> 16) >> 5] >> ((dd >> 16) & 31)) & 1u) << (2*b2 + 1);
            }
            sum += uu[r]*escha_codebook(idx);
        }
        __syncwarp();
    }
    partial[((int64_t) slice*n_rows + row)*OC + ocb*ESCHA_NT + tid] = sum;
}

static __global__ void escha_finalize_dense(
        const half * __restrict__ rout,
        const float * __restrict__ s_out,
        const float * __restrict__ partial,
        float * __restrict__ dst,
        const int OC, const int n_rows, const int n_slices,
        const int64_t nb_d1) {
    __shared__ float s_acc[ESCHA_NT];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int ocb = blockIdx.y;
    const int c = ocb*ESCHA_NT + tid;
    float sum = 0.0f;
    for (int s = 0; s < n_slices; ++s) {
        sum += partial[((int64_t) s*n_rows + row)*OC + c];
    }
    s_acc[tid] = sum;
    __syncthreads();
    escha_hadamard_128(s_acc, ESCHA_NT, tid, ESCHA_NT);
    float * dst_row = (float *) ((char *) dst + (int64_t) row*nb_d1);
    dst_row[c] = s_acc[tid]*__half2float(rout[c])*s_out[c];
}

static bool escha_launch_cubin_dense(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * code,
        const ggml_tensor * rin,
        const ggml_tensor * rout,
        const ggml_tensor * s_in,
        const ggml_tensor * s_out,
        const ggml_tensor * x,
        ggml_tensor * dst,
        const int K, const int IC, const int OC, const int n_rows) {
    // Official dense GEMV contract is decode/microbatch only (M <= 8).
    const bool driver_requested = escha_cubin_enabled();
    static bool route_trace_emitted = false;
    if (!route_trace_emitted) {
        GGML_LOG_INFO("ESCHA dense driver probe: requested=%d K=%d IC=%d OC=%d rows=%d\\n",
                driver_requested, K, IC, OC, n_rows);
        route_trace_emitted = true;
    }
    if (!driver_requested || n_rows > 8) {
        return false;
    }
    int device = -1;
    int major = 0;
    int minor = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    const bool driver_compatible = major == 12 && minor == 0;
    const bool driver_initialized = driver_compatible && escha_cubin_init();
    static bool route_result_emitted = false;
    if (!route_result_emitted) {
        GGML_LOG_INFO("ESCHA dense driver result: cc=%d.%d compatible=%d initialized=%d\\n",
                major, minor, driver_compatible, driver_initialized);
        route_result_emitted = true;
    }
    if (!driver_initialized) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t n_x = (int64_t) n_rows*IC;
    ggml_cuda_pool_alloc<half> x_f16(ctx.pool(), (size_t) n_x);
    ggml_cuda_pool_alloc<half> x_rot(ctx.pool(), (size_t) n_x);
    ggml_cuda_pool_alloc<float> mma_out(ctx.pool(), (size_t) n_rows*OC);

    escha_f32_to_f16<<<(unsigned int) ((n_x + 255)/256), 256, 0, stream>>>(
        (const float *) x->data, x_f16.get(), n_x);
    CUDA_CHECK(cudaGetLastError());

    CUdeviceptr d_rot = (CUdeviceptr) x_rot.get();
    CUdeviceptr d_x = (CUdeviceptr) x_f16.get();
    CUdeviceptr d_rin = (CUdeviceptr) rin->data;
    CUdeviceptr d_s_in = (CUdeviceptr) s_in->data;
    int n = IC;
    void * had_args[] = { &d_rot, &d_x, &d_rin, &d_s_in, &n };
    auto & state = escha_cubin();
    if (cuLaunchKernel(state.had_in,
            (unsigned int) ((IC + 127)/128), (unsigned int) n_rows, 1,
            32, 1, 1, 0, (CUstream) stream, had_args, nullptr) != CUDA_SUCCESS) {
        return false;
    }

    CUdeviceptr d_mma = (CUdeviceptr) mma_out.get();
    CUdeviceptr d_code = (CUdeviceptr) code->data;
    const int packed_groups = (IC + 15)/16;
    const int chunks = std::min(96, std::max(3, ((IC + 63)/64)/2));
    int rows_arg = n_rows;
    int oc_arg = OC;
    int ic_arg = IC;
    int packed_arg = packed_groups;
    int chunks_arg = chunks;
    void * mma_args[] = {
        &d_mma, &d_rot, &d_code,
        &rows_arg, &oc_arg, &ic_arg, &packed_arg, &chunks_arg,
    };
    CUfunction mma = K == 2 ? state.mma_k2 : K == 3 ? state.mma_k3 : nullptr;
    const unsigned int smem = K == 2 ? 2560u : 2816u;
    if (mma == nullptr || cuLaunchKernel(mma,
            (unsigned int) ((OC + 31)/32), (unsigned int) n_rows, 1,
            32, 1, 1, smem, (CUstream) stream, mma_args, nullptr) != CUDA_SUCCESS) {
        return false;
    }

    escha_finalize_dense<<<dim3(n_rows, OC/ESCHA_NT), ESCHA_NT, 0, stream>>>(
        (const half *) rout->data, (const float *) s_out->data,
        mma_out.get(), (float *) dst->data, OC, n_rows, 1, dst->nb[1]);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static bool escha_rotate_smem_ok = false;

void ggml_cuda_op_escha_moe(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin  = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * lut  = dst->src[3];
    const ggml_tensor * dep  = dst->src[4];
    const ggml_tensor * x    = dst->src[5];
    const ggml_tensor * ids  = dst->src[6];

    GGML_ASSERT(code->type == GGML_TYPE_I16 && dep->type == GGML_TYPE_I16);
    GGML_ASSERT(rin->type == GGML_TYPE_F16 && rout->type == GGML_TYPE_F16 && lut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);

    const int K   = code->ne[0]/16;
    const int OC  = code->ne[1]*16;
    const int IC  = code->ne[2]*16;
    const int nit = IC/ESCHA_TILE;

    const int64_t n_expert64 = code->ne[3];
    const int64_t n_ids64    = ids->ne[0];
    const int64_t n_tokens64 = ids->ne[1];
    GGML_ASSERT(n_expert64 > 0 && n_expert64 <= INT_MAX);
    GGML_ASSERT(n_ids64 > 0 && n_ids64 <= INT_MAX && n_tokens64 > 0 && n_tokens64 <= INT_MAX);
    GGML_ASSERT(n_ids64 <= INT64_MAX/n_tokens64 && n_ids64*n_tokens64 <= INT_MAX);
    const int n_expert = (int) n_expert64;
    const int n_ids    = (int) n_ids64;
    const int n_tokens = (int) n_tokens64;
    const int n_rows   = (int) (n_ids64*n_tokens64);
    const int n_ocb    = OC/ESCHA_NT;

    GGML_ASSERT(n_rows > 0 && n_ocb > 0);
    GGML_ASSERT((size_t) n_rows <= SIZE_MAX/(size_t) IC/sizeof(float));
    GGML_ASSERT((size_t) n_rows <= SIZE_MAX/(size_t) OC/sizeof(float));
    GGML_ASSERT(n_rows <= INT_MAX/n_ocb);

    // reuse only pays once a block can expect a decent share of ESCHA_ROWS rows for
    // its expert. below that the sliced kernel wins, and at batch 1 it is the only
    // one that fills the device at all
    const bool tiled = n_rows >= 8*n_expert;

    cudaStream_t stream = ctx.stream();

    int current_device = -1;
    int max_optin_smem = 0;
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, current_device));
    const size_t requested_smem = IC*sizeof(float);
    GGML_ASSERT(requested_smem <= (size_t) max_optin_smem);

    // the dense fold's attention-IC (17408) needs 68KB of dynamic shared memory;
    // the default 48KB cap rejects the launch with "invalid argument". Opt in once.
    if (!escha_rotate_smem_ok) {
        CUDA_CHECK(cudaFuncSetAttribute((const void*) escha_rotate_in,
            cudaFuncAttributeMaxDynamicSharedMemorySize, max_optin_smem));
        escha_rotate_smem_ok = true;
    }

    ggml_cuda_pool_alloc<float> u_buf(ctx.pool(), (size_t) n_rows*IC);

    escha_rotate_in<<<n_rows, 256, IC*sizeof(float), stream>>>(
        (const half *) rin->data, (const float *) x->data, (const int32_t *) ids->data,
        u_buf.get(), IC, (int) x->ne[1], n_ids,
        x->nb[1], x->nb[2], ids->nb[0], ids->nb[1]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaPeekAtLastError());

    const size_t smem_dep = 8*256*sizeof(uint32_t) + ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint32_t);

    if (tiled) {
        // mm_ids_helper assumes a token uses an expert at most once, which top-k routing
        // guarantees. duplicates would desync its offsets and write out of bounds
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));

        ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
        ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
        ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), n_expert + 1);

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            n_expert, n_tokens, n_ids, (int) x->ne[1], (int) (ids->nb[1]/ggml_element_size(ids)), 1,
            /*write_inverse =*/ false, stream);
        CUDA_CHECK(cudaGetLastError());

        // an expert can end with a part-full chunk, so the item count is bounded but not known
        const int n_work_max = (n_rows + ESCHA_ROWS - 1)/ESCHA_ROWS + n_expert;

        ggml_cuda_pool_alloc<int4>    work(ctx.pool(), n_work_max);
        ggml_cuda_pool_alloc<int32_t> n_work(ctx.pool(), 1);

        CUDA_CHECK(cudaMemsetAsync(n_work.get(), 0, sizeof(int32_t), stream));
        escha_build_work<<<1, 256, 0, stream>>>(bounds.get(), work.get(), n_work.get(), n_expert);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaPeekAtLastError());

        ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_rows*OC);

        const size_t smem = smem_dep + ESCHA_ROWS*ESCHA_TILE*sizeof(float);

        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_work_max, n_ocb), ESCHA_NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                u_buf.get(), ids_dst.get(), work.get(), n_work.get(), p_buf.get(), IC, OC);
        };

        switch (K) {
            case 2: launch(escha_matmul_tiled<2, ESCHA_ROWS>); break;
            case 3: launch(escha_matmul_tiled<3, ESCHA_ROWS>); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    CUDA_CHECK(cudaGetLastError());

        escha_finalize<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
            (const half *) rout->data, (const int32_t *) ids->data, p_buf.get(), (float *) dst->data,
            OC, n_ids, n_rows, 1, ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
        return;
    }

    // slice the reduction until the launch is wide enough to fill the device, but only
    // by factors that divide nit evenly
    int n_slices = 1;
    while (n_rows*n_ocb*n_slices*2 <= ESCHA_TARGET && nit % (n_slices*2) == 0) {
        n_slices *= 2;
    }

    ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_slices*n_rows*OC);

    const dim3 grid(n_rows, n_ocb, n_slices);


    auto launch = [&](auto kernel) {
        kernel<<<grid, ESCHA_NT, smem_dep, stream>>>(
            (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
            u_buf.get(), (const int32_t *) ids->data, p_buf.get(),
            IC, OC, n_ids, n_rows, n_slices, ids->nb[0], ids->nb[1]);
    };

    switch (K) {
        case 2: launch(escha_matmul_partial<2>); break;
        case 3: launch(escha_matmul_partial<3>); break;
        default: GGML_ABORT("escha: unsupported K=%d", K);
    }
    CUDA_CHECK(cudaPeekAtLastError());

    escha_finalize<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
        (const half *) rout->data, (const int32_t *) ids->data, p_buf.get(), (float *) dst->data,
        OC, n_ids, n_rows, n_slices, ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaPeekAtLastError());

}

void ggml_cuda_op_escha_linear(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin  = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * s_in = dst->src[3];
    const ggml_tensor * s_out = dst->src[4];
    const ggml_tensor * dep  = dst->src[5];
    const ggml_tensor * x    = dst->src[6];

    GGML_ASSERT(code->type == GGML_TYPE_I16);
    GGML_ASSERT(rin->type == GGML_TYPE_F16 && rout->type == GGML_TYPE_F16);
    GGML_ASSERT(s_in->type == GGML_TYPE_F32 && s_out->type == GGML_TYPE_F32);
    GGML_ASSERT(dep->type == GGML_TYPE_I16 || dep->type == GGML_TYPE_I32);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int K = code->ne[0]/16;
    const int OC = code->ne[1]*16;
    const int IC = code->ne[2]*16;
    const int nit = IC/ESCHA_TILE;
    const int64_t n_rows64 = ggml_nrows(x);
    GGML_ASSERT(n_rows64 > 0 && n_rows64 <= INT_MAX);
    GGML_ASSERT(OC % ESCHA_NT == 0);
    const int n_rows = (int) n_rows64;
    const int n_ocb = OC/ESCHA_NT;
    const bool dep_i32 = dep->type == GGML_TYPE_I32;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(escha_dense_shape_valid(IC, OC));

    // Dedicated Qwen3.8 SM120 route.  It is explicit-env opt-in and requires
    // the model-side Qwen3.8 capability marker; every other ESCHA model
    // continues through the existing cubin probe / portable implementation.
    if (ggml_cuda_op_escha_qwen38_sm120(ctx, dst)) {
        return;
    }

    if (escha_launch_cubin_dense(ctx, code, rin, rout, s_in, s_out, x, dst,
            K, IC, OC, n_rows)) {
        return;
    }

    int current_device = -1;
    int max_optin_smem = 0;
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, current_device));
    if (!escha_rotate_smem_ok) {
        CUDA_CHECK(cudaFuncSetAttribute((const void *) escha_rotate_in,
            cudaFuncAttributeMaxDynamicSharedMemorySize, max_optin_smem));
        escha_rotate_smem_ok = true;
    }

    ggml_cuda_pool_alloc<float> u_buf(ctx.pool(), (size_t) n_rows*IC);
    escha_rotate_dense<<<dim3(IC/ESCHA_NT, n_rows), ESCHA_NT, 0, stream>>>(
        (const half *) rin->data, (const float *) s_in->data,
        (const float *) x->data, u_buf.get(), IC, x->nb[1]);
    CUDA_CHECK(cudaGetLastError());

    // A Qwen3.8 W2 graph marks this operation explicitly. Its batch-1 decode
    // must avoid excessive partial slices; generic ESCHA_LINEAR is untouched.
    const int target_blocks = ggml_get_op_params_i32(dst, 0) == 1 ? 512 : ESCHA_TARGET;
    int n_slices = 1;
    while (n_rows*n_ocb*n_slices*2 <= target_blocks && nit % (n_slices*2) == 0) {
        n_slices *= 2;
    }
    ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_slices*n_rows*OC);
    const size_t smem_dep = 8*256*sizeof(uint32_t) + ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint32_t);
    const dim3 grid(n_rows, n_ocb, n_slices);

    auto launch = [&](auto kernel) {
        kernel<<<grid, ESCHA_NT, smem_dep, stream>>>(
            (const int16_t *) code->data, dep->data, dep_i32, u_buf.get(), p_buf.get(),
            IC, OC, n_rows, n_slices);
    };
    switch (K) {
        case 2: launch(escha_matmul_dense_partial<2>); break;
        case 3: launch(escha_matmul_dense_partial<3>); break;
        default: GGML_ABORT("escha: unsupported K=%d", K);
    }
    CUDA_CHECK(cudaGetLastError());

    escha_finalize_dense<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
        (const half *) rout->data, (const float *) s_out->data,
        p_buf.get(), (float *) dst->data,
        OC, n_rows, n_slices, dst->nb[1]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaPeekAtLastError());
}
