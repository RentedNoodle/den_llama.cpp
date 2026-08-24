// Additive, source-built Qwen3.8 dense ESCHA route for SM120.
// Correctness-first: consumes the existing ESCHA_LINEAR GGUF code/dependency
// and sidecar contract, without foreign cubins, inline SASS, new GGML ops, or
// any modification of the generic ESCHA fallback.

#include "common.cuh"
#include "escha-qwen38-sm120.cuh"

#include <climits>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int TILE = 16;
constexpr int THREADS = 128;
constexpr int GROUPS = THREADS/TILE;
constexpr int MAX_W = 24;
constexpr int DEP_WORDS = 8*256;
constexpr int SPLIT_TARGET_BLOCKS = 256;

static __device__ __forceinline__ float codebook(uint32_t idx) {
    const uint32_t x = ((idx*0xcbac1fedu) & 0x8fff8fffu) ^ 0x3b603b60u;
    return __half2float(__hadd(__ushort_as_half((unsigned short) x),
                               __ushort_as_half((unsigned short) (x >> 16))));
}

static __device__ __forceinline__ void hadamard128(float * v, int tid) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int j = tid; j < 64; j += THREADS) {
            const int i = (j/len)*(2*len) + j%len;
            float * b = v + i;
            const float a = b[0];
            const float z = b[len];
            b[0] = a + z;
            b[len] = a - z;
        }
        __syncthreads();
    }
    for (int i = tid; i < 128; i += THREADS) v[i] *= rsqrtf(128.0f);
    __syncthreads();
}

static __device__ __forceinline__ int dep_at(const void * dep, bool i32, int i) {
    return i32 ? ((const int32_t *) dep)[i] : ((const int16_t *) dep)[i];
}

static __global__ void rotate_dense(
        const half * rin, const float * s_in, const float * x, float * u,
        int IC, int64_t nb_x1) {
    __shared__ float tile[THREADS];
    const int tid = threadIdx.x;
    const int off = blockIdx.x*THREADS + tid;
    const int row = blockIdx.y;
    const float * xrow = (const float *) ((const char *) x + row*nb_x1);
    tile[tid] = xrow[off]*s_in[off]*__half2float(rin[off]);
    __syncthreads();
    hadamard128(tile, tid);
    u[(int64_t) row*IC + off] = tile[tid];
}

template<int K>
static __global__ void dense_gemv(
        const int16_t * code, const void * dep, bool dep_i32, const float * u,
        const half * rout, const float * s_out, float * dst,
        int IC, int OC, int64_t nb_d1) {
    extern __shared__ char raw[];
    uint32_t * sdep = (uint32_t *) raw;
    uint32_t * spay = sdep + DEP_WORDS;
    float * sacc = (float *) (spay + GROUPS*MAX_W);
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int ocb = blockIdx.y;
    const int grp = tid/TILE;
    const int cc = tid%TILE;
    const int nit = IC/TILE;
    const int nct = OC/TILE;
    const int tj = ocb*GROUPS + grp;
    const int nwd = 16*K/2;

    for (int j = tid; j < DEP_WORDS; j += THREADS) {
        const int b2 = j/256;
        const int p = j%256;
        const int d0 = dep_at(dep, dep_i32, p*16 + 2*b2);
        const int d1 = dep_at(dep, dep_i32, p*16 + 2*b2 + 1);
        sdep[j] = (uint32_t)(uint16_t)d0 | ((uint32_t)(uint16_t)d1 << 16);
    }
    __syncthreads();

    const float * urow = u + (int64_t) row*IC;
    const int16_t * code_tile = code + (int64_t)tj*(16*K);
    uint32_t * pay = spay + grp*MAX_W;
    float sum = 0.0f;
    for (int ti = 0; ti < nit; ++ti) {
        const uint32_t * src = (const uint32_t *) (
                code_tile + (int64_t)ti*nct*(16*K));
        for (int w = cc; w < nwd; w += TILE) pay[w] = src[w];
        __syncwarp();
        const float * uu = urow + ti*TILE;
#pragma unroll 4
        for (int r = 0; r < TILE; ++r) {
            const uint32_t * d = sdep + r*TILE + cc;
            uint32_t idx = 0;
#pragma unroll
            for (int b2 = 0; b2 < 8; ++b2) {
                const uint32_t dd = d[b2*256];
                const int d0 = dd & 0xffff;
                const int d1 = dd >> 16;
                idx |= ((pay[d0 >> 5] >> (d0 & 31)) & 1u) << (2*b2);
                idx |= ((pay[d1 >> 5] >> (d1 & 31)) & 1u) << (2*b2 + 1);
            }
            sum += uu[r]*codebook(idx);
        }
        __syncwarp();
    }
    sacc[tid] = sum;
    __syncthreads();
    hadamard128(sacc, tid);
    const int c = ocb*THREADS + tid;
    float * drow = (float *) ((char *) dst + (int64_t)row*nb_d1);
    drow[c] = sacc[tid]*__half2float(rout[c])*s_out[c];
}

// Split-K partial: decode exactly the same code/dep tiles as dense_gemv, but
// consume one complete, tile-aligned input interval and leave the reduction
// and output transform to dense_gemv_final.  The z dimension is deliberately
// the first layout index: [slice][row][OC].
template<int K>
static __global__ void dense_gemv_partial(
        const int16_t * code, const void * dep, bool dep_i32, const float * u,
        float * partial, int IC, int OC, int rows, int slices) {
    extern __shared__ char raw[];
    uint32_t * sdep = (uint32_t *) raw;
    uint32_t * spay = sdep + DEP_WORDS;
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int ocb = blockIdx.y;
    const int slice = blockIdx.z;
    const int grp = tid/TILE;
    const int cc = tid%TILE;
    const int nit = IC/TILE;
    const int nct = OC/TILE;
    const int tj = ocb*GROUPS + grp;
    const int nwd = 16*K/2;

    for (int j = tid; j < DEP_WORDS; j += THREADS) {
        const int b2 = j/256;
        const int p = j%256;
        const int d0 = dep_at(dep, dep_i32, p*16 + 2*b2);
        const int d1 = dep_at(dep, dep_i32, p*16 + 2*b2 + 1);
        sdep[j] = (uint32_t)(uint16_t)d0 | ((uint32_t)(uint16_t)d1 << 16);
    }
    __syncthreads();

    // Boundaries are in input-tile units, so no 16-wide tile is split.
    const int ti0 = (slice*nit)/slices;
    const int ti1 = ((slice + 1)*nit)/slices;
    const float * urow = u + (int64_t) row*IC;
    const int16_t * code_tile = code + (int64_t)tj*(16*K);
    uint32_t * pay = spay + grp*MAX_W;
    float sum = 0.0f;
    for (int ti = ti0; ti < ti1; ++ti) {
        const uint32_t * src = (const uint32_t *) (
                code_tile + (int64_t)ti*nct*(16*K));
        for (int w = cc; w < nwd; w += TILE) pay[w] = src[w];
        __syncwarp();
        const float * uu = urow + ti*TILE;
#pragma unroll 4
        for (int r = 0; r < TILE; ++r) {
            const uint32_t * d = sdep + r*TILE + cc;
            uint32_t idx = 0;
#pragma unroll
            for (int b2 = 0; b2 < 8; ++b2) {
                const uint32_t dd = d[b2*256];
                const int d0 = dd & 0xffff;
                const int d1 = dd >> 16;
                idx |= ((pay[d0 >> 5] >> (d0 & 31)) & 1u) << (2*b2);
                idx |= ((pay[d1 >> 5] >> (d1 & 31)) & 1u) << (2*b2 + 1);
            }
            sum += uu[r]*codebook(idx);
        }
        __syncwarp();
    }
    const int c = ocb*THREADS + tid;
    partial[((int64_t)slice*rows + row)*OC + c] = sum;
}

// Final split-K stage.  Slices are accumulated in increasing z order to keep
// the split path's floating-point order explicit, then the established
// Hadamard/rout/s_out transform is applied once to the complete dot product.
static __global__ void dense_gemv_final(
        const float * partial, const half * rout, const float * s_out,
        float * dst, int OC, int rows, int slices, int64_t nb_d1) {
    extern __shared__ float sacc[];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int ocb = blockIdx.y;
    const int c = ocb*THREADS + tid;
    float sum = 0.0f;
    for (int slice = 0; slice < slices; ++slice) {
        sum += partial[((int64_t)slice*rows + row)*OC + c];
    }
    sacc[tid] = sum;
    __syncthreads();
    hadamard128(sacc, tid);
    float * drow = (float *) ((char *) dst + (int64_t)row*nb_d1);
    drow[c] = sacc[tid]*__half2float(rout[c])*s_out[c];
}

static bool enabled() {
    const char * v = std::getenv("DEN_ESCHA_QWEN38_SM120");
    return v && std::strcmp(v, "1") == 0;
}

static bool sm120() {
    int dev = -1, major = 0, minor = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(
            &minor, cudaDevAttrComputeCapabilityMinor, dev));
    return major == 12 && minor == 0;
}
} // namespace

bool ggml_cuda_op_escha_qwen38_sm120(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!enabled() || ggml_get_op_params_i32(dst, 0) != 1 || !sm120()) {
        return false;
    }
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * s_in = dst->src[3];
    const ggml_tensor * s_out = dst->src[4];
    const ggml_tensor * dep = dst->src[5];
    const ggml_tensor * x = dst->src[6];
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int K = code->ne[0]/16;
    const int IC = code->ne[2]*16;
    const int OC = code->ne[1]*16;
    const int64_t rows64 = ggml_nrows(x);
    if ((K != 2 && K != 3) || rows64 <= 0 || rows64 > INT_MAX ||
            IC <= 0 || OC <= 0 || IC%128 != 0 || OC%128 != 0) {
        return false;
    }
    GGML_ASSERT(code->type == GGML_TYPE_I16);
    GGML_ASSERT(rin->type == GGML_TYPE_F16 && rout->type == GGML_TYPE_F16);
    GGML_ASSERT(s_in->type == GGML_TYPE_F32 && s_out->type == GGML_TYPE_F32);
    GGML_ASSERT(dep->type == GGML_TYPE_I16 || dep->type == GGML_TYPE_I32);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dep->ne[0] == 16 && dep->ne[1] == 256);

    const int rows = (int) rows64;
    const int ocb = OC/THREADS;
    const bool dep_i32 = dep->type == GGML_TYPE_I32;
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> u(ctx.pool(), (size_t)rows*IC);
    rotate_dense<<<dim3(IC/THREADS, rows), THREADS, 0, stream>>>(
            (const half *)rin->data, (const float *)s_in->data,
            (const float *)x->data, u.get(), IC, x->nb[1]);
    CUDA_CHECK(cudaGetLastError());
    const size_t smem = DEP_WORDS*sizeof(uint32_t) +
            GROUPS*MAX_W*sizeof(uint32_t) + THREADS*sizeof(float);
    const dim3 grid(rows, ocb);
    const int64_t natural_blocks = (int64_t)rows*ocb;
    if (natural_blocks >= SPLIT_TARGET_BLOCKS) {
        // Preserve the audited direct geometry when it already supplies the
        // target number of resident blocks; this is the unchanged path.
        if (K == 2) {
            dense_gemv<2><<<grid, THREADS, smem, stream>>>(
                    (const int16_t *)code->data, dep->data, dep_i32, u.get(),
                    (const half *)rout->data, (const float *)s_out->data,
                    (float *)dst->data, IC, OC, dst->nb[1]);
        } else {
            dense_gemv<3><<<grid, THREADS, smem, stream>>>(
                    (const int16_t *)code->data, dep->data, dep_i32, u.get(),
                    (const half *)rout->data, (const float *)s_out->data,
                    (float *)dst->data, IC, OC, dst->nb[1]);
        }
    } else {
        const int nit = IC/TILE;
        int slices = 1;
        // Double slices until the 256-block target is met.  Integer tile
        // boundaries (rather than element boundaries) preserve the complete
        // 16-wide decoding unit even when nit is not divisible by slices.
        while ((int64_t)slices*natural_blocks < SPLIT_TARGET_BLOCKS &&
                slices < nit) {
            slices *= 2;
        }
        ggml_cuda_pool_alloc<float> partial(
                ctx.pool(), (size_t)slices*rows*OC);
        const size_t smem_partial = DEP_WORDS*sizeof(uint32_t) +
                GROUPS*MAX_W*sizeof(uint32_t);
        const dim3 partial_grid(rows, ocb, slices);
        if (K == 2) {
            dense_gemv_partial<2><<<partial_grid, THREADS, smem_partial, stream>>>(
                    (const int16_t *)code->data, dep->data, dep_i32, u.get(),
                    partial.get(), IC, OC, rows, slices);
        } else {
            dense_gemv_partial<3><<<partial_grid, THREADS, smem_partial, stream>>>(
                    (const int16_t *)code->data, dep->data, dep_i32, u.get(),
                    partial.get(), IC, OC, rows, slices);
        }
        CUDA_CHECK(cudaGetLastError());
        const dim3 final_grid(rows, ocb);
        dense_gemv_final<<<final_grid, THREADS, THREADS*sizeof(float), stream>>>(
                partial.get(), (const half *)rout->data,
                (const float *)s_out->data, (float *)dst->data,
                OC, rows, slices, dst->nb[1]);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
