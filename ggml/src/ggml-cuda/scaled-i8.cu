#include "scaled-i8.cuh"

static __global__ void k_get_rows_scaled_i8(const int8_t * values, const half * scales, const int32_t * rows, float * dst, int64_t k, int64_t nr, int64_t nrows) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= k*nr) return;
    const int64_t col = i % k;
    const int64_t out_row = i / k;
    const int32_t row = rows[out_row];
    if (row < 0 || row >= nrows) {
        dst[i] = 0.0f;
        return;
    }
    dst[i] = (float) values[row*k + col] * __half2float(scales[row]);
}

static __global__ void k_mul_mat_scaled_i8(const int8_t * weights, const half * scales, const float * x, float * dst, int64_t k, int64_t m, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= m*n) return;
    const int64_t row = i % m;
    const int64_t col = i / m;
    const float scale = __half2float(scales[row]);
    float sum = 0.0f;
    for (int64_t j = 0; j < k; ++j) sum += (float) weights[row*k + j] * scale * x[col*k + j];
    dst[i] = sum;
}

// Decode-specialized rowwise-I8 GEMV.  The generic kernel above assigns one
// thread to an entire output row, serializing K dot-product terms.  Decode has
// one activation column, so give each row a warp and reduce its partial sums.
// Keep this as an additive n == 1 path; prompt/batched shapes retain the
// established generic implementation.
static __global__ void k_mul_mat_scaled_i8_warp1(
        const int8_t * __restrict__ weights,
        const half   * __restrict__ scales,
        const float  * __restrict__ x,
        float        * __restrict__ dst,
        int64_t k,
        int64_t m) {
    const int lane = threadIdx.x & 31;
    const int64_t warp = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= m) {
        return;
    }

    const int8_t * row = weights + warp*k;
    float sum = 0.0f;

    // Four adjacent elements per lane makes each warp consume 128 contiguous
    // weights and activations per iteration.  A GGML view may legally offset
    // the activation by one F32, though, so vectorize only when both bases
    // meet their actual alignment contracts.  The scalar warp path below
    // preserves valid misaligned views without changing the batched kernel.
    const bool vector_aligned =
            (k & 3) == 0 &&
            (reinterpret_cast<uintptr_t>(row) & (alignof(int) - 1)) == 0 &&
            (reinterpret_cast<uintptr_t>(x) & (alignof(float4) - 1)) == 0;
    if (vector_aligned) {
        for (int64_t j = (int64_t) lane*4; j < k; j += 128) {
            const int packed = *reinterpret_cast<const int *>(row + j);
            const float4 xv = *reinterpret_cast<const float4 *>(x + j);
            sum += (float) (int8_t) ( packed        & 0xff) * xv.x;
            sum += (float) (int8_t) ((packed >>  8) & 0xff) * xv.y;
            sum += (float) (int8_t) ((packed >> 16) & 0xff) * xv.z;
            sum += (float) (int8_t) ((packed >> 24) & 0xff) * xv.w;
        }
    } else {
        // Preserve the general operator contract without relying on aligned
        // vector loads when a row stride is not a multiple of four bytes.
        for (int64_t j = lane; j < k; j += 32) {
            sum += (float) row[j] * x[j];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        dst[warp] = sum * __half2float(scales[warp]);
    }
}

void ggml_cuda_op_get_rows_scaled_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * values = dst->src[0];
    const ggml_tensor * rows = dst->src[1];
    const ggml_tensor * scales = dst->src[2];
    const int64_t nr = ggml_nelements(rows);
    const int64_t k = values->ne[0];
    GGML_ASSERT(k > 0 && nr > 0 && k <= (INT64_MAX - 255)/nr);
    const int64_t total = k*nr;
    GGML_ASSERT((total + 255)/256 <= INT_MAX);
    k_get_rows_scaled_i8<<<(total + 255)/256, 256, 0, ctx.stream()>>>((const int8_t *) values->data, (const half *) scales->data, (const int32_t *) rows->data, (float *) dst->data, k, nr, values->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_mul_mat_scaled_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * scales = dst->src[2];
    const int64_t n = ggml_nelements(x) / x->ne[0];
    const int64_t m = weights->ne[1];
    // No output rows is a valid graph no-op (e.g. a batch without requested
    // logits).  Do not construct a zero-grid launch or reject the model.
    if (n == 0) {
        return;
    }
    const int64_t total = m*n;
    GGML_ASSERT(m > 0 && n > 0 && m <= INT64_MAX/n);
    GGML_ASSERT(total <= INT64_MAX - 255 && (total + 255)/256 <= INT_MAX);

    if (n == 1) {
        constexpr int threads = 256;
        constexpr int warps_per_block = threads/32;
        GGML_ASSERT(m <= (int64_t) INT_MAX*warps_per_block);
        k_mul_mat_scaled_i8_warp1<<<(m + warps_per_block - 1)/warps_per_block, threads, 0, ctx.stream()>>>(
            (const int8_t *) weights->data, (const half *) scales->data,
            (const float *) x->data, (float *) dst->data, weights->ne[0], m);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    k_mul_mat_scaled_i8<<<(total + 255)/256, 256, 0, ctx.stream()>>>((const int8_t *) weights->data, (const half *) scales->data, (const float *) x->data, (float *) dst->data, weights->ne[0], m, n);
    CUDA_CHECK(cudaGetLastError());
}
