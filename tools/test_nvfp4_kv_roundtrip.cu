// Standalone NVFP4 KV quant→dequant roundtrip test
// Compile: nvcc -arch=sm_120a -o test_roundtrip.exe tools/test_nvfp4_kv_roundtrip.cu
// Run: test_roundtrip.exe

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define DEN_NVFP4_KV_TILE_BYTES      160
#define DEN_NVFP4_KV_TILE_SCALES     16
#define DEN_NVFP4_KV_TILE_GROUPS     16
#define DEN_NVFP4_KV_TILE_GROUP_SZ   16
#define DEN_NVFP4_KV_TILE_NORM_OFF   144
#define DEN_NVFP4_KV_TILE_DISPATCH   148
#define DEN_NVFP4_KV_TILE_KSTRIDE    149
#define DEN_NVFP4_KV_META_SW         0x30

__device__ const float kv_e2m1_lut[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

__device__ const float kv_ue4m3_lut[16] = {
    0.0f, 0.0625f, 0.125f, 0.1875f, 0.25f, 0.3125f,
    0.375f, 0.4375f, 1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f
};

// Quantize (copied from fattn-nvfp4-kv.cu)
__device__ void kv_quantize_tile(const float * vec, uint8_t * tile, int head_dim) {
    for (int i = 0; i < DEN_NVFP4_KV_TILE_BYTES; i += 16) {
        if (i + 16 <= DEN_NVFP4_KV_TILE_BYTES) *(uint4*)(tile + i) = make_uint4(0,0,0,0);
    }
    int n_groups = (head_dim + DEN_NVFP4_KV_TILE_GROUP_SZ - 1) / DEN_NVFP4_KV_TILE_GROUP_SZ;
    if (n_groups > DEN_NVFP4_KV_TILE_GROUPS) n_groups = DEN_NVFP4_KV_TILE_GROUPS;

    for (int g = 0; g < n_groups; g++) {
        int blk_start = g * DEN_NVFP4_KV_TILE_GROUP_SZ;
        int blk_end = blk_start + DEN_NVFP4_KV_TILE_GROUP_SZ;
        if (blk_end > head_dim) blk_end = head_dim;
        int n_in_blk = blk_end - blk_start;

        float max_abs = 0.0f;
        for (int e = 0; e < n_in_blk; e++) {
            float av = vec[blk_start + e];
            if (av < 0.0f) av = -av;
            if (av > max_abs) max_abs = av;
        }

        uint8_t scale_code = 0;
        if (max_abs >= 1e-10f) {
            float ideal_scale = max_abs / 6.0f;
            float best_err = fabsf(ideal_scale - kv_ue4m3_lut[1]);
            uint8_t best_code = 1;
            #pragma unroll
            for (int c = 2; c < 16; c++) {
                float err = fabsf(ideal_scale - kv_ue4m3_lut[c]);
                if (err < best_err) { best_err = err; best_code = (uint8_t)c; }
            }
            scale_code = best_code;
        }
        tile[g] = scale_code;
        float scale = kv_ue4m3_lut[scale_code];

        for (int e = 0; e < n_in_blk; e++) {
            float val = vec[blk_start + e];
            float qval = (scale > 1e-10f) ? val / scale : 0.0f;
            if (qval > 6.0f)  qval =  6.0f;
            if (qval < -6.0f) qval = -6.0f;
            uint8_t sgn = (qval < 0.0f) ? 0x08 : 0x00;
            float abs_q = (qval < 0.0f) ? -qval : qval;
            uint8_t mag;
            if      (abs_q >= 5.0f)  mag = 7;
            else if (abs_q >= 3.5f)  mag = 6;
            else if (abs_q >= 2.5f)  mag = 5;
            else if (abs_q >= 1.75f) mag = 4;
            else if (abs_q >= 1.25f) mag = 3;
            else if (abs_q >= 0.75f) mag = 2;
            else if (abs_q >= 0.25f) mag = 1;
            else                      mag = 0;
            uint8_t nibble = sgn | mag;
            int byte_idx = DEN_NVFP4_KV_TILE_SCALES + g * 8 + (e >> 1);
            if (e & 1) tile[byte_idx] = (tile[byte_idx] & 0x0F) | (nibble << 4);
            else       tile[byte_idx] = (tile[byte_idx] & 0xF0) | (nibble & 0x0F);
        }
    }
    float tn = (head_dim > 0) ? (float)sqrt(0.0) : 1.0f; // simplified norm
    for (int i = 0; i < head_dim; i++) tn += vec[i]*vec[i]; tn = sqrtf(tn / head_dim);
    if (tn < 1e-10f) tn = 1.0f;
    *(float*)(tile + DEN_NVFP4_KV_TILE_NORM_OFF) = tn;
    tile[DEN_NVFP4_KV_TILE_DISPATCH] = DEN_NVFP4_KV_META_SW;
    tile[DEN_NVFP4_KV_TILE_KSTRIDE]  = (head_dim + 63) / 64;
}

// Dequantize (copied from fattn-nvfp4-kv.cu)
__device__ void kv_dequantize_tile(const uint8_t * tile, float * vec, int head_dim) {
    int n_groups = (head_dim + DEN_NVFP4_KV_TILE_GROUP_SZ - 1) / DEN_NVFP4_KV_TILE_GROUP_SZ;
    if (n_groups > DEN_NVFP4_KV_TILE_GROUPS) n_groups = DEN_NVFP4_KV_TILE_GROUPS;
    for (int g = 0; g < n_groups; g++) {
        float scale = kv_ue4m3_lut[tile[g] & 0x0F];
        int blk_start = g * DEN_NVFP4_KV_TILE_GROUP_SZ;
        int blk_end = blk_start + DEN_NVFP4_KV_TILE_GROUP_SZ;
        if (blk_end > head_dim) blk_end = head_dim;
        for (int e = 0; e < blk_end - blk_start; e++) {
            int idx = blk_start + e;
            int nb_byte = DEN_NVFP4_KV_TILE_SCALES + g * 8 + (e >> 1);
            uint8_t nb = tile[nb_byte];
            if (e & 1) nb >>= 4; else nb &= 0x0F;
            float val = kv_e2m1_lut[nb & 0x07];
            if (nb & 0x08) val = -val;
            vec[idx] = val * scale;
        }
    }
}

// Test kernel: quantize + dequantize each head and compare
__global__ void test_roundtrip_kernel(const float * d_input, float * d_output, float * d_error, int n_heads, int head_dim) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads) return;
    const float * in = d_input + (size_t)h * head_dim;
    float * out = d_output + (size_t)h * head_dim;
    uint8_t tile[DEN_NVFP4_KV_TILE_BYTES];
    kv_quantize_tile(in, tile, head_dim);
    kv_dequantize_tile(tile, out, head_dim);
    float max_err = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float err = fabsf(in[i] - out[i]);
        if (err > max_err) max_err = err;
    }
    d_error[h] = max_err;
}

int main() {
    const int n_heads = 8;
    const int head_dim = 128;
    const int n_elem = n_heads * head_dim;

    float * h_input = (float*)malloc(n_elem * sizeof(float));
    float * h_output = (float*)malloc(n_elem * sizeof(float));
    float * h_error = (float*)malloc(n_heads * sizeof(float));

    // Random test data
    srand(42);
    for (int i = 0; i < n_elem; i++)
        h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;

    float *d_input, *d_output, *d_error;
    cudaMalloc(&d_input, n_elem * sizeof(float));
    cudaMalloc(&d_output, n_elem * sizeof(float));
    cudaMalloc(&d_error, n_heads * sizeof(float));
    cudaMemcpy(d_input, h_input, n_elem * sizeof(float), cudaMemcpyHostToDevice);

    test_roundtrip_kernel<<<1, n_heads>>>(d_input, d_output, d_error, n_heads, head_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, n_elem * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_error, d_error, n_heads * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute statistics
    float max_err = 0.0f, sum_sq = 0.0f, sum_in = 0.0f, sum_out = 0.0f, dot = 0.0f;
    for (int i = 0; i < n_elem; i++) {
        float err = fabsf(h_input[i] - h_output[i]);
        if (err > max_err) max_err = err;
        sum_sq += err * err;
        sum_in += h_input[i] * h_input[i];
        sum_out += h_output[i] * h_output[i];
        dot += h_input[i] * h_output[i];
    }
    float rmse = sqrtf(sum_sq / n_elem);
    float cos_sim = dot / (sqrtf(sum_in) * sqrtf(sum_out));

    printf("NVFP4 KV Roundtrip Test (%d heads x %d dim)\n", n_heads, head_dim);
    printf("  Max error:   %.6f\n", max_err);
    printf("  RMSE:        %.6f\n", rmse);
    printf("  Cos sim:     %.6f\n", cos_sim);
    printf("  Per-head max errors:");
    for (int h = 0; h < n_heads; h++) printf(" %.4f", h_error[h]);
    printf("\n");

    bool pass = (cos_sim > 0.99f) && (max_err < 2.0f);
    printf("\n  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_error);
    free(h_input); free(h_output); free(h_error);
    return pass ? 0 : 1;
}
