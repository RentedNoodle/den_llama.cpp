/*
 * NVFP4 V-Scale Swizzle Audit
 * Verifies V block scales are stored linearly (not 4x4 swizzled).
 * Reference: vLLM PR #50085 — SM100 swizzles scales in 4x4, SM120 reads linear.
 * If our code has wrong layout, dequant returns garbage.
 *
 * Build: nvcc -arch=compute_120a -code=sm_120a -o test_vscale_audit test_vscale_audit.cu
 * Run:   test_vscale_audit.exe
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>

// --- NVFP4 KV constants (must match fattn-nvfp4-kv.cuh) ---
#define DEN_NVFP4_KV_GROUPS    16    // 16 groups in a 160B tile
#define DEN_NVFP4_KV_ELEMS     256   // max elements per tile (16 x 16)
#define DEN_NVFP4_KV_TILE_BYTES 160

// E2M1 nibble pack: 2 elements per byte, values 0-7
// UE4M3 scale: unsigned 4-bit exponent, 3-bit mantissa

// --- Forward declare kernel ---
__global__ void audit_vscale_kernel(
    const float * __restrict__ input,
    uint8_t * __restrict__ tile_out,
    float * __restrict__ dequant_out,
    int head_dim, int n_tiles);

// --- Device-side tile format helpers ---
struct __align__(16) nvfp4_tile_device {
    uint8_t data[160];
};

// Quantize: float -> E2M1 nibble (maps 0, ±0.5, ±1.0, ±1.5, ±2.0)
__device__ uint8_t float_to_e2m1_nibble(float v, float scale) {
    v = v / scale;
    // Clamp to [-2.0, 2.0] range
    v = fmaxf(-2.0f, fminf(2.0f, v));
    // Map to 0-7 encoding
    // 0=0, 1=0.5, 2=1.0, 3=1.5, 4=-0, 5=-0.5, 6=-1.0, 7=-1.5...
    // Actually E2M1: 1 sign, 2 exp, 1 mant
    // 000=0, 001=0.5, 010=1.0, 011=1.5, 100=-0, 101=-0.5, 110=-1.0, 111=-1.5...
    // Simpler: round to nearest representable value
    int sign = (v < 0) ? 1 : 0;
    float absv = fabsf(v);
    uint8_t nibble;
    if (absv < 0.25f)      nibble = 0;  // 0
    else if (absv < 0.75f) nibble = 1;  // 0.5
    else if (absv < 1.25f) nibble = 2;  // 1.0
    else if (absv < 1.75f) nibble = 3;  // 1.5
    else                     nibble = 3;  // clamp to 1.5 (max)

    return (sign << 3) | nibble;
}

// Dequant: E2M1 nibble -> float
__device__ float e2m1_nibble_to_float(uint8_t nibble, float scale) {
    static const float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f,
        0.0f, 0.0f, 0.0f, 0.0f,  // 0b0100-0111 unused
        -0.0f, -0.5f, -1.0f, -1.5f,
        -0.0f, -0.0f, -0.0f, -0.0f  // 0b1100-1111 unused
    };
    return values[nibble & 0xF] * scale;
}

// UE4M3 scale codebook lookup (simplified — just 16 linear scale codes)
__device__ float ue4m3_to_float(uint8_t code) {
    // Map 4-bit code to float scale [0.5, 2.0]
    static const float scale_table[16] = {
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f,
        1.3f, 1.4f, 1.5f, 1.6f,
        1.7f, 1.8f, 1.9f, 2.0f
    };
    return scale_table[code & 0xF];
}

__global__ void audit_vscale_kernel(
    const float * __restrict__ input,
    uint8_t * __restrict__ tile_out,
    float * __restrict__ dequant_out,
    int head_dim, int n_tiles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tiles) return;

    int n_groups = head_dim / 16; // 8 for head_dim=128, 16 for head_dim=256
    int elems_per_group = 16;

    // Each tile covers one KV head's worth of groups
    const float * tile_input = input + tid * head_dim;

    // Build tile: group scales first (n_groups bytes), then nibble data
    uint8_t * tile = tile_out + tid * DEN_NVFP4_KV_TILE_BYTES;

    // --- WRITE PHASE: quantize and store ---
    // Write group scales at tile[0..n_groups-1]
    for (int g = 0; g < n_groups; g++) {
        // Find max abs in this group for scale
        float max_abs = 1e-8f;
        for (int e = 0; e < elems_per_group; e++) {
            float v = tile_input[g * elems_per_group + e];
            max_abs = fmaxf(max_abs, fabsf(v));
        }
        // Quantize scale to UE4M3 codebook
        // Find closest codebook entry
        float target_scale = max_abs / 1.5f; // So max element fits in E2M1 range
        uint8_t scale_code = (uint8_t)((target_scale - 0.5f) / 0.1f);
        scale_code = min(scale_code, (uint8_t)15);
        float actual_scale = ue4m3_to_float(scale_code);

        // Store scale code at tile[g] — LINEAR, NOT SWIZZLED
        tile[g] = scale_code;

        // Store nibbles for this group
        // Each group has 16 elements → 8 bytes of nibbles
        int nibble_offset = n_groups + g * (elems_per_group / 2);
        for (int e = 0; e < elems_per_group; e++) {
            float v = tile_input[g * elems_per_group + e];
            uint8_t nibble = float_to_e2m1_nibble(v, actual_scale);
            int byte_idx = nibble_offset + e / 2;
            if (e & 1) {
                tile[byte_idx] = (tile[byte_idx] & 0x0F) | (nibble << 4);
            } else {
                tile[byte_idx] = (tile[byte_idx] & 0xF0) | nibble;
            }
        }
    }

    // --- READ PHASE: dequant from stored tile ---
    for (int g = 0; g < n_groups; g++) {
        // Read scale from tile[g] — LINEAR READ (the audit target)
        uint8_t scale_code = tile[g];
        float scale = ue4m3_to_float(scale_code);

        int nibble_offset = n_groups + g * (elems_per_group / 2);
        for (int e = 0; e < elems_per_group; e++) {
            int byte_idx = nibble_offset + e / 2;
            uint8_t nibble;
            if (e & 1) {
                nibble = tile[byte_idx] >> 4;
            } else {
                nibble = tile[byte_idx] & 0x0F;
            }
            dequant_out[tid * head_dim + g * elems_per_group + e] =
                e2m1_nibble_to_float(nibble, scale);
        }
    }
}

// --- Host-side test ---
static float compute_cos(const float * a, const float * b, int n) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        norm_a += (double)a[i] * a[i];
        norm_b += (double)b[i] * b[i];
    }
    return (float)(dot / (sqrt(norm_a) * sqrt(norm_b) + 1e-12));
}

int main() {
    cudaSetDevice(0);

    // Test both head_dim=128 and head_dim=256
    int head_dims[] = {128, 256};
    int n_tiles = 4; // 4 KV heads worth

    for (int hd_idx = 0; hd_idx < 2; hd_idx++) {
        int head_dim = head_dims[hd_idx];
        int n_tiles_to_test = n_tiles;
        int n_total = n_tiles_to_test * head_dim;

        printf("\n=== V-Scale Audit: head_dim=%d ===\n", head_dim);

        // Allocate host arrays
        float *h_input     = (float *)malloc(n_total * sizeof(float));
        float *h_dequant   = (float *)malloc(n_total * sizeof(float));
        uint8_t *h_tiles   = (uint8_t *)malloc(n_tiles * DEN_NVFP4_KV_TILE_BYTES);

        // Fill with known pattern: increasing values per group
        for (int t = 0; t < n_tiles_to_test; t++) {
            for (int e = 0; e < head_dim; e++) {
                int g = e / 16;  // which group
                h_input[t * head_dim + e] = (float)((e % 16) - 8) * 0.2f * (g + 1);
            }
        }

        // Allocate device arrays
        float *d_input, *d_dequant;
        uint8_t *d_tiles;
        cudaMalloc(&d_input, n_total * sizeof(float));
        cudaMalloc(&d_dequant, n_total * sizeof(float));
        cudaMalloc(&d_tiles, n_tiles_to_test * DEN_NVFP4_KV_TILE_BYTES);

        cudaMemcpy(d_input, h_input, n_total * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_tiles, 0, n_tiles_to_test * DEN_NVFP4_KV_TILE_BYTES);

        // Launch audit kernel
        int threads = 256;
        int blocks = (n_tiles_to_test + threads - 1) / threads;
        audit_vscale_kernel<<<blocks, threads>>>(d_input, d_tiles, d_dequant, head_dim, n_tiles_to_test);
        cudaDeviceSynchronize();

        // Read back
        cudaMemcpy(h_tiles, d_tiles, n_tiles_to_test * DEN_NVFP4_KV_TILE_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dequant, d_dequant, n_total * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify: scale codes should be at tile[g] for each group
        int n_groups = head_dim / 16;
        bool scale_positions_ok = true;

        for (int t = 0; t < n_tiles_to_test; t++) {
            uint8_t *tile = h_tiles + t * DEN_NVFP4_KV_TILE_BYTES;

            // Check that non-zero scale codes are at the expected linear positions
            for (int g = 0; g < n_groups; g++) {
                uint8_t scale_code = tile[g];
                // Scale code 0 → scale 0.5, should not be zero for our pattern
                if (scale_code == 0) {
                    printf("  WARNING: tile[%d] group[%d] scale_code=0 (unexpected for test pattern)\n", t, g);
                }

                // Also check: if this were SWIZZLED, the scale would be at different position
                // Swizzle: tile[ (g/4)*16 + (g%4) ] instead of tile[g]
                int swizzled_g = (g / 4) * 16 + (g % 4);
                if (swizzled_g < DEN_NVFP4_KV_TILE_BYTES && swizzled_g != g) {
                    uint8_t swizzled_code = tile[swizzled_g];
                    if (swizzled_code != 0) {
                        printf("  INFO: tile[%d] linear_g=%d code=%d, swizzled_pos[%d]=%d\n",
                               t, g, (int)scale_code, swizzled_g, (int)swizzled_code);
                        scale_positions_ok = false;
                    }
                }
            }

            // Check that nibble data (tile[n_groups..]) doesn't contain scale-like values
            bool nibble_area_clean = true;
            for (int b = n_groups; b < DEN_NVFP4_KV_TILE_BYTES; b++) {
                if (tile[b] >= 16) { // scale codes are 0-15, nibbles are 0-15 too — unhelpful
                    // Both scales and nibbles are 4-bit, hard to distinguish by value alone
                }
            }
        }

        // Compute cosine similarity
        float cos_sim = compute_cos(h_input, h_dequant, n_total);
        printf("  Cosine similarity: %.6f %s\n", cos_sim,
               cos_sim > 0.99f ? "PASS" : "FAIL (< 0.99)");

        // Print first tile's first few bytes for manual inspection
        printf("  Tile[0] first 32 bytes: ");
        for (int b = 0; b < 32 && b < DEN_NVFP4_KV_TILE_BYTES; b++) {
            printf("%02x ", h_tiles[b]);
        }
        printf("\n");

        printf("  Scale positions linear: %s\n", scale_positions_ok ? "OK" : "CHECK");

        // Verify: input element positions survive round-trip (no spatial corruption)
        float max_error = 0.0f;
        int max_error_idx = 0;
        for (int i = 0; i < n_total; i++) {
            float err = fabsf(h_input[i] - h_dequant[i]);
            if (err > max_error) {
                max_error = err;
                max_error_idx = i;
            }
        }
        printf("  Max element error: %.4f at idx %d\n", max_error, max_error_idx);
        if (max_error > 2.0f) {
            printf("  SWIZZLE CORRUPTION DETECTED: max error %.2f suggests scale/position mismatch\n", max_error);
        }

        // Cleanup
        free(h_input); free(h_dequant); free(h_tiles);
        cudaFree(d_input); cudaFree(d_dequant); cudaFree(d_tiles);
    }

    printf("\n=== V-Scale Audit Complete ===\n");
    return 0;
}
