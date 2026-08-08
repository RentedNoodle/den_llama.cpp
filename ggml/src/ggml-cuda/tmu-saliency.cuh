// tmu-saliency.cuh — TMU Mipmap-Based Visual Saliency Detection
// Ported from Project Den. Original at C:\Den\den-nvfp4-optimizations\cuda_kernels\vision\den_tmu_saliency.cuh
// ═══════════════════════════════════════════════════════════════════════════════
// Uses the TMU's hardware mipmapping capability for free visual saliency
// detection. Instead of running a neural saliency model (which costs GPU
// compute cycles), we exploit the fact that the TMU can sample between
// mip levels for free.
//
// Principle:
//   A mipmap pyramid naturally blurs high-frequency detail at coarser levels.
//   Regions where a pixel's value differs significantly between fine and coarse
//   mip levels correspond to high-frequency content: edges, textures, faces.
//   This IS visual saliency — computed entirely by TMU hardware.
//
// Pipeline:
//   1. Image loaded as cudaTextureObject_t with mipmaps (enableMipMaps=true)
//   2. For each pixel: sample at mip=0 (fine) and mip=L (coarse, ~16x16 average)
//   3. Compute absolute difference between fine and coarse samples
//   4. Difference magnitude = saliency score
//   5. Output: saliency map (float32, same resolution as input, or 1/4 res)
//
// Hardware: 280 TMUs on GB203-300-A1, zero CUDA core usage for saliency.
// The TMU does all the work: mipmap generation (free at texture creation),
// bilinear interpolation (free per sample), mip level selection.
//
// Novelty:
//   No vision system uses TMU mipmapping for saliency detection.
//   Standard approach: run a CNN (SALICON, DeepGaze) or hand-crafted
//   feature detectors (DoG, Sobel). Both use CUDA cores.
//   This approach = 0 CUDA cores. Entirely on TMU. Microseconds vs milliseconds.
//
//   This enables determining WHERE to look before knowing WHAT to look at
//   — the saliency map guides attention (and Lance inference) to the most
//   informative regions first.
// ═══════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Saliency configuration ────────────────────────────────────────────────────

typedef struct {
    float coarse_mip;       // Mip level for coarse sampling (default: 4.0 =
                            //   16x16 average). Higher = more blurred = more
                            //   aggressive saliency (detects only strong edges)
    float fine_mip;         // Mip level for fine sampling (default: 0.0 = full
                            //   resolution). Can be shifted up to reduce noise.
    float threshold;        // Minimum saliency value to report (default: 0.05)
    int   output_stride;    // If >1, output saliency map is 1/output_stride res.
                            //   2 = quarter resolution (default).
    int   enable_foveation; // 1 = compute foveated saliency: highest detail at
                            //   center, decreasing toward edges (biological
                            //   retina model). Default: 1.
    float fovea_radius;     // Fraction of image radius for full-detail fovea
                            //   (default: 0.3 = 30% of min(width,height)/2)
} den_tmu_saliency_config_t;

#define DEN_TMU_SALIENCY_DEFAULT_CONFIG { \
    4.0f,   /* coarse_mip = 16x16 average blocks */ \
    0.0f,   /* fine_mip = full resolution */ \
    0.05f,  /* threshold */ \
    2,      /* output_stride = quarter resolution */ \
    1,      /* enable_foveation = yes */ \
    0.3f    /* fovea_radius = 30% */ \
}

// ── Saliency result ───────────────────────────────────────────────────────────

typedef struct {
    float *d_saliency_map;  // [H/stride * W/stride] float32 GPU buffer
    int    map_width;       // Width of saliency map
    int    map_height;      // Height of saliency map
    int    full_width;      // Source image width
    int    full_height;     // Source image height
    float  max_saliency;    // Maximum saliency value in map
    float  mean_saliency;   // Mean saliency value
    float  total_saliency;  // Sum of all saliency values
} den_tmu_saliency_result_t;

// ═══════════════════════════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════════════════════════

// Compute a saliency map from a texture object using TMU mipmap comparison.
// The texture must have mipmaps enabled (cudaTextureDesc.maxMipmapLevelClamp).
// Returns 0 on success.
int den_tmu_saliency_compute(
    cudaTextureObject_t texture,
    int full_width,
    int full_height,
    den_tmu_saliency_config_t *config,
    den_tmu_saliency_result_t *result,
    cudaStream_t stream);

// Find the top-N most salient regions (bounding boxes).
// The saliency map is scanned and local maxima above threshold are returned.
// Returns the number of regions found (up to max_regions).
int den_tmu_saliency_find_regions(
    const float *d_saliency_map,
    int map_width,
    int map_height,
    int full_width,
    int full_height,
    float threshold,
    int max_regions,
    int *out_region_count,
    int *out_x, int *out_y,         // [max_regions] region centers
    int *out_w, int *out_h,         // [max_regions] region extents
    float *out_score,               // [max_regions] region saliency scores
    cudaStream_t stream);

// Apply foveated attention: given a fixation point, boost saliency near it
// and suppress far regions (central vision has higher acuity).
void den_tmu_saliency_foveate(
    float *d_saliency_map,
    int map_width,
    int map_height,
    int fix_x,
    int fix_y,
    float fovea_radius,
    cudaStream_t stream);

// Free saliency result resources.
void den_tmu_saliency_free(den_tmu_saliency_result_t *result);

#ifdef __cplusplus
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA Kernel Declarations
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef __CUDACC__

// Compute saliency by comparing fine and coarse mip levels.
// Each thread processes one output pixel of the saliency map.
//
// Saliency(x,y) = max(0, |tex2D(fine) - tex2D(coarse)| - threshold)
//
// The coarse mip level acts as a local average filter.
// The difference between fine (detailed) and coarse (blurred) is exactly the
// high-frequency content that humans find salient.
//
// This is biologically inspired: the retina's center-surround receptive fields
// (ON-center/OFF-surround ganglion cells) compute exactly this difference.
// The TMU's mipmapping implements the surround at zero compute cost.
__global__ void tmu_saliency_kernel(
    cudaTextureObject_t texture,
    float * __restrict__ saliency_out,
    int full_width,
    int full_height,
    float coarse_mip,
    float fine_mip,
    float threshold,
    int output_stride,
    int enable_foveation,
    float fovea_radius)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int map_w = full_width / output_stride;
    int map_h = full_height / output_stride;

    if (out_x >= map_w || out_y >= map_h) return;

    // Map output pixel to source image coordinates
    float sx = (float)out_x * output_stride + (float)output_stride * 0.5f;
    float sy = (float)out_y * output_stride + (float)output_stride * 0.5f;

    // Sample fine and coarse mip levels
    float4 fine   = tex2DLod<float4>(texture, sx, sy, fine_mip);
    float4 coarse = tex2DLod<float4>(texture, sx, sy, coarse_mip);

    // Saliency = L2 difference between fine and coarse across RGB channels
    float dx = fine.x - coarse.x;
    float dy = fine.y - coarse.y;
    float dz = fine.z - coarse.z;

    // Weight green channel more (human luminance sensitivity)
    float sal = sqrtf(dx * dx * 0.2126f +
                      dy * dy * 0.7152f +
                      dz * dz * 0.0722f);

    // Threshold: suppress noise
    if (sal < threshold) sal = 0.0f;
    else sal -= threshold;

    // Apply foveation if enabled: saliency falls off with eccentricity
    if (enable_foveation) {
        float cx = (float)full_width * 0.5f;
        float cy = (float)full_height * 0.5f;
        float dx_eye = (sx - cx) / (float)full_width;
        float dy_eye = (sy - cy) / (float)full_height;
        float eccentricity = sqrtf(dx_eye * dx_eye + dy_eye * dy_eye);

        // Fovea radius in normalized coordinates
        float fovea_norm = fovea_radius * 0.5f;  // fraction of image radius
        float falloff = 1.0f / (1.0f + (eccentricity / fovea_norm) *
                                       (eccentricity / fovea_norm));
        sal *= falloff;
    }

    int idx = out_y * map_w + out_x;
    saliency_out[idx] = sal;
}

// Find local maxima in saliency map (region proposals).
// Each thread examines a 3x3 neighborhood; it's a local maximum if it's the
// highest in its 3x3 block and above threshold.
__global__ void tmu_saliency_local_maxima_kernel(
    const float * __restrict__ saliency_map,
    int map_width,
    int map_height,
    float threshold,
    int * __restrict__ max_x,          // [max_candidates] output
    int * __restrict__ max_y,
    float * __restrict__ max_val,
    int * __restrict__ max_count,
    int max_candidates)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= map_width - 1 || y < 1 || y >= map_height - 1) return;

    float center = saliency_map[y * map_width + x];
    if (center < threshold) return;

    // Check 3x3 neighborhood
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (saliency_map[(y + dy) * map_width + (x + dx)] >= center)
                return; // Not a local maximum
        }
    }

    // Candidate found — atomically claim a slot
    int slot = atomicAdd(max_count, 1);
    if (slot < max_candidates) {
        max_x[slot] = x;
        max_y[slot] = y;
        max_val[slot] = center;
    }
}

// Apply foveation: reduce saliency with distance from fixation point.
// Gaussian falloff with given radius.
__global__ void tmu_saliency_foveate_kernel(
    float * __restrict__ saliency_map,
    int map_width,
    int map_height,
    int fix_x,
    int fix_y,
    float fovea_radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height) return;

    float dx = (float)(x - fix_x);
    float dy = (float)(y - fix_y);
    float dist = sqrtf(dx * dx + dy * dy);
    float attn = expf(-dist / (fovea_radius * fovea_radius));

    int idx = y * map_width + x;
    saliency_map[idx] *= attn;
}

#endif // __CUDACC__
#endif // __cplusplus
