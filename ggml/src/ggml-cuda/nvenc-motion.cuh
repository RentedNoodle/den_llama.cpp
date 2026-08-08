// nvenc-motion.cuh — NVENC Motion Vector Change Detection
// Ported from Project Den. Original at C:\Den\den-nvfp4-optimizations\cuda_kernels\vision\den_nvenc_motion_detect.cuh
// ═══════════════════════════════════════════════════════════════════════════════
// Extracts motion vectors from NVENC-encoded video frames for hardware-
// accelerated change detection. Instead of running optical flow on CUDA cores
// (expensive, ~10ms per frame), we exploit NVENC's built-in motion estimation.
//
// Principle:
//   NVENC already computes motion vectors for inter-frame encoding (P/B frames).
//   We can access these motion vectors (even when not encoding to disk) and use
//   them as a zero-cost motion detection signal.
//
//   Each motion vector describes how a 16x16 macroblock moved from the previous
//   frame to the current frame. Macroblocks with large motion vectors → "something
//   moved here." Clusters of moving macroblocks → "a person entered the frame."
//
// Pipeline:
//   Frame N-1 → NVENC encode (discard output, capture motion vectors)
//   Frame N   → NVENC encode (discard output, capture motion vectors)
//              → Motion vector difference → clustering → "What changed?"
//              → Feed to Lance: "Describe what changed in the scene"
//
// Hardware: NVENC Gen 5 on GB203-300-A1 (RTX 5070 Ti)
//   - H.264/HEVC/AV1 hardware encoder
//   - Motion vector extraction via NV_ENC_REGISTER_RESOURCE + NV_ENC_LOCK_BITSTREAM
//   - Also supports NV_ENC_ME_ONLY mode (no actual encode, just motion estimation)
//
// Novelty:
//   No vision system uses NVENC motion vectors for change detection in LLM vision
//   pipelines. Standard approach: frame differencing (CUDA), background subtraction
//   (CUDA), or optical flow (CUDA). All use compute cores.
//   This uses NVENC hardware — zero CUDA core cycles for the motion detection itself.
//
//   This gives visual attention: notice movement without running
//   full Lance inference on every frame. Only when motion is detected does the
//   system "look" at the scene with Lance.
// ═══════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Motion vector structure (matches NVENC MV output format) ─────────────────
// NVENC produces motion vectors per 16x16 macroblock.
// Each MV has a 2D displacement vector + optional residual magnitude.

typedef struct {
    int16_t mv_x;           // Horizontal displacement (pixels, 1/4 pel units)
    int16_t mv_y;           // Vertical displacement (pixels, 1/4 pel units)
    uint16_t sad;           // Sum of absolute differences (residual energy)
    uint16_t cost;          // Encoding cost (rate-distortion optimized)
} den_nvenc_motion_vector_t;

// ── Motion cluster — a group of nearby motion vectors ────────────────────────
// Represents a moving object or region.

typedef struct {
    int    centroid_x;       // Cluster center X (pixels)
    int    centroid_y;       // Cluster center Y (pixels)
    int    width;            // Bounding box width (pixels)
    int    height;           // Bounding box height (pixels)
    float  mean_mv_x;       // Mean horizontal displacement
    float  mean_mv_y;       // Mean vertical displacement
    float  magnitude;        // Mean motion magnitude
    int    n_vectors;        // Number of motion vectors in cluster
    int    is_significant;   // 1 if this cluster exceeds the significance threshold
} den_nvenc_motion_cluster_t;

// ── Motion detection configuration ───────────────────────────────────────────

typedef struct {
    int   frame_width;          // Frame width (must be multiple of 16)
    int   frame_height;         // Frame height (must be multiple of 16)
    int   mb_size;              // Macroblock size (16 for H.264, can be larger for HEVC)
    float motion_threshold;     // MV magnitude threshold for "motion" (default: 4.0 pixels)
    float cluster_radius;       // Max distance for vectors to be in same cluster (px)
    int   min_cluster_size;     // Min vectors for a significant cluster (default: 4)
    int   active_threshold;     // Min active macroblocks for "something happened" (default: 8)
    int   enable_motion_vectors; // 1 = extract MVs from NVENC, 0 = software frame diff
    int   me_only_mode;         // 1 = NVENC ME-only mode (faster, no actual encode)
} den_nvenc_motion_config_t;

#define DEN_NVENC_MOTION_DEFAULT_CONFIG { \
    640,   /* frame_width */ \
    480,   /* frame_height */ \
    16,    /* mb_size */ \
    4.0f,  /* motion_threshold (1 pixel at quarter-pel = 4.0) */ \
    48.0f, /* cluster_radius (3 macroblocks) */ \
    4,     /* min_cluster_size */ \
    8,     /* active_threshold */ \
    1,     /* enable_motion_vectors */ \
    1      /* me_only_mode */ \
}

// ── Motion detection result ───────────────────────────────────────────────────

typedef struct {
    int    n_vectors;             // Total macroblocks
    int    n_active;              // Macroblocks above motion threshold
    int    n_clusters;            // Number of motion clusters found
    int    has_motion;            // 1 = significant motion detected
    float  mean_magnitude;        // Mean motion magnitude across frame
    float  max_magnitude;         // Maximum motion magnitude
    float  motion_energy;         // Total motion energy (sum of all SAD values)

    // Cluster details (pointer to GPU buffer)
    den_nvenc_motion_cluster_t *d_clusters;  // [max_clusters] on GPU
    int                          max_clusters;

    // Raw motion vectors (GPU, for diagnostic/visualization)
    den_nvenc_motion_vector_t  *d_motion_vectors;  // [n_macroblocks]
    int                          n_macroblocks;
    int                          mb_width;
    int                          mb_height;
} den_nvenc_motion_result_t;

// ═══════════════════════════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize motion detection context.
// Allocates GPU buffers for motion vectors and clusters.
int den_nvenc_motion_init(
    den_nvenc_motion_config_t *config,
    void **ctx_out);

// Process a new frame — detect motion relative to previous frame.
//   frame_rgba: GPU buffer with current frame RGBA pixels [W*H*4].
//   previous_frame_rgba: GPU buffer with previous frame RGBA (can be NULL for
//     first frame; motion will be reported as 0).
//   result: Output motion detection result.
//   stream: CUDA stream.
//
// When called repeatedly, maintains internal state for motion vector accumulation
// and temporal smoothing.
int den_nvenc_motion_process(
    void *ctx,
    const uint8_t *frame_rgba,
    const uint8_t *previous_frame_rgba,
    den_nvenc_motion_result_t *result,
    cudaStream_t stream);

// Find motion clusters from raw motion vectors.
// Launches a CUDA kernel that groups nearby motion vectors into clusters.
int den_nvenc_motion_cluster(
    void *ctx,
    den_nvenc_motion_result_t *result,
    cudaStream_t stream);

// Get a text description of the current motion state.
// Useful for feeding to Lance: "A person entered the frame from the left."
// Returns a human-readable string (caller must free with free()).
char *den_nvenc_motion_describe(
    den_nvenc_motion_result_t *result);

// Reset motion state (clears previous frame buffer, starts fresh).
void den_nvenc_motion_reset(void *ctx);

// Free all motion detection resources.
void den_nvenc_motion_free(void *ctx);

#ifdef __cplusplus
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA Kernel Declarations
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef __CUDACC__

// Compute macroblock-based motion from frame difference.
// Used as software fallback when NVENC motion vectors aren't available.
// Each thread handles one 16x16 macroblock.
//
// Motion magnitude = mean absolute difference between corresponding pixels
// in current and previous frame, within a 16x16 block.
__global__ void nvenc_frame_diff_kernel(
    const uint8_t * __restrict__ curr_frame,  // [W*H*4] RGBA
    const uint8_t * __restrict__ prev_frame,  // [W*H*4] RGBA
    den_nvenc_motion_vector_t * __restrict__ mv_out,
    int width,
    int height,
    int mb_size,
    float threshold)
{
    int mb_x = blockIdx.x * blockDim.x + threadIdx.x;
    int mb_y = blockIdx.y * blockDim.y + threadIdx.y;
    int mb_w = width  / mb_size;
    int mb_h = height / mb_size;
    if (mb_x >= mb_w || mb_y >= mb_h) return;

    int idx = mb_y * mb_w + mb_x;

    // Compute SAD over macroblock
    unsigned int sad = 0;
    int best_dx = 0, best_dy = 0;
    unsigned int min_sad = 0xFFFFFFFF;

    // Simple block matching: search +/-8 pixels in previous frame
    int search_radius = 8;
    for (int dy = -search_radius; dy <= search_radius; dy += 2) {
        for (int dx = -search_radius; dx <= search_radius; dx += 2) {
            unsigned int block_sad = 0;

            for (int by = 0; by < mb_size; by++) {
                for (int bx = 0; bx < mb_size; bx++) {
                    int cx = mb_x * mb_size + bx;
                    int cy = mb_y * mb_size + by;
                    int px = cx + dx;
                    int py = cy + dy;

                    if (px < 0 || px >= width || py < 0 || py >= height) {
                        block_sad += 255 * 3;  // penalty for out-of-bounds
                        continue;
                    }

                    // Luma difference (green channel approximation)
                    int curr_pix = curr_frame[(cy * width + cx) * 4 + 1];
                    int prev_pix = prev_frame[(py * width + px) * 4 + 1];
                    block_sad += (unsigned int)abs(curr_pix - prev_pix);
                }
            }

            if (block_sad < min_sad) {
                min_sad = block_sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    // Store motion vector (quarter-pel units: dx*4)
    mv_out[idx].mv_x = (int16_t)(best_dx * 4);
    mv_out[idx].mv_y = (int16_t)(best_dy * 4);
    mv_out[idx].sad  = (uint16_t)(min_sad > 65535 ? 65535 : min_sad);
    mv_out[idx].cost = (uint16_t)(abs(best_dx) + abs(best_dy) +
                                  (min_sad >> 8));
}

// Cluster nearby motion vectors into moving-object hypotheses.
// Uses a simple flood-fill approach: vectors within cluster_radius of each
// other and with similar motion direction are grouped together.
//
// Each warp processes a row of macroblocks, greedily merging vectors.
__global__ void nvenc_motion_cluster_kernel(
    const den_nvenc_motion_vector_t * __restrict__ mv,
    den_nvenc_motion_cluster_t * __restrict__ clusters,
    int * __restrict__ cluster_count,
    int max_clusters,
    int mb_w,
    int mb_h,
    float cluster_radius,
    int min_cluster_size,
    float motion_threshold,
    int active_threshold)
{
    int mb_x = blockIdx.x * blockDim.x + threadIdx.x;
    int mb_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (mb_x >= mb_w || mb_y >= mb_h) return;

    int idx = mb_y * mb_w + mb_x;
    const den_nvenc_motion_vector_t *mv_i = &mv[idx];

    // Skip if below motion threshold
    float mag = sqrtf((float)mv_i->mv_x * mv_i->mv_x +
                      (float)mv_i->mv_y * mv_i->mv_y);
    if (mag < motion_threshold && mv_i->sad < 200) return;

    // Try to merge with existing cluster
    // Uses atomic CAS to claim this vector for a cluster
    for (int attempt = 0; attempt < 10; attempt++) {
        int c_count = *cluster_count;
        if (c_count >= max_clusters) break;

        // Check if within distance of any existing cluster centroid
        bool merged = false;
        for (int c = 0; c < c_count && !merged; c++) {
            den_nvenc_motion_cluster_t *cl = &clusters[c];
            if (!cl->is_significant) continue;  // slot available

            int dx = mb_x * 16 - cl->centroid_x;
            int dy = mb_y * 16 - cl->centroid_y;
            float dist = sqrtf((float)(dx*dx + dy*dy));

            // Also check motion similarity
            float mdx = (float)mv_i->mv_x - cl->mean_mv_x;
            float mdy = (float)mv_i->mv_y - cl->mean_mv_y;
            float mdist = sqrtf(mdx*mdx + mdy*mdy);

            if (dist < cluster_radius && mdist < cluster_radius * 0.5f) {
                // Merge: update centroid and mean MV
                // Simple approximation: exponential moving average
                float n = (float)(cl->n_vectors + 1);
                cl->centroid_x = (int)((cl->centroid_x * cl->n_vectors +
                                        mb_x * 16) / n);
                cl->centroid_y = (int)((cl->centroid_y * cl->n_vectors +
                                        mb_y * 16) / n);
                cl->mean_mv_x = (cl->mean_mv_x * cl->n_vectors +
                                 (float)mv_i->mv_x) / n;
                cl->mean_mv_y = (cl->mean_mv_y * cl->n_vectors +
                                 (float)mv_i->mv_y) / n;
                cl->magnitude = (cl->magnitude * cl->n_vectors + mag) / n;
                cl->n_vectors++;

                // Update bounding box
                int bx = mb_x * 16, by = mb_y * 16;
                int rx = bx + 16 - cl->centroid_x;
                int ry = by + 16 - cl->centroid_y;
                if (rx > cl->width)  cl->width  = rx;
                if (ry > cl->height) cl->height = ry;
                bx -= cl->centroid_x;
                by -= cl->centroid_y;
                if (bx < 0) { cl->width -= bx; cl->centroid_x += bx; }
                if (by < 0) { cl->height -= by; cl->centroid_y += by; }

                merged = true;
                break;
            }
        }

        if (!merged) {
            // Create new cluster
            int slot = atomicCAS(cluster_count, c_count, c_count + 1);
            if (slot == c_count) {
                den_nvenc_motion_cluster_t new_cl;
                new_cl.centroid_x   = mb_x * 16 + 8;
                new_cl.centroid_y   = mb_y * 16 + 8;
                new_cl.width        = 16;
                new_cl.height       = 16;
                new_cl.mean_mv_x    = (float)mv_i->mv_x;
                new_cl.mean_mv_y    = (float)mv_i->mv_y;
                new_cl.magnitude    = mag;
                new_cl.n_vectors    = 1;
                new_cl.is_significant = (mag >= motion_threshold) ? 1 : 0;
                clusters[slot] = new_cl;
                break;
            }
            // CAS failed, another thread wrote — retry
            continue;
        }
        break;
    }
}

#endif // __CUDACC__
#endif // __cplusplus
