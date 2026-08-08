// lance-vision.cuh — Lance Vision Pipeline
// Ported from Project Den. Original at C:\Den\den-nvfp4-optimizations\cuda_kernels\vision\den_lance_vision.cuh
// ═══════════════════════════════════════════════════════════════════════════════
// Wires a Lance vision encoder for:
//   1. Image understanding: "What's in this image?" → Lance encode → text
//   2. Visual question answering: Image + question → Lance → answer
//   3. Image editing: "Make the sky more dramatic" → Lance edit → output
//
// Lance architecture (ByteDance Research, Apache 2.0):
//   Dual-path transformer (standard + MoE generation path)
//   36 layers x 64 slots each (wider than Qwen's 32-slot stride)
//   Built-in ViT vision encoder + mmproj multimodal projection
//   Total: 6.18B params main + 0.67B ViT = 6.85B at BF16
//
// Integration with TMU:
//   NVDEC → TMU texture → tmu_extract_patches
//                       → Lance ViT patch encoder
//                       → mmproj → text embedding space
//                       → Lance LLM processes vision tokens + text tokens
//
// Hardware exploitation: GB203 TMU for patch extraction, NVFP4 OMMA for ViT.
// ═══════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Vision pipeline operation type ────────────────────────────────────────────

typedef enum {
    LANCE_VISION_UNDERSTAND  = 0,  // "What's in this image?" → text
    LANCE_VISION_VQA         = 1,  // Image + question → answer
    LANCE_VISION_EDIT        = 2,  // Image + instruction → edited image
    LANCE_VISION_CAPTION     = 3,  // Generate caption for image
    LANCE_VISION_DESCRIBE    = 4,  // Detailed description
} lance_vision_mode_t;

// ── Vision pipeline configuration ────────────────────────────────────────────

typedef struct {
    lance_vision_mode_t mode;       // Operation mode
    int                 max_patches; // Max ViT patches to process (0 = auto)
    int                 patch_size;  // ViT patch size (Lance uses 16px default)
    int                 embed_dim;   // Vision embedding dimension
    int                 use_tmu;     // 1 = use TMU for patch extraction
    int                 use_nvfp4;   // 1 = use NVFP4/OMMA for ViT (faster)
    float               temperature; // Generation temperature (0.0 = greedy)
    int                 max_tokens;  // Max output tokens for text generation
} lance_vision_config_t;

#define LANCE_VISION_DEFAULT_CONFIG { \
    LANCE_VISION_UNDERSTAND, /* mode */ \
    0,    /* max_patches */ \
    16,   /* patch_size */ \
    2048, /* embed_dim (Lance hidden_size) */ \
    1,    /* use_tmu */ \
    1,    /* use_nvfp4 */ \
    0.7f, /* temperature */ \
    128   /* max_tokens */ \
}

// ── Lance vision pipeline context ────────────────────────────────────────────
// Holds GPU pointers to Lance model weights + intermediate buffers.

typedef struct {
    // Model pointer (loaded model mapped to GPU)
    const void  *d_model;           // Raw GPU pointer to Lance weights
    const float *d_embed_tokens;    // Token embedding [V, H]
    const float *d_lm_head;         // LM head [V, H] (often tied)

    // Vision encoder weights
    const float *d_vit_patch_embed; // ViT patch projection [patch_dim, embed]
    const float *d_vit_pos_embed;   // Position embeddings [n_patches, embed]
    const float *d_vit_layers;      // ViT transformer layers (pointer)
    const float *d_mmproj;          // Multimodal projection [vit_dim, text_dim]

    // Intermediate buffers
    float *d_image_embeds;          // [max_patches, embed_dim] GPU buffer
    float *d_text_embeds;           // [max_tokens, embed_dim] GPU buffer
    float *d_output_logits;         // [vocab_size] output logits

    // Configuration
    int     n_vit_layers;           // Number of ViT transformer layers
    int     n_vit_heads;            // ViT attention heads
    int     vit_hidden;             // ViT hidden dimension
    int     vit_patches;            // Max patches for input resolution
    int     text_hidden;            // Text transformer hidden dimension (H)
    int     vocab_size;             // Vocabulary size
    int     initialized;
} lance_vision_ctx_t;

// ═══════════════════════════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize Lance vision pipeline from a loaded model.
// Returns 0 on success, negative on error.
int lance_vision_init(
    const void *d_model_weights,  // GPU pointer to Lance weight data
    const void *tensor_index,     // tensor entry index (CPU)
    int n_tensors,
    lance_vision_ctx_t *ctx);

// Run the vision pipeline on an image.
// The image should already be in GPU memory as a texture or raw patches.
//
// For UNDERSTAND/VQA/CAPTION/DESCRIBE modes:
//   output_text receives generated tokens (as int array)
//   output_count receives number of tokens generated
//
// For EDIT mode:
//   output_image receives the edited image as RGBA GPU buffer
//
// Returns 0 on success.
int lance_vision_process(
    lance_vision_ctx_t *ctx,
    cudaTextureObject_t image_texture,  // Input image (TMU-ready)
    int image_width,
    int image_height,
    const int *text_tokens,             // Optional text prompt tokens
    int n_text_tokens,
    lance_vision_config_t *config,
    int *output_tokens,                 // [max_tokens] output token IDs
    int *output_count,                  // number of output tokens
    float *output_image_rgba,           // [W*H*4] edit output (EDIT mode only)
    cudaStream_t stream);

// Extract ViT patch embeddings from an image using TMU.
// This is the first step of the vision pipeline.
// Patches are stored in ctx->d_image_embeds.
int lance_vision_extract_patches(
    lance_vision_ctx_t *ctx,
    cudaTextureObject_t image_texture,
    int image_width,
    int image_height,
    lance_vision_config_t *config,
    cudaStream_t stream);

// Run the ViT encoder on extracted patches.
// Produces vision token embeddings in ctx->d_image_embeds.
int lance_vision_encode(
    lance_vision_ctx_t *ctx,
    cudaStream_t stream);

// Project vision embeddings to text space via mmproj.
// Copies ctx->d_image_embeds through the mmproj layer.
int lance_vision_project_to_text(
    lance_vision_ctx_t *ctx,
    cudaStream_t stream);

// Run multimodal inference: vision tokens + text tokens → output tokens.
int lance_vision_infer(
    lance_vision_ctx_t *ctx,
    const int *text_tokens,
    int n_text_tokens,
    lance_vision_config_t *config,
    int *output_tokens,
    int *output_count,
    cudaStream_t stream);

// Free all GPU resources associated with the vision context.
void lance_vision_free(lance_vision_ctx_t *ctx);

#if defined(DEN_PAD_MODULATION) && !defined(DEN_PAD_STATE_T_DEFINED)
#define DEN_PAD_STATE_T_DEFINED
typedef struct { float P, A, D, scale; } den_pad_state_t;
#endif

#ifdef __cplusplus
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA Kernel Declarations
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef __CUDACC__

// Lance ViT patch embedding: projects patch pixels through ViT conv + pos embed.
// Each thread handles one patch position across embed_dim channels.
//
// Lance uses a standard ViT: image → 16x16 patches → linear projection → embed.
// The patch embed is a conv2d with kernel_size=patch_size, stride=patch_size.
//
// Kernel specific to Lance's ViT architecture:
//   Lance ViT: H_vit=1408, n_layers=32, n_heads=16, patch_size=16
//   Middle dimension through patch flax embed: 16*16*3 = 768 → 1408
__global__ void lance_patch_embed_kernel(
    const float * __restrict__ patches,    // [n_patches, 16*16*3] normalized
    const float * __restrict__ weight,     // [1408, 768] patch projection
    const float * __restrict__ pos_embed,  // [n_patches_max, 1408]
    float * __restrict__ embed_out,        // [n_patches, 1408]
    int n_patches,
    int n_patches_max,
    int vit_hidden)
{
    int p = blockIdx.x;  // patch index
    int h = threadIdx.x; // hidden dimension

    if (p >= n_patches || h >= vit_hidden) return;

    // Dot product of patch pixels with weight column h
    float sum = 0.0f;
    for (int i = 0; i < 768; i++) {
        sum += patches[p * 768 + i] * weight[h * 768 + i];
    }

    // Add position embedding
    sum += pos_embed[p * vit_hidden + h];

    embed_out[p * vit_hidden + h] = sum;
}

// Lance mmproj: project vision embeddings to text embedding space.
// Lance's mmproj is a MLP: vit_hidden → text_hidden (typically 1408 → 2048).
__global__ void lance_mmproj_kernel(
    const float * __restrict__ vit_embeds,  // [n_patches, vit_hidden]
    const float * __restrict__ mmproj_w1,   // [text_hidden, vit_hidden]
    const float * __restrict__ mmproj_b1,   // [text_hidden]
    float * __restrict__ text_embeds,       // [n_patches, text_hidden]
    int n_patches,
    int vit_hidden,
    int text_hidden)
{
    int p = blockIdx.x;
    int h = threadIdx.x;

    if (p >= n_patches || h >= text_hidden) return;

    // Linear projection with GELU activation
    float sum = mmproj_b1[h];
    for (int i = 0; i < vit_hidden; i++) {
        sum += vit_embeds[p * vit_hidden + i] * mmproj_w1[h * vit_hidden + i];
    }

    // GELU approximation
    float x = sum;
    float gelu = 0.5f * x * (1.0f + tanhf(0.79788456f *
                           (x + 0.044715f * x * x * x)));

    text_embeds[p * text_hidden + h] = gelu;
}

#endif // __CUDACC__
#endif // __cplusplus
