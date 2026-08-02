#pragma once

#include "common.cuh"

// GPU-side MoE router: fused hidden_state × router_weight^T → softmax → top-k
//
// Replaces the CPU-side router (llama_expert_cpu_router_predict) with a single
// CUDA kernel that keeps expert IDs entirely on GPU.  No CPU round-trip needed.
//
// Input:
//   hidden_state  [n_tokens, n_embd]    F32 or F16
//   router_weight [n_expert, n_embd]    F32 or F16
// Output:
//   expert_ids    [n_tokens, n_expert_used]  I32
//   expert_weights[n_tokens, n_expert_used]  F32
//
// The kernel fuses matmul, softmax, and iterative argmax top-k into one launch.
// Template-specialized for common n_expert counts (powers of 2 up to 512).

void ggml_cuda_op_gpu_router(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * hidden_state,
    const ggml_tensor * router_weight,
    ggml_tensor * expert_ids,
    ggml_tensor * expert_weights);

// Returns true if the fused GPU router should be used for this node pattern.
// Checks: contiguous inputs, supported types, n_expert is power-of-2 and <= 512.
bool ggml_cuda_should_use_gpu_router(
    const ggml_tensor * logits,      // output of mul_mat (hidden × gate_weight^T)
    const ggml_tensor * router_w,    // router weight tensor (ffn_gate_inp)
    const ggml_tensor * hidden);     // hidden state input

// ============================================================================
// Extern "C" convenience wrappers — callable from llama.cpp without ggml types
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Low-level GPU router launch: hidden_state [n_rows, n_embd] F32 × router_weight
// [n_expert, n_embd] F32 → softmax → top-k.  Writes expert_ids and expert_weights
// on GPU.  Caller must synchronize before reading back.
//
// Uses the default CUDA stream.  All pointers are GPU device pointers.
void launch_gpu_router_f32(
    const float * hidden_state,      // [n_rows, n_embd] on GPU
    const float * router_weight,     // [n_expert, n_embd] on GPU
    int32_t * expert_ids,            // [n_rows, n_expert_used] output on GPU
    float * expert_weights,          // [n_rows, n_expert_used] output on GPU
    int n_rows,
    int n_embd,
    int n_expert,
    int n_expert_used);

#ifdef __cplusplus
}
#endif
