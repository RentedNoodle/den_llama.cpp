#pragma once

struct ggml_backend_cuda_context;
struct ggml_tensor;

// Correctness-first, source-built Qwen3.8 dense ESCHA route. Returns true
// only for an explicit, SM120, Qwen3.8-marked opt-in. False preserves the
// established caller fallback unchanged.
bool ggml_cuda_op_escha_qwen38_sm120(ggml_backend_cuda_context & ctx,
                                     ggml_tensor * dst);
