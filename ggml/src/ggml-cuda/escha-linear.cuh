#pragma once

#include <cstdint>

struct ggml_backend_cuda_context;
struct ggml_tensor;

// Host-visible dense-path policy helpers. Keep these free of CUDA headers so the
// focused unit test can exercise the selector without compiling as CUDA.
inline constexpr bool escha_dense_shape_valid(const int IC, const int OC) {
    return IC > 0 && OC > 0 && IC % 128 == 0 && OC % 128 == 0;
}

void ggml_cuda_op_escha_moe(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_escha_linear(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
