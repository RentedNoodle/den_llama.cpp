#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <functional>
#include <stdexcept>

#ifdef _WIN32
#include <crtdbg.h>
#include <windows.h>
#endif

LLAMA_API extern ggml_tensor * llama_qwen35_get_rows_endpoint(
        ggml_context * ctx,
        ggml_tensor  * values,
        ggml_tensor  * row_scales,
        ggml_tensor  * rows);

LLAMA_API extern ggml_tensor * llama_qwen35_mul_mat_endpoint(
        ggml_context * ctx,
        ggml_tensor  * weights,
        ggml_tensor  * activations,
        ggml_tensor  * row_scales);

static int expect_throw(const char * name, const std::function<void()> & fn) {
    try {
        fn();
    } catch (const std::runtime_error &) {
        return 0;
    }
    std::fprintf(stderr, "%s: expected validation failure\n", name);
    return 1;
}

int main() {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    ggml_init_params params = {
        /*.mem_size   =*/ 1 << 20,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return 1;
    }

    ggml_tensor * values_i8 = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 8, 4);
    ggml_tensor * scales_f16 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 4);
    ggml_tensor * rows_i32 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_tensor * gathered = llama_qwen35_get_rows_endpoint(ctx, values_i8, scales_f16, rows_i32);
    if (gathered->op != GGML_OP_GET_ROWS_SCALED_I8 || gathered->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "scaled embedding endpoint did not select GET_ROWS_SCALED_I8\n");
        return 1;
    }
    if (gathered->src[0] != values_i8 || gathered->src[1] != rows_i32 || gathered->src[2] != scales_f16) {
        std::fprintf(stderr, "scaled embedding source pairing changed\n");
        return 1;
    }

    ggml_tensor * values_f16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 4);
    ggml_tensor * ordinary = llama_qwen35_get_rows_endpoint(ctx, values_f16, nullptr, rows_i32);
    if (ordinary->op != GGML_OP_GET_ROWS) {
        std::fprintf(stderr, "ordinary embedding endpoint changed\n");
        return 1;
    }

    ggml_tensor * activations_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3);
    ggml_tensor * logits = llama_qwen35_mul_mat_endpoint(ctx, values_i8, activations_f32, scales_f16);
    if (logits->op != GGML_OP_MUL_MAT_SCALED_I8 || logits->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "scaled lm-head endpoint did not select MUL_MAT_SCALED_I8\n");
        return 1;
    }

    ggml_tensor * ordinary_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 4);
    ggml_tensor * ordinary_head = llama_qwen35_mul_mat_endpoint(ctx, ordinary_weights, activations_f32, nullptr);
    if (ordinary_head->op != GGML_OP_MUL_MAT) {
        std::fprintf(stderr, "ordinary lm-head endpoint changed\n");
        return 1;
    }

    int failures = 0;
    failures += expect_throw("embedding I8 without scales", [&] {
        llama_qwen35_get_rows_endpoint(ctx, values_i8, nullptr, rows_i32);
    });
    failures += expect_throw("embedding scale type", [&] {
        ggml_tensor * wrong_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        llama_qwen35_get_rows_endpoint(ctx, values_i8, wrong_scale, rows_i32);
    });
    failures += expect_throw("embedding scale shape", [&] {
        ggml_tensor * wrong_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 3);
        llama_qwen35_get_rows_endpoint(ctx, values_i8, wrong_scale, rows_i32);
    });
    failures += expect_throw("lm-head activation type", [&] {
        ggml_tensor * wrong_activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 1);
        llama_qwen35_mul_mat_endpoint(ctx, values_i8, wrong_activation, scales_f16);
    });

    ggml_free(ctx);
    return failures == 0 ? 0 : 1;
}
