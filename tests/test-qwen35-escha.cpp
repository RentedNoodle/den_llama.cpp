#include "ggml.h"
#include "llama.h"
#include "llama-arch.h"

#include <cstdio>

LLAMA_API bool llama_qwen35_escha_companions_complete(
        bool code,
        bool config,
        bool rin,
        bool rout,
        bool s_in,
        bool s_out,
        bool bias);

LLAMA_API bool llama_qwen35_escha_shared_complete(
        bool lut,
        bool dep_k2,
        bool dep_k3,
        bool needs_dep_k3);

LLAMA_API bool llama_qwen35_escha_inputs_complete(
    const ggml_tensor * dep,
    const ggml_tensor * x);

static int check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return 1;
    }
    return 0;
}

int main() {
    int failures = 0;

    // Bias is optional; all non-bias companions are required.
    failures += check(llama_qwen35_escha_companions_complete(true, true, true, true, true, true, false),
            "complete ESCHA set without bias must activate");
    failures += check(!llama_qwen35_escha_companions_complete(true, false, true, true, true, true, false),
            "missing ESCHA config must not activate");
    failures += check(!llama_qwen35_escha_companions_complete(true, true, true, false, true, true, true),
            "missing ESCHA companion must not activate");

    failures += check(llama_qwen35_escha_shared_complete(true, true, false, false),
            "K2 ESCHA shared set must activate with LUT and dep");
    failures += check(!llama_qwen35_escha_shared_complete(false, true, false, false),
            "missing ESCHA LUT must be rejected");
    failures += check(!llama_qwen35_escha_shared_complete(true, false, false, false),
            "missing ESCHA K2 dep must be rejected");
    failures += check(!llama_qwen35_escha_shared_complete(true, true, false, true),
            "missing ESCHA K3 dep must be rejected");

    ggml_init_params params = {
        /*.mem_size   =*/ 1 << 20,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return 1;
    }

    ggml_tensor * dep = ggml_new_tensor_2d(ctx, GGML_TYPE_I16, 16, 256);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 3);
    failures += check(llama_qwen35_escha_inputs_complete(dep, x),
            "complete ESCHA graph inputs must be accepted");
    failures += check(!llama_qwen35_escha_inputs_complete(nullptr, x),
            "missing ESCHA dep must be rejected before graph op construction");
    failures += check(!llama_qwen35_escha_inputs_complete(dep, nullptr),
            "missing ESCHA activation must be rejected before graph op construction");

    failures += check(llm_tensor_info_for(LLM_TENSOR_ESCHA_LUT).op == GGML_OP_ESCHA_MOE,
            "ESCHA LUT must be registered for ESCHA_MOE");
    failures += check(llm_tensor_info_for(LLM_TENSOR_ESCHA_DEP_K2).op == GGML_OP_ESCHA_MOE,
            "ESCHA K2 dep must be registered for ESCHA_MOE");
    failures += check(llm_tensor_info_for(LLM_TENSOR_ESCHA_DEP_K3).op == GGML_OP_ESCHA_MOE,
            "ESCHA K3 dep must be registered for ESCHA_MOE");
    failures += check(llm_tensor_info_for(LLM_TENSOR_QWEN35_TOKEN_EMBD_SCALED).op == GGML_OP_GET_ROWS_SCALED_I8,
            "Qwen3.5 scaled embedding must be registered for GET_ROWS_SCALED_I8");
    failures += check(llm_tensor_info_for(LLM_TENSOR_QWEN35_TOKEN_EMBD_SCALED).layer == LLM_TENSOR_LAYER_INPUT,
            "Qwen3.5 scaled embedding must stay in input placement");
    failures += check(llm_tensor_info_for(LLM_TENSOR_QWEN35_OUTPUT_SCALED).op == GGML_OP_MUL_MAT_SCALED_I8,
            "Qwen3.5 scaled output must be registered for MUL_MAT_SCALED_I8");
    failures += check(llm_tensor_info_for(LLM_TENSOR_QWEN35_OUTPUT_SCALED).layer == LLM_TENSOR_LAYER_OUTPUT,
            "Qwen3.5 scaled output must stay in output placement");
    failures += check(llm_tensor_info_for(LLM_TENSOR_QWEN35_ESCHA_FFN_GATE).op == GGML_OP_ESCHA_LINEAR,
            "Qwen3.5 dense ESCHA sidecars must be registered for ESCHA_LINEAR");
    failures += check(llm_tensor_info_for(LLM_TENSOR_QWEN35_ESCHA_FFN_GATE).layer == LLM_TENSOR_LAYER_REPEATING,
            "Qwen3.5 ESCHA sidecars must use repeating placement");

    ggml_free(ctx);
    return failures == 0 ? 0 : 1;
}
