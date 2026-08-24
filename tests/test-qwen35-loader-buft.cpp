#include "ggml-cpp.h"
#include "llama-hparams.h"
#include "llama-model-loader.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <crtdbg.h>
#include <windows.h>
#endif

static void add_tensor(gguf_context * metadata, ggml_context * ctx, ggml_type type,
                       const char * name, int64_t ne0, int64_t ne1 = 1) {
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_set_name(tensor, name);
    gguf_add_tensor(metadata, tensor);
}

static int check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return 1;
    }
    return 0;
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
    ggml_context * tensor_ctx = ggml_init(params);
    if (tensor_ctx == nullptr) {
        return 1;
    }

    gguf_context_ptr metadata { gguf_init_empty() };
    gguf_set_val_str(metadata.get(), "general.architecture", "qwen35");
    add_tensor(metadata.get(), tensor_ctx, GGML_TYPE_I8,  "token_embd.weight",       8, 4);
    add_tensor(metadata.get(), tensor_ctx, GGML_TYPE_F16, "token_embd.weight_scale", 4);
    add_tensor(metadata.get(), tensor_ctx, GGML_TYPE_I8,  "output.weight",            8, 4);
    add_tensor(metadata.get(), tensor_ctx, GGML_TYPE_F16, "output.weight_scale",      4);

    std::vector<std::string> splits;
    llama_model_loader loader(
        metadata.get(), nullptr, nullptr, "", splits, nullptr,
        LLAMA_LOAD_MODE_NONE, false, true, false, nullptr, nullptr);

    const ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        ggml_free(tensor_ctx);
        return 1;
    }
    const ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(cpu_dev);
    const buft_list_t cpu_list = {{ cpu_dev, cpu_buft }};
    llama_hparams hparams;
    const LLM_TN qwen35(LLM_ARCH_QWEN35);

    int failures = 0;
    try {
        ggml_tensor * token_scale = loader.create_tensor(
            hparams, &cpu_list, &cpu_list, &cpu_list, nullptr,
            qwen35(LLM_TENSOR_QWEN35_TOKEN_EMBD_SCALED, "weight_scale"), { 4 }, 0);
        failures += check(token_scale != nullptr && token_scale->type == GGML_TYPE_F16,
                "token_embd.weight_scale must load as an F16 companion");

        ggml_tensor * output_scale = loader.create_tensor(
            hparams, &cpu_list, &cpu_list, &cpu_list, nullptr,
            qwen35(LLM_TENSOR_QWEN35_OUTPUT_SCALED, "weight_scale"), { 4 },
            llama_model_loader::TENSOR_NOT_REQUIRED);
        failures += check(output_scale != nullptr && output_scale->type == GGML_TYPE_F16,
                "output.weight_scale must load as an F16 companion");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "unexpected loader failure: %s\n", e.what());
        failures++;
    }

    ggml_free(tensor_ctx);
    return failures == 0 ? 0 : 1;
}
