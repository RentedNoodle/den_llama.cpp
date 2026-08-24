#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../ggml/src/ggml-cuda/escha-linear.cuh"

int main() {
    if (!escha_dense_shape_valid(5120, 17408) ||
        !escha_dense_shape_valid(17408, 5120) ||
        escha_dense_shape_valid(5119, 17408) ||
        escha_dense_shape_valid(5120, 17407)) {
        std::fprintf(stderr, "dense ESCHA shape validation mismatch\n");
        return 1;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 1 << 20,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return 1;
    }

    // The dense contract includes the trained per-axis scales and has no ids.
    ggml_tensor * code = ggml_new_tensor_3d(ctx, GGML_TYPE_I16, 32, 8, 16);
    ggml_tensor * rin  = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 256);
    ggml_tensor * rout = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 128);
    ggml_tensor * s_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_tensor * s_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_tensor * dep  = ggml_new_tensor_2d(ctx, GGML_TYPE_I16, 16, 256);
    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 2);

    ggml_tensor * result = ggml_escha_linear(ctx, code, rin, rout, s_in, s_out, dep, x);
    const bool ok = result != nullptr && result->op == GGML_OP_ESCHA_LINEAR &&
        result->src[0] == code && result->src[1] == rin && result->src[2] == rout &&
        result->src[3] == s_in && result->src[4] == s_out && result->src[5] == dep &&
        result->src[6] == x && result->src[7] == nullptr;

    if (!ok) {
        std::fprintf(stderr, "dense ESCHA constructor contract mismatch\n");
        ggml_free(ctx);
        return 1;
    }

    auto * code_data = (int16_t *) code->data;
    auto * rin_data = (ggml_fp16_t *) rin->data;
    auto * rout_data = (ggml_fp16_t *) rout->data;
    auto * s_in_data = (float *) s_in->data;
    auto * s_out_data = (float *) s_out->data;
    auto * dep_data = (int16_t *) dep->data;
    auto * x_data = (float *) x->data;
    for (int i = 0; i < 32*8*16; ++i) code_data[i] = 0;
    for (int i = 0; i < 256; ++i) {
        rin_data[i] = ggml_fp32_to_fp16(1.0f);
        s_in_data[i] = 0.75f + (float) (i % 7)/16.0f;
    }
    for (int i = 0; i < 128; ++i) {
        rout_data[i] = ggml_fp32_to_fp16(1.0f);
        s_out_data[i] = 0.625f + (float) (i % 5)/16.0f;
    }
    for (int i = 0; i < 16*256; ++i) dep_data[i] = (int16_t) (i & 31);
    for (int i = 0; i < 256*2; ++i) x_data[i] = (float) (i - 128) / 17.0f;

    const std::vector<int16_t> code_host(code_data, code_data + 32*8*16);
    const std::vector<ggml_fp16_t> rin_host(rin_data, rin_data + 256);
    const std::vector<ggml_fp16_t> rout_host(rout_data, rout_data + 128);
    const std::vector<float> s_in_host(s_in_data, s_in_data + 256);
    const std::vector<float> s_out_host(s_out_data, s_out_data + 128);
    const std::vector<int16_t> dep_host(dep_data, dep_data + 16*256);
    const std::vector<float> x_host(x_data, x_data + 256*2);

    ggml_tensor * lut = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 65536);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * moe_rin = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 256);
    ggml_tensor * moe_rout = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 128);
    for (int i = 0; i < 256; ++i) {
        ((ggml_fp16_t *) moe_rin->data)[i] = ggml_fp32_to_fp16(s_in_data[i]);
    }
    for (int i = 0; i < 128; ++i) {
        ((ggml_fp16_t *) moe_rout->data)[i] = ggml_fp32_to_fp16(s_out_data[i]);
    }
    std::memset(lut->data, 0, ggml_nbytes(lut));
    std::memset(ids->data, 0, ggml_nbytes(ids));
    ggml_tensor * moe = ggml_escha_moe(ctx, code, moe_rin, moe_rout, lut, dep, x, ids);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, result);
    ggml_build_forward_expand(graph, moe);
    ggml_backend_t backend = ggml_backend_cpu_init();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    bool equivalent = status == GGML_STATUS_SUCCESS;
    const float * dense_out = (const float *) result->data;
    const float * moe_out = (const float *) moe->data;
    const std::vector<float> dense_ref(dense_out, dense_out + 128*2);
    for (int i = 0; equivalent && i < 128*2; ++i) {
        equivalent = std::fabs(dense_out[i] - moe_out[i]) <= 2e-3f*(1.0f + std::fabs(dense_out[i]));
    }
    ggml_backend_free(backend);
    if (!equivalent) {
        std::fprintf(stderr, "dense ESCHA CPU result differs from zero-id MOE\n");
    }

    if (equivalent && ggml_backend_cuda_get_device_count() > 0) {
        ggml_init_params cuda_params = {
            /*.mem_size   =*/ 1 << 20,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * cuda_ctx = ggml_init(cuda_params);
        ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
        ggml_backend_buffer_t cuda_buffer = nullptr;
        ggml_tensor * cuda_result = nullptr;
        ggml_tensor * cuda_code = nullptr;
        ggml_tensor * cuda_rin = nullptr;
        ggml_tensor * cuda_rout = nullptr;
        ggml_tensor * cuda_s_in = nullptr;
        ggml_tensor * cuda_s_out = nullptr;
        ggml_tensor * cuda_dep = nullptr;
        ggml_tensor * cuda_x = nullptr;
        bool cuda_ready = cuda_ctx != nullptr && cuda_backend != nullptr;
        if (cuda_ready) {
            cuda_code = ggml_new_tensor_3d(cuda_ctx, GGML_TYPE_I16, 32, 8, 16);
            cuda_rin  = ggml_new_tensor_1d(cuda_ctx, GGML_TYPE_F16, 256);
            cuda_rout = ggml_new_tensor_1d(cuda_ctx, GGML_TYPE_F16, 128);
            cuda_s_in = ggml_new_tensor_1d(cuda_ctx, GGML_TYPE_F32, 256);
            cuda_s_out = ggml_new_tensor_1d(cuda_ctx, GGML_TYPE_F32, 128);
            cuda_dep  = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_I16, 16, 256);
            cuda_x    = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, 256, 2);
            cuda_result = ggml_escha_linear(cuda_ctx, cuda_code, cuda_rin, cuda_rout,
                cuda_s_in, cuda_s_out, cuda_dep, cuda_x);
            cuda_buffer = ggml_backend_alloc_ctx_tensors(cuda_ctx, cuda_backend);
            cuda_ready = cuda_result != nullptr && cuda_buffer != nullptr;
        }
        if (cuda_ready) {
            ggml_backend_tensor_set(cuda_code, code_host.data(), 0, ggml_nbytes(cuda_code));
            ggml_backend_tensor_set(cuda_rin, rin_host.data(), 0, ggml_nbytes(cuda_rin));
            ggml_backend_tensor_set(cuda_rout, rout_host.data(), 0, ggml_nbytes(cuda_rout));
            ggml_backend_tensor_set(cuda_s_in, s_in_host.data(), 0, ggml_nbytes(cuda_s_in));
            ggml_backend_tensor_set(cuda_s_out, s_out_host.data(), 0, ggml_nbytes(cuda_s_out));
            ggml_backend_tensor_set(cuda_dep, dep_host.data(), 0, ggml_nbytes(cuda_dep));
            ggml_backend_tensor_set(cuda_x, x_host.data(), 0, ggml_nbytes(cuda_x));

            ggml_cgraph * cuda_graph = ggml_new_graph_custom(cuda_ctx, 16, false);
            ggml_build_forward_expand(cuda_graph, cuda_result);
            const ggml_status cuda_status = ggml_backend_graph_compute(cuda_backend, cuda_graph);
            std::vector<float> cuda_out(128*2);
            ggml_backend_tensor_get(cuda_result, cuda_out.data(), 0, ggml_nbytes(cuda_result));
            equivalent = cuda_status == GGML_STATUS_SUCCESS;
            for (size_t i = 0; equivalent && i < cuda_out.size(); ++i) {
                equivalent = std::fabs(cuda_out[i] - dense_ref[i]) <= 1e-4f*(1.0f + std::fabs(dense_ref[i]));
            }
        }

        if (cuda_buffer != nullptr) {
            ggml_backend_buffer_free(cuda_buffer);
        }
        if (cuda_backend != nullptr) {
            ggml_backend_free(cuda_backend);
        }
        ggml_free(cuda_ctx);
        if (!equivalent) {
            std::fprintf(stderr, "dense ESCHA CUDA result differs from CPU\n");
        }
    }

    ggml_free(ctx);
    return equivalent ? 0 : 1;
}
