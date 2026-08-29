#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "testing.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int64_t CACHE_WIDTH = 256*4;
constexpr int64_t CACHE_ROWS = 512;
constexpr int64_t UPDATE_ROWS = 3;

std::vector<uint8_t> run_set_rows(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t cache_buft,
        ggml_type cache_type) {
    constexpr size_t N_TENSORS = 16;
    const size_t context_bytes = ggml_tensor_overhead()*N_TENSORS +
        ggml_graph_overhead_custom(N_TENSORS, false);
    const ggml_init_params params{context_bytes, nullptr, true};

    ggml_context_ptr cache_ctx(ggml_init(params));
    ggml_context_ptr compute_ctx(ggml_init(params));
    GGML_ASSERT(cache_ctx && compute_ctx);

    ggml_tensor * cache = ggml_new_tensor_2d(cache_ctx.get(), cache_type, CACHE_WIDTH, CACHE_ROWS);
    ggml_tensor * values = ggml_new_tensor_2d(compute_ctx.get(), GGML_TYPE_F32, CACHE_WIDTH, UPDATE_ROWS);
    ggml_tensor * indices = ggml_new_tensor_1d(compute_ctx.get(), GGML_TYPE_I32, UPDATE_ROWS);
    ggml_tensor * updated = ggml_set_rows(compute_ctx.get(), cache, values, indices);

    ggml_backend_buffer_ptr cache_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(cache_ctx.get(), cache_buft));
    ggml_backend_buffer_ptr compute_buffer(
        ggml_backend_alloc_ctx_tensors(compute_ctx.get(), backend));
    GGML_ASSERT(cache_buffer && compute_buffer);
    ggml_backend_buffer_clear(cache_buffer.get(), 0);

    std::vector<float> source(CACHE_WIDTH*UPDATE_ROWS);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = 0.5f*std::sin(float(i)*0.01953125f) - 0.25f*std::cos(float(i)*0.00390625f);
    }
    const std::vector<int32_t> rows{1, 257, 511};
    ggml_backend_tensor_set(values, source.data(), 0, source.size()*sizeof(float));
    ggml_backend_tensor_set(indices, rows.data(), 0, rows.size()*sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph_custom(compute_ctx.get(), N_TENSORS, false);
    ggml_build_forward_expand(graph, updated);
    GGML_ASSERT(ggml_backend_supports_op(backend, updated));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<uint8_t> result(ggml_nbytes(cache));
    ggml_backend_tensor_get(cache, result.data(), 0, result.size());
    return result;
}

} // namespace

int main() {
    testing t;

    t.test("mapped authoritative cache SET_ROWS matches CUDA quantization", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        ggml_backend_cuda_kv_stream_params params{};
        params.device = 0;
        params.stage_bytes = 512*1024;
        params.stage_slots = 1;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        for (ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
            const std::vector<uint8_t> expected = run_set_rows(
                backend.get(), ggml_backend_get_default_buffer_type(backend.get()), type);
            const std::vector<uint8_t> actual = run_set_rows(
                backend.get(), ggml_backend_cuda_kv_stream_buffer_type(runtime), type);
            t.assert_true(
                type == GGML_TYPE_Q8_0 ? "Q8 rows and untouched bytes match" : "Q4 rows and untouched bytes match",
                expected == actual);
        }

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    return t.summary();
}
