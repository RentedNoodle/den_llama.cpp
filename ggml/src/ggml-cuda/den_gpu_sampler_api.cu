// den_gpu_sampler_api.cu — host API bridge for the GPU-resident argmax sampler.
// Exposes den_gpu_argmax (a greedy on-device argmax over the vocab) to llama.cpp
// so the per-token CPU sampler + 608KB logits D2H can be replaced by a 4-byte
// token readback (U3: the host-serialization fix, 2026-08-03).
#include "den_gpu_sampler.cuh"
#include "ggml-cuda.h"

extern "C" GGML_CALL void den_gpu_argmax(
    const float * logits, uint32_t * token_out_host, int vocab_size, cudaStream_t stream) {
    if (!logits || !token_out_host || vocab_size <= 0) return;
    static uint32_t * s_token_dev = nullptr;
    if (!s_token_dev) {
        cudaMalloc(&s_token_dev, sizeof(uint32_t));
    }
    GPUSamplerConfig cfg;
    cfg.temperature = 0.0f;  // greedy
    cfg.top_k       = 1;
    cfg.vocab_size  = vocab_size;
    cfg.seed        = 0;     // auto-seed from clock64 on device
    den_gpu_greedy_kernel<<<1, GPU_SAMPLE_THREADS, 0, stream>>>(logits, s_token_dev, cfg);
    cudaMemcpy(token_out_host, s_token_dev, sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
