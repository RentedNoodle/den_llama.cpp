// den_expert_cpu_stubs.cpp — CPU-only stubs for the CUDA expert-offload engine.
// The scheduler (ggml-backend.cpp) references these extern "C" helpers unconditionally,
// but they are only implemented in ggml-cuda/den_expert_offload.cu. When GGML_CUDA is
// OFF the .cu is not compiled, so provide no-op/error stubs to keep the CPU build linkable.
// On CPU-only decode the expert cache is never used (no GPU tensors), so these are inert.
#include <cstddef>

extern "C" {
    int     den_expert_offload_init(int device)          { (void)device; return -1; }
    int     den_expert_h2d_async(int device, void * dst, const void * src, size_t bytes) { (void)device; (void)dst; (void)src; (void)bytes; return -1; }
    int     den_expert_h2d_sync (int device, void * dst, const void * src, size_t bytes) { (void)device; (void)dst; (void)src; (void)bytes; return -1; }
    int     den_expert_d2d_async(int device, void * dst, const void * src, size_t bytes) { (void)device; (void)dst; (void)src; (void)bytes; return -1; }
    void    den_expert_offload_wait(int device)          { (void)device; }
    void    den_expert_d2d_wait (int device)             { (void)device; }
    int64_t den_expert_cache_alloc(int device, size_t bytes) { (void)device; (void)bytes; return -1; }
    void    den_expert_cache_free (int device, int64_t offset, size_t bytes) { (void)device; (void)offset; (void)bytes; }
    void *  den_expert_cache_base(int device)            { (void)device; return nullptr; }
    size_t  den_expert_cache_capacity(int device)        { (void)device; return 0; }
    size_t  den_expert_cache_used(int device)            { (void)device; return 0; }
    void    den_expert_cache_truncate(int device, size_t new_used) { (void)device; (void)new_used; }
    void    den_expert_offload_shutdown(int device)      { (void)device; }
}
