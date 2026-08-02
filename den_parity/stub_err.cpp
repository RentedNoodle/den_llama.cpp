#include <cstdio>
void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line, const char* msg) {
    fprintf(stderr, "CUDA error %s at %s:%d %s: %s\n", stmt, file, line, func, msg);
}
