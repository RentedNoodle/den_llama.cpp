/**
 * profile_launch_overhead.cu — Minimal CUDA driver launch overhead profiler
 *
 * Measures pure kernel launch overhead: CPU time to enqueue 1000 empty kernels,
 * GPU time on stream, and launch + sync overhead.
 *
 * BUILD (Windows, pip nvcc — NO CUPTI, NO WSL):
 *   C:\Users\james\AppData\Local\Programs\Python\Python314\Lib\site-packages\nvidia\cu13\bin\nvcc.exe ^
 *     -arch=sm_120a -o profile_launch_overhead.exe profile_launch_overhead.cu
 *
 * BUILD (WSL, CUDA 13.3):
 *   /usr/local/cuda-13.3/bin/nvcc -arch=sm_120a -o profile_launch_overhead profile_launch_overhead.cu
 *
 * USAGE:
 *   profile_launch_overhead.exe
 *
 * OUTPUT (example, RTX 5070 Ti):
 *   empty kernel x 10000, no events: 12500 us total, 1.25 us/launch
 *   empty kernel x 10000, with events: 21000 us total, 2.10 us/launch (events add 0.85 us each)
 *   single-kernel GPU time (event pair): 3.20 us
 *   DRIVER OVERHEAD (per launch, no events): ~1.25 us
 */

#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

__global__ void empty_kernel() {}

int main() {
    const int N = 10000;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ── Warmup ─────────────────────────────────────────────────────
    for (int i = 0; i < 10; i++)
        empty_kernel<<<1, 1, 0, stream>>>();
    cudaStreamSynchronize(stream);

    // ── Test 1: pure launch loop (no events) ───────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        empty_kernel<<<1, 1, 0, stream>>>();
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double no_event_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // ── Test 2: launch loop WITH cudaEvent record ──────────────────
    cudaEvent_t ev;
    cudaEventCreate(&ev);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        empty_kernel<<<1, 1, 0, stream>>>();
        cudaEventRecord(ev, stream);
    }
    cudaStreamSynchronize(stream);
    auto t3 = std::chrono::high_resolution_clock::now();
    double with_event_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

    // ── Test 3: single kernel GPU time (start→end event pair) ─────
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start, stream);
    empty_kernel<<<1, 1, 0, stream>>>();
    cudaEventRecord(ev_stop, stream);
    cudaStreamSynchronize(stream);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);

    // ── Test 4: measure launch-to-launch gap GPU-side ──────────────
    // Record start of kernel #1 → end of kernel #1 → start of kernel #2
    // Gap = start_of_kernel_2 - end_of_kernel_1 (GPU time)
    cudaEvent_t a, b, c;
    cudaEventCreate(&a); cudaEventCreate(&b); cudaEventCreate(&c);

    cudaEventRecord(a, stream);  // before launch
    empty_kernel<<<1, 1, 0, stream>>>();
    cudaEventRecord(b, stream);  // after first kernel ends
    empty_kernel<<<1, 1, 0, stream>>>();
    cudaEventRecord(c, stream);  // after second kernel ends
    cudaStreamSynchronize(stream);

    float kernel1_ms = 0, gap_ms = 0, kernel2_ms = 0;
    cudaEventElapsedTime(&kernel1_ms, a, b);
    cudaEventElapsedTime(&gap_ms, b, c);
    // kernel2 duration ≈ kernel1, launch gap is gap_ms - kernel1_ms
    float launch_gap_us = (gap_ms - kernel1_ms) * 1000.0f;

    // ── Report ─────────────────────────────────────────────────────
    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  CUDA DRIVER LAUNCH OVERHEAD — standalone measurement\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("  Test 1: %d empty kernel launches (no events)\n", N);
    printf("    CPU wall time:  %.1f us total, %.3f us/launch\n",
           no_event_us, no_event_us / N);
    printf("    This is BEST CASE launch overhead (CPU-side).\n");
    printf("\n");
    printf("  Test 2: %d empty kernel launches (with cudaEvent)\n", N);
    printf("    CPU wall time:  %.1f us total, %.3f us/launch\n",
           with_event_us, with_event_us / N);
    printf("    Event overhead:  %.3f us/event\n",
           (with_event_us - no_event_us) / N);
    printf("\n");
    printf("  Test 3: single empty kernel GPU execution time\n");
    printf("    GPU time:       %.3f us (event pair)\n", gpu_ms * 1000.0f);
    printf("\n");
    printf("  Test 4: GPU-side launch gap (end_k1 to start_k2)\n");
    printf("    Kernel 1 GPU:   %.3f us\n", kernel1_ms * 1000.0f);
    printf("    Launch gap:     %.3f us (GPU idle between launches)\n",
           launch_gap_us);
    printf("\n");
    printf("  ─── ESTIMATE FOR LLM DECODE ───\n");
    printf("  Per-launch driver overhead:   ~%.2f us\n", no_event_us / N);
    printf("  35B MoE decode (~1640 launches/token):\n");
    printf("    Est. overhead:  ~%.3f ms/token\n", no_event_us / N * 1640 / 1000);
    printf("  At 160 tok/s target (6.25ms/token budget):\n");
    printf("    Overhead %%:     ~%.1f%%\n",
           (no_event_us / N * 1640 / 1000) / 6.25 * 100);
    printf("\n");
    printf("  ─── INTERPRETATION ───\n");
    printf("  <1 us/launch  = EXCELLENT (driver optimized)\n");
    printf("  1-3 us/launch = NORMAL (typical consumer GPU)\n");
    printf("  3-10 us/launch = HIGH (consider CUDA graphs)\n");
    printf("  >10 us/launch = CRITICAL (driver issue, WDDM/TDR?)\n");
    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");

    // Cleanup
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaEventDestroy(ev);
    cudaEventDestroy(a); cudaEventDestroy(b); cudaEventDestroy(c);
    cudaStreamDestroy(stream);

    return 0;
}
