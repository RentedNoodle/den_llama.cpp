/**
 * probe_noc_topology.cu — GB203 NoC / Crossbar Topology Probe
 *
 * Maps the SM-to-SM communication latency matrix and SM-to-L2-slice
 * affinity on Blackwell consumer silicon (RTX 5070 Ti, sm_120a, 70 SMs).
 *
 * Two tests:
 *   TEST 1: SM↔SM ping-pong latency matrix (70×70)
 *   TEST 2: SM→L2 slice affinity via strided load latency
 *
 * Build:
 *   nvcc -arch=sm_120a -o probe_noc_topology.exe probe_noc_topology.cu
 *
 * Run:
 *   probe_noc_topology.exe
 *
 * Output files:
 *   noc_sm_latency_cycles.csv   — 70×70 raw min-cycles matrix
 *   noc_sm_latency_us.csv       — 70×70 in microseconds
 *   noc_sm_latency_report.txt   — human-readable summary
 *   l2_affinity_cycles.csv      — SM×stride load latencies
 *   l2_affinity_report.txt      — per-SM home slice
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ──────────────────────────────────────────────────────────────
// Hardware constants (GB203 / RTX 5070 Ti)
// ──────────────────────────────────────────────────────────────

#define N_SM           70          // GB203 SMs (total)
#define N_SAMPLE_SMS   23          // sampled: every 3rd SM (70/3 = 23)
#define N_SM_MAX       128         // safe array sizing
#define N_ITERS        3           // ping-pong iterations per pair (reduced from 10)
#define N_PAIRS        ((N_SAMPLE_SMS) * ((N_SAMPLE_SMS) - 1) / 2)  // 253
#define CLOCK_MHZ      2482        // RTX 5070 Ti boost clock (approx)
#define TIMEOUT_CYCLES 24820000ULL // 10ms timeout per pair @ 2482 MHz
#define PROGRESS_EVERY 50          // print progress every N pairs
#define L2_SLICE_STRIDE 256        // bytes — one cache line maps to one slice
#define N_L2_STRIDES   256         // probe up to 256 slices (overprovision)

// ──────────────────────────────────────────────────────────────
// Test 1: SM↔SM ping-pong latency
// ──────────────────────────────────────────────────────────────

/**
 * Ping-pong atomic protocol between SM pairs.
 *
 * Each pair (src,dst) is tested sequentially. The block on SM 0 acts
 * as coordinator, publishing the current pair and advancing an epoch
 * counter that all other SMs spin on. Within each pair epoch, the
 * source SM does an atomicAdd on the destination's pong flag, then
 * spins waiting for its own ping flag to be set. The destination SM
 * spins waiting for its pong flag, then does atomicAdd on the source's
 * ping flag. The coordinator SM waits for ping[src] to be set, then
 * advances to the next pair.
 *
 * Epoch numbering: pair p of iteration it has go_epoch = it*2*N_PAIRS + 2*p + 1.
 * Coordinator sets epoch to go_epoch after publishing src/dst; other SMs
 * wait for epoch >= go_epoch before reading the pair.
 */

__global__ void noc_sm_pingpong(
    uint64_t*    latency,      // [N_SM_MAX * N_SM_MAX] — min cycles
    uint32_t*    ping,         // [N_SM_MAX] — ping flag per SM
    uint32_t*    pong,         // [N_SM_MAX] — pong flag per SM
    uint32_t*    epoch,        // single counter
    uint32_t*    cur_src,      // current source SM (published by coordinator)
    uint32_t*    cur_dst,      // current destination SM
    const uint32_t* pairs_src, // [N_PAIRS] — precomputed pair sources (sampled SMs)
    const uint32_t* pairs_dst, // [N_PAIRS] — precomputed pair destinations (sampled SMs)
    uint32_t*    skipped       // [1] — count of timed-out pairs
) {
    unsigned sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id >= N_SM_MAX) return;

    for (int it = 0; it < N_ITERS; it++) {
        for (int p = 0; p < N_PAIRS; p++) {
            uint32_t go_epoch = (uint32_t)(it * 2ULL * N_PAIRS + 2ULL * p + 1);
            uint32_t src, dst;
            int timed_out = 0;

            if (sm_id == 0) {
                // ── Coordinator: publish pair, advance epoch ──
                src = pairs_src[p];
                dst = pairs_dst[p];
                ping[src] = 0;
                ping[dst] = 0;
                pong[src] = 0;
                pong[dst] = 0;
                cur_src[0] = src;
                cur_dst[0] = dst;
                __threadfence();
                epoch[0] = go_epoch;

                // ── Progress print ──
                if (it == 0 && (p % PROGRESS_EVERY) == 0) {
                    printf("  pair %d/%d done\n", p, N_PAIRS);
                }
            } else {
                // ── Wait for coordinator to publish ──
                while (atomicAdd(epoch, 0) < go_epoch) {
                    __threadfence();
                }
                src = atomicAdd(cur_src, 0);
                dst = atomicAdd(cur_dst, 0);
            }

            // ── Ping-pong (with timeout on busy SMs) ──
            if (sm_id == src) {
                uint64_t t0 = clock64();
                atomicAdd(&pong[dst], 1);               // PING
                while (atomicAdd(&ping[src], 0) == 0) { // wait PONG
                    __threadfence();
                    // Timeout: if dst SM is busy (WDDM/display), bail after 10ms
                    if ((clock64() - t0) > TIMEOUT_CYCLES) {
                        timed_out = 1;
                        break;
                    }
                }
                if (!timed_out) {
                    uint64_t t1 = clock64();
                    uint64_t dt  = t1 - t0;
                    uint64_t idx = (uint64_t)src * N_SM_MAX + dst;
                    if (it == 0 || dt < latency[idx]) {
                        latency[idx] = dt;
                    }
                }
            } else if (sm_id == dst) {
                uint64_t t_wait = clock64();
                while (atomicAdd(&pong[dst], 0) == 0) { // wait PING
                    __threadfence();
                    if ((clock64() - t_wait) > TIMEOUT_CYCLES) {
                        timed_out = 1;
                        break;
                    }
                }
                if (!timed_out) {
                    atomicAdd(&ping[src], 1);           // PONG
                }
            }
            // other SMs: nothing to do

            // ── Coordinator: wait for ping-pong completion (with timeout) ──
            if (sm_id == 0) {
                uint64_t t_wait = clock64();
                while (atomicAdd(&ping[src], 0) == 0) {
                    __threadfence();
                    if ((clock64() - t_wait) > TIMEOUT_CYCLES) {
                        timed_out = 1;
                        break;
                    }
                }
                if (timed_out) {
                    atomicAdd(skipped, 1);
                    // Reset epoch so other SMs don't spin on stale state
                    epoch[0] = go_epoch + 1;
                }
            }
        }
    }
    // Final progress print for last batch
    if (sm_id == 0) {
        printf("  pair %d/%d done (all complete)\n", N_PAIRS, N_PAIRS);
    }
}

// ──────────────────────────────────────────────────────────────
// Test 2: SM→L2 slice affinity via strided load latency
// ──────────────────────────────────────────────────────────────

/**
 * Each SM loads from every stride offset and records clock64() delta.
 * Stride = L2_SLICE_STRIDE bytes ensures each access targets a
 * different L2 slice (hash-distributed by GPU MMU).
 *
 * We do multiple samples per stride and take the minimum to filter
 * scheduling noise. Before each timed load, we touch a distant
 * address to evict the target line from L1.
 */

__global__ void noc_l2_affinity(
    uint64_t*       latencies,   // [N_SM_MAX * N_L2_STRIDES] — min cycles
    volatile uint32_t* buf,      // large buffer: buf[offset] at each stride
    int             buf_elems    // total uint32 elements in buf
) {
    unsigned sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id >= N_SM_MAX) return;

    const int elems_per_stride = L2_SLICE_STRIDE / sizeof(uint32_t); // 64
    const int n_strides        = buf_elems / elems_per_stride;
    if (n_strides > N_L2_STRIDES) return; // safety

    // Warm up: touch all strides to ensure pages are mapped
    for (int s = 0; s < n_strides; s++) {
        volatile uint32_t tmp = buf[s * elems_per_stride];
        (void)tmp;
    }
    __threadfence();

    // Measure: for each stride, do multiple samples, take min
    const int samples_per_stride = 16;

    for (int s = 0; s < n_strides; s++) {
        int offset = s * elems_per_stride;
        uint64_t best = UINT64_MAX;

        for (int sample = 0; sample < samples_per_stride; sample++) {
            // Evict L1: touch a far-away line
            int far_offset = ((s + 1) % n_strides) * elems_per_stride;
            volatile uint32_t evict = buf[far_offset];
            (void)evict;
            __threadfence();

            uint64_t t0 = clock64();
            volatile uint32_t val = buf[offset];
            (void)val;
            uint64_t t1 = clock64();

            // Only count if we got a reasonable measurement
            uint64_t dt = t1 - t0;
            if (dt > 10 && dt < best) { // filter clock64() artifacts
                best = dt;
            }
        }

        if (best != UINT64_MAX) {
            latencies[(uint64_t)sm_id * N_L2_STRIDES + s] = best;
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Host: pair index computation
// ──────────────────────────────────────────────────────────────

/**
 * Compute (src, dst) from a linear pair index [0, N*(N-1)/2).
 * Pairs are enumerated row-major: (0,1),(0,2),...,(0,N-1),(1,2),...
 */
static void pair_from_index(uint32_t idx, uint32_t n, uint32_t* src, uint32_t* dst) {
    // k = largest integer such that k*(2*n - k - 1)/2 <= idx
    // src = k, dst = src + 1 + (idx - k*(2*n - k - 1)/2)
    uint32_t k = 0;
    uint32_t accum = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t next = accum + (n - i - 1);
        if (idx < next) {
            k = i;
            break;
        }
        accum = next;
    }
    *src = k;
    *dst = k + 1 + (idx - accum);
}

// ──────────────────────────────────────────────────────────────
// CUDA error checking
// ──────────────────────────────────────────────────────────────

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while (0)

// ──────────────────────────────────────────────────────────────
// Output helpers
// ──────────────────────────────────────────────────────────────

static void write_csv_matrix(const char* path,
                              const uint64_t* data,
                              int rows, int cols, int stride,
                              double scale) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }
    fprintf(f, "SM");
    for (int c = 0; c < cols; c++) fprintf(f, ",%d", c);
    fprintf(f, "\n");
    for (int r = 0; r < rows; r++) {
        fprintf(f, "%d", r);
        for (int c = 0; c < cols; c++) {
            uint64_t v = data[(uint64_t)r * stride + c];
            if (v == 0 || v == UINT64_MAX) {
                fprintf(f, ",");
            } else if (scale != 1.0) {
                fprintf(f, ",%.4f", (double)v * scale);
            } else {
                fprintf(f, ",%llu", (unsigned long long)v);
            }
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("  Wrote %s\n", path);
}

static void write_report_sm_latency(const char* path,
                                     const uint64_t* latency,
                                     double clock_mhz) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }

    fprintf(f, "=== GB203 NoC SM-to-SM Latency Report ===\n\n");
    fprintf(f, "GPU clock: %.0f MHz (%.2f ns/cycle)\n", clock_mhz, 1000.0 / clock_mhz);
    fprintf(f, "SMs: %d (sampled: %d, every 3rd)  Pairs measured: %d  Iterations per pair: %d\n\n",
            N_SM, N_SAMPLE_SMS, N_PAIRS, N_ITERS);

    // Compute stats
    uint64_t total = 0, min_val = UINT64_MAX, max_val = 0;
    int count = 0;
    for (int i = 0; i < N_SM; i++) {
        for (int j = 0; j < N_SM; j++) {
            uint64_t v = latency[(uint64_t)i * N_SM_MAX + j];
            if (v > 0) {
                total += v;
                count++;
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
        }
    }
    double avg_cycles = (count > 0) ? (double)total / count : 0;
    fprintf(f, "Global stats (all directed pairs):\n");
    fprintf(f, "  Min:   %llu cycles (%.3f us)\n",
            (unsigned long long)min_val, (double)min_val * 1000.0 / clock_mhz);
    fprintf(f, "  Max:   %llu cycles (%.3f us)\n",
            (unsigned long long)max_val, (double)max_val * 1000.0 / clock_mhz);
    fprintf(f, "  Avg:   %.1f cycles (%.3f us)\n", avg_cycles, avg_cycles * 1000.0 / clock_mhz);
    fprintf(f, "  Count: %d\n\n", count);

    // Per-SM average outgoing latency
    fprintf(f, "Per-SM average OUTGOING latency (us):\n");
    fprintf(f, "  SM  AvgOut_us  MinOut_us  MaxOut_us\n");
    for (int i = 0; i < N_SM; i++) {
        uint64_t sm_total = 0, sm_min = UINT64_MAX, sm_max = 0;
        int sm_count = 0;
        for (int j = 0; j < N_SM; j++) {
            uint64_t v = latency[(uint64_t)i * N_SM_MAX + j];
            if (v > 0) {
                sm_total += v;
                sm_count++;
                if (v < sm_min) sm_min = v;
                if (v > sm_max) sm_max = v;
            }
        }
        if (sm_count > 0) {
            fprintf(f, "  %3d  %9.3f  %9.3f  %9.3f\n",
                    i,
                    (double)sm_total / sm_count * 1000.0 / clock_mhz,
                    (double)sm_min * 1000.0 / clock_mhz,
                    (double)sm_max * 1000.0 / clock_mhz);
        }
    }

    // Closest 5 neighbors per SM (lowest latency outgoing)
    fprintf(f, "\nTop-5 nearest neighbors per SM (by outgoing latency):\n");
    for (int i = 0; i < N_SM; i++) {
        fprintf(f, "  SM %2d → ", i);
        // selection sort for top 5 from this row
        struct { uint32_t dst; uint64_t lat; } top[5];
        int top_n = 0;
        for (int j = 0; j < N_SM; j++) {
            if (j == i) continue;
            uint64_t v = latency[(uint64_t)i * N_SM_MAX + j];
            if (v == 0) continue;
            // insert sorted
            int pos = top_n;
            while (pos > 0 && v < top[pos - 1].lat) pos--;
            if (pos < 5) {
                if (top_n < 5) top_n++;
                for (int k = top_n - 1; k > pos; k--) top[k] = top[k - 1];
                top[pos].dst = (uint32_t)j;
                top[pos].lat = v;
            }
        }
        for (int k = 0; k < top_n; k++) {
            fprintf(f, "SM%d(%.3fus) ", top[k].dst,
                    (double)top[k].lat * 1000.0 / clock_mhz);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    printf("  Wrote %s\n", path);
}

static void write_report_l2_affinity(const char* path,
                                      const uint64_t* latencies,
                                      double clock_mhz) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }

    fprintf(f, "=== GB203 SM → L2 Slice Affinity Report ===\n\n");
    fprintf(f, "Stride: %d bytes  Strides probed: %d  Clock: %.0f MHz\n\n",
            L2_SLICE_STRIDE, N_L2_STRIDES, clock_mhz);

    // For each SM, find the stride with minimum latency (its home slice)
    fprintf(f, "Per-SM home L2 slice (lowest load latency):\n");
    fprintf(f, "  SM   HomeSlice  Latency(us)  Latency(cycles)\n");

    int home_slice_counts[N_L2_STRIDES] = {0};

    for (int i = 0; i < N_SM; i++) {
        uint64_t best_lat = UINT64_MAX;
        int      best_slice = -1;
        for (int s = 0; s < N_L2_STRIDES; s++) {
            uint64_t v = latencies[(uint64_t)i * N_L2_STRIDES + s];
            if (v > 0 && v < best_lat) {
                best_lat = v;
                best_slice = s;
            }
        }
        if (best_slice >= 0) {
            fprintf(f, "  %3d  %10d  %11.4f  %15llu\n",
                    i, best_slice,
                    (double)best_lat * 1000.0 / clock_mhz,
                    (unsigned long long)best_lat);
            if (best_slice < N_L2_STRIDES) {
                home_slice_counts[best_slice]++;
            }
        }
    }

    // Slice occupancy
    fprintf(f, "\nL2 slice occupancy (number of SMs that prefer each slice):\n");
    for (int s = 0; s < N_L2_STRIDES; s++) {
        if (home_slice_counts[s] > 0) {
            fprintf(f, "  Slice %3d: %d SMs\n", s, home_slice_counts[s]);
        }
    }

    // Detect topology pattern
    fprintf(f, "\nTopology inference:\n");
    int total_assigned = 0;
    int max_per_slice = 0;
    int slices_used = 0;
    for (int s = 0; s < N_L2_STRIDES; s++) {
        if (home_slice_counts[s] > 0) {
            slices_used++;
            total_assigned += home_slice_counts[s];
            if (home_slice_counts[s] > max_per_slice)
                max_per_slice = home_slice_counts[s];
        }
    }
    fprintf(f, "  Total SMs assigned: %d\n", total_assigned);
    fprintf(f, "  L2 slices in use: %d\n", slices_used);
    fprintf(f, "  Max SMs per slice: %d\n", max_per_slice);
    if (slices_used > 0) {
        fprintf(f, "  Avg SMs per slice: %.1f\n", (double)total_assigned / slices_used);
    }
    fprintf(f, "  Pattern: %s\n",
            slices_used == N_SM ? "1:1 (one SM per L2 slice)" :
            slices_used == N_SM / 2 ? "2:1 (two SMs per L2 slice)" :
            slices_used == N_SM / 4 ? "4:1 (four SMs per L2 slice)" :
            "custom mapping (see per-SM table)");

    fclose(f);
    printf("  Wrote %s\n", path);
}

// ──────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    printf("=== GB203 NoC / Crossbar Topology Probe ===\n");
    printf("Target: RTX 5070 Ti (sm_120a, %d SMs)\n", N_SM);
    printf("Clock: ~%d MHz (assumed)\n\n", CLOCK_MHZ);

    // ── Device info ──
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
    printf("Device: %s\n", props.name);
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    printf("SMs: %d  Clock: %d MHz (rated)\n\n",
           props.multiProcessorCount, clock_khz / 1000);

    // Use actual clock rate if available
    double clock_mhz = (clock_khz > 0) ? (double)clock_khz / 1000.0 : CLOCK_MHZ;

    // ═══════════════════════════════════════
    // TEST 1: SM↔SM Ping-Pong Latency Matrix
    // ═══════════════════════════════════════
    printf("═══ Test 1: SM↔SM Ping-Pong Latency ═══\n\n");

    // Build sampled SM list: every 3rd SM (always include SM 0)
    uint32_t sampled_sm[N_SAMPLE_SMS];
    for (uint32_t i = 0; i < N_SAMPLE_SMS; i++) {
        sampled_sm[i] = i * 3;  // SM 0, 3, 6, 9, ..., 66
    }
    printf("Sampled %u SMs out of %u (every 3rd): ", N_SAMPLE_SMS, N_SM);
    for (uint32_t i = 0; i < N_SAMPLE_SMS; i++) printf("%u ", sampled_sm[i]);
    printf("\n");

    // Precompute pair arrays on host (pairs of sampled SM indices)
    uint32_t* h_pairs_src = (uint32_t*)malloc(N_PAIRS * sizeof(uint32_t));
    uint32_t* h_pairs_dst = (uint32_t*)malloc(N_PAIRS * sizeof(uint32_t));
    if (!h_pairs_src || !h_pairs_dst) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }
    for (uint32_t p = 0; p < N_PAIRS; p++) {
        uint32_t a, b;
        pair_from_index(p, N_SAMPLE_SMS, &a, &b);
        h_pairs_src[p] = sampled_sm[a];  // map to actual SM ID
        h_pairs_dst[p] = sampled_sm[b];
    }
    printf("Pair index computed: %u pairs\n", N_PAIRS);

    // Allocate device memory for Test 1
    uint64_t* d_latency;
    uint32_t* d_ping, *d_pong, *d_epoch, *d_cur_src, *d_cur_dst;
    uint32_t* d_pairs_src, *d_pairs_dst;
    uint32_t* d_skipped;

    CUDA_CHECK(cudaMalloc(&d_latency,  N_SM_MAX * N_SM_MAX * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ping,     N_SM_MAX * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_pong,     N_SM_MAX * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_epoch,    sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_cur_src,  sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_cur_dst,  sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_skipped,  sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_pairs_src, N_PAIRS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_pairs_dst, N_PAIRS * sizeof(uint32_t)));

    // Initialize
    CUDA_CHECK(cudaMemset(d_latency,  0, N_SM_MAX * N_SM_MAX * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_ping,     0, N_SM_MAX * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_pong,     0, N_SM_MAX * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_epoch,    0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_skipped,  0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_pairs_src, h_pairs_src, N_PAIRS * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pairs_dst, h_pairs_dst, N_PAIRS * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // No warmup — SM ping-pong is latency-bound, not memory/cache-bound.
    // Measurements start immediately.

    printf("Running ping-pong probe (%d iterations x %d sampled pairs)...\n",
           N_ITERS, N_PAIRS);

    cudaEvent_t t1_start, t1_stop;
    CUDA_CHECK(cudaEventCreate(&t1_start));
    CUDA_CHECK(cudaEventCreate(&t1_stop));
    CUDA_CHECK(cudaEventRecord(t1_start));

    noc_sm_pingpong<<<N_SM, 1>>>(d_latency, d_ping, d_pong, d_epoch,
                                   d_cur_src, d_cur_dst,
                                   d_pairs_src, d_pairs_dst,
                                   d_skipped);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(t1_stop));
    CUDA_CHECK(cudaEventSynchronize(t1_stop));
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t1_start, t1_stop));
    printf("Done in %.2f seconds\n\n", elapsed_ms / 1000.0f);

    // Report skipped pairs
    uint32_t h_skipped = 0;
    CUDA_CHECK(cudaMemcpy(&h_skipped, d_skipped, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    if (h_skipped > 0) {
        printf("  ⚠ %u pairs timed out (skipped — likely WDDM-occupied SMs)\n\n", h_skipped);
    }

    // Copy results to host
    uint64_t* h_latency = (uint64_t*)calloc(N_SM_MAX * N_SM_MAX, sizeof(uint64_t));
    CUDA_CHECK(cudaMemcpy(h_latency, d_latency,
                          N_SM_MAX * N_SM_MAX * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    // Write outputs
    printf("Writing Test 1 results:\n");
    write_csv_matrix("noc_sm_latency_cycles.csv",
                     h_latency, N_SM, N_SM, N_SM_MAX, 1.0);
    write_csv_matrix("noc_sm_latency_us.csv",
                     h_latency, N_SM, N_SM, N_SM_MAX, 1000.0 / clock_mhz);
    write_report_sm_latency("noc_sm_latency_report.txt", h_latency, clock_mhz);

    // Cleanup Test 1
    CUDA_CHECK(cudaFree(d_latency));
    CUDA_CHECK(cudaFree(d_ping));
    CUDA_CHECK(cudaFree(d_pong));
    CUDA_CHECK(cudaFree(d_epoch));
    CUDA_CHECK(cudaFree(d_cur_src));
    CUDA_CHECK(cudaFree(d_cur_dst));
    CUDA_CHECK(cudaFree(d_skipped));
    CUDA_CHECK(cudaFree(d_pairs_src));
    CUDA_CHECK(cudaFree(d_pairs_dst));
    free(h_latency);
    free(h_pairs_src);
    free(h_pairs_dst);
    cudaEventDestroy(t1_start);
    cudaEventDestroy(t1_stop);

    // ═══════════════════════════════════════
    // TEST 2: SM→L2 Slice Affinity
    // ═══════════════════════════════════════
    printf("\n═══ Test 2: SM→L2 Slice Affinity ═══\n\n");

    // Allocate a large buffer: ~64 MB, one page per stride
    size_t buf_elems = N_L2_STRIDES * (L2_SLICE_STRIDE / sizeof(uint32_t));
    size_t buf_bytes = buf_elems * sizeof(uint32_t);
    printf("Allocating %.1f MB L2 probe buffer...\n",
           (double)buf_bytes / (1024.0 * 1024.0));

    volatile uint32_t* d_buf;
    CUDA_CHECK(cudaMalloc((void**)&d_buf, buf_bytes));
    // Initialize buffer with non-zero pattern (forces page allocation)
    uint32_t* h_buf = (uint32_t*)malloc(buf_bytes);
    if (h_buf) {
        for (size_t i = 0; i < buf_elems; i++) {
            h_buf[i] = (uint32_t)(i * 0x9E3779B1u); // golden ratio hash
        }
        CUDA_CHECK(cudaMemcpy((void*)d_buf, h_buf, buf_bytes,
                              cudaMemcpyHostToDevice));
        free(h_buf);
    } else {
        CUDA_CHECK(cudaMemset((void*)d_buf, 0xAB, buf_bytes));
    }

    // Allocate latency output
    uint64_t* d_l2_lat;
    CUDA_CHECK(cudaMalloc(&d_l2_lat, N_SM_MAX * N_L2_STRIDES * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_l2_lat, 0, N_SM_MAX * N_L2_STRIDES * sizeof(uint64_t)));

    // Warmup
    printf("Warming up L2 probe...\n");
    noc_l2_affinity<<<N_SM, 1>>>(d_l2_lat, d_buf, (int)buf_elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaMemset(d_l2_lat, 0, N_SM_MAX * N_L2_STRIDES * sizeof(uint64_t)));
    printf("Running L2 affinity probe (16 samples x %d strides)...\n", N_L2_STRIDES);

    cudaEvent_t t2_start, t2_stop;
    CUDA_CHECK(cudaEventCreate(&t2_start));
    CUDA_CHECK(cudaEventCreate(&t2_stop));
    CUDA_CHECK(cudaEventRecord(t2_start));

    noc_l2_affinity<<<N_SM, 1>>>(d_l2_lat, d_buf, (int)buf_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(t2_stop));
    CUDA_CHECK(cudaEventSynchronize(t2_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t2_start, t2_stop));
    printf("Done in %.2f seconds\n\n", elapsed_ms / 1000.0f);

    // Copy results
    uint64_t* h_l2_lat = (uint64_t*)calloc(N_SM_MAX * N_L2_STRIDES, sizeof(uint64_t));
    CUDA_CHECK(cudaMemcpy(h_l2_lat, d_l2_lat,
                          N_SM_MAX * N_L2_STRIDES * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    // Write outputs
    printf("Writing Test 2 results:\n");
    write_csv_matrix("l2_affinity_cycles.csv",
                     h_l2_lat, N_SM, N_L2_STRIDES, N_L2_STRIDES, 1.0);
    write_report_l2_affinity("l2_affinity_report.txt", h_l2_lat, clock_mhz);

    // Cleanup Test 2
    CUDA_CHECK(cudaFree((void*)d_buf));
    CUDA_CHECK(cudaFree(d_l2_lat));
    free(h_l2_lat);
    cudaEventDestroy(t2_start);
    cudaEventDestroy(t2_stop);

    printf("\n═══ All probes complete ═══\n");
    printf("Files written:\n");
    printf("  noc_sm_latency_cycles.csv   — 70×70 matrix, raw clock cycles\n");
    printf("  noc_sm_latency_us.csv       — 70×70 matrix, microseconds\n");
    printf("  noc_sm_latency_report.txt   — summary with per-SM stats\n");
    printf("  l2_affinity_cycles.csv      — SM×stride load latency\n");
    printf("  l2_affinity_report.txt      — per-SM home L2 slice\n");

    return 0;
}
