/**
 * profile_driver_overhead.cu — CUPTI Activity-based CUDA driver overhead profiler
 *
 * Measures kernel launch-to-launch gaps during LLM decode to quantify
 * per-launch driver overhead (CPU time lost in cudaLaunchKernel + stream mgmt).
 *
 * BUILD (WSL ext4, CUDA 13.3):
 *   /usr/local/cuda-13.3/bin/nvcc -arch=sm_120a -o profile_driver_overhead \
 *     -I"/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/extras/CUPTI/include" \
 *     -L"/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/extras/CUPTI/lib64" \
 *     -lcupti -lcuda \
 *     profile_driver_overhead.cu
 *
 * USAGE:
 *   ./profile_driver_overhead -- llama-cli -m model.gguf -n 128 -ngl 99 ...
 *   ./profile_driver_overhead --exe llama-cli -- -m model.gguf -n 128 ...
 *
 * OUTPUT:
 *   total_kernels: 164000
 *   total_wall_ms: 543.2
 *   total_gpu_active_ms: 492.1
 *   total_gpu_gap_ms: 51.1
 *   avg_launch_gap_us: 0.31
 *   pct_overhead: 9.4%
 *   verdict: WORTH_OPTIMIZING (5-15%)
 *
 * THRESHOLDS:
 *   <5%  = FINE (driver overhead negligible)
 *   5-15% = WORTH_OPTIMIZING (batch launches, CUDA graphs)
 *   >15% = URGENT (driver is bottleneck, use persistent kernel or graphs)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#define sleep_ms(ms) Sleep(ms)
typedef HANDLE pid_t;
#define get_pid() GetCurrentProcessId()
#else
#include <unistd.h>
#include <sys/wait.h>
#include <spawn.h>
#define sleep_ms(ms) usleep((ms) * 1000)
typedef int pid_t;
#define get_pid() getpid()
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

/* ── Configuration ─────────────────────────────────────────────────── */

#define CUPTI_BUF_SIZE      (32 * 1024 * 1024)  /* 32 MB activity buffer */
#define CUPTI_ALIGN_SIZE     8
#define ALIGN_BUFFER(buf, align) \
    (((uintptr_t)(buf) & ((align)-1)) \
        ? ((buf) + (align) - ((uintptr_t)(buf) & ((align)-1))) \
        : (buf))

/* ── Structures ────────────────────────────────────────────────────── */

typedef struct {
    uint64_t start;       /* GPU timestamp (ns) */
    uint64_t end;         /* GPU timestamp (ns) */
    uint32_t streamId;
    uint32_t contextId;
} KernelRecord;

typedef struct {
    KernelRecord *records;
    size_t        count;
    size_t        capacity;
    uint64_t      first_timestamp;   /* earliest kernel start */
    uint64_t      last_timestamp;    /* latest kernel end */
    uint64_t      total_gpu_ns;      /* sum of (end - start) for all kernels */
    uint64_t      total_gap_ns;      /* sum of gaps between consecutive kernels on same stream */
    int           recorded_streams[64];
    int           num_streams;
} TraceData;

/* ── Globals ───────────────────────────────────────────────────────── */

static TraceData g_trace = {0};
static FILE *g_csv = NULL;
static int g_verbose = 0;
static const char *g_csv_path = NULL;

/* ── CUPTI Buffer Callback ─────────────────────────────────────────── */

static void CUPTIAPI BufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    uint8_t *raw = (uint8_t *)malloc(CUPTI_BUF_SIZE + CUPTI_ALIGN_SIZE);
    if (!raw) {
        fprintf(stderr, "CUPTI: buffer alloc failed\n");
        exit(1);
    }
    *buffer = ALIGN_BUFFER(raw, CUPTI_ALIGN_SIZE);
    *size = CUPTI_BUF_SIZE;
    *maxNumRecords = 0;
}

static void CUPTIAPI BufferCompleted(CUcontext ctx, uint32_t streamId,
                                      uint8_t *buffer, size_t size, size_t validSize) {
    if (validSize == 0) return;

    CUpti_Activity *record = NULL;
    CUptiResult status;

    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS && record) {
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
                record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {

                CUpti_ActivityKernel11 *k = (CUpti_ActivityKernel11 *)record;

                /* Only record kernels with valid timestamps */
                if (k->start > 0 && k->end > 0 && k->end > k->start) {
                    if (g_trace.count >= g_trace.capacity) {
                        size_t new_cap = g_trace.capacity ? g_trace.capacity * 2 : 1048576;
                        KernelRecord *new_recs = (KernelRecord *)realloc(
                            g_trace.records, new_cap * sizeof(KernelRecord));
                        if (!new_recs) {
                            fprintf(stderr, "CUPTI: record array realloc failed at %zu\n",
                                    g_trace.count);
                            return;
                        }
                        g_trace.records = new_recs;
                        g_trace.capacity = new_cap;
                    }

                    g_trace.records[g_trace.count].start    = k->start;
                    g_trace.records[g_trace.count].end      = k->end;
                    g_trace.records[g_trace.count].streamId = k->streamId;
                    g_trace.records[g_trace.count].contextId = k->contextId;
                    g_trace.count++;

                    if (g_trace.first_timestamp == 0 || k->start < g_trace.first_timestamp) {
                        g_trace.first_timestamp = k->start;
                    }
                    if (k->end > g_trace.last_timestamp) {
                        g_trace.last_timestamp = k->end;
                    }
                    g_trace.total_gpu_ns += (k->end - k->start);
                }
            }
        } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        } else if (status != CUPTI_SUCCESS) {
            break;
        }
    } while (1);

    free(buffer);
}

/* ── CUPTI Init / Deinit ───────────────────────────────────────────── */

static int cupti_init(void) {
    CUptiResult res;

    /* Enable CONCURRENT_KERNEL tracing */
    res = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (res != CUPTI_SUCCESS) {
        fprintf(stderr, "cuptiActivityEnable(KERNEL) failed: %d\n", res);
        return 1;
    }

    /* Also enable driver API to capture launch calls (optional, for CPU-side timing) */
    res = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
    if (res != CUPTI_SUCCESS) {
        /* Non-fatal — driver API tracing may not be available */
        if (g_verbose) fprintf(stderr, "cuptiActivityEnable(DRIVER) failed: %d (non-fatal)\n", res);
    }

    /* Don't auto-flush — we want full buffer dumps */
    res = cuptiActivityFlushAll(0);
    if (res != CUPTI_SUCCESS) {
        fprintf(stderr, "cuptiActivityFlushAll failed: %d\n", res);
    }

    /* Register buffer callbacks */
    res = cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);
    if (res != CUPTI_SUCCESS) {
        fprintf(stderr, "cuptiActivityRegisterCallbacks failed: %d\n", res);
        return 1;
    }

    return 0;
}

static void cupti_deinit(void) {
    /* Force flush all remaining activity records */
    cuptiActivityFlushAll(1);

    /* Disable activity tracing */
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
}

/* ── Analysis ──────────────────────────────────────────────────────── */

static int compare_by_stream_start(const void *a, const void *b) {
    const KernelRecord *ka = (const KernelRecord *)a;
    const KernelRecord *kb = (const KernelRecord *)b;
    if (ka->streamId != kb->streamId)
        return (ka->streamId > kb->streamId) ? 1 : -1;
    if (ka->start < kb->start) return -1;
    if (ka->start > kb->start) return 1;
    return 0;
}

static void analyze_trace(void) {
    if (g_trace.count == 0) {
        printf("NO KERNEL RECORDS CAPTURED. Check CUPTI setup.\n");
        return;
    }

    /* Sort by stream then start time */
    qsort(g_trace.records, g_trace.count, sizeof(KernelRecord), compare_by_stream_start);

    /* Compute gaps between consecutive kernels on same stream */
    uint64_t total_gap_ns = 0;
    uint64_t gap_count = 0;
    uint64_t min_gap_ns = UINT64_MAX;
    uint64_t max_gap_ns = 0;
    int current_stream = -1;
    uint64_t prev_end = 0;

    for (size_t i = 0; i < g_trace.count; i++) {
        KernelRecord *kr = &g_trace.records[i];

        if (kr->streamId != (uint32_t)current_stream) {
            current_stream = kr->streamId;
            prev_end = kr->end;
            continue;
        }

        /* Same stream: gap between prev end and this start */
        if (kr->start > prev_end) {
            uint64_t gap = kr->start - prev_end;
            total_gap_ns += gap;
            gap_count++;
            if (gap < min_gap_ns) min_gap_ns = gap;
            if (gap > max_gap_ns) max_gap_ns = gap;
        }

        prev_end = kr->end;
    }

    g_trace.total_gap_ns = total_gap_ns;

    /* Metrics */
    double wall_ms = (g_trace.last_timestamp - g_trace.first_timestamp) / 1e6;
    double gpu_active_ms = g_trace.total_gpu_ns / 1e6;
    double gap_ms = total_gap_ns / 1e6;
    double avg_gap_us = (gap_count > 0) ? (total_gap_ns / (double)gap_count / 1000.0) : 0.0;
    double pct_overhead = (wall_ms > 0) ? (gap_ms / wall_ms * 100.0) : 0.0;

    /* ── OUTPUT ────────────────────────────────────────────────── */
    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  CUDA DRIVER OVERHEAD PROFILE (CUPTI Activity Trace)\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("  total_kernels:         %zu\n", g_trace.count);
    printf("  unique_streams:        %d  (kernels scattered across streams)\n", g_trace.num_streams);
    printf("  wall_clock_ms:         %.2f  (first kernel start → last kernel end)\n", wall_ms);
    printf("  gpu_active_ms:         %.2f  (sum of all kernel durations)\n", gpu_active_ms);
    printf("  gpu_gap_ms:            %.2f  (sum of inter-kernel gaps)\n", gap_ms);
    printf("  gap_count:             %" PRIu64 "\n", gap_count);
    printf("  avg_launch_gap_us:     %.3f  (mean gap between consecutive kernels)\n", avg_gap_us);
    printf("  min_launch_gap_us:     %.3f\n", min_gap_ns / 1000.0);
    printf("  max_launch_gap_us:     %.3f\n", max_gap_ns / 1000.0);
    printf("  pct_overhead:          %.1f%%  (gpu_gap / wall_clock)\n", pct_overhead);
    printf("\n");

    /* Verdict */
    printf("  ─── VERDICT ───\n");
    if (pct_overhead < 5.0) {
        printf("  <5%%  = FINE — driver overhead negligible\n");
    } else if (pct_overhead < 15.0) {
        printf("  5-15%% = WORTH_OPTIMIZING — batch launches, CUDA graphs\n");
    } else {
        printf("  >15%% = URGENT — driver is bottleneck\n");
        printf("          RECOMMEND: persistent kernel or CUDA graphs\n");
    }
    printf("\n");

    /* Per-token estimate */
    printf("  ─── PER-TOKEN ESTIMATE (if from decode) ───\n");
    printf("  If this trace covers N tokens, divide by N for per-token overhead.\n");
    printf("  Example: 100 tokens → %.2f ms/token overhead\n", gap_ms / 100.0);
    printf("  At 186.7 tok/s (5.36ms/token), overhead = %.1f%% of budget\n", gap_ms / 100.0 / 5.36 * 100.0);
    printf("\n");

    /* Histogram of gap sizes */
    printf("  ─── GAP HISTOGRAM (log10 buckets, ns) ───\n");
    uint64_t buckets[10] = {0}; /* 10^0 to 10^9 ns */
    current_stream = -1;
    prev_end = 0;
    for (size_t i = 0; i < g_trace.count; i++) {
        KernelRecord *kr = &g_trace.records[i];
        if (kr->streamId != (uint32_t)current_stream) {
            current_stream = kr->streamId;
            prev_end = kr->end;
            continue;
        }
        if (kr->start > prev_end) {
            uint64_t gap = kr->start - prev_end;
            int bucket = 0;
            uint64_t g = gap;
            while (g >= 10 && bucket < 9) { g /= 10; bucket++; }
            buckets[bucket]++;
        }
        prev_end = kr->end;
    }
    printf("     <10ns: %8" PRIu64 "\n", buckets[0]);
    printf("   10-99ns: %8" PRIu64 "\n", buckets[1]);
    printf("  100ns-1us: %8" PRIu64 "\n", buckets[2]);
    printf("   1-10us:   %8" PRIu64 "  <-- typical driver overhead lives here\n", buckets[3]);
    printf("  10-100us:  %8" PRIu64 "  <-- stream sync or memory stall\n", buckets[4]);
    printf(" 100us-1ms:  %8" PRIu64 "\n", buckets[5]);
    printf("    >1ms:    %8" PRIu64 "  <-- context sync or layer boundary\n", buckets[6]);
    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");

    /* Optional CSV export */
    if (g_csv_path) {
        g_csv = fopen(g_csv_path, "w");
        if (g_csv) {
            fprintf(g_csv, "index,streamId,start_ns,end_ns,duration_ns\n");
            for (size_t i = 0; i < g_trace.count && i < 100000; i++) {
                KernelRecord *kr = &g_trace.records[i];
                fprintf(g_csv, "%zu,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                        i, kr->streamId, kr->start, kr->end, kr->end - kr->start);
            }
            fclose(g_csv);
            printf("  CSV export: %s (first 100K records)\n", g_csv_path);
        }
    }
}

/* ── Simple wall-clock timer (no CUPTI, just for comparison) ───────── */

static double wall_clock_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart * 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

/* ── Main ──────────────────────────────────────────────────────────── */

static void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS] -- <command> [args...]\n", prog);
    printf("\n");
    printf("Measures CUDA driver overhead during GPU inference via CUPTI activity tracing.\n");
    printf("Runs <command> under a CUPTI trace, capturing all kernel timestamps,\n");
    printf("then computes inter-kernel launch gaps and overhead percentage.\n");
    printf("\n");
    printf("OPTIONS:\n");
    printf("  --csv PATH       Export kernel records to CSV (first 100K)\n");
    printf("  --verbose        Print verbose CUPTI status\n");
    printf("  --self           Profile CUPTI-enabled kernels in THIS process\n");
    printf("                   (requires target to be linked with cupti)\n");
    printf("  --help           This help\n");
    printf("\n");
    printf("EXAMPLES:\n");
    printf("  %s -- llama-cli -m model.gguf -n 128 -ngl 99\n", prog);
    printf("\n");
    printf("NOTE: CUPTI traces activity WITHIN the calling process.\n");
    printf("To profile llama.cpp, either:\n");
    printf("  A) Link llama.cpp with cupti and set CUPTI_TRACE=1 env var, OR\n");
    printf("  B) Use the Python wrapper: tools/profile_driver_overhead.py\n");
    printf("\n");
}

int main(int argc, char **argv) {
    const char *cmd = NULL;
    int cmd_argc = 0;
    const char **cmd_argv = NULL;
    int self_mode = 0;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            g_verbose = 1;
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            g_csv_path = argv[++i];
        } else if (strcmp(argv[i], "--self") == 0) {
            self_mode = 1;
        } else if (strcmp(argv[i], "--") == 0) {
            cmd = argv[i + 1];
            cmd_argc = argc - i - 1;
            cmd_argv = (const char **)&argv[i + 1];
            break;
        }
    }

    if (!cmd && !self_mode) {
        fprintf(stderr, "ERROR: No command specified. Use -- <command> [args...]\n");
        fprintf(stderr, "For self-profiling mode, use --self and call CUPTI init manually.\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Initialize CUPTI */
    printf("profile_driver_overhead: Initializing CUPTI activity tracing...\n");
    if (cupti_init() != 0) {
        fprintf(stderr, "CUPTI initialization FAILED.\n");
        fprintf(stderr, "Ensure CUPTI DLLs are accessible (CUDA 13.3 extras/CUPTI/lib64/).\n");
        return 1;
    }
    printf("profile_driver_overhead: CUPTI active. Launching target...\n\n");

    double t0 = wall_clock_ms();

    /* ── Launch child process ───────────────────────────────────── */
    if (self_mode) {
        printf("Self-profiling mode: run your CUDA kernels here.\n");
        printf("Press Enter after your kernel work completes...\n");
        getchar();
    } else {
#ifdef _WIN32
        /* Build command line */
        char cmdline[32768] = {0};
        int pos = 0;
        for (int i = 0; i < cmd_argc; i++) {
            if (i > 0) cmdline[pos++] = ' ';
            /* Quote arguments with spaces */
            int needs_quote = (strchr(cmd_argv[i], ' ') != NULL);
            if (needs_quote) cmdline[pos++] = '"';
            size_t len = strlen(cmd_argv[i]);
            memcpy(cmdline + pos, cmd_argv[i], len);
            pos += (int)len;
            if (needs_quote) cmdline[pos++] = '"';
        }
        cmdline[pos] = '\0';

        STARTUPINFO si = { sizeof(si) };
        PROCESS_INFORMATION pi = {0};
        si.dwFlags = STARTF_USESHOWWINDOW;
        si.wShowWindow = SW_HIDE;

        if (!CreateProcess(NULL, cmdline, NULL, NULL, FALSE,
                           CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
            fprintf(stderr, "CreateProcess failed: %lu\n", GetLastError());
            cupti_deinit();
            return 1;
        }

        /* Wait for child to finish */
        WaitForSingleObject(pi.hProcess, INFINITE);

        DWORD exit_code = 0;
        GetExitCodeProcess(pi.hProcess, &exit_code);

        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);

        if (g_verbose) printf("Child process exited with code %lu\n", exit_code);
#else
        pid_t child_pid;
        char **child_argv = (char **)malloc((cmd_argc + 1) * sizeof(char *));
        for (int i = 0; i < cmd_argc; i++) {
            child_argv[i] = (char *)cmd_argv[i];
        }
        child_argv[cmd_argc] = NULL;

        int ret = posix_spawnp(&child_pid, cmd_argv[0], NULL, NULL,
                                child_argv, NULL);
        if (ret != 0) {
            fprintf(stderr, "posix_spawnp failed: %d\n", ret);
            free(child_argv);
            cupti_deinit();
            return 1;
        }

        int status;
        waitpid(child_pid, &status, 0);
        free(child_argv);

        if (g_verbose) printf("Child process exited with status %d\n", status);
#endif
    }

    double t1 = wall_clock_ms();

    /* ── Flush and deinit CUPTI ─────────────────────────────────── */
    cupti_deinit();

    printf("profile_driver_overhead: Trace complete (%.0f ms wall clock)\n", t1 - t0);
    printf("profile_driver_overhead: Captured %zu kernel records\n", g_trace.count);

    /* ── Analyze ────────────────────────────────────────────────── */
    analyze_trace();

    /* Cleanup */
    if (g_trace.records) free(g_trace.records);

    return 0;
}
