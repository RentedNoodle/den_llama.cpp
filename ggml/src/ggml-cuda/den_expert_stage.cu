// den_expert_stage.cu - L3-resident CPU staging tier for expert offload (Den)
//
// Implements the host-half of Blocker 22.2 on MAINLINE llama.cpp.
//
// When MoE experts are host-resident (--cpu-moe / --n-cpu-moe / -ot exps=CPU),
// the ggml scheduler copies only the active expert slices to the GPU
// (ggml_backend_sched_compute_splits -> copy_experts).  Those H2D copies are
// served from the large unpinned host expert buffer, so each DMA bounces 4KB
// pages out of DRAM.  This file adds a small pinned (cudaMallocHost) staging
// area, pre-filled by a dedicated CPU thread pinned to physical core 0, so the
// DMA reads hit a hot, pinned source (CPU L3 on Zen 4) instead.
//
// Design deltas vs. the 22.2 plan (see blockers-22-24 plan):
//   - 3 rotating buffers instead of 2x32MB.  Mainline's scheduler does NOT
//     synchronize between a layer's gate_up and down MUL_MAT_ID splits (they
//     share the same ids tensor, so the ids read-back is skipped).  With two
//     buffers the GPU's async H2D from a just-promoted buffer can still be in
//     flight while the staging thread overwrites it.  Three rotating buffers
//     guarantee a ggml_backend_synchronize (from the next different-ids split)
//     always lands between any buffer's read and its reuse.
//   - Predictor: first cut = temporal expert-locality.  The thread learns the
//     split key sequence (gate_up L -> down L -> gate_up L+1 ...) and predicts
//     the next split's ids from the last time that key fired (previous token),
//     or the current ids for the same-layer down split (exact match).

#include "den_expert_stage.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

// AVX-512 copy path (Zen 4).  Enabled whenever the host compiler sees
// -mavx512f — on MSVC that is /arch:AVX512, which also defines __AVX512F__ and
// provides _mm512_loadu_si512/_mm512_stream_si512.  Non-temporal stores (see
// den_stage_copy) are why this path matters on both toolchains.
#if defined(__AVX512F__)
#define DEN_STAGE_AVX512 1
#include <immintrin.h>
#else
#define DEN_STAGE_AVX512 0
#endif

// ---------------------------------------------------------------------------
// static state (single instance - one context per process is the norm)
// ---------------------------------------------------------------------------

static std::mutex              g_mutex;
static std::condition_variable g_cv_job;   // job enqueue / staging thread wake
static std::condition_variable g_cv_done;  // job completion

static bool  g_enabled = false;
static bool  g_inited  = false;
static bool  g_shutdown = false;

static std::thread g_stage_thread;

// per-expert slot: expert_size bytes + 512B zero padding (MMQ over-fetch safety)
static const size_t DEN_STAGE_SLOT_PAD = 512;
static const size_t DEN_STAGE_DEFAULT_BYTES = 96 * 1024 * 1024; // 3 x 32 MiB
static const int    DEN_STAGE_DEFAULT_N_BUFFERS = 3;

struct stage_buffer {
    uint8_t * base = nullptr;
    size_t    size = 0;         // bytes
    size_t    slot_size = 0;    // per-expert slot bytes (expert_size + pad, aligned)
    int       max_slots = 0;
    // owner + staged expert ids (sorted ascending)
    std::string owner_key;
    int32_t     layer = -1;
    std::vector<int32_t> ids;   // sorted staged expert ids
    bool        ready = false;  // data fully staged
};

// in-flight job (what the staging thread is filling right now)
struct stage_job {
    bool         active = false;
    bool         done   = false;
    std::string  key;
    int          target = 0;    // buffer index being filled
    std::vector<int32_t> ids;   // predicted expert ids (sorted)
    const void * host_src = nullptr;
    size_t       expert_size = 0;
    size_t       slot_size = 0;
};

static stage_buffer g_bufs[DEN_STAGE_DEFAULT_N_BUFFERS];
static stage_job    g_job;
static int          g_read_idx  = 0;
static int          g_write_idx = 1;
static int          g_spare_idx = 2;

// predictor state
static std::string g_prev_key;                 // previously observed key
static std::vector<int32_t> g_prev_ids;        // ids of the previously observed key
static std::vector<std::string> g_next_of;     // learned key -> next key
static std::vector<std::string> g_next_keys;   // (parallel array for lookup)
static std::vector<std::string> g_key_names;   // known keys
static std::vector<std::vector<int32_t>> g_key_ids; // key -> last observed ids

// stats
static std::atomic<size_t> g_hits{0};
static std::atomic<size_t> g_misses{0};
static std::atomic<size_t> g_bytes_staged{0};

static void den_stage_copy(void * dst, const void * src, size_t n);
static void den_stage_predict_and_enqueue_locked(std::unique_lock<std::mutex> & lock,
                                                 const std::string & key, int layer,
                                                 const int32_t * ids, int n_ids,
                                                 const void * host_src, size_t expert_size);

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

// parse "blk.<layer>.ffn_<type>exps.weight" -> layer index and type
static bool den_stage_parse_key(const std::string & key, int & layer, std::string & type) {
    const char * s = key.c_str();
    const char * blk = strstr(s, "blk.");
    if (!blk) {
        return false;
    }
    blk += 4;
    char * end = nullptr;
    long l = strtol(blk, &end, 10);
    if (end == blk) {
        return false;
    }
    layer = (int) l;
    const char * ffn = strstr(end, "ffn_");
    if (!ffn) {
        return false;
    }
    ffn += 4;
    const char * us = strchr(ffn, '_');
    const char * ex = strstr(ffn, "exps");
    if (!us || !ex) {
        return false;
    }
    type.assign(ffn, us - ffn);
    return true;
}

static bool den_stage_is_input_side(const std::string & type) {
    // gate_up / up / gate are the input side; down is the output side.
    return type != "down";
}

static void den_stage_learn(const std::string & key, const std::vector<int32_t> & ids) {
    // learn key -> next-key transition
    if (!g_prev_key.empty() && g_prev_key != key) {
        bool found = false;
        for (size_t i = 0; i < g_next_keys.size(); ++i) {
            if (g_next_keys[i] == g_prev_key) {
                g_next_of[i] = key;
                found = true;
                break;
            }
        }
        if (!found) {
            g_next_keys.push_back(g_prev_key);
            g_next_of.push_back(key);
        }
    }
    g_prev_key = key;
    g_prev_ids = ids;

    // remember last ids per key
    for (size_t i = 0; i < g_key_names.size(); ++i) {
        if (g_key_names[i] == key) {
            g_key_ids[i] = ids;
            return;
        }
    }
    g_key_names.push_back(key);
    g_key_ids.push_back(ids);
}

static std::string den_stage_next_key(const std::string & key) {
    // learned mapping first
    for (size_t i = 0; i < g_next_keys.size(); ++i) {
        if (g_next_keys[i] == key) {
            return g_next_of[i];
        }
    }
    // heuristic: gate_up/up/gate L -> down L ; down L -> gate_up L+1
    int layer = 0;
    std::string type;
    if (!den_stage_parse_key(key, layer, type)) {
        return "";
    }
    if (den_stage_is_input_side(type)) {
        // same layer, output side
        const char * pos = key.c_str();
        const char * ffn = strstr(pos, "ffn_");
        if (!ffn) {
            return "";
        }
        return key.substr(0, ffn - pos) + "ffn_down_exps.weight";
    }
    // down L -> gate_up L+1 (merged). learned map corrects for separate gate/up.
    const char * pos = key.c_str();
    const char * ffn = strstr(pos, "ffn_");
    if (!ffn) {
        return "";
    }
    char next[64];
    snprintf(next, sizeof(next), "blk.%d.ffn_gate_up_exps.weight", layer + 1);
    return std::string(next);
}

// predicted ids for the next key
static std::vector<int32_t> den_stage_predict_ids(const std::string & next_key,
                                                  const std::vector<int32_t> & cur_ids) {
    // if the next key is the same layer (down side), the current ids are exact
    int cur_layer = 0, next_layer = 0;
    std::string cur_type, next_type;
    bool cur_ok = den_stage_parse_key(g_prev_key.empty() ? "" : g_prev_key, cur_layer, cur_type);
    bool next_ok = den_stage_parse_key(next_key, next_layer, next_type);
    if (cur_ok && next_ok && next_layer == cur_layer) {
        return cur_ids; // same layer: same ids tensor, exact prediction
    }
    // temporal locality: what did this key use last time?
    for (size_t i = 0; i < g_key_names.size(); ++i) {
        if (g_key_names[i] == next_key) {
            return g_key_ids[i];
        }
    }
    return cur_ids; // first-cut: next layer = current layer's ids
}

// ---------------------------------------------------------------------------
// staging thread
// ---------------------------------------------------------------------------

static void den_stage_thread_main() {
    // pin to physical core 0
#ifdef _WIN32
    // physical core 0 = CPU 0; leave the mask to the first core of the primary group
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1);
#else
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(0, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#endif

    for (;;) {
        stage_job job;
        {
            std::unique_lock<std::mutex> lock(g_mutex);
            g_cv_job.wait(lock, [] { return g_job.active || g_shutdown; });
            if (g_shutdown) {
                return;
            }
            job = g_job; // copy the job under lock
        }

        // fill the target buffer
        stage_buffer & buf = g_bufs[job.target];
        const uint8_t * src_base = (const uint8_t *) job.host_src;
        for (size_t i = 0; i < job.ids.size(); ++i) {
            const int32_t id = job.ids[i];
            uint8_t * dst = buf.base + i * job.slot_size;
            const uint8_t * src = src_base + (size_t) id * job.expert_size;
            den_stage_copy(dst, src, job.expert_size);
            // zero the MMQ padding after each slot
            std::memset(dst + job.expert_size, 0, job.slot_size - job.expert_size);
            g_bytes_staged.fetch_add(job.expert_size, std::memory_order_relaxed);
        }

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            buf.ready = true;
            g_job.done = true;
            g_job.active = false;
        }
        g_cv_done.notify_all();
    }
}

// wait for the in-flight job (bounded)
static bool den_stage_wait_job_locked(std::unique_lock<std::mutex> & lock, int timeout_us) {
    if (!g_job.active || g_job.done) {
        return true;
    }
    if (timeout_us <= 0) {
        g_cv_done.wait(lock, [] { return !g_job.active || g_job.done; });
    } else {
        g_cv_done.wait_for(lock, std::chrono::microseconds(timeout_us),
                           [] { return !g_job.active || g_job.done; });
    }
    return !g_job.active || g_job.done;
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

static void den_stage_teardown_locked(std::unique_lock<std::mutex> & lock) {
    if (!g_inited) {
        return;
    }
    g_shutdown = true;
    g_cv_job.notify_all();
    lock.unlock();
    if (g_stage_thread.joinable()) {
        g_stage_thread.join();
    }
    lock.lock();
    for (auto & b : g_bufs) {
        if (b.base) {
            cudaFreeHost(b.base);
            b.base = nullptr;
        }
        b.owner_key.clear();
        b.ids.clear();
        b.ready = false;
        b.slot_size = 0;
        b.max_slots = 0;
    }
    g_job = stage_job();
    g_inited = false;
    g_shutdown = false;
}

// assumes g_mutex is held
static void den_stage_init_locked(size_t bytes_total, int n_buffers) {
    if (g_inited) {
        return;
    }
    if (n_buffers <= 0 || n_buffers > DEN_STAGE_DEFAULT_N_BUFFERS) {
        n_buffers = DEN_STAGE_DEFAULT_N_BUFFERS;
    }
    if (bytes_total == 0) {
        bytes_total = DEN_STAGE_DEFAULT_BYTES;
    }
    const size_t buf_size = bytes_total / n_buffers;
    for (int i = 0; i < n_buffers; ++i) {
        stage_buffer & b = g_bufs[i];
        cudaError_t err = cudaMallocHost((void **) &b.base, buf_size);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            // not fatal: fall back to disabled staging
            b.base = nullptr;
            g_enabled = false;
            return;
        }
        std::memset(b.base, 0, buf_size);
        b.size = buf_size;
    }
    g_read_idx  = 0;
    g_write_idx = 1;
    g_spare_idx = n_buffers >= 3 ? 2 : 0;
    g_inited = true;
    g_shutdown = false;
    g_stage_thread = std::thread(den_stage_thread_main);
}

void den_expert_stage_set_enabled(bool enabled) {
    std::unique_lock<std::mutex> lock(g_mutex);
    if (enabled == g_enabled) {
        return;
    }
    g_enabled = enabled;
    if (enabled) {
        if (!g_inited) {
            den_stage_init_locked(DEN_STAGE_DEFAULT_BYTES, DEN_STAGE_DEFAULT_N_BUFFERS);
        }
    } else {
        den_stage_teardown_locked(lock);
    }
}

void den_expert_stage_init(size_t bytes_total, int n_buffers) {
    std::unique_lock<std::mutex> lock(g_mutex);
    if (g_inited) {
        return;
    }
    if (!g_enabled) {
        g_enabled = true;
    }
    den_stage_init_locked(bytes_total, n_buffers);
}

int den_expert_stage_submit(const char * key, int layer, const int32_t * ids, int n_ids,
                            const void * host_src, size_t expert_size) {
    if (!g_enabled || !g_inited || key == nullptr || ids == nullptr || n_ids <= 0 ||
        host_src == nullptr || expert_size == 0) {
        return -1;
    }
    std::unique_lock<std::mutex> lock(g_mutex);

    // learn + compute the next-key prediction, then enqueue it (replaces any
    // pending job - the previous job's target was already promoted by the
    // scheduler when the previous key's split ran).
    den_stage_predict_and_enqueue_locked(lock, key, layer, ids, n_ids, host_src, expert_size);
    return 0;
}

// core of submit: under lock, learn the key sequence and enqueue the next key
static void den_stage_predict_and_enqueue_locked(std::unique_lock<std::mutex> & lock,
                                                 const std::string & key, int layer,
                                                 const int32_t * ids, int n_ids,
                                                 const void * host_src, size_t expert_size) {
    std::vector<int32_t> cur_ids(ids, ids + n_ids);
    std::sort(cur_ids.begin(), cur_ids.end());
    cur_ids.erase(std::unique(cur_ids.begin(), cur_ids.end()), cur_ids.end());

    den_stage_learn(key, cur_ids);

    const std::string next_key = den_stage_next_key(key);
    if (next_key.empty()) {
        return;
    }

    std::vector<int32_t> pred = den_stage_predict_ids(next_key, cur_ids);
    if (pred.empty()) {
        return;
    }

    // fit into the target buffer's capacity
    const size_t slot_size = (expert_size + DEN_STAGE_SLOT_PAD + 63) & ~((size_t) 63);

    // Wait for the in-flight job (the CURRENT split's key, started during the
    // previous split's GPU compute) to finish before we rotate buffers.  This is
    // required: the staging thread is writing the target buffer, and we cannot
    // overwrite g_job or rotate that buffer out from under it.
    den_stage_wait_job_locked(lock, 0);

    // Rotate 3 buffers.  The completed job in write_idx becomes the read buffer
    // (the current split's find_span will serve it).  The old read becomes the
    // spare.  The old spare becomes the new write target.  The new write target
    // was last read >= 2 splits ago, so at least one ggml_backend_synchronize
    // (from a different-ids split's ids read-back) has landed since - the GPU is
    // guaranteed done with it.
    const int old_read  = g_read_idx;
    const int old_write = g_write_idx;
    const int old_spare = g_spare_idx;
    g_read_idx  = old_write;
    g_spare_idx = old_read;
    g_write_idx = old_spare;

    stage_buffer & target = g_bufs[g_write_idx];
    target.slot_size = slot_size;
    target.max_slots = (int) (target.size / slot_size);
    if ((int) pred.size() > target.max_slots) {
        pred.resize(target.max_slots);
    }
    if (pred.empty()) {
        return;
    }

    g_job = stage_job();
    g_job.active = true;
    g_job.done = false;
    g_job.key = next_key;
    g_job.target = g_write_idx;
    g_job.ids = pred;
    g_job.host_src = host_src;
    g_job.expert_size = expert_size;
    g_job.slot_size = slot_size;

    // prepare the target buffer
    stage_buffer & tb = g_bufs[g_write_idx];
    tb.owner_key = next_key;
    tb.layer = layer;
    tb.ids = pred;
    tb.slot_size = slot_size;
    tb.ready = false;

    g_cv_job.notify_one();
}

const void * den_expert_stage_find(const char * key, int32_t expert_id) {
    return den_expert_stage_find_span(key, expert_id, expert_id);
}

const void * den_expert_stage_find_span(const char * key, int32_t first_id, int32_t last_id) {
    if (!g_enabled || !g_inited || key == nullptr || last_id < first_id) {
        return nullptr;
    }
    std::unique_lock<std::mutex> lock(g_mutex);

    // the current split's key is either the promoted read buffer or an in-flight job
    const stage_buffer * b = nullptr;
    if (g_bufs[g_read_idx].ready && g_bufs[g_read_idx].owner_key == key) {
        b = &g_bufs[g_read_idx];
    } else if (g_job.active && g_job.key == key) {
        // wait for the staging thread (bounded) - it was started during the
        // previous split's GPU compute, so this should be nearly instant.
        den_stage_wait_job_locked(lock, 2000);
        if (g_job.done && g_bufs[g_write_idx].owner_key == key) {
            b = &g_bufs[g_write_idx];
        }
    }

    if (b == nullptr || !b->ready) {
        g_misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    // locate first_id in the sorted staged ids and verify [first..last] is present
    const std::vector<int32_t> & staged = b->ids;
    auto it = std::lower_bound(staged.begin(), staged.end(), first_id);
    const size_t pos = it - staged.begin();
    const size_t range = (size_t)(last_id - first_id);
    if (it == staged.end() || *it != first_id || pos + range >= staged.size() ||
        staged[pos + range] != last_id) {
        g_misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    g_hits.fetch_add(1, std::memory_order_relaxed);
    return b->base + pos * b->slot_size;
}

void den_expert_stage_wait_layer(void) {
    std::unique_lock<std::mutex> lock(g_mutex);
    den_stage_wait_job_locked(lock, 0);
}

double den_expert_stage_probe(void) {
    // Timed re-read of a pinned buffer -> GB/s (L3-resident vs DRAM).
    //
    // The probe MUST NOT touch the live working buffers g_bufs[].  Those are the
    // H2D source for expert weights; overwriting one with the 0xAB sentinel would
    // serve corrupt bytes to the GPU (decoded as NaN -> swiglu assert).  Use a
    // dedicated pinned allocation for the timing read and free it before return.
    std::unique_lock<std::mutex> lock(g_mutex);
    if (!g_inited) {
        if (!g_enabled) {
            g_enabled = true;
        }
        den_stage_init_locked(DEN_STAGE_DEFAULT_BYTES, DEN_STAGE_DEFAULT_N_BUFFERS);
    }

    // dedicated probe buffer (same size as one working buffer; freed below)
    const size_t probe_size = DEN_STAGE_DEFAULT_BYTES / DEN_STAGE_DEFAULT_N_BUFFERS;
    void * probe = nullptr;
    cudaError_t err = cudaMallocHost(&probe, probe_size);
    if (err != cudaSuccess || probe == nullptr) {
        (void) cudaGetLastError();
        return 0.0;
    }

    // fill with a pattern
    std::memset(probe, 0xAB, probe_size);

    // warm + timed read
    const int reps = 8;
    auto t0 = std::chrono::steady_clock::now();
    volatile uint64_t sink = 0;
    for (int r = 0; r < reps; ++r) {
        const uint64_t * p = (const uint64_t *) probe;
        const size_t n = probe_size / sizeof(uint64_t);
        for (size_t i = 0; i < n; ++i) {
            sink ^= p[i];
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double gbps = (double) probe_size * reps / (secs * 1e9);
    (void) sink;
    cudaFreeHost(probe);
    return gbps;
}

void den_expert_stage_stats(size_t * hits, size_t * misses, size_t * bytes_staged) {
    if (hits)        { *hits        = g_hits.load(std::memory_order_relaxed); }
    if (misses)      { *misses      = g_misses.load(std::memory_order_relaxed); }
    if (bytes_staged){ *bytes_staged = g_bytes_staged.load(std::memory_order_relaxed); }
}

void den_expert_stage_shutdown(void) {
    std::unique_lock<std::mutex> lock(g_mutex);
    if (g_inited) {
        g_shutdown = true;
        g_cv_job.notify_all();
        lock.unlock();
        if (g_stage_thread.joinable()) {
            g_stage_thread.join();
        }
        lock.lock();
        for (auto & b : g_bufs) {
            if (b.base) {
                cudaFreeHost(b.base);
                b.base = nullptr;
            }
            b.owner_key.clear();
            b.ids.clear();
            b.ready = false;
        }
        g_job = stage_job();
        g_inited = false;
        g_shutdown = false;
    }
}

// ---------------------------------------------------------------------------
// copy kernel (memcpy, with an AVX-512 fast path on GCC/Clang + -mavx512f)
// ---------------------------------------------------------------------------

static void den_stage_copy(void * dst, const void * src, size_t n) {
#if DEN_STAGE_AVX512
    uint8_t * d = (uint8_t *) dst;
    const uint8_t * s = (const uint8_t *) src;
    size_t i = 0;
    // Den /btw #4: NON-TEMPORAL stores for the staging fills.  The staging
    // buffers sit in the pinned L3-adjacent zone that is meant to hold the HOT
    // expert resident set; a plain store write-allocates and evicts the very
    // lines the tier exists to keep hot.  _mm512_stream_si512 (MOVNTDQ) writes
    // through to DRAM without touching the cache, so the cold expert stream
    // flows into pinned memory without thrashing the hot set.
    // d is 64-byte aligned: slot_size is 64-aligned and cudaMallocHost returns
    // cache-line aligned memory, so every 64B store lands on MOVNTDQ's bounds.
    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512((const __m512i *) (s + i));
        _mm512_stream_si512((void *) (d + i), v);
    }
    // tail: small, cache-warm, plain store is fine
    if (i < n) {
        std::memcpy(d + i, s + i, n - i);
    }
#else
    std::memcpy(dst, src, n);
#endif
}

// ---------------------------------------------------------------------------
// registration into ggml-base
// ---------------------------------------------------------------------------

static void den_stage_iface_set_enabled(bool enabled) {
    den_expert_stage_set_enabled(enabled);
}

static const ggml_den_stage_iface den_stage_iface_impl = {
    /* .set_enabled = */ den_stage_iface_set_enabled,
    /* .submit      = */ den_expert_stage_submit,
    /* .find_span   = */ den_expert_stage_find_span,
    /* .wait_layer  = */ den_expert_stage_wait_layer,
    /* .probe       = */ den_expert_stage_probe,
    /* .stats       = */ den_expert_stage_stats,
    /* .shutdown    = */ den_expert_stage_shutdown,
};

// called by ggml_backend_cuda_reg() to install this backend's staging impl
extern "C" void den_expert_stage_register_ggml(void) {
    ggml_backend_den_stage_register(&den_stage_iface_impl);
}
