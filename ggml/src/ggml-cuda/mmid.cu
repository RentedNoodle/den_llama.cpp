#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation.
        // SER pioneer: match slots are assigned per-THREAD ballot ranks, not one per
        // token: sanitized ids repeat one expert many times per token, so several
        // threads can match the same (token, expert) and need distinct compact slots.
        // Loop trip counts are uniform so __ballot_sync never diverges.
        const int nsteps = (n_expert_used + warp_size - 1) / warp_size;
        for (int it = 0; it < n_tokens; ++it) {
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
            }
            for (int s = 0; s < nsteps; ++s) {
                const int iex = threadIdx.x + s*warp_size;
                const bool match = iex < n_expert_used && ids[it*si1 + iex] == expert;
                const unsigned match_ballot = __ballot_sync(0xFFFFFFFFu, match);
                const unsigned lane = (unsigned) threadIdx.x;
                const unsigned lane_lt = lane == 0 ? 0u : ((1u << lane) - 1u);
                if (match) {
                    store[it_compact + (int) __popc(match_ballot & lane_lt)] = mm_ids_helper_store(it, iex);
                }
                it_compact += (int) __popc(match_ballot);
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // SER pioneer: per-thread ballot rank gives every matching thread a distinct
            // compact slot. The legacy per-token-group scan assumed one match per token
            // per expert, which sanitized ids (repeated expert per token) violate.
            static_assert(warp_size <= 32, "ballot mask assumes warp <= 32 lanes");
            const unsigned match_ballot = __ballot_sync(0xFFFFFFFFu, iex_used != -1);
            const unsigned lane = (unsigned) threadIdx.x;
            const unsigned lane_lt = lane == 0 ? 0u : ((1u << lane) - 1u);
            const int rank_in_warp = (int) __popc(match_ballot & lane_lt);
            if (iex_used != -1) {
                store[it_compact + rank_in_warp] = mm_ids_helper_store(it, iex_used);
            }
            it_compact += (int) __shfl_sync(0xFFFFFFFFu, __popc(match_ballot), warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    // SER note: sanitized ids can repeat one expert many times per token (dropped slots
    // remap to expert 0), so one expert may own up to n_tokens*n_expert_used rows here.
    // The legacy n_tokens sizing assumed distinct experts per token and overflows otherwise.
    const size_t nbytes_shared = ((size_t) n_tokens) * n_expert_used_var * sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

// SER SoT pioneer: remap dropped slots (id -1) to expert 0 so every downstream
// MoE path (helper maps, quantize scatter, fused match loop) keeps its denseployer
// invariant: all slots valid, all outputs written and finite. The weights path keeps
// the original -1 ids (get_rows zero-fills them), so dropped slots contribute exactly 0.
static __global__ void k_sanitize_moe_ids(const int32_t * __restrict__ ids_in, int32_t * __restrict__ ids_out,
        const int n_tokens, const int n_expert_used, const int si1) {
    // One thread per token: remap dropped slots (-1) round-robin over the token's
    // KEPT experts (hot rows, balanced load). Falls back to expert 0 if none kept.
    // All experts named here are active this token, hence GPU-resident by definition.
    const int it = blockIdx.x * blockDim.x + threadIdx.x;
    if (it >= n_tokens) {
        return;
    }
    int kept[64];
    int n_kept = 0;
    for (int iex = 0; iex < n_expert_used && n_kept < 64; ++iex) {
        const int v = ids_in[it * si1 + iex];
        if (v >= 0) {
            kept[n_kept++] = v;
        }
    }
    for (int iex = 0; iex < n_expert_used; ++iex) {
        const int v = ids_in[it * si1 + iex];
        ids_out[it * n_expert_used + iex] = v >= 0 ? v : (n_kept > 0 ? kept[iex % n_kept] : 0);
    }
}

void ggml_cuda_sanitize_moe_ids(const int32_t * ids_in, int32_t * ids_out,
        const int n_tokens, const int n_expert_used, const int si1, cudaStream_t stream) {
    if (n_tokens <= 0 || n_expert_used <= 0) {
        return;
    }
    GGML_ASSERT(n_expert_used <= 64);
    constexpr int block_size = 256;
    const int num_blocks = (n_tokens + block_size - 1) / block_size;
    k_sanitize_moe_ids<<<num_blocks, block_size, 0, stream>>>(ids_in, ids_out, n_tokens, n_expert_used, si1);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
