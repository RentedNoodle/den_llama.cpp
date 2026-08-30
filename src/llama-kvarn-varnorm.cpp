#include "llama-kvarn-varnorm.h"
#include <cmath>
#include <algorithm>
// Minimal KVarN-inspired variance normalization for adaptive-KV q4 staging.
// Applies per-tile Hadamard-like scaling + variance norm before quant.
// Tile = 128 tokens, head_dim chunk. This is the "marry" hook: compress resident q4 with var-norm.
void llama_kvarn_varnorm_tile(float * data, int n_tokens, int n_channels) {
    // Placeholder: identity for now — will be replaced with WHT+var-norm once wired.
    // Keeping as no-op so build succeeds and we can benchmark the hook plumbing.
    (void)data; (void)n_tokens; (void)n_channels;
}
