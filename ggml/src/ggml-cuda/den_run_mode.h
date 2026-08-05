// den_run_mode.h — runtime engine mode switch (2026-08-04).
//
// den_llama.cpp is a super-fork (llama.cpp + beellama.cpp + ik_llama.cpp +
// den NVFP4). Three runtime tiers, selected by `--mode` (no build change):
//   DEN_MODE_BASIC  — stock llama.cpp / ik_llama.cpp base. No beellama/ik
//       fork extras, no den NVFP4. Closest to upstream behavior.
//   DEN_MODE_NORMAL — the fork: base + beellama/ik features (TurboQuant KV,
//       DFlash/MTP spec, expert offload), with den's NVFP4-specific additions
//       BYPASSED. Apples-to-apples parity baseline vs beellama/ik/mainline.
//       DEFAULT (den's NVFP4 not yet trusted as default).
//   DEN_MODE_SUPER  — DEN_MODE_NORMAL + all den NVFP4 additions (OMMA cubin,
//       NULLGLASS mmq loader, expert offload, NVFP4 KV, MTP, persistent kernel,
//       cognitive hooks). The full-den differentiator. Opt-in.
//
// The global lives in the ggml lib (ggml-cuda.cu) so both ggml dispatch AND
// llama/common can read/set it. Gated at each override site by comparing
// den_get_run_mode(): den NVFP4 additions behind `== DEN_MODE_SUPER`, beellama/
// ik fork extras behind `>= DEN_MODE_NORMAL`. Nothing is deleted/renamed — the
// lower modes just bypass features.
#pragma once

enum den_run_mode {
    DEN_MODE_BASIC  = 0, // stock llama.cpp / ik_llama.cpp base
    DEN_MODE_NORMAL = 1, // fork (llama/beellama/ik), den NVFP4 OFF (default)
    DEN_MODE_SUPER  = 2, // fork + den NVFP4 ON
};

void den_set_run_mode(den_run_mode mode);
den_run_mode den_get_run_mode(void);

// ── mode policy: single source of truth for which den features each mode
// enables. Gates read these named flags instead of raw enum comparisons, so
// adding a feature = add one flag here + one line in den_mode_policy_get().
struct den_mode_policy {
    bool nvfp4_omma;      // OMMA.SF.16864 cubin tensor-core GEMV (super)
    bool nvfp4_mmq;       // NULLGLASS mmq fp4 batched path (super)
    bool nvfp4_softgemv;  // den software-dequant GEMV for NVFP4 attn/lm_head (coherence fix; normal+super)
    bool expert_offload;  // DEN_EXPERT_OFFLOAD / only_active_experts (normal+super)
    bool nvfp4_kv;        // NVFP4 KV cache (super)
    bool mtp;             // MTP / draft speculative decode (normal+super)
    bool persistent;      // persistent single-launch kernel (super)
    bool cognitive;       // cognitive / living-kernel hooks (super)
};
den_mode_policy den_mode_policy_get(void);
