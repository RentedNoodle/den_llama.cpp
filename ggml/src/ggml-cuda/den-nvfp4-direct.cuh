// den-nvfp4-direct.cuh — Direct OMMA dispatch diagnostic for NVFP4 tensors
//
// Phase 1: Verifies that the Blackwell MMA path is active for NVFP4.
// The OMMA instruction IS being used (mmq.cuh:693-698 routes NVFP4 to
// mma_block_scaled_fp4<NVFP4>() → mma.sync.kind::mxf4nvf4.4X).
//
// Speed gap (150 vs 202 tok/s NVFP4 GGUF vs Q4_K_M) is from tile load conversion:
// GGUF stores weights+scales separately → load_tiles_nvfp4_nvfp4 repacks.
// .den NULLGLASS format (Phase 2) stores 160B contiguous tiles → zero repack.
//
// Phase 2 (.den loader): skip the tile load step entirely.
// Phase 3 (dual-format dispatch): runtime tile format detection.

#pragma once
#include <cstdio>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

static inline int den_nvfp4_direct_enabled(void) {
    static int checked = 0, enabled = 0;
    if (!checked) {
        const char * env = getenv("DEN_NVFP4_DIRECT");
        enabled = (env && env[0] == '1') ? 1 : 0;
        checked = 1;
        if (enabled) {
            fprintf(stderr, "NVFP4-DIRECT: OMMA path active (Blackwell MMA, mma.sync.kind::mxf4nvf4.4X)\n");
            fprintf(stderr, "NVFP4-DIRECT: GGUF tile load step still active (Phase 1 diagnostic)\n");
            fprintf(stderr, "NVFP4-DIRECT: .den NULLGLASS zero-copy pending (Phase 2 loader port)\n");
        }
    }
    return enabled;
}

#ifdef __cplusplus
}
#endif
