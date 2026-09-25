// Hand-authored mirror of fattn-vec-instance-q8_0-q4_0.cu (fa_fix instance pattern).
// Provides the NVFP4 x NVFP4 FA-VEC template instantiations; declared extern in fattn-vec.cuh
// and referenced by the FATTN_VEC_CASES_ALL_D(GGML_TYPE_NVFP4, GGML_TYPE_NVFP4) table line.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4);
