// Hand-authored NVFP4 mixed-pair VEC instance: K=q8_0, V=nvfp4. Mirrors the matched-pair
// instance (fattn-vec-instance-nvfp4-nvfp4.cu) and the fa_fix pattern
// (fattn-vec-instance-q8_0-q4_0.cu). Declared extern in fattn-vec.cuh and referenced by the
// FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_NVFP4) table line.
//
// K side uses the existing vec_dot_fattn_vec_KQ_q8_0 hook; V side uses dequantize_V_nvfp4.
// Both are selected by get_vec_dot_KQ/get_dequantize_V on the (type_K, type_V) template
// params alone, so no cross-type kernel change is required.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_NVFP4);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_NVFP4);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_NVFP4);
