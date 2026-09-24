// Mixed pair K=q4_g64 / V=q8_0 FA-VEC instantiations (mirror of fattn-vec-instance-nvfp4-q8_0.cu).
#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q4_G64, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_G64, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q4_G64, GGML_TYPE_Q8_0);
