// Mixed pair K=q8_0 / V=q4_g64 FA-VEC instantiations (mirror of fattn-vec-instance-q8_0-nvfp4.cu).
#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_Q4_G64);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_G64);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_Q4_G64);
