// TQ3_0 x Q8_0 FA-VEC instantiations (V-side precision isolation; mirror of
// fattn-vec-instance-q4_g64-q8_0.cu). TQ block = 256 = one head vector, D=256 only.
#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ3_0, GGML_TYPE_Q8_0);
