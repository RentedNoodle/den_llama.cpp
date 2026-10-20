// Q8_0 x TQ4_0 FA-VEC instantiations (K-side precision isolation; mirror of
// fattn-vec-instance-q8_0-q4_g64.cu). TQ block = 256 = one head vector, D=256 only.
#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TQ4_0);
