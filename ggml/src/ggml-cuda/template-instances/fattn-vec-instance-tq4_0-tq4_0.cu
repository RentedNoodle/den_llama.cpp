// TQ4_0 x TQ4_0 FA-VEC instantiations (mirror of fattn-vec-instance-q4_g64-q4_g64.cu).
// TQ block = 256 = one head vector, so only D=256 is valid.
#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ4_0, GGML_TYPE_TQ4_0);
