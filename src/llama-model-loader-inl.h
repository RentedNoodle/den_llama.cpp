// llama-model-loader-inl.h
// Template implementations for llama_model_loader
// Uses GGUF API from ggml.c

// Helper: read array element from gguf_arr_data
template<typename T>
static inline T gguf_get_arr_value(const struct gguf_context * ctx, int key_id, int i) {
    const void * data = gguf_get_arr_data(ctx, key_id);
    if (!data) return T{};
    return ((const T*)data)[i];
}

// get_key by enum llm_kv
template<typename T>
bool llama_model_loader::get_key(const enum llm_kv kid, T & result, const bool required) {
    if (!meta) { if (required) fprintf(stderr, "%s: meta not loaded\n", __func__); return false; }
    std::string kstr = llm_kv(kid);
    const char * key = kstr.c_str();
    if (!key || !*key) { if (required) fprintf(stderr, "%s: unknown KV key %d\n", __func__, kid); return false; }
    const int idx = gguf_find_key(meta, key);
    if (idx < 0) { if (required) fprintf(stderr, "%s: key '%s' not found\n", __func__, key); return false; }
    const int val_type = gguf_get_kv_type(meta, idx);
    if constexpr (std::is_same_v<T, std::string>) {
        if (val_type == GGUF_TYPE_STRING) { result = gguf_get_val_str(meta, idx); return true; }
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        if (val_type == GGUF_TYPE_UINT32) { result = gguf_get_val_u32(meta, idx); return true; }
        if (val_type == GGUF_TYPE_INT32)  { result = (uint32_t)gguf_get_val_i32(meta, idx); return true; }
    } else if constexpr (std::is_same_v<T, int32_t>) {
        if (val_type == GGUF_TYPE_INT32)  { result = gguf_get_val_i32(meta, idx); return true; }
        if (val_type == GGUF_TYPE_UINT32) { result = (int32_t)gguf_get_val_u32(meta, idx); return true; }
    } else if constexpr (std::is_same_v<T, float>) {
        if (val_type == GGUF_TYPE_FLOAT32) { result = gguf_get_val_f32(meta, idx); return true; }
    } else if constexpr (std::is_same_v<T, bool>) {
        if (val_type == GGUF_TYPE_BOOL) { result = gguf_get_val_bool(meta, idx); return true; }
    } else if constexpr (std::is_same_v<T, enum llama_pooling_type>) {
        if (val_type == GGUF_TYPE_UINT32 || val_type == GGUF_TYPE_INT32) {
            result = (enum llama_pooling_type)gguf_get_val_u32(meta, idx);
            return true;
        }
    }
    (void)val_type;
    if (required) fprintf(stderr, "%s: key '%s' type mismatch\n", __func__, key);
    return false;
}

// get_key by string key
template<typename T>
bool llama_model_loader::get_key(const std::string & key, T & result, const bool required) {
    if (!meta) { if (required) fprintf(stderr, "%s: meta not loaded\n", __func__); return false; }
    const int idx = gguf_find_key(meta, key.c_str());
    if (idx < 0) { if (required) fprintf(stderr, "%s: key '%s' not found\n", __func__, key.c_str()); return false; }
    const int val_type = gguf_get_kv_type(meta, idx);
    if constexpr (std::is_same_v<T, uint32_t>) {
        if (val_type == GGUF_TYPE_UINT32) { result = gguf_get_val_u32(meta, idx); return true; }
    } else if constexpr (std::is_same_v<T, int32_t>) {
        if (val_type == GGUF_TYPE_INT32) { result = gguf_get_val_i32(meta, idx); return true; }
    } else if constexpr (std::is_same_v<T, float>) {
        if (val_type == GGUF_TYPE_FLOAT32) { result = gguf_get_val_f32(meta, idx); return true; }
    } else if constexpr (std::is_same_v<T, bool>) {
        if (val_type == GGUF_TYPE_BOOL) { result = gguf_get_val_bool(meta, idx); return true; }
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (val_type == GGUF_TYPE_STRING) { result = gguf_get_val_str(meta, idx); return true; }
    }
    (void)val_type;
    return false;
}

// get_arr_n (integral) by enum llm_kv
template<typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
llama_model_loader::get_arr_n(const enum llm_kv kid, T & result, const bool required) {
    if (!meta) { if (required) fprintf(stderr, "%s: meta not loaded\n", __func__); return false; }
    std::string kstr = llm_kv(kid);
    const char * key = kstr.c_str();
    if (!key || !*key) { if (required) fprintf(stderr, "%s: unknown KV key %d\n", __func__, kid); return false; }
    const int idx = gguf_find_key(meta, key);
    if (idx < 0) { if (required) fprintf(stderr, "%s: key '%s' not found\n", __func__, key); return false; }
    result = (T)gguf_get_arr_n(meta, idx);
    return true;
}

// get_arr_n (integral) by string key
template<typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
llama_model_loader::get_arr_n(const std::string & key, T & result, const bool required) {
    if (!meta) { if (required) fprintf(stderr, "%s: meta not loaded\n", __func__); return false; }
    const int idx = gguf_find_key(meta, key.c_str());
    if (idx < 0) { if (required) fprintf(stderr, "%s: key '%s' not found\n", __func__, key.c_str()); return false; }
    result = (T)gguf_get_arr_n(meta, idx);
    return true;
}

// get_arr (vector) by string key
template<typename T>
bool llama_model_loader::get_arr(const std::string & key, std::vector<T> & result, const bool required) {
    if (!meta) { if (required) fprintf(stderr, "%s: meta not loaded\n", __func__); return false; }
    const int idx = gguf_find_key(meta, key.c_str());
    if (idx < 0) { if (required) fprintf(stderr, "%s: key '%s' not found\n", __func__, key.c_str()); return false; }
    const int n = gguf_get_arr_n(meta, idx);
    result.resize(n);
    for (int i = 0; i < n; i++) result[i] = gguf_get_arr_value<T>(meta, idx, i);
    return true;
}

// get_arr (vector) by enum llm_kv
template<typename T>
bool llama_model_loader::get_arr(const enum llm_kv kid, T & result, const bool required) {
    if (!meta) { if (required) fprintf(stderr, "%s: meta not loaded\n", __func__); return false; }
    std::string kstr = llm_kv(kid);
    const char * key = kstr.c_str();
    if (!key || !*key) { if (required) fprintf(stderr, "%s: unknown KV key %d\n", __func__, kid); return false; }
    return get_arr(key, result, required);
}

// get_arr (array) by string key
template<typename T, size_t N_MAX>
bool llama_model_loader::get_arr(const std::string & key, std::array<T, N_MAX> & result, const bool required) {
    if (!meta) { if (required) fprintf(stderr, "%s: meta not loaded\n", __func__); return false; }
    const int idx = gguf_find_key(meta, key.c_str());
    if (idx < 0) { if (required) fprintf(stderr, "%s: key '%s' not found\n", __func__, key.c_str()); return false; }
    const int n = gguf_get_arr_n(meta, idx);
    for (int i = 0; i < n && (size_t)i < N_MAX; i++) result[i] = gguf_get_arr_value<T>(meta, idx, i);
    return true;
}

// get_key_or_arr by string key
template<typename T, size_t N_MAX>
bool llama_model_loader::get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, const bool required) {
    if (!meta) { if (required) fprintf(stderr, "%s: meta not loaded\n", __func__); return false; }
    const int idx = gguf_find_key(meta, key.c_str());
    if (idx < 0) { if (required) fprintf(stderr, "%s: key '%s' not found\n", __func__, key.c_str()); return false; }
    const int val_type = gguf_get_kv_type(meta, idx);
    if (val_type == GGUF_TYPE_ARRAY) {
        const int arr_n = gguf_get_arr_n(meta, idx);
        for (int i = 0; i < arr_n && (size_t)i < N_MAX; i++) result[i] = gguf_get_arr_value<T>(meta, idx, i);
        return true;
    }
    T val{};
    if constexpr (std::is_same_v<T, uint32_t>) val = (T)gguf_get_val_u32(meta, idx);
    else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) val = (T)gguf_get_val_i32(meta, idx);
    else if constexpr (std::is_same_v<T, float>) val = (T)gguf_get_val_f32(meta, idx);
    for (uint32_t i = 0; i < n && (size_t)i < N_MAX; i++) result[i] = val;
    return true;
}

// get_key_or_arr by enum llm_kv
template<typename T>
bool llama_model_loader::get_key_or_arr(const enum llm_kv kid, T & result, uint32_t n, const bool required) {
    if (!meta) { if (required) fprintf(stderr, "%s: meta not loaded\n", __func__); return false; }
    std::string kstr = llm_kv(kid);
    const char * key = kstr.c_str();
    if (!key || !*key) { if (required) fprintf(stderr, "%s: unknown KV key %d\n", __func__, kid); return false; }
    return get_key_or_arr(key, result, n, required);
}
