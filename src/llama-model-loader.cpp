#include "llama-model-loader.h"
#include <cstring>
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "den_loader.h"
#include "ggml.h"
//#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#elif defined(GGML_USE_VULKAN)
#  include "ggml-vulkan.h"
#elif defined(GGML_USE_SYCL)
#  include "ggml-sycl.h"
#elif defined(GGML_USE_KOMPUTE)
#   include "ggml-kompute.h"
#elif defined(GGML_USE_CANN)
#   include "ggml-cann.h"
#endif

#include <set>
#include <map>
#include <array>
#include <charconv>
#include <future>
#include <regex>
#include <algorithm>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

#define LLAMA_API_INTERNAL

namespace GGUFMeta {
    template <typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int)>
    struct GKV_Base_Type {
        static constexpr gguf_type gt = gt_;

        static T getter(const gguf_context * ctx, const int kid) {
            return gfun(ctx, kid);
        }
    };

    template<typename T> struct GKV_Base;

    template<> struct GKV_Base<bool        >: GKV_Base_Type<bool,         GGUF_TYPE_BOOL,    gguf_get_val_bool> {};
    template<> struct GKV_Base<uint8_t     >: GKV_Base_Type<uint8_t,      GGUF_TYPE_UINT8,   gguf_get_val_u8  > {};
    template<> struct GKV_Base<uint16_t    >: GKV_Base_Type<uint16_t,     GGUF_TYPE_UINT16,  gguf_get_val_u16 > {};
    template<> struct GKV_Base<uint32_t    >: GKV_Base_Type<uint32_t,     GGUF_TYPE_UINT32,  gguf_get_val_u32 > {};
    template<> struct GKV_Base<uint64_t    >: GKV_Base_Type<uint64_t,     GGUF_TYPE_UINT64,  gguf_get_val_u64 > {};
    template<> struct GKV_Base<int8_t      >: GKV_Base_Type<int8_t,       GGUF_TYPE_INT8,    gguf_get_val_i8  > {};
    template<> struct GKV_Base<int16_t     >: GKV_Base_Type<int16_t,      GGUF_TYPE_INT16,   gguf_get_val_i16 > {};
    template<> struct GKV_Base<int32_t     >: GKV_Base_Type<int32_t,      GGUF_TYPE_INT32,   gguf_get_val_i32 > {};
    template<> struct GKV_Base<int64_t     >: GKV_Base_Type<int64_t,      GGUF_TYPE_INT64,   gguf_get_val_i64 > {};
    template<> struct GKV_Base<float       >: GKV_Base_Type<float,        GGUF_TYPE_FLOAT32, gguf_get_val_f32 > {};
    template<> struct GKV_Base<double      >: GKV_Base_Type<double,       GGUF_TYPE_FLOAT64, gguf_get_val_f64 > {};
    template<> struct GKV_Base<const char *>: GKV_Base_Type<const char *, GGUF_TYPE_STRING,  gguf_get_val_str > {};

    template<> struct GKV_Base<std::string> {
        static constexpr gguf_type gt = GGUF_TYPE_STRING;

        static std::string getter(const gguf_context * ctx, const int kid) {
            return gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo {
        const gguf_type gt;
        const size_t length;
        const void * data;
    };

    template<> struct GKV_Base<ArrayInfo> {
        public:
        static constexpr gguf_type gt = GGUF_TYPE_ARRAY;
        static ArrayInfo getter(const gguf_context *ctx, const int k) {
            return ArrayInfo {
                gguf_get_arr_type(ctx, k),
                size_t(gguf_get_arr_n(ctx, k)),
                gguf_get_arr_data(ctx, k),
            };
        }
    };

    template<typename T>
    class GKV : public GKV_Base<T> {
        GKV() = delete;

        public:
        static T get_kv(const gguf_context * ctx, const int k) {
            const enum gguf_type kt = gguf_get_kv_type(ctx, k);

            if (kt != GKV::gt) {
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                    gguf_get_key(ctx, k), gguf_type_name(kt), gguf_type_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static const char * override_type_to_str(const llama_model_kv_override_type ty) {
            switch (ty) {
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:  return "bool";
                case LLAMA_KV_OVERRIDE_TYPE_INT:   return "int";
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT: return "float";
                case LLAMA_KV_OVERRIDE_TYPE_STR:   return "str";
            }
            return "unknown";
        }

        static bool validate_override(const llama_model_kv_override_type expected_type, const struct llama_model_kv_override * ovrd) {
            if (!ovrd) { return false; }
            if (ovrd->tag == expected_type) {
                LLAMA_LOG_INFO("%s: Using metadata override (%5s) '%s' = ",
                    __func__, override_type_to_str(ovrd->tag), ovrd->key);
                switch (ovrd->tag) {
                    case LLAMA_KV_OVERRIDE_TYPE_BOOL:  {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_bool ? "true" : "false");
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_INT:   {
                        LLAMA_LOG_INFO("%" PRId64 "\n", ovrd->val_i64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_FLOAT: {
                        LLAMA_LOG_INFO("%.6f\n", ovrd->val_f64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_STR: {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_str);
                    } break;
                    default:
                        // Shouldn't be possible to end up here, but just in case...
                        throw std::runtime_error(
                            format("Unsupported attempt to override %s type for metadata key %s\n",
                                override_type_to_str(ovrd->tag), ovrd->key));
                }
                return true;
            }
            LLAMA_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n",
                __func__, ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) {
                target = ovrd->val_bool;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
                target = ovrd->val_i64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) {
                target = ovrd->val_f64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_STR, ovrd)) {
                target = ovrd->val_str;
                return true;
            }
            return false;
        }

        static bool set(const gguf_context * ctx, const int k, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            if (try_override<T>(target, ovrd)) {
                return true;
            }
            if (k < 0) { return false; }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const gguf_context * ctx, const char * key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, gguf_find_key(ctx, key), target, ovrd);
        }

        static bool set(const gguf_context * ctx, const std::string & key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, key.c_str(), target, ovrd);
        }
    };
}

static bool parse_tensor_layer_index(const std::string & name, uint32_t & layer) {
    if (name.rfind("blk.", 0) != 0) {
        return false;
    }

    const char * first = name.data() + 4;
    const char * last  = first;
    const char * end   = name.data() + name.size();

    while (last < end && *last != '.') {
        ++last;
    }

    if (last == first || last == end) {
        return false;
    }

    auto result = std::from_chars(first, last, layer);
    return result.ec == std::errc() && result.ptr == last;
}

static bool is_split_expert_tensor(const std::string & name, uint32_t & expert) {
    static const char * prefixes[] = { "ffn_gate.", "ffn_down.", "ffn_up." };

    const size_t layer_end = name.find('.', 4);
    if (layer_end == std::string::npos) {
        return false;
    }

    const size_t prefix_begin = layer_end + 1;

    for (const char * prefix : prefixes) {
        const size_t prefix_len = std::char_traits<char>::length(prefix);
        if (name.compare(prefix_begin, prefix_len, prefix) != 0) {
            continue;
        }

        const size_t expert_begin = prefix_begin + prefix_len;
        const size_t expert_end = name.find('.', expert_begin);
        if (expert_end == std::string::npos || expert_end == expert_begin) {
            continue;
        }

        auto result = std::from_chars(name.data() + expert_begin, name.data() + expert_end, expert);
        if (result.ec == std::errc() && result.ptr == name.data() + expert_end) {
            return true;
        }
    }

    return false;
}

static bool is_merged_expert_tensor(llm_tensor tensor_type) {
    switch (tensor_type) {
        case LLM_TENSOR_FFN_NORM_EXPS:
        case LLM_TENSOR_FFN_DOWN_EXPS:
        case LLM_TENSOR_FFN_GATE_EXPS:
        case LLM_TENSOR_FFN_UP_EXPS:
        case LLM_TENSOR_FFN_GATE_UP_EXPS:
        case LLM_TENSOR_FFN_EXP_PROBS_B:
            return true;
        default:
            return false;
    }
}

static void coalesce_ranges(std::vector<llama_file_range> & ranges) {
    ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [](const llama_file_range & range) {
        return range.empty();
    }), ranges.end());

    std::sort(ranges.begin(), ranges.end(), [](const llama_file_range & lhs, const llama_file_range & rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return lhs.last < rhs.last;
    });

    std::vector<llama_file_range> merged;
    merged.reserve(ranges.size());

    for (const auto & range : ranges) {
        if (merged.empty() || range.first > merged.back().last) {
            merged.push_back(range);
            continue;
        }
        merged.back().last = std::max(merged.back().last, range.last);
    }

    ranges = std::move(merged);
}

llama_model_loader::llama_model_loader(const std::string & fname, int ncmoe, bool use_mmap, bool check_tensors,
        bool repack_tensors, bool use_thp, bool merge_qkv, bool merge_up_gate_exps, bool defer_experts,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p) {
    int trace = 0;
    if (getenv("LLAMA_TRACE")) {
        trace = atoi(getenv("LLAMA_TRACE"));
    }

#ifdef _WIN32
    // Only bump maxstdio if the user really wants large contexts:
#if defined(GGML_MAX_CONTEXTS) && (GGML_MAX_CONTEXTS > 512)
    // Cap at MSVC's hard limit of 8192 - https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/setmaxstdio?view=msvc-160
#if (GGML_MAX_CONTEXTS > 8192)
#define _GGML_STDIO_TARGET 8192
#else
#define _GGML_STDIO_TARGET GGML_MAX_CONTEXTS
#endif
    int _setmaxstdio_ret = _setmaxstdio(_GGML_STDIO_TARGET);
    if (_setmaxstdio_ret == -1) {
        LLAMA_LOG_INFO("%s: failed to set max stdio to %d. (setmaxstdio returned -1)\n", __func__, _GGML_STDIO_TARGET);
    } else {
        LLAMA_LOG_INFO("%s: max stdio successfully set to %d\n", __func__, _setmaxstdio_ret);
    }
#endif // GGML_MAX_CONTEXTS > 512
#endif // _WIN32

    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            kv_overrides.insert({std::string(p->key), *p});
        }
    }

    tensor_buft_overrides = param_tensor_buft_overrides_p;

    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    // ── .den native format detection ─────────────────────────────────────
    {
        FILE * fm = fopen(fname.c_str(), "rb");
        uint32_t magic = 0;
        if (fm) { size_t r = fread(&magic, 1, 4, fm); (void) r; fclose(fm); }
        is_den = (magic == denrt::DEN_MAGIC);
    }

    if (is_den) {
        if (!load_den_model(fname)) {
            throw std::runtime_error(format("%s: failed to load .den model from %s\n", __func__, fname.c_str()));
        }
    } else {
        meta = gguf_init_from_file(fname.c_str(), params);
        if (!meta) {
            throw std::runtime_error(format("%s: failed to load model from %s\n", __func__, fname.c_str()));
        }

        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));

        files.emplace_back(new llama_file(fname.c_str(), "rb"));
        contexts.emplace_back(ctx);

        // Save tensors data offset of the main file.
        // For subsidiary files, `meta` tensor data offset must not be used,
        // so we build a unified tensors index for weights.
        for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
            weights.emplace_back(files.back().get(), 0, cur->name, meta, cur);
        }
    }
    uint16_t n_split = 0;
    get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

    // Load additional GGML contexts
    if (!is_den && n_split > 1) {
        uint16_t idx = 0;
        get_key(llm_kv(LLM_KV_SPLIT_NO), idx);
        if (idx != 0) {
            throw std::runtime_error(format("illegal split file: %d, model must be loaded with the first split", idx));
        }

        char split_prefix[PATH_MAX] = {0};
        if (!llama_split_prefix(split_prefix, sizeof(split_prefix), fname.c_str(), idx, n_split)) {
            throw std::runtime_error(format("invalid split file: %s", fname.c_str()));
        }

        if (trace > 0) {
            LLAMA_LOG_INFO("%s: loading additional %d GGUFs\n", __func__, n_split);
        }

        char split_path[PATH_MAX] = {0};
        for (idx = 1; idx < n_split; idx++) {
            llama_split_path(split_path, sizeof(split_path), split_prefix, idx, n_split);

            struct gguf_init_params split_params = {
                /*.no_alloc = */ true,
                /*.ctx      = */ &ctx,
            };
            struct gguf_context * ctx_gguf = gguf_init_from_file(split_path, split_params);
            if (!ctx_gguf) {
                throw std::runtime_error(format("%s: failed to load GGUF split from %s\n", __func__, split_path));
            }

            files.emplace_back(new llama_file(split_path, "rb"));
            contexts.emplace_back(ctx);

            // Save tensors data offset info of the shard.
            for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
                weights.emplace_back(files.back().get(), idx, cur->name, ctx_gguf, cur);
            }

            gguf_free(ctx_gguf);
        }

        get_key(llm_kv(LLM_KV_SPLIT_TENSORS_COUNT), n_tensors);

        // sanity check
        {
            const int n_tensors_loaded = (int) weights.size();
            if (n_tensors != n_tensors_loaded) {
                throw std::runtime_error(format("corrupted model: %d tensors expected but %d found", n_tensors, n_tensors_loaded));
            }
        }

        LLAMA_LOG_INFO("%s: additional %d GGUFs metadata loaded.\n",  __func__, n_split - 1);
    }

    n_kv      = gguf_get_n_kv(meta);
    n_tensors = weights.size();

    fver = (enum llama_fver) gguf_get_version(meta);

    std::set<std::string> tensor_names;
    for (auto & w : weights) {
        n_elements += ggml_nelements(w.tensor);
        n_bytes    += ggml_nbytes(w.tensor);
        // make sure there is no duplicated tensor names
        const std::string name(w.tensor->name);
        auto found = tensor_names.find(name);
        if (found != tensor_names.end()) {
            throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", w.tensor->name));
        }
        tensor_names.insert(name);
    }

    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
            __func__, n_kv, n_tensors, fname.c_str(), llama_file_version_name(fver));

    // determine file type based on the number of tensors for each quantization and print meta data
    // TODO: make optional
    {
        std::map<enum ggml_type, uint32_t> n_type;

        uint32_t n_type_max = 0;
        enum ggml_type type_max = GGML_TYPE_F32;

        for (int i = 0; i < n_tensors; i++) {
            const ggml_tensor * tensor = weights.at(i).tensor;
            enum ggml_type type = tensor->type;

            n_type[type]++;

            if (n_type_max < n_type[type]) {
                n_type_max = n_type[type];
                type_max   = type;
            }

            if (trace > 0) {
                const uint16_t sid = weights.at(i).idx;
                LLAMA_LOG_INFO("%s: - tensor %4d, split %2d: %32s %-8s [ %s ]\n", __func__, i, sid, ggml_get_name(tensor), ggml_type_name(type), llama_format_tensor_shape(tensor).c_str());
            }
        }

        switch (type_max) {
            case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
            case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
            case GGML_TYPE_BF16:    ftype = LLAMA_FTYPE_MOSTLY_BF16;    break;
            case GGML_TYPE_BF16_R16:ftype = LLAMA_FTYPE_MOSTLY_BF16_R16;break;
            case GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
            case GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
            case GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
            case GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
            case GGML_TYPE_Q6_0:    ftype = LLAMA_FTYPE_MOSTLY_Q6_0;    break;
            case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
            case GGML_TYPE_Q8_KV:   ftype = LLAMA_FTYPE_MOSTLY_Q8_KV;   break;
            case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
            case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
            case GGML_TYPE_Q3_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q3_K_R4; break;
            case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
            case GGML_TYPE_Q4_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q4_K_R4; break;
            case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
            case GGML_TYPE_Q5_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q5_K_R4; break;
            case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
            case GGML_TYPE_Q6_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q6_K_R4; break;
            case GGML_TYPE_Q8_K_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_K_R8; break;
            case GGML_TYPE_Q8_KV_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_KV_R8; break;
            case GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
            case GGML_TYPE_IQ2_XXS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4; break;
            case GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
            case GGML_TYPE_IQ2_XS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS_R4; break;
            case GGML_TYPE_IQ2_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KS;  break;
            case GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_M;   break;
            case GGML_TYPE_IQ2_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_M_R4;break;
            case GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
            case GGML_TYPE_IQ3_XXS_R4: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4; break;
            case GGML_TYPE_IQ1_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ1_KT;  break;
            case GGML_TYPE_IQ2_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KT;  break;
            case GGML_TYPE_IQ3_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ3_KT;  break;
            case GGML_TYPE_IQ4_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_KT;  break;
            case GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
            case GGML_TYPE_IQ1_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ1_S_R4;break;
            case GGML_TYPE_IQ1_M_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ1_M_R4;break;
            case GGML_TYPE_IQ1_M:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;   break;
            case GGML_TYPE_IQ1_BN:  ftype = LLAMA_FTYPE_MOSTLY_IQ1_BN;  break;
            case GGML_TYPE_IQ2_BN:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_BN;  break;
            case GGML_TYPE_IQ2_BN_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_BN_R4;break;
            case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
            case GGML_TYPE_IQ4_NL_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL_R4;break;
            case GGML_TYPE_IQ4_XS_R8:ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS_R8;break;
            case GGML_TYPE_Q4_0_R8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_R8; break;
            case GGML_TYPE_Q5_0_R4: ftype = LLAMA_FTYPE_MOSTLY_Q5_0_R4; break;
            case GGML_TYPE_Q6_0_R4: ftype = LLAMA_FTYPE_MOSTLY_Q6_0_R4; break;
            case GGML_TYPE_Q8_0_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_0_R8; break;
            case GGML_TYPE_MXFP4:   ftype = LLAMA_FTYPE_MOSTLY_MXFP4;   break;
            case GGML_TYPE_NVFP4:   ftype = LLAMA_FTYPE_MOSTLY_NVFP4;   break;
            case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
            case GGML_TYPE_IQ4_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_KS;  break;
            case GGML_TYPE_IQ4_KS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_KS_R4;  break;
            case GGML_TYPE_IQ5_KS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ5_KS_R4;  break;
            case GGML_TYPE_IQ4_KSS: ftype = LLAMA_FTYPE_MOSTLY_IQ4_KSS; break;
            case GGML_TYPE_IQ5_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ5_KS;  break;
            case GGML_TYPE_IQ2_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_K;   break;
            case GGML_TYPE_IQ2_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_K_R4;break;
            case GGML_TYPE_IQ3_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ3_KS;  break;
            case GGML_TYPE_IQ2_KL:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KL;  break;
            case GGML_TYPE_IQ3_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_K;   break;
            case GGML_TYPE_IQ3_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ3_K_R4;break;
            case GGML_TYPE_IQ4_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ4_K;   break;
            case GGML_TYPE_IQ4_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_K_R4;break;
            case GGML_TYPE_IQ5_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ5_K;   break;
            case GGML_TYPE_IQ5_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ5_K_R4;break;
            case GGML_TYPE_IQ6_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ6_K;   break;
            case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
            case GGML_TYPE_IQ3_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ3_S_R4;break;
            case GGML_TYPE_Q4_0_4_4: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_4_4; break;
            case GGML_TYPE_Q4_0_4_8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_4_8; break;
            case GGML_TYPE_Q4_0_8_8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_8_8; break;
            default:
                {
                     LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                     ftype = LLAMA_FTYPE_ALL_F32;
                } break;
        }

        // this is a way to mark that we have "guessed" the file type
        ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

        {
            const int kid = gguf_find_key(meta, "general.file_type"); // TODO: use LLM_KV
            if (kid >= 0) {
                ftype = (llama_ftype) gguf_get_val_u32(meta, kid);
            }
        }

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(meta, i);
            const enum gguf_type type   = gguf_get_kv_type(meta, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%d]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(meta, i)), gguf_get_arr_n(meta, i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(meta, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    if (!llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        use_mmap = false;
    }
    if (repack_tensors) {
        use_mmap = false;
    }

    this->ncmoe = ncmoe;
    // NVFP4 tensors need 144->160 byte expansion at load, which requires the
    // non-mmap file path (cur->data preallocated). Disable mmap if any NVFP4
    // tensors are present so the expansion branch in load_all_data works.
    for (const auto & w : weights) {
        if (w.tensor != nullptr && w.tensor->type == GGML_TYPE_NVFP4) {
            use_mmap = false;
            break;
        }
    }
    // .den tensors that need an on-load transform (A_log -> -exp) cannot be
    // mapped in place; use the read/copy path.
    if (is_den && den_has_xform) {
        use_mmap = false;
    }
    this->use_mmap = use_mmap;
    this->check_tensors = check_tensors;
    this->repack_tensors = repack_tensors;
    this->use_thp = use_thp;
    this->merge_qkv = merge_qkv;

    this->merge_up_gate_exps = merge_up_gate_exps;
    this->defer_experts = defer_experts;
}

bool llama_model_loader::load_den_model(const std::string & fname) {
    denrt::den_header hdr;
    std::vector<denrt::den_tensor> entries;
    if (!denrt::load(fname.c_str(), &hdr, &entries)) {
        return false;
    }

    uint32_t ssm_dt_rank = 0, ssm_d_inner = 0, head_dim = 0;
    denrt::derive_hparams(hdr, entries, &ssm_dt_rank, &ssm_d_inner, &head_dim);
    if (ssm_dt_rank == 0 || ssm_d_inner == 0 || head_dim == 0) {
        LLAMA_LOG_ERROR("%s: could not derive .den hparams (dt_rank=%u inner=%u head_dim=%u)\n",
                __func__, ssm_dt_rank, ssm_d_inner, head_dim);
        return false;
    }

    // .den header computes n_rot from hidden//n_heads which is wrong for the
    // rope kernel (Qwen3.5 uses partial_rotary_factor * head_dim, e.g. 0.25*256=64).
    uint32_t n_rot = hdr.n_rot;
    if (n_rot > head_dim) {
        n_rot = head_dim / 4; // Qwen3.5 partial_rotary_factor == 0.25
        LLAMA_LOG_INFO("%s: .den n_rot %u exceeds head_dim %u -- clamped to %u\n",
                __func__, hdr.n_rot, head_dim, n_rot);
    }

    meta = gguf_init_empty();
    // MoE slot-.den (n_experts>0) loads as the qwen35moe arch (fused expert
    // gate_up, shared experts, ffn_gate_inp); dense .den stays qwen35.
    const bool den_moe = hdr.n_experts > 0;
    llm_kv = LLM_KV(den_moe ? LLM_ARCH_QWEN35MOE : LLM_ARCH_QWEN35);
    arch_name = den_moe ? "qwen35moe" : "qwen35";

    // ── Populate architecture + hparams KV (what llm_load_hparams reads) ─
#define DEN_SET_U32(kid, val) gguf_set_val_u32(meta, llm_kv(kid).c_str(), (uint32_t)(val))
#define DEN_SET_F32(kid, val) gguf_set_val_f32(meta, llm_kv(kid).c_str(), (float)(val))
#define DEN_SET_STR(kid, val) gguf_set_val_str(meta, llm_kv(kid).c_str(), (val))
    DEN_SET_STR(LLM_KV_GENERAL_ARCHITECTURE, "qwen35");
    DEN_SET_U32(LLM_KV_BLOCK_COUNT,          hdr.n_layers);
    DEN_SET_U32(LLM_KV_CONTEXT_LENGTH,       hdr.max_seq_len);
    DEN_SET_U32(LLM_KV_EMBEDDING_LENGTH,     hdr.hidden_size);
    DEN_SET_U32(LLM_KV_FEED_FORWARD_LENGTH,  hdr.ffn_size);
    DEN_SET_U32(LLM_KV_VOCAB_SIZE,           hdr.vocab_size);
    DEN_SET_U32(LLM_KV_ATTENTION_HEAD_COUNT,     hdr.n_heads);
    DEN_SET_U32(LLM_KV_ATTENTION_HEAD_COUNT_KV,  hdr.n_kv_heads);
    DEN_SET_F32(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hdr.rms_norm_eps);
    DEN_SET_U32(LLM_KV_ATTENTION_KEY_LENGTH,  head_dim);
    DEN_SET_U32(LLM_KV_ATTENTION_VALUE_LENGTH, head_dim);
    {
        const uint32_t sections[4] = { 11, 11, 10, 0 };
        gguf_set_arr_data(meta, llm_kv(LLM_KV_ROPE_DIMENSION_SECTIONS).c_str(), GGUF_TYPE_UINT32, sections, 4);
    }
    DEN_SET_U32(LLM_KV_ROPE_DIMENSION_COUNT, n_rot);
    DEN_SET_F32(LLM_KV_ROPE_FREQ_BASE,       hdr.rope_theta);
    DEN_SET_U32(LLM_KV_SSM_CONV_KERNEL,      hdr.ssm_conv_kernel);
    DEN_SET_U32(LLM_KV_SSM_STATE_SIZE,       hdr.ssm_state_size);
    DEN_SET_U32(LLM_KV_SSM_GROUP_COUNT,      hdr.ssm_group_count);
    DEN_SET_U32(LLM_KV_SSM_TIME_STEP_RANK,   ssm_dt_rank);
    DEN_SET_U32(LLM_KV_SSM_INNER_SIZE,       ssm_d_inner);
    DEN_SET_U32(LLM_KV_FULL_ATTENTION_INTERVAL, hdr.full_attention_interval);
    DEN_SET_U32(LLM_KV_EXPERT_COUNT,         hdr.n_experts);
    DEN_SET_U32(LLM_KV_EXPERT_USED_COUNT,    hdr.n_experts_used);
#undef DEN_SET_U32
#undef DEN_SET_F32
#undef DEN_SET_STR

    // ── Graft tokenizer KV from a companion vocab GGUF ──────────────────
    // .den stores weights only; llama.cpp requires the tokenizer.ggml.* KV.
    // Look for <model>.vocab.gguf next to the .den file.
    {
        std::string vocab_path = fname;
        const size_t dot = vocab_path.find_last_of('.');
        if (dot != std::string::npos) vocab_path = vocab_path.substr(0, dot);
        vocab_path += ".vocab.gguf";
        struct ggml_context * vocab_ctx = NULL;
        struct gguf_init_params vparams = { /*.no_alloc=*/ true, /*.ctx=*/ &vocab_ctx };
        struct gguf_context * vocab_meta = gguf_init_from_file(vocab_path.c_str(), vparams);
        if (vocab_meta) {
            gguf_set_kv(meta, vocab_meta);
            gguf_free(vocab_meta);
            if (vocab_ctx) ggml_free(vocab_ctx);
            LLAMA_LOG_INFO("%s: grafted tokenizer KV from %s\n", __func__, vocab_path.c_str());
        } else {
            LLAMA_LOG_WARN("%s: no companion vocab GGUF at %s -- tokenizer will be missing\n",
                    __func__, vocab_path.c_str());
        }
    }

    // ── Create tensor metas + weights ───────────────────────────────────
    files.emplace_back(new llama_file(fname.c_str(), "rb"));
    {
        struct ggml_init_params den_params = {
            /*.mem_size   =*/ 128 * 1024 * 1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx_den = ggml_init(den_params);
        contexts.emplace_back(ctx_den);
        for (const auto & e : entries) {
            const std::string name = denrt::slot_to_name(e.slot, hdr.n_layers);
            if (name.empty()) {
                continue;
            }
            const int64_t ndim = std::min((int64_t) e.ndim, (int64_t) GGML_MAX_DIMS);
            const int64_t ne[GGML_MAX_DIMS] = {
                ndim >= 1 ? e.dims[ndim - 1] : 1,
                ndim >= 2 ? e.dims[ndim - 2] : 1,
                ndim >= 3 ? e.dims[ndim - 3] : 1,
                ndim >= 4 ? e.dims[ndim - 4] : 1,
            };
            enum ggml_type type;
            switch (e.hw_target) {
                case denrt::DEN_HW_BF16:  type = GGML_TYPE_BF16;  break;
                case denrt::DEN_HW_F32:   type = GGML_TYPE_F32;   break;
                case denrt::DEN_HW_F16:   type = GGML_TYPE_F16;   break;
                case denrt::DEN_HW_NVFP4: type = GGML_TYPE_NVFP4; break;
                default:
                    LLAMA_LOG_WARN("%s: .den slot %u unsupported hw_target %u -- skipped\n",
                            __func__, e.slot, e.hw_target);
                    continue;
            }
            const uint32_t sub = e.slot >= 3 ? (e.slot - 3) % denrt::DEN_LAYER_STRIDE : 0;
            // ssm_conv1d must be F32 for the delta-net conv kernel
            if (sub == 17) {
                type = GGML_TYPE_F32;
            }
            ggml_tensor * t = ggml_new_tensor(ctx_den, type, (int) ndim, ne);
            ggml_set_name(t, name.c_str());
            weights.emplace_back(0, (size_t)(hdr.data_offset + e.data_offset), t);
            if (sub == 15) { // A_log -> ssm_a transform
                weights.back().xform = llama_tensor_weight::XFORM_NEG_EXP;
                den_has_xform = true;
            } else if (sub == 17) { // conv1d BF16 -> F32
                weights.back().xform = llama_tensor_weight::XFORM_BF16_TO_F32;
                den_has_xform = true;
            }
            if (e.hw_target == denrt::DEN_HW_NVFP4) {
                weights.back().nvfp4_160b = true;
            }
        }
    }

    LLAMA_LOG_INFO("%s: loaded .den header: layers=%u hidden=%u heads=%u kv=%u ffn=%u vocab=%u experts=%u/%u "
            "ssm_state=%u ssm_grp=%u ssm_dt=%u ssm_inner=%u head_dim=%u\n",
            __func__, hdr.n_layers, hdr.hidden_size, hdr.n_heads, hdr.n_kv_heads, hdr.ffn_size,
            hdr.vocab_size, hdr.n_experts, hdr.n_experts_used, hdr.ssm_state_size, hdr.ssm_group_count,
            ssm_dt_rank, ssm_d_inner, head_dim);
    return true;
}

llama_model_loader::~llama_model_loader() {
    if (meta) {
        gguf_free(meta);
    }
    for (auto * ctx : contexts) {
        ggml_free(ctx);
    }
}

void llama_model_loader::build_expert_tensor_index(const llama_hparams & hparams) {
    expert_tensor_index = {};

    if (hparams.n_expert == 0 || hparams.n_layer == 0) {
        return;
    }

    expert_tensor_index.file_ranges.resize(files.size());

    size_t deferred_bytes = 0;
    const llm_arch arch = get_arch();

    for (const auto & weight : weights) {
        const std::string name(weight.tensor->name);
        uint32_t layer = 0;
        if (!parse_tensor_layer_index(name, layer)) {
            continue;
        }

        if (layer >= hparams.n_layer) {
            throw std::runtime_error(format("expert tensor '%s' has invalid layer index %u", name.c_str(), layer));
        }

        // check for split expert tensors (blk.N.ffn_gate.E.weight) by name pattern,
        // since llm_tensor_type can't resolve these (two %d in the format string)
        uint32_t expert = 0;
        if (is_split_expert_tensor(name, expert)) {
            if (expert >= hparams.n_expert) {
                throw std::runtime_error(format("expert tensor '%s' has invalid expert index %u", name.c_str(), expert));
            }
        } else {
            const llm_tensor tensor_type = llm_tensor_type(arch, name, int(layer));
            if (!is_merged_expert_tensor(tensor_type)) {
                continue;
            }
        }

        const size_t tensor_bytes = ggml_nbytes(weight.tensor);
        deferred_bytes += tensor_bytes;
        expert_tensor_index.file_ranges.at(weight.idx).push_back({ weight.offs, weight.offs + tensor_bytes });
    }

    for (auto & ranges : expert_tensor_index.file_ranges) {
        coalesce_ranges(ranges);
    }

    expert_tensor_index.deferred_bytes = deferred_bytes;
    expert_tensor_index.dense_bytes = n_bytes > deferred_bytes ? n_bytes - deferred_bytes : 0;
}

bool llama_model_loader::should_defer_expert_mmaps() const {
    return defer_experts && use_mmap && !expert_tensor_index.empty();
}

void llama_model_loader::drop_mmap_expert_pages() const {
    if (!use_mmap || mappings.empty() || expert_tensor_index.file_ranges.empty()) {
        return;
    }

    const size_t n_range_sets = std::min(mappings.size(), expert_tensor_index.file_ranges.size());
    for (size_t idx = 0; idx < n_range_sets; ++idx) {
        const auto & ranges = expert_tensor_index.file_ranges[idx];
        for (const auto & range : ranges) {
            mappings[idx]->dontneed_fragment(range.first, range.last);
        }
    }
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
llama_model_loader::get_arr_n(const std::string & key, T & result, const bool required) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }
        return false;
    }

    struct GGUFMeta::ArrayInfo arr_info =
        GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);


    result = arr_info.length;
    return true;
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
llama_model_loader::get_arr_n(const enum llm_kv kid, T & result, const bool required) {
    return get_arr_n(llm_kv(kid), result, required);
}

template<typename T>
bool llama_model_loader::get_arr(const std::string & key, std::vector<T> & result, const bool required) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY) {
        if (required) {
            throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
        }
        return false;
    }

    struct GGUFMeta::ArrayInfo arr_info =
        GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);

    switch (arr_info.gt) {
        case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same<T, float>::value)); break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_BOOL:
        case GGUF_TYPE_INT32:   GGML_ASSERT((std::is_same_v<T,  int32_t>) || (std::is_same_v<T, uint32_t>));  break;
        default:
            throw std::runtime_error(format("%s is not a float32, int32, uint32 or bool array", key.c_str()));
    }

    result.resize(arr_info.length);
    if (arr_info.gt == GGUF_TYPE_BOOL) {
        std::transform((const int8_t *)arr_info.data, (const int8_t *)arr_info.data + arr_info.length, result.begin(),
                [] (int8_t x) { return static_cast<T>(x != 0); });

    } else {
        result.assign((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length);
    }

    return true;
}

template<typename T, size_t N_MAX>
bool llama_model_loader::get_arr(const std::string & key, std::array<T, N_MAX> & result, const bool required) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY) {
        if (required) {
            throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
        }
        return false;
    }

    struct GGUFMeta::ArrayInfo arr_info =
        GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);

    switch (arr_info.gt) {
        case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same_v<T, float>)); break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_BOOL:
        case GGUF_TYPE_INT32:   GGML_ASSERT((std::is_same_v<T,  int32_t>) || (std::is_same_v<T, uint32_t>));  break;
        default:
            throw std::runtime_error(format("%s is not a float32, int32 array", key.c_str()));
    }

    if (arr_info.length > N_MAX) {
        throw std::runtime_error(format("array length %u for key %s exceeds max %u", (uint32_t) arr_info.length, key.c_str(), (uint32_t) N_MAX));
    }

    if (arr_info.gt == GGUF_TYPE_BOOL) {
        std::transform((const int8_t *)arr_info.data, (const int8_t *)arr_info.data + arr_info.length, result.begin(),
                [] (int8_t x) { return static_cast<T>(x != 0); });
    } else {
        std::copy((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length, result.begin());
    }

    return true;
}

template<typename T>
bool llama_model_loader::get_arr(const enum llm_kv kid, T & result, const bool required) {
    return get_arr(llm_kv(kid), result, required);
}

template<typename T>
bool llama_model_loader::get_key(const std::string & key, T & result, const bool required) {
    auto it = kv_overrides.find(key);

    const struct llama_model_kv_override * override =
        it != kv_overrides.end() ? &it->second : nullptr;

    const bool found = GGUFMeta::GKV<T>::set(meta, key, result, override);

    if (required && !found) {
        throw std::runtime_error(format("key not found in model: %s", key.c_str()));
    }

    return found;
}

template<typename T>
bool llama_model_loader::get_key(const enum llm_kv kid, T & result, const bool required) {
    return get_key(llm_kv(kid), result, required);
}

// get array of n <= N_MAX elements, or a single element repeated n times
template<typename T, size_t N_MAX>
bool llama_model_loader::get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, const bool required) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }
        return false;
    }

    if (n > N_MAX) {
        throw std::runtime_error(format("n > N_MAX: %u > %u for key %s", (uint32_t) n, (uint32_t) N_MAX, key.c_str()));
    }

    if (gguf_get_kv_type(meta, kid) == GGUF_TYPE_ARRAY) {
        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);

        if (n != arr_info.length) {
            throw std::runtime_error(format("key %s has wrong array length; expected %u, got %u", key.c_str(), n, (uint32_t) arr_info.length));
        }

        return get_arr(key, result, required);
    } else {
        T value;

        bool ok = get_key(key, value, required);
        if (!ok) {
            return false;
        }

        for (uint32_t i = 0; i < n; i++) {
            result[i] = value;
        }

        return true;
    }
}

template<typename T>
bool llama_model_loader::get_key_or_arr(const enum llm_kv kid, T & result, uint32_t n, const bool required) {
    return get_key_or_arr(llm_kv(kid), result, n, required);
}

const char * llama_model_loader::get_tensor_name(int i) const {
    return weights.at(i).tensor->name;
}

const llama_model_loader::llama_tensor_weight * llama_model_loader::get_weight(const char * name) const {
    for (const auto & weight : weights) {
        if (strcmp(name, weight.tensor->name) == 0) {
            return &weight;
        }
    }
    return nullptr;
}

const llama_model_loader::llama_tensor_weight & llama_model_loader::require_weight(const char * name) const {
    const llama_tensor_weight * weight = get_weight(name);
    if (!weight) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
    }
    return *weight;
}

struct ggml_tensor * llama_model_loader::get_tensor_meta(const char * name) const {
    const auto * weight = get_weight(name);
    if (!weight) {
        return nullptr;
    }
    return weight->tensor;
}

struct ggml_tensor * llama_model_loader::require_tensor_meta(const char * name) const {
    struct ggml_tensor * tensor = get_tensor_meta(name);
    if (!tensor) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
    }
    return tensor;
}

struct ggml_tensor * llama_model_loader::create_tensor_for(struct ggml_context * ctx, const struct ggml_tensor * cur, bool duplicated) {
    struct ggml_tensor * tensor = ggml_dup_tensor(ctx, cur);
    ggml_set_name(tensor, ggml_get_name(cur));

    if (duplicated) {
        size_data += ggml_nbytes(cur);
    } else {
        n_created++;
    }

    return tensor;
}

const struct ggml_tensor * llama_model_loader::check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required) const {
    const struct ggml_tensor * cur = get_tensor_meta(name.c_str());

    if (cur == NULL) {
        if (!required) {
            return NULL;
        }
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }

    {
        bool is_ok = true;
        for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
            if ((i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)) {
                is_ok = false;
                break;
            }
        }
        if (!is_ok) {
            throw std::runtime_error(
                    format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                        __func__, name.c_str(),
                        llama_format_tensor_shape(ne).c_str(),
                        llama_format_tensor_shape(cur).c_str()));
        }
    }

    return cur;
}

struct ggml_tensor * llama_model_loader::create_tensor(struct ggml_context * ctx, const std::string & name,
        const std::vector<int64_t> & ne, int flags) {
    const struct ggml_tensor * cur = check_tensor_dims(name, ne, !(flags & TENSOR_NOT_REQUIRED));

    if (cur == NULL) {
        return NULL;
    }

    // skip unused tensors
    if (flags & TENSOR_SKIP) {
        const size_t nbytes = ggml_nbytes(cur);
        LLAMA_LOG_WARN("model has unused tensor %s (size = %zu bytes) -- ignoring\n", name.c_str(), nbytes);

        size_data -= nbytes;
        n_created++;

        return nullptr;
    }

    return create_tensor_for(ctx, cur, flags & TENSOR_DUPLICATED);
}

struct ggml_tensor * llama_model_loader::create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base,
        const std::string & name, const std::vector<int64_t> & ne, size_t offset, bool required) {
    const struct ggml_tensor * cur = check_tensor_dims(name, ne, required);

    if (cur == NULL) {
        return NULL;
    }

    if (cur->type != base->type) {
        throw std::runtime_error(format("%s: tensor '%s' has wrong type; expected %s, got %s", __func__, name.c_str(), ggml_type_name(base->type), ggml_type_name(cur->type)));
    }

    std::array<int64_t, GGML_MAX_DIMS> dims;
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        dims[i] = i < ne.size() ? ne[i] : 1;
    }

    struct ggml_tensor * tensor = ggml_view_4d(ctx, base,
            dims[0], dims[1], dims[2], dims[3],
            cur->nb[1], cur->nb[2], cur->nb[3],
            offset);

    ggml_set_name(tensor, name.c_str());

    n_created++;

    return tensor;
}

void llama_model_loader::done_getting_tensors() const {
    // Exclude tensors that are consumed as companion metadata rather than
    // created as standalone ggml tensors:
    //   1. "_n" suffix scalars   : NVFP4 per-tile norm factors (fork converter format)
    //   2. ".scale"/".input_scale" : per-tensor scale factors emitted next to every
    //      NVFP4 weight by upstream converters. The arch loaders do not create them
    //      as ggml tensors (scales are embedded in the 144B/160B NVFP4 blocks).
    int n_meta = 0;
    for (const auto & w : weights) {
        const std::string name = ggml_get_name(w.tensor);
        const size_t len = name.size();
        if (len > 2 && name[len-2] == '_' && name[len-1] == 'n') {
            n_meta++;
            continue;
        }
        static const char * const suffixes[] = { ".input_scale", ".scale" };
        for (const char * suf : suffixes) {
            const size_t slen = strlen(suf);
            if (len > slen && name.compare(len - slen, slen, suf) == 0) {
                // only treat it as a companion when the sibling weight is NVFP4
                const std::string wname = name.substr(0, len - slen) + ".weight";
                const ggml_tensor * wt = get_tensor_meta(wname.c_str());
                if (wt && wt->type == GGML_TYPE_NVFP4) {
                    n_meta++;
                }
                break;
            }
        }
    }
    const int n_expected = n_tensors - n_meta;
    if (n_created != n_expected) {
        throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_expected, n_created));
    }
}

void llama_model_loader::init_mappings(bool prefetch, llama_mlocks * mlock_mmaps, bool use_thp) {
    if (use_mmap) {
        mappings.reserve(files.size());
        mmaps_used.reserve(files.size());
        for (const auto & file : files) {
            std::unique_ptr<llama_mmap> mapping(new llama_mmap(file.get(), prefetch ? -1 : 0, ggml_is_numa(), use_thp));
            mmaps_used.emplace_back(mapping->size(), 0);
            if (mlock_mmaps) {
                std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                mlock_mmap->init(mapping->addr());
                mlock_mmaps->emplace_back(std::move(mlock_mmap));
            }
            mappings.emplace_back(std::move(mapping));
        }
    }

    // compute the total size of all tensors for progress reporting
    for (auto & w : weights) {
        size_data += ggml_nbytes(w.tensor);
    }
}

void llama_model_loader::get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const {
    GGML_ASSERT(!mappings.empty());
    const auto & mapping = mappings.at(idx);

    *first = mapping->size();
    *last  = 0;
    *addr = mapping->addr();
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        try {
            const auto * weight = get_weight(ggml_get_name(tensor));
            if (!weight) {
                continue;
            }
            if (weight->idx != idx) {
                continue;
            }
            *first = std::min(*first, weight->offs);
            *last  = std::max(*last,  weight->offs + ggml_nbytes(tensor));
        } catch(...) {
            // the tensor is not in the model
        }
    }
}

// for backwards compatibility, does not support ggml-backend
void llama_model_loader::load_data_for(struct ggml_tensor * cur) const {
    const auto & w = require_weight(ggml_get_name(cur));

    if (w.xform == llama_tensor_weight::XFORM_NEG_EXP) {
        // A_log -> ssm_a = -exp(A_log)
        GGML_ASSERT(cur->data != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(cur->data, ggml_nbytes(cur));
        float * p = (float *) cur->data;
        for (size_t i = 0; i < ggml_nelements(cur); ++i) {
            p[i] = -expf(p[i]);
        }
        return;
    }

    if (w.xform == llama_tensor_weight::XFORM_BF16_TO_F32) {
        // ssm_conv1d: on-disk BF16, engine requires F32
        GGML_ASSERT(cur->data != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        const size_t n_elem = ggml_nelements(cur);
        std::vector<uint16_t> raw(n_elem);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(raw.data(), n_elem * sizeof(uint16_t));
        float * p = (float *) cur->data;
        for (size_t i = 0; i < n_elem; ++i) {
            uint32_t u = ((uint32_t) raw[i]) << 16;
            float f; memcpy(&f, &u, 4); p[i] = f;
        }
        return;
    }

    if (use_mmap) {
        const auto & mapping = mappings.at(w.idx);
        if (cur->data == nullptr) {
            cur->data = (uint8_t *)mapping->addr() + w.offs;
        } else {
            memcpy(cur->data, (uint8_t *)mapping->addr() + w.offs, ggml_nbytes(cur));
        }
    } else {
        GGML_ASSERT(cur->data != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(cur->data, ggml_nbytes(cur));
    }

    if (check_tensors && !ggml_validate_row_data(cur->type, cur->data, ggml_nbytes(cur))) {
        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
    }
}

// Returns false if cancelled by progress_callback
bool llama_model_loader::load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs_mmap,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data) {
    GGML_ASSERT(size_data != 0 && "call init_mappings() first");

    std::vector<no_init<uint8_t>> read_buf;
    std::vector<std::future<std::pair<ggml_tensor *, bool>>> validation_result;

#if defined(GGML_USE_CUDA)
    // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.
    // NVMe raid configurations might require more / larger buffers.
    constexpr size_t n_buffers = 4;
    constexpr size_t buffer_size = 1 * 1024 * 1024; // 1MB

    std::vector<ggml_backend_buffer_t> host_buffers;
    std::vector<void*> host_ptrs;
    std::vector<ggml_backend_event_t> events;
    size_t buffer_idx = 0; // buffer to use for async loads

    ggml_backend_t cuda_backend = nullptr;
    if (!use_mmap && !check_tensors) {
        // When not using mmaped io use async uploads from pinned memory to GPU memory.
        // First determine if the CUDA backend is active, and if so, determine the device ID.
        ggml_backend_buffer_t buf = bufs_mmap.count(0) ? bufs_mmap.at(0) : nullptr;
        if (buf) {
            ggml_backend_buffer_type_t buffer_type = ggml_backend_buffer_get_type(buf);
            for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
                auto * cuda_buffer_type = ggml_backend_cuda_buffer_type(i);
                if (buffer_type == cuda_buffer_type) {
                    cuda_backend = ggml_backend_cuda_init(i, nullptr);
                    break;
                }
            }
        }

        // If the cuda backend is active create pinned memory buffers and events for synchronisation.
        if (cuda_backend) {
            for (size_t idx = 0; idx < n_buffers; ++idx) {
                host_buffers.emplace_back(ggml_backend_buft_alloc_buffer(llama_default_buffer_type_cpu(true), buffer_size));
                host_ptrs.emplace_back(ggml_backend_buffer_get_base(host_buffers[idx]));
                events.emplace_back(ggml_backend_event_new(cuda_backend));
            }
        }
    }
#endif
    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        const auto * weight = get_weight(ggml_get_name(cur));
        if (weight == nullptr) {
            // this can happen with split experts models
            continue;
        }

        if (progress_callback) {
            if (!progress_callback((float) size_done / size_data, progress_callback_user_data)) {
                return false;
            }
        }

        size_t n_size = ggml_nbytes(cur);

        // NVFP4: on-disk blocks are 144B (4x [4 scales][32 nibbles] per 64-elem
        // sub-block), in-memory NULLGLASS blocks are 160B (+16B header).
        // We expand ALWAYS via the non-mmap file path, staging the expansion in a
        // HOST buffer first. cur->data is NOT host-writable for GPU-offloaded
        // tensors (it is a CUDA device pointer once the buffer is allocated in
        // llm_load_tensors), so writing the 144B->160B expansion directly into it
        // segfaults/hangs. Stage in host memory, then upload with
        // ggml_backend_tensor_set for device buffers / memcpy for host buffers.
        if (cur->type == GGML_TYPE_NVFP4) {
            if (weight->nvfp4_160b) {
                // .den NVFP4: blocks are already 160B expanded NULLGLASS.
                const int64_t nblocks = ggml_nelements(cur) / ggml_blck_size(cur->type);
                const size_t n_mem = (size_t) nblocks * 160;
                const auto & file = files.at(weight->idx);
                std::vector<uint8_t> raw(n_mem);
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(raw.data(), n_mem);
                // Ensure the GEMV dispatch byte is set on empty headers.
                for (int64_t b = 0; b < nblocks; ++b) {
                    uint8_t * blk = raw.data() + b * 160;
                    if (blk[148] == 0) blk[148] = 0x10;
                }
                if (check_tensors && !ggml_validate_row_data(cur->type, (const uint8_t *) raw.data(), n_mem)) {
                    throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                }
                if (cur->buffer != nullptr && !ggml_backend_buffer_is_host(cur->buffer)) {
                    ggml_backend_tensor_set(cur, raw.data(), 0, n_mem);
                } else {
                    memcpy(cur->data, raw.data(), n_mem);
                }
                continue;
            }
            const int64_t nblocks = ggml_nelements(cur) / ggml_blck_size(cur->type);
            const size_t n_file  = (size_t) nblocks * 144;
            const size_t n_mem   = (size_t) nblocks * 160;
            GGML_ASSERT(n_mem == ggml_nbytes(cur) && "NVFP4 expansion size mismatch");

            // Force the file path even if the loader uses mmap for other types
            const auto & file = files.at(weight->idx);
            std::vector<uint8_t> raw(n_file);
            file->seek(weight->offs, SEEK_SET);
            file->read_raw(raw.data(), n_file);

            GGML_ASSERT(cur->data != nullptr && "NVFP4 tensor data not allocated");
            std::vector<uint8_t> expanded(n_mem);
            uint8_t * dst = expanded.data();
            const uint8_t * srcp = raw.data();
            for (int64_t b = 0; b < nblocks; ++b) {
                // De-interleave 4x [4 scales][32 nibbles] -> scales[0:16] + nibbles[16:144]
                for (int sb = 0; sb < 4; ++sb) {
                    const uint8_t * s = srcp + sb * 36;
                    memcpy(dst + sb * 4, s, 4);            // scales -> dst[0:16]
                    memcpy(dst + 16 + sb * 32, s + 4, 32); // nibbles -> dst[16:144]
                }
                memset(dst + 144, 0, 16);                  // NULLGLASS header
                srcp += 144; dst += 160;
            }
            // ── Global scale fold (OMMA_NATIVE_FUSED_SCALE step 1) ──
            // Companion .scale/.input_scale F32 tensor holds the per-tensor
            // global (~0.000138). Write it into tile_norm[144:147] (float32),
            // the NULLGLASS-designed field. Dequant multiplies by tile_norm.
            // Keeps block UE4M3 scales relative -> no precision underflow.
            const std::string tname = ggml_get_name(cur);
            // The companion tensors are '<base>.scale' / '<base>.input_scale'
            // where '<base>' is the weight name WITHOUT '.weight'. Appending
            // the suffix to the full name ('...weight.scale') never matches.
            // Prefer .scale (WEIGHT global): .input_scale is the ACTIVATION
            // scale, a different quantity. Models emit both (ModelOpt).
            std::string base_name = tname;
            static const char * const wSuffix = ".weight";
            const size_t wlen = strlen(wSuffix);
            if (base_name.size() > wlen && base_name.compare(base_name.size() - wlen, wlen, wSuffix) == 0) {
                base_name.erase(base_name.size() - wlen);
            }
            static const char * const gsuffixes[] = { ".scale" };  // weight-global ONLY
            std::vector<float> gval;
            for (const char * suf : gsuffixes) {
                const std::string gname = base_name + suf;
                const ggml_tensor * gt = get_tensor_meta(gname.c_str());
                if (gt && gt->type == GGML_TYPE_F32 && gt->ne[0] >= 1) {
                    const auto & wg = require_weight(gname.c_str());
                    gval.resize((size_t) ggml_nelements(gt));
                    const auto & gfile = files.at(wg.idx);
                    gfile->seek(wg.offs, SEEK_SET);
                    gfile->read_raw(gval.data(), gval.size() * sizeof(float));
                    break;
                }
            }
            // OMMA_NATIVE_FUSED_SCALE: fold per-expert (or per-tensor) global
            // scale into NULLGLASS header bytes 152:155. Bytes 144-145 are the
            // policy bits (null_skip/budget) — overlapping them with a float
            // made policy reads depend on the scale value. 152:155 is clean.
            // For MoE expert tensors (.scale has n_experts values), write each
            // expert's own scale into its blocks so the kernel reads the right
            // norm directly from the tile — no separate norm array, no OOB.
            const int64_t n_experts = cur->ne[2] > 1 ? cur->ne[2] : 1;
            const bool per_expert = (int64_t) gval.size() == n_experts && n_experts > 1;
            const int64_t blocks_per_expert = per_expert ? nblocks / n_experts : nblocks;
            const float default_scale = gval.empty() ? 1.0f : gval[0];
            for (int64_t b = 0; b < nblocks; ++b) {
                uint8_t * blk = expanded.data() + b * 160;
                float scale = default_scale;
                if (per_expert) {
                    scale = gval[b / blocks_per_expert];
                }
                if (scale != 1.0f) {
                    memcpy(blk + 152, &scale, 4);            // tile_norm (unified)
                }
                blk[148] = 0x10;                              // dispatch: GEMV
            }
            if (check_tensors && !ggml_validate_row_data(cur->type, (const uint8_t *)expanded.data(), n_mem)) {
                throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
            }
            // Upload: host buffers get a direct memcpy; device (CUDA) buffers need
            // a backend copy because cur->data is a device pointer there.
            if (cur->buffer != nullptr && !ggml_backend_buffer_is_host(cur->buffer)) {
                ggml_backend_tensor_set(cur, expanded.data(), 0, n_mem);
            } else {
                memcpy(cur->data, expanded.data(), n_mem);
            }
            continue;
        }

        if (weight->xform == llama_tensor_weight::XFORM_NEG_EXP) {
            // A_log -> ssm_a = -exp(A_log). Read raw F32, transform, upload.
            const auto & file = files.at(weight->idx);
            std::vector<float> raw(ggml_nelements(cur));
            file->seek(weight->offs, SEEK_SET);
            file->read_raw(raw.data(), raw.size() * sizeof(float));
            for (float & v : raw) {
                v = -expf(v);
            }
            if (cur->buffer != nullptr && !ggml_backend_buffer_is_host(cur->buffer)) {
                ggml_backend_tensor_set(cur, raw.data(), 0, n_size);
            } else {
                memcpy(cur->data, raw.data(), n_size);
            }
            continue;
        }

        if (weight->xform == llama_tensor_weight::XFORM_BF16_TO_F32) {
            // ssm_conv1d: on-disk BF16, engine requires F32.
            const auto & file = files.at(weight->idx);
            const size_t n_elem = ggml_nelements(cur);
            std::vector<uint16_t> raw(n_elem);
            file->seek(weight->offs, SEEK_SET);
            file->read_raw(raw.data(), n_elem * sizeof(uint16_t));
            std::vector<float> f32(n_elem);
            for (size_t i = 0; i < n_elem; ++i) {
                uint32_t u = ((uint32_t) raw[i]) << 16;
                float f; memcpy(&f, &u, 4); f32[i] = f;
            }
            if (cur->buffer != nullptr && !ggml_backend_buffer_is_host(cur->buffer)) {
                ggml_backend_tensor_set(cur, f32.data(), 0, n_size);
            } else {
                memcpy(cur->data, f32.data(), n_size);
            }
            continue;
        }

        if (use_mmap) {
            const auto & mapping = mappings.at(weight->idx);
            ggml_backend_buffer_t buf_mmap = nullptr;
            if (bufs_mmap.count(weight->idx)) {
                buf_mmap = bufs_mmap.at(weight->idx);
            }
            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;

            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                            return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                            }));
            }

            GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
            if (buf_mmap && cur->data == nullptr) {
                ggml_backend_tensor_alloc(buf_mmap, cur, data);
                if (lmlocks) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + n_size);
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
        } else {
            GGML_ASSERT(weight->idx < files.size());
            const auto & file = files.at(weight->idx);
            if (ggml_backend_buffer_is_host(cur->buffer)) {
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(cur->data, n_size);
                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                                return std::make_pair(cur, ggml_validate_row_data(cur->type, cur->data, n_size));
                                }));
                }
            } else {
#if defined(GGML_USE_CUDA)
                // If cuda_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the GPU.
                if (cuda_backend) {
                    file->seek(weight->offs, SEEK_SET);

                    size_t bytes_read = 0;

                    while (bytes_read < n_size) {
                        size_t read_iteration = std::min<size_t>(buffer_size, n_size - bytes_read);

                        ggml_backend_event_synchronize(events[buffer_idx]);
                        file->read_raw(host_ptrs[buffer_idx], read_iteration);
                        ggml_backend_tensor_set_async(cuda_backend, cur, host_ptrs[buffer_idx], bytes_read, read_iteration);
                        ggml_backend_event_record(events[buffer_idx]);

                        bytes_read += read_iteration;
                        ++buffer_idx;
                        buffer_idx %= n_buffers;
                    }
                }
                else
#endif
                {
                    read_buf.resize(n_size);
                    file->seek(weight->offs, SEEK_SET);
                    file->read_raw(read_buf.data(), n_size);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                    if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                    }
                }
            }
        }

        size_done += n_size;
    }

#if defined(GGML_USE_CUDA)
    // free temporary resources used for async cuda uploads
    if (cuda_backend) {
        for (size_t idx = 0; idx < n_buffers;++idx) {
            ggml_backend_event_synchronize(events[idx]);
            ggml_backend_event_free(events[idx]);
            ggml_backend_buffer_free(host_buffers[idx]);
        }
        ggml_backend_free(cuda_backend);
    }
#endif

    // check validation results
    bool validation_failed = false;
    for (auto & future : validation_result) {
        auto result = future.get();
        if (!result.second) {
            LLAMA_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, ggml_get_name(result.first));
            validation_failed = true;
        }
    }
    if (validation_failed) {
        throw std::runtime_error("found tensors with invalid data");
    }

    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto & mapping = mappings.at(idx);
                mapping->unmap_fragment(0, mmap_used.first);
                if (mmap_used.second != 0) {
                    mapping->unmap_fragment(mmap_used.second, mapping->size());
                }
            }
        }
        if (progress_callback) {
            // Even though the model is done loading, we still honor
            // cancellation since we need to free allocations.
            return progress_callback(1.0f, progress_callback_user_data);
        }
    }

    return true;
}

template<>
bool llama_model_loader::get_key(const enum llm_kv kid, enum llama_pooling_type & result, const bool required) {
    uint32_t tmp;
    const bool found = get_key(kid, tmp, required);
    if (found) {
        result = (enum llama_pooling_type) tmp;
    } else {
        result = LLAMA_POOLING_TYPE_UNSPECIFIED;
    }
    return found;
}
template bool llama_model_loader::get_key<bool>       (enum llm_kv kid, bool & result,        bool required);
template bool llama_model_loader::get_key<float>      (enum llm_kv kid, float & result,       bool required);
template bool llama_model_loader::get_key<uint32_t>   (enum llm_kv kid, uint32_t & result,    bool required);
template bool llama_model_loader::get_key<std::string>(enum llm_kv kid, std::string & result, bool required);

template bool llama_model_loader::get_key_or_arr<std::array<int, 4>>(enum llm_kv kid, std::array<int, 4> & result, uint32_t n, bool required);
template bool llama_model_loader::get_key_or_arr<std::array<uint32_t, 512>>(enum llm_kv kid, std::array<uint32_t, 512> & result, uint32_t n, bool required);
template bool llama_model_loader::get_key_or_arr<std::array<float, 512>>(enum llm_kv kid, std::array<float, 512> & result, uint32_t n, bool required);

template std::enable_if<std::is_integral<unsigned int>::value, bool>::type llama_model_loader::get_arr_n<unsigned int>(enum llm_kv, unsigned int&, bool);

