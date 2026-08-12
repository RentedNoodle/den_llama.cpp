// llama-den-loader.cpp — .den file loader implementation
//
// Opens .den files (heap-loaded, not mmap), parses the 4096B header + tensor
// index, and converts .den tensor formats to GGML tensor formats on read.
//
// NVFP4 conversion (converter v5): nibbles and scales are stored SEPARATELY.
//   nibble region: N rows × (K/2) bytes of interleaved E2M1 nibbles
//                  (elem 2i in the low nibble of byte i, elem 2i+1 in high).
//   scale region:  ceil(N/128) × (4 + 128 × K/16) bytes, per 128-row block:
//                  [4-byte boost float][128 × K/16 UE8M0 exponent codes].
// Conversion → GGML block_nvfp4 blocks (64 elements, 36 bytes each): reorder
// the nibbles from interleaved to the split layout GGML expects, and encode
// boost × 2^(code−127) as a UE4M3 scale per 16-element sub-block.
//
// Ported from dengine/src/den_core.c — Project Den

#include "llama-den-loader.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// Forward declare GGML NVFP4 type traits (from ggml-common.h)
#define QK_NVFP4     64
#define QK_NVFP4_SUB 16

struct block_nvfp4 {
    uint8_t d[QK_NVFP4 / QK_NVFP4_SUB]; // 4 scale bytes (UE4M3)
    uint8_t qs[QK_NVFP4 / 2];           // 32 bytes E2M1 nibbles
};

static_assert(sizeof(block_nvfp4) == 36, "block_nvfp4 must be 36 bytes");

// ── .den on-disk header (from denrt_format.h) ─────────────────────────

struct den_header_t {
    uint32_t magic;
    uint32_t version;
    uint32_t arch;
    uint32_t flags;

    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t hidden_size;
    uint32_t ffn_size;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t n_rot;
    uint32_t n_experts;
    uint32_t n_experts_used;
    float    rope_theta;
    float    rms_norm_eps;

    uint32_t ssm_state_size;
    uint32_t ssm_conv_kernel;
    uint32_t ssm_inner_size;
    uint32_t ssm_group_count;
    uint32_t ssm_time_step_rank;
    uint32_t full_attention_interval;

    uint32_t mtp_layer_count;
    uint32_t ssm_value_size;
    uint32_t _padding[2];

    uint32_t tensor_count;
    uint32_t index_offset;
    uint64_t data_offset;
    uint64_t total_data_size;

    uint32_t hot_tier_count;
    uint32_t warm_tier_count;
    uint32_t cold_tier_count;
    uint64_t hot_tier_size;
    uint64_t warm_tier_size;
    uint64_t cold_tier_size;

    uint8_t _reserved[4096 - 168];
};

static_assert(sizeof(den_header_t) == 4096, "den_header_t must be 4096 bytes");

// ── .den tensor index entry (128 bytes) ────────────────────────────────

struct den_tensor_entry_t {
    uint32_t slot;
    uint32_t hw_target;
    uint32_t ndim;
    uint32_t flags;
    int64_t  dims[4];
    uint64_t numel;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t scale_offset;
    uint64_t scale_size;
    uint32_t tile_k;
    uint32_t tile_n;
    uint32_t n_tiles;
    uint32_t scale_count;
    uint64_t norm_offset;
    uint32_t norm_size;
    uint32_t block_size;
    uint32_t grid_size;
    uint32_t smem_bytes;
};

static_assert(sizeof(den_tensor_entry_t) == 128, "den_tensor_entry_t must be 128 bytes");

// ═══════════════════════════════════════════════════════════════════════════
// Internal tensor descriptor (populated at open time, used for reads)
// ═══════════════════════════════════════════════════════════════════════════

struct den_tensor_desc {
    std::string name;               // llama.cpp tensor name
    ggml_type   ggml_type;          // GGML type code
    int64_t     ne[4];              // logical dimensions (ne[0..ndim-1])
    uint64_t    numel;              // total logical elements
    size_t      src_offset;         // byte offset of nibble region (relative to data_region)
    size_t      src_size;           // byte size of nibble region
    size_t      scale_offset;       // byte offset of separate scale region (relative to data_region)
    size_t      scale_size;         // byte size of separate scale region
    size_t      dst_size;           // byte size in GGML format (output buffer)
    uint32_t    hw_target;          // original .den hw_target
    bool        is_wh4;             // true if WH4 WHT-domain tensor
};

// ═══════════════════════════════════════════════════════════════════════════
// Private implementation
// ═══════════════════════════════════════════════════════════════════════════

struct llama_den_loader::impl {
    // Raw file data (heap-allocated)
    uint8_t * file_data    = nullptr;
    size_t    file_size    = 0;

    // Pointers into file_data
    const den_header_t       * header = nullptr;
    const den_tensor_entry_t * index  = nullptr;
    const uint8_t            * data_region = nullptr;

    // Parsed header fields
    uint32_t n_layers          = 0;
    uint32_t n_heads           = 0;
    uint32_t n_kv_heads        = 0;
    uint32_t hidden_size       = 0;
    uint32_t ffn_size          = 0;
    uint32_t vocab_size        = 0;
    uint32_t n_experts         = 0;
    uint32_t n_experts_used    = 0;
    uint32_t n_rot             = 0;
    float    rope_theta        = 1e6f;
    float    rms_norm_eps      = 1e-6f;
    uint32_t ssm_state_size    = 0;
    uint32_t ssm_conv_kernel   = 0;
    uint32_t ssm_inner_size    = 0;
    uint32_t ssm_value_size    = 0;
    uint32_t full_attn_interval = 0;
    int      head_dim          = 0;
    uint32_t actual_tensor_count = 0;
    uint32_t actual_index_offset = 0;
    uint64_t actual_data_offset  = 0;

    // Tensor inventory
    std::vector<den_tensor_desc> tensors;
    std::string arch_name;
    std::string error_msg;
    bool        is_open = false;

    // ── Methods ────────────────────────────────────────────────────────

    void clear() {
        free(file_data);
        file_data = nullptr;
        file_size = 0;
        header = nullptr;
        index  = nullptr;
        data_region = nullptr;
        tensors.clear();
        arch_name.clear();
        error_msg.clear();
        is_open = false;
    }

    // Validate and parse the header
    bool parse_header();

    // Derive model info from v1 format (where header fields may be zero)
    bool derive_v1_model_info(uint32_t tensor_count, uint32_t index_offset);

    // Populate the tensor inventory from the index
    bool build_tensor_inventory();

    // Read and convert an NVFP4 tensor from .den → GGML format
    void read_nvfp4_tensor(const den_tensor_desc & desc, void * dst, size_t dst_size) const;

    // Slot ID → llama.cpp tensor name
    static std::string slot_to_llama_name(uint32_t slot, uint32_t n_layers);

    // Architecture enum → name string
    static std::string arch_to_name(uint32_t arch);
};

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

llama_den_loader::llama_den_loader()
    : pimpl_(new impl())
{}

llama_den_loader::~llama_den_loader() {
    close();
}

bool llama_den_loader::open(const char * path) {
    pimpl_->clear();

    if (!path || !*path) {
        pimpl_->error_msg = "null or empty path";
        return false;
    }

    // ── Open file ────────────────────────────────────────────────────
#ifdef _WIN32
    int fd = _open(path, _O_RDONLY | _O_BINARY);
#else
    int fd = open(path, O_RDONLY);
#endif
    if (fd < 0) {
        pimpl_->error_msg = std::string("cannot open file: ") + path;
        return false;
    }

    // ── Get file size ─────────────────────────────────────────────────
#ifdef _WIN32
    struct _stat64 st;
    if (_fstat64(fd, &st) < 0) {
        _close(fd);
#else
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
#endif
        pimpl_->error_msg = "cannot stat file";
        return false;
    }
    pimpl_->file_size = (size_t)st.st_size;

    if (pimpl_->file_size < DEN_HEADER_SIZE) {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        pimpl_->error_msg = "file too small for .den header (min 4096 bytes)";
        return false;
    }

    // ── Heap-load entire file (chunked read for >2GB safety) ──────────
    pimpl_->file_data = (uint8_t *)malloc(pimpl_->file_size);
    if (!pimpl_->file_data) {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        pimpl_->error_msg = "out of memory loading .den file";
        return false;
    }

    size_t total_read = 0;
    while (total_read < pimpl_->file_size) {
        size_t chunk = pimpl_->file_size - total_read;
        if (chunk > (size_t)1024 * 1024 * 1024) {
            chunk = (size_t)1024 * 1024 * 1024; // 1 GB chunks
        }
#ifdef _WIN32
        int rd = (int)_read(fd, pimpl_->file_data + total_read, (unsigned int)chunk);
#else
        ssize_t rd = read(fd, pimpl_->file_data + total_read, chunk);
#endif
        if (rd <= 0) {
            free(pimpl_->file_data);
            pimpl_->file_data = nullptr;
#ifdef _WIN32
            _close(fd);
#else
            close(fd);
#endif
            pimpl_->error_msg = "read error or truncated file";
            return false;
        }
        total_read += (size_t)rd;
    }

#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif

    // ── Parse header ──────────────────────────────────────────────────
    if (!pimpl_->parse_header()) {
        pimpl_->clear();
        return false;
    }

    // ── Build tensor inventory ─────────────────────────────────────────
    if (!pimpl_->build_tensor_inventory()) {
        pimpl_->clear();
        return false;
    }

    pimpl_->is_open = true;
    return true;
}

size_t llama_den_loader::get_tensor_count() const {
    return pimpl_->is_open ? pimpl_->tensors.size() : 0;
}

std::string llama_den_loader::get_tensor_name(size_t i) const {
    if (!pimpl_->is_open || i >= pimpl_->tensors.size()) return {};
    return pimpl_->tensors[i].name;
}

ggml_type llama_den_loader::get_tensor_type(size_t i) const {
    if (!pimpl_->is_open || i >= pimpl_->tensors.size()) return GGML_TYPE_F32;
    return pimpl_->tensors[i].ggml_type;
}

std::vector<int64_t> llama_den_loader::get_tensor_shape(size_t i) const {
    if (!pimpl_->is_open || i >= pimpl_->tensors.size()) return {};
    const auto & t = pimpl_->tensors[i];
    std::vector<int64_t> shape;
    for (int d = 0; d < 4 && t.ne[d] > 0; d++) {
        shape.push_back(t.ne[d]);
    }
    // llama.cpp convention: ne[0] is innermost, report in same order
    if (shape.empty()) shape.push_back(1);
    return shape;
}

size_t llama_den_loader::get_tensor_size(size_t i) const {
    if (!pimpl_->is_open || i >= pimpl_->tensors.size()) return 0;
    return pimpl_->tensors[i].dst_size;
}

uint64_t llama_den_loader::get_tensor_data_offset(size_t i) const {
    if (!pimpl_->is_open || i >= pimpl_->tensors.size()) return 0;
    return pimpl_->tensors[i].src_offset;
}

uint64_t llama_den_loader::get_tensor_data_size(size_t i) const {
    if (!pimpl_->is_open || i >= pimpl_->tensors.size()) return 0;
    return pimpl_->tensors[i].src_size;
}

uint64_t llama_den_loader::get_header_data_offset() const {
    return pimpl_->actual_data_offset;
}

void llama_den_loader::read_tensor_data(size_t i, void * dst, size_t size) const {
    if (!pimpl_->is_open || i >= pimpl_->tensors.size()) return;
    if (!dst || size == 0) return;

    const auto & t = pimpl_->tensors[i];

    switch (t.hw_target) {
    case DEN_TARGET_NVFP4:
        if (t.is_wh4) {
            // WH4: keep raw tiles, copy as-is (WHT-domain OMMA dispatch handles them directly)
            size_t copy_size = std::min(size, t.src_size);
            memcpy(dst, pimpl_->data_region + t.src_offset, copy_size);
        } else {
            pimpl_->read_nvfp4_tensor(t, dst, size);
        }
        break;
    case DEN_TARGET_BF16:
    case DEN_TARGET_F16:
    case DEN_TARGET_F32:
    case DEN_TARGET_INT8:
        // Direct copy (same layout in .den and GGML for these types)
        {
            size_t copy_size = std::min(size, t.src_size);
            memcpy(dst, pimpl_->data_region + t.src_offset, copy_size);
        }
        break;
    default:
        memset(dst, 0, size);
        break;
    }
}

const std::string & llama_den_loader::get_arch_name() const {
    return pimpl_->arch_name;
}

const std::string & llama_den_loader::get_error() const {
    return pimpl_->error_msg;
}

void llama_den_loader::close() {
    pimpl_->clear();
}

// ── Model info accessors ──────────────────────────────────────────────

#define DEN_GETTER(name, field) \
    uint32_t llama_den_loader::get_##name() const { \
        return pimpl_->is_open ? pimpl_->field : 0; \
    }

DEN_GETTER(n_layers,          n_layers)
DEN_GETTER(hidden_size,       hidden_size)
DEN_GETTER(ffn_size,          ffn_size)
DEN_GETTER(n_heads,           n_heads)
DEN_GETTER(n_kv_heads,        n_kv_heads)
DEN_GETTER(n_rot,             n_rot)
DEN_GETTER(vocab_size,        vocab_size)
DEN_GETTER(n_experts,         n_experts)
DEN_GETTER(n_experts_used,    n_experts_used)
DEN_GETTER(ssm_state_size,    ssm_state_size)
DEN_GETTER(ssm_conv_kernel,   ssm_conv_kernel)
DEN_GETTER(ssm_inner_size,    ssm_inner_size)
DEN_GETTER(ssm_value_size,    ssm_value_size)
DEN_GETTER(full_attn_interval, full_attn_interval)

#undef DEN_GETTER

uint32_t llama_den_loader::get_head_dim() const {
    if (!pimpl_->is_open) return 0;
    if (pimpl_->head_dim > 0) return (uint32_t)pimpl_->head_dim;
    if (pimpl_->n_heads > 0) return pimpl_->hidden_size / pimpl_->n_heads;
    return 256;
}

float llama_den_loader::get_rope_theta() const {
    return pimpl_->is_open ? pimpl_->rope_theta : 0.0f;
}

float llama_den_loader::get_rms_norm_eps() const {
    return pimpl_->is_open ? pimpl_->rms_norm_eps : 1e-6f;
}

// ═══════════════════════════════════════════════════════════════════════════
// Header parsing
// ═══════════════════════════════════════════════════════════════════════════

bool llama_den_loader::impl::parse_header() {
    const den_header_t * hdr = (const den_header_t *)file_data;

    // Validate magic
    if (hdr->magic != DEN_MAGIC) {
        error_msg = "invalid .den magic (expected 0x4E454400)";
        return false;
    }

    // Accept v5, v1, and v0
    if (hdr->version != DEN_VERSION_V5 &&
        hdr->version != DEN_VERSION_V1 &&
        hdr->version != 0x00000001) {
        char buf[128];
        snprintf(buf, sizeof(buf), "unsupported .den version 0x%08X", hdr->version);
        error_msg = buf;
        return false;
    }

    header = hdr;

    // ── Handle v1 format (compact header) ──────────────────────────
    actual_tensor_count = hdr->tensor_count;
    actual_index_offset = hdr->index_offset;
    actual_data_offset  = hdr->data_offset;

    if (hdr->version == DEN_VERSION_V1 || hdr->version == 0x00000001) {
        // v1 layout: magic(4) + version(4) + arch(4) + tensor_count_at_flags(4) = 16
        // tensor_count stored where v5 has 'flags'
        if (hdr->version == DEN_VERSION_V1) {
            actual_tensor_count = hdr->flags; // v1: tensor_count in flags slot
        }
        actual_index_offset = 16;
        actual_data_offset  = 16 + actual_tensor_count * sizeof(den_tensor_entry_t);
        // Page-align data
        actual_data_offset  = ((actual_data_offset + 4095) / 4096) * 4096;

        // Validate v1: tensor_count check
        if (actual_tensor_count == 0 || actual_tensor_count > 4096) {
            error_msg = "v1 header: invalid tensor count";
            return false;
        }
    } else {
        // v5 validation
        if (hdr->tensor_count == 0 || hdr->tensor_count > 4096) {
            error_msg = "invalid tensor count in header";
            return false;
        }
        if (hdr->index_offset != DEN_HEADER_SIZE) {
            char buf[128];
            snprintf(buf, sizeof(buf), "unexpected index offset %u (expected %u)",
                     hdr->index_offset, DEN_HEADER_SIZE);
            error_msg = buf;
            return false;
        }
    }

    // Validate bounds
    if (actual_data_offset > file_size) {
        error_msg = "data offset exceeds file size";
        return false;
    }
    size_t index_end = actual_index_offset + actual_tensor_count * sizeof(den_tensor_entry_t);
    if (index_end > file_size) {
        error_msg = "tensor index exceeds file size";
        return false;
    }

    // ── Set up pointers ──────────────────────────────────────────────
    index       = (const den_tensor_entry_t *)(file_data + actual_index_offset);
    data_region = file_data + actual_data_offset;

    // ── Copy model info from header ──────────────────────────────────
    n_layers          = hdr->n_layers;
    n_heads           = hdr->n_heads;
    n_kv_heads        = hdr->n_kv_heads;
    hidden_size       = hdr->hidden_size;
    ffn_size          = hdr->ffn_size;
    vocab_size        = hdr->vocab_size;
    n_experts         = hdr->n_experts;
    n_experts_used    = hdr->n_experts_used;
    n_rot             = hdr->n_rot;
    rope_theta        = hdr->rope_theta;
    rms_norm_eps      = hdr->rms_norm_eps;
    ssm_state_size    = hdr->ssm_state_size;
    ssm_conv_kernel   = hdr->ssm_conv_kernel;
    ssm_inner_size    = hdr->ssm_inner_size;
    ssm_value_size    = hdr->ssm_value_size;
    full_attn_interval = hdr->full_attention_interval;

    // ── Compute head_dim ─────────────────────────────────────────────
    head_dim = hidden_size / (n_heads ? n_heads : 1);

    // ── Derive v1 model info if needed ───────────────────────────────
    if ((hdr->version == DEN_VERSION_V1 || hdr->version == 0x00000001) &&
        (n_layers == 0 || hidden_size == 0)) {
        if (!derive_v1_model_info(actual_tensor_count, actual_index_offset)) {
            return false;
        }
    }

    // ── Architecture name ─────────────────────────────────────────────
    arch_name = arch_to_name(hdr->arch);

    // Log
    fprintf(stderr,
            "den: arch=%u (%s) layers=%u hidden=%u ffn=%u heads=%u kv=%u "
            "vocab=%u experts=%u/%u tensors=%u\n",
            hdr->arch, arch_name.c_str(),
            n_layers, hidden_size, ffn_size,
            n_heads, n_kv_heads, vocab_size,
            n_experts, n_experts_used, actual_tensor_count);

    return true;
}

bool llama_den_loader::impl::derive_v1_model_info(uint32_t tensor_count,
                                                   uint32_t index_offset) {
    const den_tensor_entry_t * entries =
        (const den_tensor_entry_t *)(file_data + index_offset);

    int max_layer = -1;

    for (uint32_t i = 0; i < tensor_count; i++) {
        uint32_t slot = entries[i].slot;

        // Scan for layer index from slot number
        if (slot >= 3) {
            int layer = (int)((slot - 3) / DEN_LAYER_STRIDE);
            if (layer > max_layer) max_layer = layer;
        }

        // hidden_size from embedding table: token_embd.weight
        if (slot == 0 && entries[i].ndim == 2) {
            hidden_size = (uint32_t)entries[i].dims[1];
            vocab_size  = (uint32_t)entries[i].dims[0];
        }

        // ffn_size from first MLP gate tensor
        if (ffn_size == 0 && slot >= 11 && slot <= 13 + DEN_LAYER_STRIDE * 64) {
            int sub = (int)((slot - 3) % DEN_LAYER_STRIDE);
            if (sub == 8 && entries[i].ndim >= 1) {
                ffn_size = (uint32_t)entries[i].dims[0];
            }
        }
    }

    if (max_layer >= 0) n_layers = (uint32_t)(max_layer + 1);

    // Sensible defaults
    if (n_heads    == 0)   n_heads    = 32;
    if (n_kv_heads == 0)   n_kv_heads = 4;
    if (n_rot      == 0)   n_rot      = 128;
    if (rope_theta == 0.0f) rope_theta = 1000000.0f;
    if (rms_norm_eps == 0.0f) rms_norm_eps = 1e-6f;
    if (n_experts == 0 && n_layers > 32) {
        n_experts      = 128;
        n_experts_used = 8;
    }

    fprintf(stderr, "den: v1 derived: layers=%u hidden=%u ffn=%u heads=%u kv=%u\n",
            n_layers, hidden_size, ffn_size, n_heads, n_kv_heads);

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// NVFP4 (converter v5) → GGML conversion
// ═══════════════════════════════════════════════════════════════════════════
//
// .den v5 stores NVFP4 weights as two separate row-major regions:
//   nibble region: N rows × (K/2) bytes. Each byte holds two E2M1 nibbles,
//                  INTERLEAVED — element 2i in the low nibble, 2i+1 in high.
//   scale region:  ceil(N/128) × (4 + 128 × K/16) bytes. Per 128-row block:
//                  a 4-byte boost float followed by 128 × K/16 UE8M0 exponent
//                  codes (value = 2^(code−127)); a partial trailing block is
//                  padded to the full 128-row stride.
//
// GGML block_nvfp4 (64 elements / 36 bytes) expects, per 16-element sub-block s:
//   d[s]:      UE4M3 scale byte.
//   qs[s*8..s*8+7]: SPLIT nibbles — byte s*8+j holds elem 16s+j in the low
//                  nibble and elem 16s+8+j in the high nibble.
//
// The nibble VALUES themselves (E2M1, sign bit 3) are identical between the two
// layouts; only their byte positions differ, hence the interleave→split reorder.

// UE8M0: value = 2^(code − 127)  (matches the converter's _ue8m0_to_f32)
static inline float den_ue8m0_to_fp32(uint8_t code) {
    if (code == 0) {
        return ldexpf(1.0f, -127); // 2^-127 (subnormal)
    }
    uint32_t bits = (uint32_t)code << 23;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// FP32 → UE4M3 code (unsigned 4-bit exponent, bias 7; 3-bit mantissa).
// Mirrors ggml_fp32_to_ue4m3() so that GGML's dequantize_row_nvfp4, which
// halves the decoded UE4M3 value, recovers the intended scale.
static inline uint8_t den_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    if (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    int fp32_exp  = ((int)((bits >> 23) & 0xFF)) - 127;
    int fp32_man  = (int)((bits >> 20) & 0x7);
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9
        int man = (int)(x * 512.0f + 0.5f);
        if (man > 7) man = 7;
        if (man < 1) return 0;
        return (uint8_t) man;
    }
    if (ue4m3_exp >= 15) {
        return 0x7E;
    }
    int round_bit = (int)((bits >> 19) & 1);
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        if (ue4m3_exp >= 15) {
            return 0x7E;
        }
    }
    return (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
}

// Convert one 16-element sub-block: reorder .den INTERLEAVED nibbles to GGML
// SPLIT layout and encode the sub-block scale as UE4M3.
//   src_nibbles: 8 bytes of .den nibble data for this sub-block
//   scale_f32:   effective scale = boost * 2^(UE8M0 code − 127)
//   dst_d:       1 byte output (block_nvfp4.d[sub])
//   dst_qs:      8 bytes output (block_nvfp4.qs[sub*8 .. sub*8+7])
static void convert_nvfp4_subblock(const uint8_t * src_nibbles, float scale_f32,
                                   uint8_t * dst_d, uint8_t * dst_qs) {
    for (int j = 0; j < 8; j++) {
        // GGML byte j: low nibble = elem 16s+j, high nibble = elem 16s+8+j.
        // Source elem e lives in byte e/2 (interleaved), low if e even, high if odd.
        const uint8_t lo_byte = src_nibbles[j / 2];
        const uint8_t hi_byte = src_nibbles[4 + j / 2];
        const uint8_t lo = (j & 1) ? (uint8_t)(lo_byte >> 4) : (uint8_t)(lo_byte & 0x0F);
        const uint8_t hi = (j & 1) ? (uint8_t)(hi_byte >> 4) : (uint8_t)(hi_byte & 0x0F);
        dst_qs[j] = (uint8_t) (lo | (hi << 4));
    }
    *dst_d = den_fp32_to_ue4m3(scale_f32);
}

void llama_den_loader::impl::read_nvfp4_tensor(
        const den_tensor_desc & desc, void * dst, size_t dst_size) const {

    // Source: two separate row-major regions (see section comment above).
    //   nibble region: N rows × (K/2) bytes
    //   scale region:  ceil(N/128) × (4 + 128 × K/16) bytes (per-128-row-block)
    // Destination: N rows × (K/64) GGML block_nvfp4 blocks (36 bytes each).

    const uint8_t * src = data_region + desc.src_offset;
    const uint8_t * scl = data_region + desc.scale_offset;
    uint8_t * d = (uint8_t *)dst;

    const int64_t K = desc.ne[1]; // columns (inner dimension)

    // Total rows = ne[0] * ne[2] * ne[3] (flat row-major layout for multi-dim tensors)
    int64_t total_rows = desc.ne[0];
    if (desc.ne[2] > 0) total_rows *= desc.ne[2];
    if (desc.ne[3] > 0) total_rows *= desc.ne[3];
    if (total_rows <= 0) total_rows = 1;

    if (K <= 0) {
        memset(dst, 0, dst_size);
        return;
    }

    const int64_t blocks_per_row = (K + QK_NVFP4 - 1) / QK_NVFP4;
    const int64_t sub_per_row    = K / QK_NVFP4_SUB;        // K/16
    const int64_t sub_per_block  = QK_NVFP4 / QK_NVFP4_SUB; // 4
    const size_t  dst_row_stride = (size_t)blocks_per_row * sizeof(block_nvfp4);
    const int64_t ROWS_PER_BLK   = 128;
    const int64_t n_blk          = (total_rows + ROWS_PER_BLK - 1) / ROWS_PER_BLK;

    // Guard against malformed entries (nibble and/or scale regions too small)
    if ((size_t)(total_rows * (K / 2)) > desc.src_size ||
        desc.scale_size < (size_t)(n_blk * (4 + ROWS_PER_BLK * sub_per_row))) {
        memset(dst, 0, dst_size);
        return;
    }

    for (int64_t row = 0; row < total_rows; row++) {
        const uint8_t * src_row = src + row * (K / 2);

        // Per-128-row-block scale region: [4B boost][128 rows × SP codes].
        const int64_t blk     = row / ROWS_PER_BLK;
        const int64_t in_blk  = row % ROWS_PER_BLK;
        const uint8_t * blk_start = scl + blk * (4 + ROWS_PER_BLK * sub_per_row);
        float boost = 1.0f;
        memcpy(&boost, blk_start, sizeof(boost));
        const uint8_t * codes = blk_start + 4 + in_blk * sub_per_row;   // SP codes for this row

        uint8_t * dst_row = d + row * dst_row_stride;

        // Zero the (possibly partial) last block so padding sub-blocks read as 0
        memset(dst_row + (blocks_per_row - 1) * sizeof(block_nvfp4),
               0, sizeof(block_nvfp4));

        for (int64_t g = 0; g < sub_per_row; g++) {
            const int64_t block = g / sub_per_block;
            const int64_t sub   = g % sub_per_block;
            uint8_t * blk = dst_row + block * sizeof(block_nvfp4);

            // effective scale = boost * 2^(code − 127)
            const float scale_f32 = den_ue8m0_to_fp32(codes[g]) * boost;

            convert_nvfp4_subblock(src_row + g * (QK_NVFP4_SUB / 2),
                                   scale_f32,
                                   blk + sub,
                                   blk + (QK_NVFP4 / QK_NVFP4_SUB) + sub * (QK_NVFP4_SUB / 2));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tensor inventory builder
// ═══════════════════════════════════════════════════════════════════════════

bool llama_den_loader::impl::build_tensor_inventory() {
    uint32_t n_nvfp4 = 0, n_bf16 = 0, n_f32 = 0;

    for (uint32_t i = 0; i < actual_tensor_count; i++) {
        const den_tensor_entry_t & te = index[i];
        den_tensor_desc desc;

        // ── Name ────────────────────────────────────────────────────
        desc.name = slot_to_llama_name(te.slot, n_layers);

        // ── Shape ───────────────────────────────────────────────────
        memset(desc.ne, 0, sizeof(desc.ne));
        for (uint32_t d = 0; d < te.ndim && d < 4; d++) {
            desc.ne[d] = te.dims[d];
        }
        if (te.ndim == 0) {
            desc.ne[0] = 1;
        }
        desc.numel    = te.numel;
        desc.src_offset = (size_t)te.data_offset;
        desc.src_size   = (size_t)te.data_size;
        desc.hw_target  = te.hw_target;
        desc.is_wh4     = false;

        // ── Determine GGML type and destination size ─────────────────
        switch (te.hw_target) {
        case DEN_TARGET_NVFP4: {
            desc.ggml_type = GGML_TYPE_NVFP4;

            // Converter v5 stores nibbles and scales in separate regions — there
            // are no inline 160B NULLGLASS tiles. WH4 tile detection only applies
            // to legacy .den files that carry no separate scale region.
            desc.scale_offset = (size_t)(te.data_offset + te.scale_offset);
            desc.scale_size   = (size_t)te.scale_size;

            if (desc.scale_size == 0 && desc.src_size >= DEN_NVFP4_TILE_SIZE) {
                // Check for WH4: tile[149] == 8 means WHT-domain
                const uint8_t * first_tile = data_region + desc.src_offset;
                uint8_t kstride = first_tile[DEN_TILE_KSTRIDE];
                if (kstride == DEN_KSTRIDE_WH4) {
                    desc.is_wh4 = true;
                    // WH4: keep raw tiles, use NVFP4 type for dispatch
                    desc.dst_size = desc.src_size;
                }
            }

            if (!desc.is_wh4) {
                // Standard v5 → GGML conversion
                // Elements per block: QK_NVFP4 = 64
                // Bytes per block:   sizeof(block_nvfp4) = 36
                // Total rows = ne[1] * ne[2] * ne[3] (flat row-major layout)
                int64_t K = desc.ne[0];
                int64_t blocks_per_row = (K + QK_NVFP4 - 1) / QK_NVFP4;
                if (blocks_per_row == 0) blocks_per_row = 1;
                int64_t total_rows = desc.ne[1];
                if (desc.ne[2] > 0) total_rows *= desc.ne[2];
                if (desc.ne[3] > 0) total_rows *= desc.ne[3];
                if (total_rows <= 0) total_rows = 1;
                desc.dst_size = (size_t)(blocks_per_row * sizeof(block_nvfp4) * total_rows);
            }

            n_nvfp4++;
            break;
        }
        case DEN_TARGET_BF16:
            desc.ggml_type = GGML_TYPE_BF16;
            desc.dst_size  = desc.src_size; // 2 bytes per element
            n_bf16++;
            break;
        case DEN_TARGET_F16:
            desc.ggml_type = GGML_TYPE_F16;
            desc.dst_size  = desc.src_size; // 2 bytes per element
            break;
        case DEN_TARGET_F32:
            desc.ggml_type = GGML_TYPE_F32;
            desc.dst_size  = desc.src_size; // 4 bytes per element
            n_f32++;
            break;
        case DEN_TARGET_INT8:
            desc.ggml_type = GGML_TYPE_Q8_0;
            desc.dst_size  = desc.src_size;
            break;
        default:
            // Unknown — treat as F32
            desc.ggml_type = GGML_TYPE_F32;
            desc.dst_size  = desc.src_size;
            break;
        }

        tensors.push_back(desc);
    }

    fprintf(stderr, "den: %zu tensors: %u NVFP4 + %u BF16 + %u F32\n",
            tensors.size(), n_nvfp4, n_bf16, n_f32);

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Slot → llama.cpp tensor name mapping
// ═══════════════════════════════════════════════════════════════════════════

static const char * den_sub_to_llama_name(int sub) {
    switch (sub) {
    // Attention
    case DEN_SUB_ATTN_QKV:      return "attn_qkv";
    case DEN_SUB_ATTN_Q:        return "attn_q";
    case DEN_SUB_ATTN_K:        return "attn_k";
    case DEN_SUB_ATTN_V:        return "attn_v";
    case DEN_SUB_ATTN_O:        return "attn_output";
    case DEN_SUB_ATTN_Q_NORM:   return "attn_q_norm";
    case DEN_SUB_ATTN_K_NORM:   return "attn_k_norm";

    // MoE router
    case DEN_SUB_MOE_ROUTER:    return "ffn_gate_inp";

    // Dense MLP
    case DEN_SUB_MLP_GATE:      return "ffn_gate";
    case DEN_SUB_MLP_UP:        return "ffn_up";
    case DEN_SUB_MLP_DOWN:      return "ffn_down";

    // MoE expert weights
    case DEN_SUB_MOE_GATE_UP:   return "ffn_gate_up_exps";   // fused gate+up per expert
    case DEN_SUB_MOE_DOWN:      return "ffn_down_exps";

    // GDN / SSM (linear attention)
    case DEN_SUB_GDN_IN_X:      return "ssm_alpha";    // in_proj_a
    case DEN_SUB_GDN_IN_Z:      return "attn_gate";    // in_proj_z
    case DEN_SUB_GDN_OUT:       return "ssm_out";
    case DEN_SUB_GDN_A_LOG:     return "ssm_a";
    case DEN_SUB_GDN_DT_BIAS:   return "ssm_dt";
    case DEN_SUB_GDN_CONV1D:    return "ssm_conv1d";
    case DEN_SUB_GDN_D_PROJ:    return "ssm_d";
    case DEN_SUB_GDN_NORM:      return "ssm_norm";

    // Sub-slot 20: input_layernorm (used by both attention and SSM layers as attn_norm)
    case DEN_SUB_GDN_A_NORM:    return "attn_norm";

    case DEN_SUB_POST_ATTN_NORM:  return "post_attention_norm";
    case DEN_SUB_PRE_MLP_NORM:    return "pre_mlp_norm";
    case DEN_SUB_POST_MLP_NORM:   return "post_mlp_norm";

    // Shared between GDN QKV (SSM) and MOE GATE (MoE)
    case DEN_SUB_GDN_QKV:       return "attn_qkv";     // in_proj_qkv for GDN layers

    case DEN_SUB_GDN_PROJ_B:    return "ssm_beta";

    // Shared expert weights
    case DEN_SUB_SHARED_GATE:   return "ffn_gate_shexp";
    case DEN_SUB_SHARED_UP:     return "ffn_up_shexp";
    case DEN_SUB_SHARED_DOWN:   return "ffn_down_shexp";
    case DEN_SUB_SHARED_GATE_W: return "ffn_gate_inp_shexp";

    default: return nullptr;
    }
}

std::string llama_den_loader::impl::slot_to_llama_name(
        uint32_t slot, uint32_t n_layers) {

    if (slot == DEN_SLOT_TOKEN_EMBD)   return "token_embd.weight";
    if (slot == DEN_SLOT_OUTPUT_NORM)  return "output_norm.weight";
    if (slot == DEN_SLOT_OUTPUT)       return "output.weight";

    // Per-layer slots
    if (slot >= DEN_SLOT_LAYER_BASE(0) &&
        slot <  DEN_SLOT_LAYER_BASE(n_layers)) {

        int layer = (int)((slot - 3) / DEN_LAYER_STRIDE);
        int sub   = (int)((slot - 3) % DEN_LAYER_STRIDE);

        const char * tensor_name = den_sub_to_llama_name(sub);

        if (tensor_name) {
            // Bias/1D parameters (special suffixes)
            if (sub == 16) {
                return "blk." + std::to_string(layer) + "." + tensor_name + ".bias";
            }
            if (sub == 15 || sub == 18) {
                return "blk." + std::to_string(layer) + "." + tensor_name;
            }
            return "blk." + std::to_string(layer) + "." + tensor_name + ".weight";
        }

        // Unknown sub-slot — use generic name
        return "blk." + std::to_string(layer) + ".slot_" + std::to_string(sub) + ".weight";
    }

    // Unknown global slot
    return "slot_" + std::to_string(slot) + ".weight";
}

// ═══════════════════════════════════════════════════════════════════════════
// Architecture name
// ═══════════════════════════════════════════════════════════════════════════

std::string llama_den_loader::impl::arch_to_name(uint32_t arch) {
    switch (arch) {
    case DEN_ARCH_QWEN35:       return "qwen35";
    case DEN_ARCH_QWEN36_MOE:   return "qwen36-moe";
    case DEN_ARCH_QWEN36_DENSE: return "qwen36-dense";
    case DEN_ARCH_DEEPSEEK_V4:  return "deepseek-v4";
    default:                    return "custom-" + std::to_string(arch);
    }
}
