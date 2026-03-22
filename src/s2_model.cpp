#include "../include/s2_model.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <stdexcept>
#ifdef __linux__
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace s2 {

// ---------------------------------------------------------------------------
// Helpers (graph‑level, no side effects)
// ---------------------------------------------------------------------------

static ggml_tensor * repeat_checked(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                                    const char * label = "repeat") {
    if (!ggml_can_repeat(a, b)) {
        std::fprintf(stderr, "%s a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
            label,
            (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
            (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
        std::fflush(stderr);
    }
    return ggml_repeat(ctx, a, b);
}

static ggml_tensor * mul_mat_checked(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                                     const char * label = "mul_mat") {
    const bool can_mul =
        a->ne[0] == b->ne[0] &&
        (b->ne[2] % a->ne[2] == 0) &&
        (b->ne[3] % a->ne[3] == 0);
    if (!can_mul || ggml_is_transposed(a)) {
        std::fprintf(stderr,
            "%s transposed=%d a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
            label, ggml_is_transposed(a) ? 1 : 0,
            (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
            (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
        std::fflush(stderr);
    }
    return ggml_mul_mat(ctx, a, b);
}

static ggml_tensor * rms_norm_weighted(ggml_context * ctx, ggml_tensor * x,
                                       ggml_tensor * weight, float eps) {
    ggml_tensor * cur = ggml_rms_norm(ctx, x, eps);
    ggml_tensor * w = weight;
    if (w->type != cur->type) {
        w = ggml_cast(ctx, w, cur->type);
    }
    w = repeat_checked(ctx, w, cur, "repeat:rms_norm");
    return ggml_mul(ctx, cur, w);
}

static ggml_tensor * repeat_interleave_heads(ggml_context * ctx, ggml_tensor * x,
                                              int32_t repeat_factor) {
    if (repeat_factor == 1) return x;
    ggml_tensor * xf = (x->type != GGML_TYPE_F32) ? ggml_cast(ctx, x, GGML_TYPE_F32) : x;
    ggml_tensor * x4 = ggml_reshape_4d(ctx, ggml_cont(ctx, xf),
                                        xf->ne[0], 1, xf->ne[1], xf->ne[2]);
    ggml_tensor * target = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                               xf->ne[0], repeat_factor, xf->ne[1], xf->ne[2]);
    ggml_tensor * repeated = ggml_repeat(ctx, x4, target);
    return ggml_reshape_3d(ctx, ggml_cont(ctx, repeated),
                           xf->ne[0], xf->ne[1] * repeat_factor, xf->ne[2]);
}

static ggml_tensor * last_token_view(ggml_context * ctx, ggml_tensor * x, int32_t n_tokens) {
    return ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], (n_tokens - 1) * x->nb[1]);
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

SlowARModel::SlowARModel() {}

SlowARModel::~SlowARModel() {
    if (ctx_kv_)          ggml_free(ctx_kv_);
    if (kv_buf_)          ggml_backend_buffer_free(kv_buf_);
    if (weights_.ctx_w)   ggml_free(weights_.ctx_w);
    if (weights_.model_buf) ggml_backend_buffer_free(weights_.model_buf);
    if (fast_allocr_)     ggml_gallocr_free(fast_allocr_);
    if (allocr_)          ggml_gallocr_free(allocr_);
    if (backend_)         ggml_backend_free(backend_);
}

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------

bool SlowARModel::load(const std::string & gguf_path, int32_t gpu_device, int32_t backend_type) {
    if (gpu_device >= 0) {

#ifdef GGML_USE_VULKAN
        if(!backend_ && backend_type == 0)
        {
            backend_ = ggml_backend_vk_init(static_cast<size_t>(gpu_device));
            if (!backend_) {
                std::cerr << "[Model] Vulkan init failed, falling back to CPU." << std::endl;
            }
        }
#endif
#ifdef GGML_USE_CUDA
        if(!backend_ && backend_type == 1)
        {
            backend_ = ggml_backend_cuda_init(static_cast<size_t>(gpu_device));
            if (!backend_) {
                std::cerr << "[Model] Cuda init failed, falling back to CPU." << std::endl;
            }
        }
#endif
        if (!backend_)
        {
            std::cerr << "[Model] NPU not compiled, falling back to CPU." << std::endl;
        }
    }
    if (!backend_) {
        backend_ = ggml_backend_cpu_init();
    }
    if (!backend_) {
        std::cerr << "[Model] Failed to init any GGML backend." << std::endl;
        return false;
    }

    allocr_      = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    fast_allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));

    struct gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/&weights_.ctx_w };
    gguf_context * ctx_gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf) {
        std::cerr << "[Model] Failed to load GGUF from " << gguf_path << std::endl;
        return false;
    }

    std::cout << "[Model] Reading metadata from " << gguf_path << std::endl;

    // Helpers to read GGUF metadata
    auto get_u32 = [&](const char * key, uint32_t def) -> uint32_t {
        int id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { std::cerr << "[GGUF] missing key: " << key << " (using default " << def << ")\n"; return def; }
        uint32_t v = gguf_get_val_u32(ctx_gguf, id);
        std::cout << "[GGUF] " << key << " = " << v << "\n";
        return v;
    };
    auto get_f32 = [&](const char * key, float def) -> float {
        int id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { std::cerr << "[GGUF] missing key: " << key << " (using default " << def << ")\n"; return def; }
        float v = gguf_get_val_f32(ctx_gguf, id);
        std::cout << "[GGUF] " << key << " = " << v << "\n";
        return v;
    };
    auto get_bool = [&](const char * key, bool def) -> bool {
        int id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { std::cerr << "[GGUF] missing key: " << key << " (using default " << (def?"true":"false") << ")\n"; return def; }
        bool v = gguf_get_val_bool(ctx_gguf, id);
        std::cout << "[GGUF] " << key << " = " << (v?"true":"false") << "\n";
        return v;
    };

    hparams_ = ModelHParams();

    // Determine architecture prefix from the file
    std::string arch_prefix = "fish-speech.";
    {
        int arch_id = gguf_find_key(ctx_gguf, "general.architecture");
        if (arch_id >= 0) {
            std::string arch = gguf_get_val_str(ctx_gguf, arch_id);
            arch_prefix = arch + ".";
            hparams_.has_fast_decoder = (arch == "fish-speech");
            std::cout << "[Model] Architecture: " << arch << std::endl;
        }
    }

    // Main model hparams (from arch-prefixed keys)
    hparams_.context_length      = (int32_t)get_u32((arch_prefix + "context_length").c_str(), 32768);
    hparams_.vocab_size          = (int32_t)get_u32((arch_prefix + "vocab_size").c_str(), 155776);
    hparams_.embedding_length    = (int32_t)get_u32((arch_prefix + "embedding_length").c_str(), 2560);
    hparams_.feed_forward_length = (int32_t)get_u32((arch_prefix + "feed_forward_length").c_str(), 9728);
    hparams_.block_count         = (int32_t)get_u32((arch_prefix + "block_count").c_str(), 36);
    hparams_.head_count          = (int32_t)get_u32((arch_prefix + "attention.head_count").c_str(), 32);
    hparams_.head_count_kv       = (int32_t)get_u32((arch_prefix + "attention.head_count_kv").c_str(), 8);
    hparams_.rope_freq_base      = get_f32((arch_prefix + "rope.freq_base").c_str(), 1e6f);
    hparams_.rms_norm_eps        = get_f32((arch_prefix + "attention.layer_norm_rms_epsilon").c_str(), 1e-6f);

    // Fish-speech specific keys
    hparams_.codebook_size            = (int32_t)get_u32("fish_speech.codebook_size", 4096);
    hparams_.num_codebooks            = (int32_t)get_u32("fish_speech.num_codebooks", 10);
    hparams_.semantic_begin_id        = (int32_t)get_u32("fish_speech.semantic_begin_id", 151678);
    hparams_.semantic_end_id          = (int32_t)get_u32("fish_speech.semantic_end_id", 155773);
    hparams_.tie_word_embeddings      = get_bool("fish_speech.tie_word_embeddings", true);
    hparams_.attention_qk_norm        = get_bool("fish_speech.attention_qk_norm", false);
    hparams_.scale_codebook_embeddings = get_bool("fish_speech.scale_codebook_embeddings", false);

    // Fast decoder hparams
    if (hparams_.has_fast_decoder) {
        hparams_.fast_context_length   = (int32_t)get_u32("fish_speech.fast_context_length", 11);
        hparams_.fast_embedding_length = (int32_t)get_u32("fish_speech.fast_embedding_length", 2560);
        hparams_.fast_feed_forward_length = (int32_t)get_u32("fish_speech.fast_feed_forward_length", 9728);
        hparams_.fast_block_count      = (int32_t)get_u32("fish_speech.fast_block_count", 4);
        hparams_.fast_head_count       = (int32_t)get_u32("fish_speech.fast_head_count", 32);
        hparams_.fast_head_count_kv    = (int32_t)get_u32("fish_speech.fast_head_count_kv", 8);
        hparams_.fast_head_dim         = (int32_t)get_u32("fish_speech.fast_head_dim", 128);
        hparams_.fast_rope_freq_base   = get_f32("fish_speech.fast_rope_freq_base", 1e6f);
        hparams_.fast_rms_norm_eps     = get_f32("fish_speech.fast_layer_norm_rms_eps", 1e-6f);
        hparams_.fast_attention_qk_norm = get_bool("fish_speech.fast_attention_qk_norm", false);
        hparams_.fast_has_project_in   = get_bool("fish_speech.fast_project_in", false);
    }

    std::cout << "[Model] Layers: " << hparams_.block_count
              << ", Dim: " << hparams_.embedding_length
              << ", Vocab: " << hparams_.vocab_size
              << ", head_count: " << hparams_.head_count
              << ", has_fast_decoder: " << hparams_.has_fast_decoder << std::endl;

    // ---------------------------------------------------------------------------
    // Load tensor pointers (metadata only — data loaded below)
    // ---------------------------------------------------------------------------
    auto req_t = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(weights_.ctx_w, name.c_str());
        if (!t) {
            throw std::runtime_error("missing tensor: " + name);
        }
        return t;
    };
    auto opt_t = [&](const std::string & name) -> ggml_tensor * {
        return ggml_get_tensor(weights_.ctx_w, name.c_str());
    };

    try {
        weights_.embeddings          = req_t("embeddings.weight");
        weights_.codebook_embeddings = req_t("codebook_embeddings.weight");
        weights_.norm                = req_t("norm.weight");

        weights_.layers.resize(hparams_.block_count);
        for (int32_t i = 0; i < hparams_.block_count; ++i) {
            auto & layer = weights_.layers[i];
            std::string stem = "layers." + std::to_string(i) + ".";

            layer.attention_norm = req_t(stem + "attention_norm.weight");
            layer.ffn_norm       = req_t(stem + "ffn_norm.weight");
            layer.wqkv           = req_t(stem + "attention.wqkv.weight");
            layer.wo             = req_t(stem + "attention.wo.weight");
            layer.w1             = req_t(stem + "feed_forward.w1.weight");
            layer.w2             = req_t(stem + "feed_forward.w2.weight");
            layer.w3             = req_t(stem + "feed_forward.w3.weight");

            if (hparams_.attention_qk_norm) {
                layer.q_norm = req_t(stem + "attention.q_norm.weight");
                layer.k_norm = req_t(stem + "attention.k_norm.weight");
            }
        }

        if (hparams_.has_fast_decoder) {
            if (hparams_.fast_has_project_in) {
                weights_.fast_project_in = req_t("fast_project_in.weight");
            }
            weights_.fast_embeddings = req_t("fast_embeddings.weight");
            weights_.fast_norm       = req_t("fast_norm.weight");
            weights_.fast_output     = req_t("fast_output.weight");

            weights_.fast_layers.resize(hparams_.fast_block_count);
            for (int32_t i = 0; i < hparams_.fast_block_count; ++i) {
                auto & layer = weights_.fast_layers[i];
                std::string stem = "fast_layers." + std::to_string(i) + ".";

                layer.attention_norm = req_t(stem + "attention_norm.weight");
                layer.ffn_norm       = req_t(stem + "ffn_norm.weight");
                layer.wqkv           = req_t(stem + "attention.wqkv.weight");
                layer.wo             = req_t(stem + "attention.wo.weight");
                layer.w1             = req_t(stem + "feed_forward.w1.weight");
                layer.w2             = req_t(stem + "feed_forward.w2.weight");
                layer.w3             = req_t(stem + "feed_forward.w3.weight");

                if (hparams_.fast_attention_qk_norm) {
                    layer.q_norm = req_t(stem + "attention.q_norm.weight");
                    layer.k_norm = req_t(stem + "attention.k_norm.weight");
                }
            }
        }
    } catch (const std::exception & e) {
        std::cerr << "[Model] " << e.what() << std::endl;
        gguf_free(ctx_gguf);
        return false;
    }

    // Allocate backend buffer for all weight tensors
    weights_.model_buf = ggml_backend_alloc_ctx_tensors(weights_.ctx_w, backend_);
    if (!weights_.model_buf) {
        std::cerr << "[Model] Failed to allocate backend buffer for weights." << std::endl;
        gguf_free(ctx_gguf);
        return false;
    }

    // Load tensor data from GGUF file
    const size_t data_offset = gguf_get_data_offset(ctx_gguf);
    const int64_t n_tensors  = gguf_get_n_tensors(ctx_gguf);
    std::FILE * f = std::fopen(gguf_path.c_str(), "rb");
    if (!f) {
        std::cerr << "[Model] Cannot reopen " << gguf_path << " for data loading." << std::endl;
        gguf_free(ctx_gguf);
        return false;
    }
    std::vector<uint8_t> tmp;
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * tname = gguf_get_tensor_name(ctx_gguf, ti);
        ggml_tensor * t = ggml_get_tensor(weights_.ctx_w, tname);
        if (!t) continue;
        const size_t toff  = data_offset + gguf_get_tensor_offset(ctx_gguf, ti);
        const size_t tsize = ggml_nbytes(t);
        if (tmp.size() < tsize) tmp.resize(tsize);
#ifdef _WIN32
        _fseeki64(f, (int64_t)toff, SEEK_SET);
#else
        fseeko(f, (off_t)toff, SEEK_SET);
#endif
        if (std::fread(tmp.data(), 1, tsize, f) != tsize) {
            std::cerr << "[Model] Failed to read tensor: " << tname << std::endl;
            std::fclose(f);
            gguf_free(ctx_gguf);
            return false;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, tsize);
    }
    tmp.clear();
    tmp.shrink_to_fit();
    std::fclose(f);

    // Advise the kernel to drop the file pages from page cache — the weights
    // are now in the backend buffer (VRAM) and we no longer need the cached
    // file data in RAM.
#ifdef __linux__
    {
        int fd = ::open(gguf_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
        }
    }
#endif

    std::cout << "[Model] Weights loaded. Total tensors: " << n_tensors << std::endl;

    gguf_free(ctx_gguf);
    return true;
}

// ---------------------------------------------------------------------------
// init_kv_cache()
// ---------------------------------------------------------------------------

bool SlowARModel::init_kv_cache(int32_t max_seq_len) {
    if (ctx_kv_) {
        ggml_free(ctx_kv_);
        ctx_kv_ = nullptr;
    }
    if (kv_buf_) {
        ggml_backend_buffer_free(kv_buf_);
        kv_buf_ = nullptr;
    }

    max_seq_len_ = max_seq_len;
    n_past_      = 0;

    const int32_t dim = hparams_.embedding_length;
    if (dim == 0) return true;

    // head_dim: if attention_qk_norm, get from q_norm weight shape; else dim/head_count
    int32_t head_dim = 0;
    if (hparams_.attention_qk_norm && !weights_.layers.empty() && weights_.layers[0].q_norm) {
        head_dim = static_cast<int32_t>(weights_.layers[0].q_norm->ne[0]);
    } else {
        head_dim = hparams_.embedding_length / hparams_.head_count;
    }

    const int32_t n_head_kv = hparams_.head_count_kv;
    const int32_t n_layer   = hparams_.block_count;

    const size_t ctx_kv_size = 2ull * ggml_tensor_overhead() + (1ull << 20);
    ggml_init_params p = {
        /*.mem_size =*/ ctx_kv_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    ctx_kv_ = ggml_init(p);
    if (!ctx_kv_) {
        std::cerr << "[Model] Failed to init KV context." << std::endl;
        return false;
    }

    memory_k_ = ggml_new_tensor_4d(ctx_kv_, GGML_TYPE_F16, head_dim, n_head_kv, max_seq_len, n_layer);
    memory_v_ = ggml_new_tensor_4d(ctx_kv_, GGML_TYPE_F16, head_dim, n_head_kv, max_seq_len, n_layer);

    kv_buf_ = ggml_backend_alloc_ctx_tensors(ctx_kv_, backend_);
    if (!kv_buf_) {
        std::cerr << "[Model] Failed to allocate KV cache buffer." << std::endl;
        return false;
    }

    ggml_backend_tensor_memset(memory_k_, 0, 0, ggml_nbytes(memory_k_));
    ggml_backend_tensor_memset(memory_v_, 0, 0, ggml_nbytes(memory_v_));

    return true;
}

// ---------------------------------------------------------------------------
// reset()
// ---------------------------------------------------------------------------

void SlowARModel::reset() {
    n_past_ = 0;
}

// ---------------------------------------------------------------------------
// prefill() / step()
// ---------------------------------------------------------------------------

bool SlowARModel::prefill(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
                          int32_t n_threads, StepResult & result) {
    // Use a temporary gallocr for prefill so the large compute buffer
    // (sized for n_tokens) is freed immediately after, not kept for steps.
    ggml_gallocr_t prefill_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!prefill_allocr) return false;
    std::swap(allocr_, prefill_allocr);
    bool ok = eval_cached(flat_tokens, n_tokens, n_threads, result);
    std::swap(allocr_, prefill_allocr);
    ggml_gallocr_free(prefill_allocr);
    return ok;
}

bool SlowARModel::step(const std::vector<int32_t> & flat_tokens, int32_t n_threads,
                       StepResult & result) {
    return eval_cached(flat_tokens, 1, n_threads, result);
}

// ---------------------------------------------------------------------------
// eval_cached() — main inference path with KV cache
// ---------------------------------------------------------------------------

bool SlowARModel::eval_cached(const std::vector<int32_t> & flat_tokens,
                               int32_t n_tokens, int32_t n_threads,
                               StepResult & result) {
    if (n_tokens <= 0) return false;

    const int32_t codebook_dim = hparams_.num_codebooks + 1;
    if (static_cast<int32_t>(flat_tokens.size()) != n_tokens * codebook_dim) {
        std::fprintf(stderr, "[eval_cached] expected %d ints for %d tokens, got %zu\n",
            n_tokens * codebook_dim, n_tokens, flat_tokens.size());
        return false;
    }
    if (n_past_ + n_tokens > max_seq_len_) {
        std::fprintf(stderr, "[eval_cached] KV cache overflow (%d + %d > %d)\n",
            n_past_, n_tokens, max_seq_len_);
        return false;
    }

    const int32_t dim       = hparams_.embedding_length;
    const int32_t n_head    = hparams_.head_count;
    const int32_t n_head_kv = hparams_.head_count_kv;

    // head_dim: from q_norm when qk_norm, else wo/head_count
    int32_t head_dim = 0;
    if (hparams_.attention_qk_norm && !weights_.layers.empty() && weights_.layers[0].q_norm) {
        head_dim = static_cast<int32_t>(weights_.layers[0].q_norm->ne[0]);
    } else {
        head_dim = static_cast<int32_t>(weights_.layers[0].wo->ne[0] / n_head);
    }

    const int32_t q_size   = n_head * head_dim;
    const int32_t kv_size  = n_head_kv * head_dim;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float sem_scale  = 1.0f / std::sqrt(static_cast<float>(codebook_dim));

    // Prepare host-side input arrays
    std::vector<int32_t> semantic_vals(n_tokens);
    std::vector<int32_t> pos_vals(n_tokens);
    std::vector<float>   semantic_mask_vals(n_tokens, 0.0f);
    std::vector<float>   token_scale_vals;
    std::vector<std::vector<int32_t>> cb_vals(hparams_.num_codebooks, std::vector<int32_t>(n_tokens, 0));

    if (hparams_.scale_codebook_embeddings) {
        token_scale_vals.resize(n_tokens);
    }

    for (int32_t t = 0; t < n_tokens; ++t) {
        const int32_t semantic = flat_tokens[t * codebook_dim];
        const bool is_semantic = (semantic >= hparams_.semantic_begin_id &&
                                  semantic <= hparams_.semantic_end_id);

        semantic_vals[t]      = semantic;
        pos_vals[t]           = n_past_ + t;
        semantic_mask_vals[t] = is_semantic ? 1.0f : 0.0f;

        if (!token_scale_vals.empty()) {
            token_scale_vals[t] = is_semantic ? sem_scale : 1.0f;
        }

        for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
            if (!is_semantic) continue;
            const int32_t v = flat_tokens[t * codebook_dim + cb + 1];
            cb_vals[cb][t] = v + cb * hparams_.codebook_size;
        }
    }

    // Build computation graph
    static size_t ctx_size = 0;
    static std::vector<uint8_t> ctx_buf;
    if (ctx_size == 0) {
        ctx_size = 10u * 1024u * 1024u;
        ctx_buf.resize(ctx_size);
    }
    ggml_init_params p = { ctx_size, ctx_buf.data(), true };
    ggml_context * ctx0 = ggml_init(p);
    if (!ctx0) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor * semantic_ids   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_tensor * positions      = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_tensor * semantic_mask  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    ggml_tensor * token_scale    = nullptr;
    if (hparams_.scale_codebook_embeddings) {
        token_scale = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    }

    ggml_tensor * x = ggml_get_rows(ctx0, weights_.embeddings, semantic_ids);
    if (x->type != GGML_TYPE_F32) x = ggml_cast(ctx0, x, GGML_TYPE_F32);

    std::vector<ggml_tensor *> cb_id_tensors(hparams_.num_codebooks);
    ggml_tensor * codebook_sum = nullptr;
    for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
        ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        cb_id_tensors[cb] = ids;
        ggml_tensor * emb = ggml_get_rows(ctx0, weights_.codebook_embeddings, ids);
        if (emb->type != GGML_TYPE_F32) emb = ggml_cast(ctx0, emb, GGML_TYPE_F32);
        codebook_sum = (codebook_sum == nullptr) ? emb : ggml_add(ctx0, codebook_sum, emb);
    }

    if (codebook_sum != nullptr) {
        // Mask out codebook embeddings for non-semantic positions
        codebook_sum = ggml_mul(ctx0, codebook_sum,
                                ggml_repeat(ctx0, semantic_mask, codebook_sum));
        x = ggml_add(ctx0, x, codebook_sum);
    }
    if (token_scale != nullptr) {
        x = ggml_mul(ctx0, x, ggml_repeat(ctx0, token_scale, x));
    }

    for (int32_t il = 0; il < hparams_.block_count; ++il) {
        const auto & layer = weights_.layers[il];

        ggml_tensor * attn_in = rms_norm_weighted(ctx0, x, layer.attention_norm, hparams_.rms_norm_eps);
        ggml_tensor * qkv     = mul_mat_checked(ctx0, layer.wqkv, attn_in, "mul_mat:wqkv");
        const size_t elem_size = ggml_element_size(qkv);

        ggml_tensor * q2d = ggml_view_2d(ctx0, qkv, q_size, n_tokens, qkv->nb[1], 0);
        ggml_tensor * k2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], q_size * elem_size);
        ggml_tensor * v2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], (q_size + kv_size) * elem_size);

        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q2d), head_dim, n_head, n_tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_cont(ctx0, k2d), head_dim, n_head_kv, n_tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_cont(ctx0, v2d), head_dim, n_head_kv, n_tokens);

        // QK norm (applied before RoPE)
        if (hparams_.attention_qk_norm) {
            q = rms_norm_weighted(ctx0, q, layer.q_norm, hparams_.rms_norm_eps);
            k = rms_norm_weighted(ctx0, k, layer.k_norm, hparams_.rms_norm_eps);
        }

        // RoPE
        q = ggml_rope_ext(ctx0, q, positions, nullptr, head_dim, 0,
                          hparams_.context_length, hparams_.rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        k = ggml_rope_ext(ctx0, k, positions, nullptr, head_dim, 0,
                          hparams_.context_length, hparams_.rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

        // Write K/V into KV cache
        const size_t layer_off_k = static_cast<size_t>(il) * memory_k_->nb[3];
        const size_t layer_off_v = static_cast<size_t>(il) * memory_v_->nb[3];
        const size_t token_off_k = static_cast<size_t>(n_past_) * memory_k_->nb[2];
        const size_t token_off_v = static_cast<size_t>(n_past_) * memory_v_->nb[2];

        ggml_tensor * k_slot = ggml_view_3d(ctx0, memory_k_,
            head_dim, n_head_kv, n_tokens,
            memory_k_->nb[1], memory_k_->nb[2],
            layer_off_k + token_off_k);
        ggml_tensor * v_slot = ggml_view_3d(ctx0, memory_v_,
            head_dim, n_head_kv, n_tokens,
            memory_v_->nb[1], memory_v_->nb[2],
            layer_off_v + token_off_v);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, k, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, v, v_slot));

        ggml_tensor * k_mem = k;
        ggml_tensor * v_mem = v;
        if (n_past_ > 0) {
            ggml_tensor * k_past = ggml_reshape_3d(ctx0,
                ggml_view_1d(ctx0, memory_k_, static_cast<int64_t>(n_past_) * kv_size, layer_off_k),
                head_dim, n_head_kv, n_past_);
            ggml_tensor * v_past = ggml_reshape_3d(ctx0,
                ggml_view_1d(ctx0, memory_v_, static_cast<int64_t>(n_past_) * kv_size, layer_off_v),
                head_dim, n_head_kv, n_past_);
            if (k_past->type != k->type) k_past = ggml_cast(ctx0, k_past, k->type);
            if (v_past->type != v->type) v_past = ggml_cast(ctx0, v_past, v->type);
            k_mem = ggml_concat(ctx0, k_past, k, 2);
            v_mem = ggml_concat(ctx0, v_past, v, 2);
        }

        if (n_head != n_head_kv && q->type != GGML_TYPE_F32) {
            q = ggml_cast(ctx0, q, GGML_TYPE_F32);
        }
        ggml_tensor * k_rep = repeat_interleave_heads(ctx0, k_mem, n_head / n_head_kv);
        ggml_tensor * v_rep = repeat_interleave_heads(ctx0, v_mem, n_head / n_head_kv);

        ggml_tensor * Q   = ggml_permute(ctx0, q,     0, 2, 1, 3);
        ggml_tensor * K   = ggml_permute(ctx0, k_rep, 0, 2, 1, 3);
        ggml_tensor * KQ  = mul_mat_checked(ctx0, K, Q, "mul_mat:kq");
        ggml_tensor * KQs = ggml_scale(ctx0, KQ, attn_scale);
        ggml_tensor * KQm = ggml_diag_mask_inf(ctx0, KQs, n_past_);
        ggml_tensor * KQf = ggml_soft_max(ctx0, KQm);

        ggml_tensor * V       = ggml_cont(ctx0, ggml_permute(ctx0, v_rep, 1, 2, 0, 3));
        ggml_tensor * KQV     = mul_mat_checked(ctx0, V, KQf, "mul_mat:kqv");
        ggml_tensor * KQVm    = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        ggml_tensor * attn_cur = ggml_cpy(ctx0, KQVm,
                                          ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, q_size, n_tokens));
        ggml_tensor * attn_out = mul_mat_checked(ctx0, layer.wo, attn_cur, "mul_mat:wo");

        ggml_tensor * h     = ggml_add(ctx0, x, attn_out);
        ggml_tensor * ff_in = rms_norm_weighted(ctx0, h, layer.ffn_norm, hparams_.rms_norm_eps);
        ggml_tensor * gate  = mul_mat_checked(ctx0, layer.w1, ff_in, "mul_mat:w1");
        ggml_tensor * up    = mul_mat_checked(ctx0, layer.w3, ff_in, "mul_mat:w3");
        ggml_tensor * ff_h  = ggml_swiglu_split(ctx0, gate, up);
        ggml_tensor * ff_out = mul_mat_checked(ctx0, layer.w2, ff_h, "mul_mat:w2");

        x = ggml_add(ctx0, h, ff_out);
    }

    ggml_tensor * slow_out  = rms_norm_weighted(ctx0, x, weights_.norm, hparams_.rms_norm_eps);
    ggml_tensor * slow_cont = ggml_cont(ctx0, slow_out);
    ggml_tensor * hidden_last = ggml_cpy(ctx0,
        last_token_view(ctx0, slow_cont, n_tokens),
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, 1));

    ggml_tensor * logits = mul_mat_checked(ctx0, weights_.embeddings, hidden_last, "mul_mat:logits");
    ggml_build_forward_expand(gf, logits);

    // Allocate and run
    if (!ggml_gallocr_alloc_graph(allocr_, gf)) {
        std::fprintf(stderr, "[eval_cached] gallocr alloc failed\n");
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(semantic_ids,  semantic_vals.data(), 0, n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(positions,     pos_vals.data(),       0, n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(semantic_mask, semantic_mask_vals.data(), 0, n_tokens * sizeof(float));
    if (token_scale) {
        ggml_backend_tensor_set(token_scale, token_scale_vals.data(), 0, n_tokens * sizeof(float));
    }
    for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
        ggml_backend_tensor_set(cb_id_tensors[cb], cb_vals[cb].data(), 0, n_tokens * sizeof(int32_t));
    }

    if (ggml_backend_is_cpu(backend_)) {
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[eval_cached] compute failed\n");
        ggml_free(ctx0);
        return false;
    }

    result.hidden.resize(dim);
    result.logits.resize(hparams_.vocab_size);
    ggml_backend_tensor_get(hidden_last, result.hidden.data(), 0, dim * sizeof(float));
    ggml_backend_tensor_get(logits,      result.logits.data(), 0, hparams_.vocab_size * sizeof(float));

    ggml_free(ctx0);
    n_past_ += n_tokens;
    return true;
}

// ---------------------------------------------------------------------------
// fast_decode() — fast AR decoder (matches eval_fast_prefix from reference)
// ---------------------------------------------------------------------------

bool SlowARModel::fast_decode(const std::vector<float> & hidden_in,
                               const std::vector<int32_t> & prefix_tokens,
                               int32_t n_threads,
                               std::vector<float> & logits_out) {
    if (!hparams_.has_fast_decoder) {
        std::fprintf(stderr, "[fast_decode] model has no fast decoder\n");
        return false;
    }
    if (static_cast<int32_t>(hidden_in.size()) != hparams_.embedding_length) {
        std::fprintf(stderr, "[fast_decode] expected hidden size %d, got %zu\n",
            hparams_.embedding_length, hidden_in.size());
        return false;
    }
    if (static_cast<int32_t>(prefix_tokens.size()) >= hparams_.num_codebooks) {
        std::fprintf(stderr, "[fast_decode] prefix too long (%zu >= %d)\n",
            prefix_tokens.size(), hparams_.num_codebooks);
        return false;
    }

    const int32_t fast_dim  = hparams_.fast_embedding_length;
    const int32_t n_head    = hparams_.fast_head_count;
    const int32_t n_head_kv = hparams_.fast_head_count_kv;
    const int32_t head_dim  = (hparams_.fast_head_dim > 0)
                                ? hparams_.fast_head_dim
                                : fast_dim / n_head;
    const int32_t q_size    = n_head * head_dim;
    const int32_t kv_size   = n_head_kv * head_dim;
    const float attn_scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int32_t n_tokens  = static_cast<int32_t>(prefix_tokens.size()) + 1;

    static size_t fast_ctx_size = 0;
    static std::vector<uint8_t> fast_ctx_buf;
    if (fast_ctx_size == 0) {
        fast_ctx_size = 8u * 1024u * 1024u;
        fast_ctx_buf.resize(fast_ctx_size);
    }
    ggml_init_params p = { fast_ctx_size, fast_ctx_buf.data(), true };
    ggml_context * ctx0 = ggml_init(p);
    if (!ctx0) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    // hidden input (from slow decoder)
    ggml_tensor * hidden0 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams_.embedding_length, 1);

    // Optional project_in (not present for this model, but handle it)
    ggml_tensor * projected = (weights_.fast_project_in != nullptr)
        ? mul_mat_checked(ctx0, weights_.fast_project_in, hidden0, "mul_mat:fast_project_in")
        : hidden0;
    if (projected->type != GGML_TYPE_F32) {
        projected = ggml_cast(ctx0, projected, GGML_TYPE_F32);
    }

    // Build sequence: [projected_hidden; prefix_embeddings]
    ggml_tensor * x = projected;
    ggml_tensor * prefix_ids = nullptr;
    if (!prefix_tokens.empty()) {
        prefix_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)prefix_tokens.size());
        ggml_tensor * prefix_emb = ggml_get_rows(ctx0, weights_.fast_embeddings, prefix_ids);
        if (prefix_emb->type != GGML_TYPE_F32) {
            prefix_emb = ggml_cast(ctx0, prefix_emb, GGML_TYPE_F32);
        }
        x = ggml_concat(ctx0, x, prefix_emb, 1);
    }

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    std::vector<int32_t> pos_vals(n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) pos_vals[i] = i;

    for (int32_t il = 0; il < hparams_.fast_block_count; ++il) {
        const auto & layer = weights_.fast_layers[il];

        ggml_tensor * attn_in = rms_norm_weighted(ctx0, x, layer.attention_norm, hparams_.fast_rms_norm_eps);
        ggml_tensor * qkv     = mul_mat_checked(ctx0, layer.wqkv, attn_in, "mul_mat:fast_wqkv");
        const size_t elem_size = ggml_element_size(qkv);

        ggml_tensor * q2d = ggml_view_2d(ctx0, qkv, q_size, n_tokens, qkv->nb[1], 0);
        ggml_tensor * k2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], q_size * elem_size);
        ggml_tensor * v2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], (q_size + kv_size) * elem_size);

        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q2d), head_dim, n_head, n_tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_cont(ctx0, k2d), head_dim, n_head_kv, n_tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_cont(ctx0, v2d), head_dim, n_head_kv, n_tokens);

        if (hparams_.fast_attention_qk_norm) {
            q = rms_norm_weighted(ctx0, q, layer.q_norm, hparams_.fast_rms_norm_eps);
            k = rms_norm_weighted(ctx0, k, layer.k_norm, hparams_.fast_rms_norm_eps);
        }

        q = ggml_rope_ext(ctx0, q, positions, nullptr, head_dim, 0,
                          hparams_.fast_context_length, hparams_.fast_rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        k = ggml_rope_ext(ctx0, k, positions, nullptr, head_dim, 0,
                          hparams_.fast_context_length, hparams_.fast_rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

        ggml_tensor * k_rep = repeat_interleave_heads(ctx0, k, n_head / n_head_kv);
        ggml_tensor * v_rep = repeat_interleave_heads(ctx0, v, n_head / n_head_kv);

        ggml_tensor * Q   = ggml_permute(ctx0, q,     0, 2, 1, 3);
        ggml_tensor * K   = ggml_permute(ctx0, k_rep, 0, 2, 1, 3);
        ggml_tensor * KQ  = mul_mat_checked(ctx0, K, Q, "mul_mat:fast_kq");
        ggml_tensor * KQs = ggml_scale(ctx0, KQ, attn_scale);
        ggml_tensor * KQm = ggml_diag_mask_inf(ctx0, KQs, 0);
        ggml_tensor * KQf = ggml_soft_max(ctx0, KQm);

        ggml_tensor * V       = ggml_cont(ctx0, ggml_permute(ctx0, v_rep, 1, 2, 0, 3));
        ggml_tensor * KQV     = mul_mat_checked(ctx0, V, KQf, "mul_mat:fast_kqv");
        ggml_tensor * KQVm    = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        ggml_tensor * attn_cur = ggml_cpy(ctx0, KQVm,
                                          ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, q_size, n_tokens));
        ggml_tensor * attn_out = mul_mat_checked(ctx0, layer.wo, attn_cur, "mul_mat:fast_wo");

        ggml_tensor * h     = ggml_add(ctx0, x, attn_out);
        ggml_tensor * ff_in = rms_norm_weighted(ctx0, h, layer.ffn_norm, hparams_.fast_rms_norm_eps);
        ggml_tensor * gate  = mul_mat_checked(ctx0, layer.w1, ff_in, "mul_mat:fast_w1");
        ggml_tensor * up    = mul_mat_checked(ctx0, layer.w3, ff_in, "mul_mat:fast_w3");
        ggml_tensor * ff_h  = ggml_swiglu_split(ctx0, gate, up);
        ggml_tensor * ff_out = mul_mat_checked(ctx0, layer.w2, ff_h, "mul_mat:fast_w2");

        x = ggml_add(ctx0, h, ff_out);
    }

    ggml_tensor * fast_out  = rms_norm_weighted(ctx0, x, weights_.fast_norm, hparams_.fast_rms_norm_eps);
    ggml_tensor * fast_cont = ggml_cont(ctx0, fast_out);
    ggml_tensor * fast_last = ggml_cpy(ctx0,
        last_token_view(ctx0, fast_cont, n_tokens),
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, fast_dim, 1));
    ggml_tensor * logits = mul_mat_checked(ctx0, weights_.fast_output, fast_last, "mul_mat:fast_logits");
    ggml_build_forward_expand(gf, logits);

    if (!ggml_gallocr_alloc_graph(fast_allocr_, gf)) {
        std::fprintf(stderr, "[fast_decode] gallocr alloc failed\n");
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(hidden0,   hidden_in.data(),    0, hidden_in.size() * sizeof(float));
    ggml_backend_tensor_set(positions, pos_vals.data(),     0, pos_vals.size() * sizeof(int32_t));
    if (prefix_ids) {
        ggml_backend_tensor_set(prefix_ids, prefix_tokens.data(), 0,
                                prefix_tokens.size() * sizeof(int32_t));
    }

    if (ggml_backend_is_cpu(backend_)) {
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[fast_decode] compute failed\n");
        ggml_free(ctx0);
        return false;
    }

    // logits_out has codebook_size elements (NOT vocab_size)
    logits_out.resize(hparams_.codebook_size);
    ggml_backend_tensor_get(logits, logits_out.data(), 0, hparams_.codebook_size * sizeof(float));

    ggml_free(ctx0);
    return true;
}

} // namespace s2
