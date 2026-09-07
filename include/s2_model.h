#pragma once

#include "s2_backend.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace s2 {

struct ModelLayer {
    ggml_tensor * attention_norm = nullptr;
    ggml_tensor * ffn_norm       = nullptr;
    ggml_tensor * q_norm         = nullptr;
    ggml_tensor * k_norm         = nullptr;
    ggml_tensor * wqkv           = nullptr;
    ggml_tensor * wo             = nullptr;
    ggml_tensor * w1             = nullptr;
    ggml_tensor * w2             = nullptr;
    ggml_tensor * w3             = nullptr;
};

struct ModelHParams {
    int32_t context_length     = 0;
    int32_t vocab_size         = 0;
    int32_t embedding_length   = 0;
    int32_t feed_forward_length = 0;
    int32_t block_count        = 0;
    int32_t head_count         = 0;
    int32_t head_count_kv      = 0;
    int32_t codebook_size      = 0;
    int32_t num_codebooks      = 0;
    int32_t semantic_begin_id  = 0;
    int32_t semantic_end_id    = 0;
    float   rope_freq_base     = 10000.0f;
    float   rms_norm_eps       = 1e-5f;
    bool    tie_word_embeddings = true;
    bool    attention_qk_norm  = false;
    bool    scale_codebook_embeddings = false;

    int32_t fast_context_length     = 0;
    int32_t fast_embedding_length   = 0;
    int32_t fast_feed_forward_length = 0;
    int32_t fast_block_count        = 0;
    int32_t fast_head_count         = 0;
    int32_t fast_head_count_kv      = 0;
    int32_t fast_head_dim           = 0;
    float   fast_rope_freq_base     = 10000.0f;
    float   fast_rms_norm_eps       = 1e-5f;
    bool    fast_attention_qk_norm  = false;
    bool    fast_has_project_in     = false;
    bool    has_fast_decoder        = false;
};

struct ModelWeights {
    ggml_context * ctx_w = nullptr;
    std::vector<ggml_backend_buffer_t> model_bufs_cpu;
    std::vector<ggml_backend_buffer_t> model_bufs_gpu;

    ggml_tensor * embeddings           = nullptr;
    ggml_tensor * codebook_embeddings  = nullptr;
    ggml_tensor * norm                 = nullptr;
    ggml_tensor * fast_project_in      = nullptr;
    ggml_tensor * fast_embeddings      = nullptr;
    ggml_tensor * fast_norm            = nullptr;
    ggml_tensor * fast_output          = nullptr;

    std::vector<ModelLayer> layers;
    std::vector<ModelLayer> fast_layers;
};

struct StepResult {
    std::vector<float> hidden;
    // The sampling window is the semantic range plus the end-of-sequence token
    // (im_end). logits holds one score per window entry; window_ids maps each
    // entry back to its full-vocab id. This keeps the logits head, the mask
    // application and the sampler all O(window) instead of O(vocab_size).
    std::vector<int32_t> window_ids;
    std::vector<float> logits;
};

class SlowARModel {
public:
    SlowARModel();
    ~SlowARModel();

    bool load(const std::string & gguf_path, int32_t gpu_device = -1, BackendType backend_type = BackendType::CPU, int32_t n_gpu_layers = -1);

    bool load_shared(gguf_context * gguf_ctx, const std::string & gguf_path, int32_t gpu_device = -1, BackendType backend_type = BackendType::CPU, int32_t n_gpu_layers = -1);

    bool read_tensor_data(const std::string & gguf_path, gguf_context * gguf_ctx);

    ggml_context * weights_ctx() { return weights_.ctx_w; }
    const std::unordered_set<ggml_tensor *> & weight_tensor_set() const { return weight_tensor_set_; }

    bool init_kv_cache(int32_t max_seq_len);

    void reset();

    void clear_kv_cache();

private:
    bool eval_cached(const std::vector<int32_t> & flat_tokens,
        int32_t n_tokens, int32_t n_threads,
        StepResult & result);
public:
    bool prefill_fast(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
        int32_t n_threads, StepResult & result);

    bool prefill(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
                 int32_t n_threads, StepResult & result);

    bool step(const std::vector<int32_t> & flat_tokens, int32_t n_threads,
              StepResult & result);

    bool fast_decode(const std::vector<float> & hidden,
                     const std::vector<int32_t> & prefix_codes,
                     int32_t n_threads,
                     std::vector<float> & logits_out);

    const ModelHParams & hparams() const { return hparams_; }

    // The end-of-sequence token id (from the tokenizer config). Needed so the
    // logits head only materializes the sampling window (semantic range + im_end)
    // instead of the full vocab. Set once before generating.
    void set_im_end_id(int32_t id) { im_end_id_ = id; }

    ModelWeights   weights_;

private:
    ModelHParams   hparams_;
    int32_t        im_end_id_ = -1;
    ggml_backend_t backend_cpu_  = nullptr;
    ggml_backend_t backend_gpu_  = nullptr;
    ggml_backend_sched_t sched_       = nullptr;
    ggml_backend_sched_t fast_sched_  = nullptr;
    ggml_context * ctx_kv_      = nullptr;
    ggml_backend_buffer_t kv_buf_ = nullptr;
    ggml_tensor *  memory_k_   = nullptr;
    ggml_tensor *  memory_v_   = nullptr;
    int32_t        max_seq_len_ = 0;
    int32_t        n_gpu_layers_ = -1;
    int32_t        n_past_     = 0;
    size_t         ctx_size_   = 0;
    std::vector<uint8_t> ctx_buf_;
    size_t         fast_ctx_size_ = 0;
    std::vector<uint8_t> fast_ctx_buf_;

    // Stage 2A: cached decode graphs. The slow decode (n_tokens == 1) and fast
    // decode (fixed num_codebooks) graphs are shape-stable, so they are built once
    // and reused across frames — only the input tensor VALUES are re-uploaded.
    // Cached tensors reference memory_k_/memory_v_ (KV cache) and the scheduler's
    // compute buffers, so this state is invalidated whenever those are rebuilt.
    ggml_context * slow_ctx_   = nullptr;
    ggml_cgraph  * slow_gf_    = nullptr;
    ggml_tensor  * slow_semantic_ids_  = nullptr;
    ggml_tensor  * slow_positions_     = nullptr;
    ggml_tensor  * slow_semantic_mask_ = nullptr;
    ggml_tensor  * slow_n_past_idx_    = nullptr;
    ggml_tensor  * slow_token_scale_   = nullptr;
    ggml_tensor  * slow_attn_mask_     = nullptr;
    ggml_tensor  * slow_hidden_last_   = nullptr;
    ggml_tensor  * slow_logits_        = nullptr;
    ggml_tensor  * slow_window_ids_t_  = nullptr;
    std::vector<ggml_tensor *> slow_cb_id_tensors_;

    ggml_context * fast_ctx_   = nullptr;
    ggml_cgraph  * fast_gf_    = nullptr;
    ggml_tensor  * fast_hidden0_     = nullptr;
    ggml_tensor  * fast_prefix_ids_  = nullptr;
    ggml_tensor  * fast_positions_   = nullptr;
    ggml_tensor  * fast_out_idx_     = nullptr;
    ggml_tensor  * fast_logits_      = nullptr;

    void invalidate_cached_graphs();

    std::unordered_set<ggml_tensor *> weight_tensor_set_;

};

}
