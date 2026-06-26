#include "../include/s2_codec.h"
#include "../include/s2_log.h"
#include "s2_ggml_utils.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>

namespace s2 {

struct vq_cache {
    int32_t input_dim    = 0;
    int32_t codebook_dim = 0;
    int32_t codebook_size = 0;

    std::vector<float> in_proj_weight;
    std::vector<float> in_proj_bias;
    std::vector<float> out_proj_weight;
    std::vector<float> out_proj_bias;
    std::vector<float> codebook;
    std::vector<float> codebook_norm;
};

struct transformer_inputs {
    ggml_tensor * positions = nullptr;
    ggml_tensor * mask      = nullptr;
    std::vector<int32_t> position_values;
    std::vector<float>   mask_values;
};

struct codec_decode_cache {
    int32_t n_frames = 0;
    int32_t n_samples = 0;
    int32_t failed_n_frames = 0;
    size_t ctx_size = 0;

    std::vector<uint8_t> ctx_buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocr = nullptr;

    std::vector<ggml_tensor *> code_id_tensors;
    transformer_inputs inp;
    ggml_tensor * audio = nullptr;
    std::vector<int32_t> sanitized_codes;
};

struct AudioCodec::Impl {
    ggml_backend_t        backend   = nullptr;
    ggml_context *        ctx_w     = nullptr;
    ggml_backend_buffer_t model_buf = nullptr;
    std::string tprefix;

    int32_t sample_rate   = 0;
    int32_t hop_length    = 0;
    int32_t frame_length  = 0;
    int32_t encoder_dim   = 0;
    int32_t decoder_dim   = 0;
    int32_t latent_dim    = 0;

    std::vector<int32_t> encoder_rates;
    std::vector<int32_t> decoder_rates;
    std::vector<int32_t> encoder_transformer_layers;

    int32_t quantizer_input_dim              = 0;
    int32_t quantizer_codebook_dim           = 0;
    int32_t quantizer_residual_codebooks     = 0;
    int32_t quantizer_residual_codebook_size = 0;
    int32_t quantizer_semantic_codebook_size = 0;
    std::vector<int32_t> quantizer_downsample_factor;

    int32_t transformer_block_size   = 0;
    int32_t transformer_n_local_heads = 0;
    int32_t transformer_head_dim     = 0;
    float   transformer_rope_base    = 10000.0f;
    float   transformer_norm_eps     = 1e-5f;

    int32_t rvq_transformer_window_size   = 0;
    int32_t rvq_transformer_block_size    = 0;
    int32_t rvq_transformer_n_layer       = 0;
    int32_t rvq_transformer_n_local_heads = 0;
    int32_t rvq_transformer_head_dim      = 0;
    int32_t rvq_transformer_dim           = 0;
    float   rvq_transformer_rope_base     = 10000.0f;
    float   rvq_transformer_norm_eps      = 1e-5f;

    vq_cache semantic_vq;
    std::vector<vq_cache> residual_vq;

    codec_decode_cache decode_cache;
};

static const char * backend_type_name(BackendType backend_type) {
    switch (backend_type) {
        case BackendType::CPU:    return "CPU";
        case BackendType::Vulkan: return "Vulkan";
        case BackendType::CUDA:   return "CUDA";
        case BackendType::Metal:  return "Metal";
    }
    return "Unknown";
}

static void reset_decode_cache(codec_decode_cache & cache, bool preserve_failed_n_frames = true) {
    if (cache.allocr) {
        ggml_gallocr_free(cache.allocr);
        cache.allocr = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }

    const int32_t failed_n_frames = cache.failed_n_frames;
    cache = codec_decode_cache();
    if (preserve_failed_n_frames) {
        cache.failed_n_frames = failed_n_frames;
    }
}

static void reset_codec_impl(AudioCodec::Impl & impl) {
    reset_decode_cache(impl.decode_cache, false);
    if (impl.model_buf) {
        ggml_backend_buffer_free(impl.model_buf);
        impl.model_buf = nullptr;
    }
    if (impl.ctx_w) {
        ggml_free(impl.ctx_w);
        impl.ctx_w = nullptr;
    }
    if (impl.backend) {
        ggml_backend_free(impl.backend);
        impl.backend = nullptr;
    }
    impl = AudioCodec::Impl();
}

static bool backend_requires_explicit_causal_mask(ggml_backend_t backend) {
#ifdef GGML_USE_METAL
    return backend != nullptr && ggml_backend_is_metal(backend);
#else
    (void) backend;
    return false;
#endif
}

static ggml_tensor * reshape_vector_cl(ggml_context * ctx, ggml_tensor * t, int64_t channels) {
    ggml_tensor * cur = (t->type == GGML_TYPE_F32) ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
    return ggml_reshape_2d(ctx, cur, channels, 1);
}

static ggml_tensor * reshape_vector_lc(ggml_context * ctx, ggml_tensor * t, int64_t channels) {
    ggml_tensor * cur = (t->type == GGML_TYPE_F32) ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
    return ggml_reshape_2d(ctx, cur, 1, channels);
}

static ggml_tensor * add_channel_bias_cl(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    return ggml_add(ctx, x, repeat_checked(ctx, reshape_vector_cl(ctx, bias, x->ne[0]), x, "repeat:bias_cl"));
}

static ggml_tensor * add_channel_bias_lc(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    return ggml_add(ctx, x, repeat_checked(ctx, reshape_vector_lc(ctx, bias, x->ne[1]), x, "repeat:bias_lc"));
}

static ggml_tensor * scale_channels_cl(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale) {
    return ggml_mul(ctx, repeat_checked(ctx, reshape_vector_cl(ctx, scale, x->ne[0]), x, "repeat:scale"), x);
}

static ggml_tensor * rms_norm_weighted_cl(ggml_context * ctx, ggml_tensor * x,
                                           ggml_tensor * weight, float eps) {
    ggml_tensor * cur = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, repeat_checked(ctx, reshape_vector_cl(ctx, weight, cur->ne[0]), cur, "repeat:rms"), cur);
}

static ggml_tensor * layer_norm_affine(ggml_context * ctx, ggml_tensor * x,
                                        ggml_tensor * weight, ggml_tensor * bias, float eps) {
    ggml_tensor * cur = ggml_norm(ctx, x, eps);
    cur = ggml_mul(ctx, repeat_checked(ctx, reshape_vector_cl(ctx, weight, cur->ne[0]), cur, "repeat:ln_w"), cur);
    cur = ggml_add(ctx, cur, repeat_checked(ctx, reshape_vector_cl(ctx, bias, cur->ne[0]), cur, "repeat:ln_b"));
    return cur;
}

static ggml_tensor * snake_activation(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha) {
    ggml_tensor * alpha_2d = reshape_vector_cl(ctx, alpha, x->ne[0]);
    ggml_tensor * alpha_rep = repeat_checked(ctx, alpha_2d, x, "repeat:snake_alpha");
    ggml_tensor * ax     = ggml_mul(ctx, alpha_rep, x);
    ggml_tensor * sin_ax = ggml_sin(ctx, ax);
    ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);
    return ggml_add(ctx, x, ggml_div(ctx, sin_sq, alpha_rep));
}

static int64_t extra_padding_for_conv1d(int64_t length, int kernel_size, int stride, int padding_total) {
    const float n_frames = (static_cast<float>(length - kernel_size + padding_total) / stride) + 1.0f;
    const int64_t ideal  = (static_cast<int64_t>(std::ceil(n_frames)) - 1) * stride + (kernel_size - padding_total);
    return ideal - length;
}

static ggml_tensor * cl_to_lc(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

static ggml_tensor * lc_to_cl(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
    ggml_tensor * x2 = ggml_reshape_2d(ctx, xc, xc->ne[0], xc->ne[1]);
    ggml_tensor * tr = ggml_cont(ctx, ggml_transpose(ctx, x2));
    return ggml_reshape_2d(ctx, tr, tr->ne[0], tr->ne[1]);
}

static ggml_tensor * causal_conv_1d(ggml_context * ctx,
                                     ggml_tensor * weight, ggml_tensor * bias,
                                     ggml_tensor * x, int stride, int dilation) {
    const int kernel_size = static_cast<int>((weight->ne[0] - 1) * dilation + 1);
    const int pad   = kernel_size - stride;
    const int extra = static_cast<int>(extra_padding_for_conv1d(x->ne[1], kernel_size, stride, pad));
    ggml_tensor * x_lc = cl_to_lc(ctx, x);
    x_lc = ggml_pad_ext(ctx, x_lc, pad, extra, 0, 0, 0, 0, 0, 0);
    ggml_tensor * y = ggml_conv_1d(ctx, weight, x_lc, stride, 0, dilation);
    y = add_channel_bias_lc(ctx, y, bias);
    return lc_to_cl(ctx, y);
}

static ggml_tensor * causal_conv_transpose_1d(ggml_context * ctx,
                                               ggml_tensor * weight, ggml_tensor * bias,
                                               ggml_tensor * x, int stride, int crop_right) {
    if (weight->type != GGML_TYPE_F32) weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
    ggml_tensor * x_lc = cl_to_lc(ctx, x);
    if (x_lc->type != GGML_TYPE_F32) x_lc = ggml_cast(ctx, x_lc, GGML_TYPE_F32);
    ggml_tensor * y = ggml_conv_transpose_1d(ctx, weight, x_lc, stride, 0, 1);
    y = add_channel_bias_lc(ctx, y, bias);
    if (crop_right > 0) {
        y = ggml_view_2d(ctx, y, y->ne[0] - crop_right, y->ne[1], y->nb[1], 0);
    }
    return lc_to_cl(ctx, y);
}

static ggml_tensor * linear_bias(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias,
                                  ggml_tensor * x, const char * label) {
    return add_channel_bias_cl(ctx, mul_mat_checked(ctx, weight, x, label), bias);
}

static ggml_tensor * repeat_interleave_heads(ggml_context * ctx, ggml_tensor * x, int32_t rep) {
    if (rep == 1) return x;
    ggml_tensor * xf = (x->type != GGML_TYPE_F32) ? ggml_cast(ctx, x, GGML_TYPE_F32) : x;
    ggml_tensor * x4 = ggml_reshape_4d(ctx, ggml_cont(ctx, xf), xf->ne[0], 1, xf->ne[1], xf->ne[2]);
    ggml_tensor * tgt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, xf->ne[0], rep, xf->ne[1], xf->ne[2]);
    ggml_tensor * rp  = ggml_repeat(ctx, x4, tgt);
    return ggml_reshape_3d(ctx, ggml_cont(ctx, rp), xf->ne[0], xf->ne[1] * rep, xf->ne[2]);
}

static void prepare_transformer_inputs(ggml_context * ctx, transformer_inputs & inp,
                                        int32_t seq_len, int32_t window_size,
                                        bool force_explicit_causal_mask) {
    if (inp.positions != nullptr) return;

    inp.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    inp.position_values.resize(seq_len);
    for (int32_t i = 0; i < seq_len; ++i) inp.position_values[i] = i;

    const bool use_window_mask = window_size > 0 && window_size < seq_len;
    const bool use_full_causal_mask = force_explicit_causal_mask && !use_window_mask;
    if (use_window_mask || use_full_causal_mask) {
        inp.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);
        inp.mask_values.resize(static_cast<size_t>(seq_len) * seq_len);
        for (int32_t q = 0; q < seq_len; ++q) {
            const int32_t min_k = use_window_mask ? std::max(0, q - window_size + 1) : 0;
            for (int32_t k = 0; k < seq_len; ++k) {
                const bool allowed = (k >= min_k && k <= q);
                inp.mask_values[static_cast<size_t>(q) * seq_len + k] = allowed ? 0.0f : -1e9f;
            }
        }
    }
}

static ggml_tensor * build_transformer(ggml_context * ctx, ggml_context * ctx_w,
                                        const std::string & prefix, ggml_tensor * x,
                                        int32_t block_size, int32_t n_local_heads, int32_t head_dim,
                                        float rope_base, float norm_eps, int32_t window_size,
                                        bool force_explicit_causal_mask,
                                        transformer_inputs & inp) {
    const int32_t dim     = static_cast<int32_t>(x->ne[0]);
    const int32_t seq_len = static_cast<int32_t>(x->ne[1]);
    const int32_t n_head  = dim / head_dim;
    if (n_local_heads < 1) n_local_heads = n_head;

    prepare_transformer_inputs(ctx, inp, seq_len, window_size, force_explicit_causal_mask);

    for (int32_t i = 0;; ++i) {
        const std::string stem = prefix + ".layers." + std::to_string(i);
        ggml_tensor * wqkv = ggml_get_tensor(ctx_w, (stem + ".attention.wqkv.weight").c_str());
        if (!wqkv) break;

        auto req = [&](const std::string & n) -> ggml_tensor * {
            ggml_tensor * t = ggml_get_tensor(ctx_w, n.c_str());
            if (!t) throw std::runtime_error("missing codec transformer tensor: " + n);
            return t;
        };

        ggml_tensor * wo       = req(stem + ".attention.wo.weight");
        ggml_tensor * w1       = req(stem + ".feed_forward.w1.weight");
        ggml_tensor * w2       = req(stem + ".feed_forward.w2.weight");
        ggml_tensor * w3       = req(stem + ".feed_forward.w3.weight");
        ggml_tensor * ffn_norm = req(stem + ".ffn_norm.weight");
        ggml_tensor * attn_norm = req(stem + ".attention_norm.weight");
        ggml_tensor * attn_gamma = req(stem + ".attention_layer_scale.gamma");
        ggml_tensor * ffn_gamma  = req(stem + ".ffn_layer_scale.gamma");

        const int32_t q_size  = n_head * head_dim;
        const int32_t kv_size = n_local_heads * head_dim;

        ggml_tensor * attn_in = rms_norm_weighted_cl(ctx, x, attn_norm, norm_eps);
        ggml_tensor * qkv = mul_mat_checked(ctx, wqkv, attn_in, "mul_mat:codec_wqkv");
        const size_t es = ggml_element_size(qkv);

        ggml_tensor * q2d = ggml_view_2d(ctx, qkv, q_size,  seq_len, qkv->nb[1], 0);
        ggml_tensor * k2d = ggml_view_2d(ctx, qkv, kv_size, seq_len, qkv->nb[1], q_size * es);
        ggml_tensor * v2d = ggml_view_2d(ctx, qkv, kv_size, seq_len, qkv->nb[1], (q_size + kv_size) * es);

        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_cont(ctx, q2d), head_dim, n_head,        seq_len);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_cont(ctx, k2d), head_dim, n_local_heads, seq_len);
        ggml_tensor * v = ggml_reshape_3d(ctx, ggml_cont(ctx, v2d), head_dim, n_local_heads, seq_len);

        q = ggml_rope_ext(ctx, q, inp.positions, nullptr, head_dim, 0,
                          block_size, rope_base, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        k = ggml_rope_ext(ctx, k, inp.positions, nullptr, head_dim, 0,
                          block_size, rope_base, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

        ggml_tensor * k_rep = repeat_interleave_heads(ctx, k, n_head / n_local_heads);
        ggml_tensor * v_rep = repeat_interleave_heads(ctx, v, n_head / n_local_heads);

        ggml_tensor * Q  = ggml_permute(ctx, q,     0, 2, 1, 3);
        ggml_tensor * K  = ggml_permute(ctx, k_rep, 0, 2, 1, 3);
        ggml_tensor * KQ = mul_mat_checked(ctx, K, Q, "mul_mat:codec_kq");
        ggml_tensor * KQs = ggml_scale(ctx, KQ, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        ggml_tensor * KQm;
        if (inp.mask) {
            KQm = ggml_add(ctx, KQs, repeat_checked(ctx, inp.mask, KQs, "repeat:attn_mask"));
        } else {
            KQm = ggml_diag_mask_inf(ctx, KQs, 0);
        }
        ggml_tensor * KQf = ggml_soft_max(ctx, KQm);

        ggml_tensor * V       = ggml_cont(ctx, ggml_permute(ctx, v_rep, 1, 2, 0, 3));
        ggml_tensor * KQV     = mul_mat_checked(ctx, V, KQf, "mul_mat:codec_kqv");
        ggml_tensor * KQVm    = ggml_permute(ctx, KQV, 0, 2, 1, 3);
        ggml_tensor * attn_cur = ggml_cpy(ctx, KQVm, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, q_size, seq_len));
        ggml_tensor * attn_out = mul_mat_checked(ctx, wo, attn_cur, "mul_mat:codec_wo");
        ggml_tensor * attn_sc  = scale_channels_cl(ctx, attn_out, attn_gamma);

        ggml_tensor * h    = ggml_add(ctx, x, attn_sc);
        ggml_tensor * ff_in = rms_norm_weighted_cl(ctx, h, ffn_norm, norm_eps);
        ggml_tensor * gate  = mul_mat_checked(ctx, w1, ff_in, "mul_mat:codec_w1");
        ggml_tensor * up    = mul_mat_checked(ctx, w3, ff_in, "mul_mat:codec_w3");
        ggml_tensor * ff_h  = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        ggml_tensor * ff_out = mul_mat_checked(ctx, w2, ff_h, "mul_mat:codec_w2");
        ggml_tensor * ff_sc  = scale_channels_cl(ctx, ff_out, ffn_gamma);

        x = ggml_add(ctx, h, ff_sc);
    }

    ggml_tensor * norm_w = ggml_get_tensor(ctx_w, (prefix + ".norm.weight").c_str());
    if (!norm_w) throw std::runtime_error("missing tensor: " + prefix + ".norm.weight");
    return rms_norm_weighted_cl(ctx, x, norm_w, norm_eps);
}

static ggml_tensor * build_residual_unit(ggml_context * ctx, ggml_context * ctx_w,
                                          const std::string & prefix, ggml_tensor * x, int dilation) {
    auto req = [&](const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(ctx_w, n.c_str());
        if (!t) throw std::runtime_error("missing tensor: " + n);
        return t;
    };
    ggml_tensor * y = snake_activation(ctx, x, req(prefix + ".block.0.alpha"));
    y = causal_conv_1d(ctx, req(prefix + ".block.1.conv.weight"), req(prefix + ".block.1.conv.bias"), y, 1, dilation);
    y = snake_activation(ctx, y, req(prefix + ".block.2.alpha"));
    y = causal_conv_1d(ctx, req(prefix + ".block.3.conv.weight"), req(prefix + ".block.3.conv.bias"), y, 1, 1);
    return ggml_add(ctx, x, y);
}

static ggml_tensor * build_convnext_block(ggml_context * ctx, ggml_context * ctx_w,
                                           const std::string & prefix, ggml_tensor * x) {
    auto req = [&](const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(ctx_w, n.c_str());
        if (!t) throw std::runtime_error("missing tensor: " + n);
        return t;
    };

    ggml_tensor * dw_w = req(prefix + ".dwconv.conv.weight");
    ggml_tensor * dw_b = req(prefix + ".dwconv.conv.bias");
    const int kernel_size_dw = static_cast<int>(dw_w->ne[0]);
    const int pad_dw = kernel_size_dw - 1;
    const int extra_dw = static_cast<int>(extra_padding_for_conv1d(x->ne[1], kernel_size_dw, 1, pad_dw));
    ggml_tensor * x_lc = cl_to_lc(ctx, x);
    x_lc = ggml_pad_ext(ctx, x_lc, pad_dw, extra_dw, 0, 0, 0, 0, 0, 0);
    ggml_tensor * y_lc = ggml_conv_1d_dw(ctx, dw_w, x_lc, 1, 0, 1);
    y_lc = add_channel_bias_lc(ctx, y_lc, dw_b);
    ggml_tensor * y = lc_to_cl(ctx, y_lc);

    y = layer_norm_affine(ctx, y, req(prefix + ".norm.weight"), req(prefix + ".norm.bias"), 1e-6f);
    y = linear_bias(ctx, req(prefix + ".pwconv1.weight"), req(prefix + ".pwconv1.bias"), y, "mul_mat:cn_pw1");
    y = ggml_gelu_erf(ctx, y);
    y = linear_bias(ctx, req(prefix + ".pwconv2.weight"), req(prefix + ".pwconv2.bias"), y, "mul_mat:cn_pw2");
    y = scale_channels_cl(ctx, y, req(prefix + ".gamma"));
    return ggml_add(ctx, x, y);
}

static ggml_tensor * build_encoder_block(ggml_context * ctx, AudioCodec::Impl & impl,
                                          const std::string & prefix, ggml_tensor * x,
                                          int stride, int32_t n_transformer_layers,
                                          transformer_inputs & inp) {
    auto req = [&](const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(impl.ctx_w, n.c_str());
        if (!t) throw std::runtime_error("missing tensor: " + n);
        return t;
    };
    x = build_residual_unit(ctx, impl.ctx_w, prefix + ".0", x, 1);
    x = build_residual_unit(ctx, impl.ctx_w, prefix + ".1", x, 3);
    x = build_residual_unit(ctx, impl.ctx_w, prefix + ".2", x, 9);
    x = snake_activation(ctx, x, req(prefix + ".3.alpha"));
    x = causal_conv_1d(ctx, req(prefix + ".4.conv.weight"), req(prefix + ".4.conv.bias"), x, stride, 1);

    if (n_transformer_layers > 0) {
        x = build_transformer(ctx, impl.ctx_w, prefix + ".5", x,
                               impl.transformer_block_size,
                               impl.transformer_n_local_heads,
                               impl.transformer_head_dim,
                               impl.transformer_rope_base,
                               impl.transformer_norm_eps,
                               512,
                               backend_requires_explicit_causal_mask(impl.backend),
                               inp);
    }
    return x;
}

static ggml_tensor * build_quantizer_stage_up(ggml_context * ctx, AudioCodec::Impl & impl,
                                               const std::string & prefix, ggml_tensor * x, int factor) {
    auto req = [&](const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(impl.ctx_w, n.c_str());
        if (!t) throw std::runtime_error("missing tensor: " + n);
        return t;
    };
    x = causal_conv_transpose_1d(ctx, req(prefix + ".0.conv.weight"), req(prefix + ".0.conv.bias"), x, factor, 0);
    x = build_convnext_block(ctx, impl.ctx_w, prefix + ".1", x);
    return x;
}

static ggml_tensor * build_quantizer_decode_stage(ggml_context * ctx, AudioCodec::Impl & impl,
                                                   ggml_tensor * z, transformer_inputs & inp) {
    ggml_tensor * x = build_transformer(ctx, impl.ctx_w, impl.tprefix + "quantizer.post_module", z,
                                         impl.rvq_transformer_block_size,
                                         impl.rvq_transformer_n_local_heads,
                                         impl.rvq_transformer_head_dim,
                                         impl.rvq_transformer_rope_base,
                                         impl.rvq_transformer_norm_eps,
                                         impl.rvq_transformer_window_size,
                                         backend_requires_explicit_causal_mask(impl.backend),
                                         inp);
    const size_t n = impl.quantizer_downsample_factor.size();
    for (size_t i = 0; i < n; ++i) {
        int factor = impl.quantizer_downsample_factor[n - 1 - i];
        x = build_quantizer_stage_up(ctx, impl, impl.tprefix + "quantizer.upsample." + std::to_string(i), x, factor);
    }
    return x;
}

static ggml_tensor * build_decoder_block(ggml_context * ctx, AudioCodec::Impl & impl,
                                          const std::string & prefix, ggml_tensor * x, int stride) {
    auto req = [&](const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(impl.ctx_w, n.c_str());
        if (!t) throw std::runtime_error("missing tensor: " + n);
        return t;
    };
    x = snake_activation(ctx, x, req(prefix + ".block.0.alpha"));
    x = causal_conv_transpose_1d(ctx, req(prefix + ".block.1.conv.weight"),
                                       req(prefix + ".block.1.conv.bias"),
                                       x, stride, stride);
    x = build_residual_unit(ctx, impl.ctx_w, prefix + ".block.2", x, 1);
    x = build_residual_unit(ctx, impl.ctx_w, prefix + ".block.3", x, 3);
    x = build_residual_unit(ctx, impl.ctx_w, prefix + ".block.4", x, 9);
    return x;
}

static ggml_tensor * build_decoder(ggml_context * ctx, AudioCodec::Impl & impl, ggml_tensor * z) {
    auto req = [&](const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(impl.ctx_w, n.c_str());
        if (!t) throw std::runtime_error("missing tensor: " + n);
        return t;
    };

    ggml_tensor * x = causal_conv_1d(ctx,
        req(impl.tprefix + "decoder.model.0.conv.weight"),
        req(impl.tprefix + "decoder.model.0.conv.bias"),
        z, 1, 1);

    for (size_t i = 0; i < impl.decoder_rates.size(); ++i) {
        x = build_decoder_block(ctx, impl,
                                 impl.tprefix + "decoder.model." + std::to_string(i + 1),
                                 x, impl.decoder_rates[i]);
    }

    const int last_idx = static_cast<int>(impl.decoder_rates.size()) + 1;
    x = snake_activation(ctx, x, req(impl.tprefix + "decoder.model." + std::to_string(last_idx) + ".alpha"));
    x = causal_conv_1d(ctx,
        req(impl.tprefix + "decoder.model." + std::to_string(last_idx + 1) + ".conv.weight"),
        req(impl.tprefix + "decoder.model." + std::to_string(last_idx + 1) + ".conv.bias"),
        x, 1, 1);
    return ggml_tanh(ctx, x);
}

static std::vector<float> tensor_to_f32(ggml_tensor * t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        throw std::runtime_error("unsupported tensor type for host copy: " +
                                 std::string(ggml_type_name(t->type)));
    }
    return out;
}

static vq_cache load_vq_cache(ggml_context * ctx_w, const std::string & prefix,
                               int32_t in_dim, int32_t cb_dim, int32_t cb_size) {
    auto req = [&](const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(ctx_w, n.c_str());
        if (!t) throw std::runtime_error("missing vq tensor: " + n);
        return t;
    };
    vq_cache vq;
    vq.input_dim    = in_dim;
    vq.codebook_dim = cb_dim;
    vq.codebook_size = cb_size;
    vq.in_proj_weight  = tensor_to_f32(req(prefix + ".in_proj.weight"));
    vq.in_proj_bias    = tensor_to_f32(req(prefix + ".in_proj.bias"));
    vq.out_proj_weight = tensor_to_f32(req(prefix + ".out_proj.weight"));
    vq.out_proj_bias   = tensor_to_f32(req(prefix + ".out_proj.bias"));
    vq.codebook        = tensor_to_f32(req(prefix + ".codebook.weight"));

    vq.codebook_norm.resize(vq.codebook.size());
    for (int32_t code = 0; code < vq.codebook_size; ++code) {
        float norm = 0.0f;
        const size_t base = static_cast<size_t>(code) * vq.codebook_dim;
        for (int32_t d = 0; d < vq.codebook_dim; ++d) norm += vq.codebook[base+d] * vq.codebook[base+d];
        norm = std::sqrt(std::max(norm, 1e-12f));
        for (int32_t d = 0; d < vq.codebook_dim; ++d) vq.codebook_norm[base+d] = vq.codebook[base+d] / norm;
    }
    return vq;
}

static void project_1x1(const std::vector<float> & input, int32_t frames,
                         int32_t in_dim, int32_t out_dim,
                         const std::vector<float> & weight, const std::vector<float> & bias,
                         std::vector<float> & output) {
    output.assign(static_cast<size_t>(frames) * out_dim, 0.0f);
    for (int32_t t = 0; t < frames; ++t) {
        const float * src = input.data() + static_cast<size_t>(t) * in_dim;
        float * dst = output.data() + static_cast<size_t>(t) * out_dim;
        for (int32_t o = 0; o < out_dim; ++o) {
            float v = bias[o];
            const float * w = weight.data() + static_cast<size_t>(o) * in_dim;
            for (int32_t i = 0; i < in_dim; ++i) v += w[i] * src[i];
            dst[o] = v;
        }
    }
}

static void quantize_with_vq(const vq_cache & vq, const std::vector<float> & input, int32_t frames,
                               std::vector<int32_t> & codes, std::vector<float> & projected_out) {
    std::vector<float> latents;
    project_1x1(input, frames, vq.input_dim, vq.codebook_dim, vq.in_proj_weight, vq.in_proj_bias, latents);

    codes.resize(frames);
    projected_out.assign(static_cast<size_t>(frames) * vq.input_dim, 0.0f);

    for (int32_t t = 0; t < frames; ++t) {
        const float * lat = latents.data() + static_cast<size_t>(t) * vq.codebook_dim;
        float lat_norm = 0.0f;
        for (int32_t d = 0; d < vq.codebook_dim; ++d) lat_norm += lat[d] * lat[d];
        lat_norm = std::sqrt(std::max(lat_norm, 1e-12f));

        int32_t best = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        for (int32_t code = 0; code < vq.codebook_size; ++code) {
            const float * cb = vq.codebook_norm.data() + static_cast<size_t>(code) * vq.codebook_dim;
            float score = 0.0f;
            for (int32_t d = 0; d < vq.codebook_dim; ++d) score += (lat[d] / lat_norm) * cb[d];
            if (score > best_score) { best_score = score; best = code; }
        }
        codes[t] = best;

        const float * cb = vq.codebook.data() + static_cast<size_t>(best) * vq.codebook_dim;
        float * dst = projected_out.data() + static_cast<size_t>(t) * vq.input_dim;
        for (int32_t o = 0; o < vq.input_dim; ++o) {
            float v = vq.out_proj_bias[o];
            const float * w = vq.out_proj_weight.data() + static_cast<size_t>(o) * vq.codebook_dim;
            for (int32_t d = 0; d < vq.codebook_dim; ++d) v += w[d] * cb[d];
            dst[o] = v;
        }
    }
}

static ggml_tensor * build_vq_decode_stage(ggml_context * ctx, ggml_context * ctx_w,
                                            const std::string & prefix, ggml_tensor * code_ids) {
    auto req = [&](const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(ctx_w, n.c_str());
        if (!t) throw std::runtime_error("missing vq decode tensor: " + n);
        return t;
    };

    ggml_tensor * codebook = req(prefix + ".codebook.weight");
    ggml_tensor * out_proj_weight = req(prefix + ".out_proj.weight");
    ggml_tensor * out_proj_bias = req(prefix + ".out_proj.bias");

    if (out_proj_weight->ne[0] == 1 && out_proj_weight->ne[1] > 0 &&
        out_proj_weight->ne[2] > 0 && out_proj_weight->ne[3] == 1) {
        out_proj_weight = ggml_reshape_2d(ctx, out_proj_weight,
                                          out_proj_weight->ne[1], out_proj_weight->ne[2]);
    }

    ggml_tensor * gathered = ggml_get_rows(ctx, codebook, code_ids);
    if (gathered->type != GGML_TYPE_F32) {
        gathered = ggml_cast(ctx, gathered, GGML_TYPE_F32);
    }

    return linear_bias(ctx, out_proj_weight, out_proj_bias, gathered, "mul_mat:vq_out_proj");
}

static ggml_tensor * build_decode_codes_stage_backend(ggml_context * ctx, AudioCodec::Impl & impl,
                                                       const std::vector<ggml_tensor *> & code_id_tensors) {
    if (code_id_tensors.empty()) {
        throw std::runtime_error("missing code tensors for decode stage");
    }

    ggml_tensor * stage = build_vq_decode_stage(
        ctx, impl.ctx_w, impl.tprefix + "quantizer.semantic_quantizer.quantizers.0", code_id_tensors[0]);

    for (int32_t i = 0; i < impl.quantizer_residual_codebooks; ++i) {
        const size_t tensor_index = static_cast<size_t>(i + 1);
        if (tensor_index >= code_id_tensors.size()) {
            throw std::runtime_error("insufficient residual code tensors for decode stage");
        }

        ggml_tensor * residual = build_vq_decode_stage(
            ctx, impl.ctx_w,
            impl.tprefix + "quantizer.quantizer.quantizers." + std::to_string(i),
            code_id_tensors[tensor_index]);
        stage = ggml_add(ctx, stage, residual);
    }

    return stage;
}

AudioCodec::AudioCodec()  { impl_ = new Impl(); }
AudioCodec::~AudioCodec() {
    if (impl_) {
        reset_codec_impl(*impl_);
        delete impl_;
        impl_ = nullptr;
    }
}

const char * AudioCodec::backend_name() const {
    if (!impl_ || !impl_->backend) {
        return "unknown";
    }
    return ggml_backend_name(impl_->backend);
}

void AudioCodec::clear_decode_cache() {
    if (impl_) {
        reset_decode_cache(impl_->decode_cache, false);
    }
}

bool AudioCodec::load_shared(SlowARModel* Model, gguf_context * shared_gguf_ctx, const std::string & gguf_path, int32_t gpu_device, BackendType backend_type) {
    if (!impl_) {
        impl_ = new Impl();
    } else {
        reset_codec_impl(*impl_);
    }
    sample_rate_ = 44100;
    hop_length_ = 512;
    num_codebooks_ = 10;
    samples_per_code_frame_ = 2048;
    streaming_history_frames_ = 160;
    bool warned_backend_fallback = false;

    const bool wants_gpu_backend =
        backend_type != BackendType::CPU &&
        (gpu_device >= 0 || backend_type == BackendType::Metal);

    if (wants_gpu_backend) {
#ifdef GGML_USE_VULKAN
        if (!impl_->backend && backend_type == BackendType::Vulkan) {
            impl_->backend = ggml_backend_vk_init(static_cast<size_t>(gpu_device));
            if (!impl_->backend) {
                S2_LOG_WARN_STREAM("[Codec] Vulkan init failed, falling back to CPU." << std::endl);
                warned_backend_fallback = true;
            }
        }
#endif
#ifdef GGML_USE_CUDA
        if (!impl_->backend && backend_type == BackendType::CUDA) {
            impl_->backend = ggml_backend_cuda_init(static_cast<size_t>(gpu_device));
            if (!impl_->backend) {
                S2_LOG_WARN_STREAM("[Codec] Cuda init failed, falling back to CPU." << std::endl);
                warned_backend_fallback = true;
            }
        }
#endif
#ifdef GGML_USE_METAL
        if (!impl_->backend && backend_type == BackendType::Metal) {
            impl_->backend = ggml_backend_metal_init();
            if (!impl_->backend) {
                S2_LOG_WARN_STREAM("[Codec] Metal init failed, falling back to CPU." << std::endl);
                warned_backend_fallback = true;
            }
        }
#endif
    }
    if (!impl_->backend && wants_gpu_backend && !warned_backend_fallback) {
        S2_LOG_WARN_STREAM("[Codec] Requested " << backend_type_name(backend_type)
                  << " backend unavailable, falling back to CPU." << std::endl);
    }
    if (!impl_->backend) impl_->backend = ggml_backend_cpu_init();
    if (!impl_->backend) {
        std::cerr << "[Codec] No backend." << std::endl;
        reset_codec_impl(*impl_);
        return false;
    }

    if(!Model)
    {
        struct gguf_init_params params = { true, &impl_->ctx_w };
        gguf_context * local_gguf = gguf_init_from_file(gguf_path.c_str(), params);
        if (!local_gguf) {
            std::cerr << "[Codec] Failed to open " << gguf_path << std::endl;
            reset_codec_impl(*impl_);
            return false;
        }

        gguf_free(local_gguf);
    }
    else { impl_->ctx_w = Model->weights_.ctx_w; }

    gguf_context * gguf_ctx = shared_gguf_ctx;

    try {
        auto req_str = [&](const char * k) -> std::string {
            int id = gguf_find_key(gguf_ctx, k);
            if (id < 0) throw std::runtime_error(std::string("missing key: ") + k);
            return gguf_get_val_str(gguf_ctx, id);
        };
        auto req_u32 = [&](const char * k) -> uint32_t {
            int id = gguf_find_key(gguf_ctx, k);
            if (id < 0) throw std::runtime_error(std::string("missing key: ") + k);
            return gguf_get_val_u32(gguf_ctx, id);
        };
        auto opt_u32 = [&](const char * k, uint32_t def) -> uint32_t {
            int id = gguf_find_key(gguf_ctx, k);
            return (id < 0) ? def : gguf_get_val_u32(gguf_ctx, id);
        };
        auto req_f32 = [&](const char * k) -> float {
            int id = gguf_find_key(gguf_ctx, k);
            if (id < 0) throw std::runtime_error(std::string("missing key: ") + k);
            return gguf_get_val_f32(gguf_ctx, id);
        };
        auto req_i32_or_u32 = [&](const char * k) -> int32_t {
            int id = gguf_find_key(gguf_ctx, k);
            if (id < 0) throw std::runtime_error(std::string("missing key: ") + k);
            auto type = gguf_get_kv_type(gguf_ctx, id);
            if (type == GGUF_TYPE_INT32)  return gguf_get_val_i32(gguf_ctx, id);
            if (type == GGUF_TYPE_UINT32) return static_cast<int32_t>(gguf_get_val_u32(gguf_ctx, id));
            throw std::runtime_error(std::string("expected INT32/UINT32 for key: ") + k);
        };
        auto req_u32_arr = [&](const char * k) -> std::vector<int32_t> {
            int id = gguf_find_key(gguf_ctx, k);
            if (id < 0) throw std::runtime_error(std::string("missing key: ") + k);
            auto type = gguf_get_arr_type(gguf_ctx, id);
            size_t n = gguf_get_arr_n(gguf_ctx, id);
            std::vector<int32_t> v(n);
            if (type == GGUF_TYPE_UINT32) {
                const auto * d = static_cast<const uint32_t *>(gguf_get_arr_data(gguf_ctx, id));
                for (size_t i = 0; i < n; ++i) v[i] = static_cast<int32_t>(d[i]);
            } else if (type == GGUF_TYPE_INT32) {
                const auto * d = static_cast<const int32_t *>(gguf_get_arr_data(gguf_ctx, id));
                for (size_t i = 0; i < n; ++i) v[i] = d[i];
            } else if (type == GGUF_TYPE_UINT64) {
                const auto * d = static_cast<const uint64_t *>(gguf_get_arr_data(gguf_ctx, id));
                for (size_t i = 0; i < n; ++i) v[i] = static_cast<int32_t>(d[i]);
            } else {
                throw std::runtime_error(std::string("unexpected array type for key: ") + k);
            }
            return v;
        };

        const std::string arch = req_str("general.architecture");
        if (arch == "fish-speech") {
            impl_->tprefix = "c.";
        } else if (arch == "fish-speech-codec") {
            impl_->tprefix = "";
        } else {
            throw std::runtime_error("unexpected architecture: " + arch);
        }

        impl_->sample_rate    = static_cast<int32_t>(req_u32("fish_speech.codec.sample_rate"));
        impl_->hop_length     = static_cast<int32_t>(req_u32("fish_speech.codec.hop_length"));
        impl_->frame_length   = static_cast<int32_t>(opt_u32("fish_speech.codec.frame_length", 512));
        impl_->encoder_dim    = static_cast<int32_t>(req_u32("fish_speech.codec.encoder_dim"));
        impl_->decoder_dim    = static_cast<int32_t>(req_u32("fish_speech.codec.decoder_dim"));
        impl_->latent_dim     = static_cast<int32_t>(req_u32("fish_speech.codec.latent_dim"));
        impl_->encoder_rates  = req_u32_arr("fish_speech.codec.encoder_rates");
        impl_->decoder_rates  = req_u32_arr("fish_speech.codec.decoder_rates");
        impl_->encoder_transformer_layers = req_u32_arr("fish_speech.codec.encoder_transformer_layers");

        impl_->quantizer_input_dim              = static_cast<int32_t>(req_u32("fish_speech.codec.quantizer_input_dim"));
        impl_->quantizer_codebook_dim           = static_cast<int32_t>(req_u32("fish_speech.codec.quantizer_codebook_dim"));
        impl_->quantizer_residual_codebooks     = static_cast<int32_t>(req_u32("fish_speech.codec.quantizer_residual_codebooks"));
        impl_->quantizer_residual_codebook_size = static_cast<int32_t>(req_u32("fish_speech.codec.quantizer_residual_codebook_size"));
        impl_->quantizer_semantic_codebook_size = static_cast<int32_t>(req_u32("fish_speech.codec.quantizer_semantic_codebook_size"));
        impl_->quantizer_downsample_factor      = req_u32_arr("fish_speech.codec.quantizer_downsample_factor");

        impl_->transformer_block_size    = static_cast<int32_t>(req_u32("fish_speech.codec.transformer.block_size"));
        impl_->transformer_n_local_heads = req_i32_or_u32("fish_speech.codec.transformer.n_local_heads");
        impl_->transformer_head_dim      = static_cast<int32_t>(req_u32("fish_speech.codec.transformer.head_dim"));
        impl_->transformer_rope_base     = req_f32("fish_speech.codec.transformer.rope_freq_base");
        impl_->transformer_norm_eps      = req_f32("fish_speech.codec.transformer.layer_norm_rms_eps");

        impl_->rvq_transformer_window_size   = static_cast<int32_t>(req_u32("fish_speech.codec.rvq_transformer.window_size"));
        impl_->rvq_transformer_block_size    = static_cast<int32_t>(req_u32("fish_speech.codec.rvq_transformer.block_size"));
        impl_->rvq_transformer_n_layer       = static_cast<int32_t>(req_u32("fish_speech.codec.rvq_transformer.n_layer"));
        impl_->rvq_transformer_n_local_heads = req_i32_or_u32("fish_speech.codec.rvq_transformer.n_local_heads");
        impl_->rvq_transformer_head_dim      = static_cast<int32_t>(req_u32("fish_speech.codec.rvq_transformer.head_dim"));
        impl_->rvq_transformer_dim           = static_cast<int32_t>(req_u32("fish_speech.codec.rvq_transformer.dim"));
        impl_->rvq_transformer_rope_base     = req_f32("fish_speech.codec.rvq_transformer.rope_freq_base");
        impl_->rvq_transformer_norm_eps      = req_f32("fish_speech.codec.rvq_transformer.layer_norm_rms_eps");

        sample_rate_    = impl_->sample_rate;
        hop_length_     = impl_->hop_length;
        num_codebooks_  = impl_->quantizer_residual_codebooks + 1;
        int32_t code_frame_factor = 1;
        for (int32_t factor : impl_->quantizer_downsample_factor) {
            if (factor > 0 && code_frame_factor <= (std::numeric_limits<int32_t>::max() / factor)) {
                code_frame_factor *= factor;
            }
        }
        if (code_frame_factor <= 0) {
            code_frame_factor = 1;
        }
        samples_per_code_frame_ = hop_length_ * code_frame_factor;

        const int32_t rvq_history = std::max(0, impl_->rvq_transformer_window_size - 1);
        streaming_history_frames_ = ((rvq_history + 16 + 7) / 8) * 8;
        if (streaming_history_frames_ <= 0) {
            streaming_history_frames_ = 160;
        }

        impl_->model_buf = ggml_backend_alloc_ctx_tensors(impl_->ctx_w, impl_->backend);
        if (!impl_->model_buf) throw std::runtime_error("ggml_backend_alloc_ctx_tensors() failed");

        impl_->semantic_vq = vq_cache();
        impl_->residual_vq.clear();
    } catch (const std::exception & e) {
        std::cerr << "[Codec] " << e.what() << std::endl;
        reset_codec_impl(*impl_);
        return false;
    }
    S2_LOG_INFO_STREAM("[Codec] Backend: " << backend_name() << std::endl);
    return true;
}

bool AudioCodec::refresh_host_caches() {
    if (!impl_ || !impl_->ctx_w) {
        return false;
    }

    try {
        impl_->semantic_vq = load_vq_cache(impl_->ctx_w,
            impl_->tprefix + "quantizer.semantic_quantizer.quantizers.0",
            impl_->quantizer_input_dim, impl_->quantizer_codebook_dim,
            impl_->quantizer_semantic_codebook_size);

        impl_->residual_vq.clear();
        impl_->residual_vq.reserve(impl_->quantizer_residual_codebooks);
        for (int32_t i = 0; i < impl_->quantizer_residual_codebooks; ++i) {
            impl_->residual_vq.push_back(load_vq_cache(impl_->ctx_w,
                impl_->tprefix + "quantizer.quantizer.quantizers." + std::to_string(i),
                impl_->quantizer_input_dim, impl_->quantizer_codebook_dim,
                impl_->quantizer_residual_codebook_size));
        }
    } catch (const std::exception & e) {
        std::cerr << "[Codec] Failed to refresh VQ caches: " << e.what() << std::endl;
        impl_->semantic_vq = vq_cache();
        impl_->residual_vq.clear();
        return false;
    }

    return true;
}

bool AudioCodec::read_tensor_data(const std::string & gguf_path, gguf_context * gguf_ctx) {
    if (!impl_ || !impl_->ctx_w) return false;

    const size_t data_offset = gguf_get_data_offset(gguf_ctx);
    const int64_t n_tensors  = gguf_get_n_tensors(gguf_ctx);

    std::FILE * f = std::fopen(gguf_path.c_str(), "rb");
    if (!f) {
        std::cerr << "[Codec] Failed to reopen " << gguf_path << " for data loading." << std::endl;
        return false;
    }
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(gguf_ctx, ti);
        ggml_tensor * t = ggml_get_tensor(impl_->ctx_w, name);
        if (!t) continue;
        const size_t off   = data_offset + gguf_get_tensor_offset(gguf_ctx, ti);
        const size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> tmp(nbytes);
#ifdef _WIN32
        _fseeki64(f, (int64_t)off, SEEK_SET);
#else
        fseeko(f, (off_t)off, SEEK_SET);
#endif
        if (std::fread(tmp.data(), 1, nbytes, f) != nbytes) {
            std::fclose(f);
            std::cerr << "[Codec] Failed to read tensor: " << name << std::endl;
            return false;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
    }
    std::fclose(f);
    return refresh_host_caches();
}

bool AudioCodec::load(const std::string & gguf_path, int32_t gpu_device, BackendType backend_type) {

    struct gguf_init_params params = { true, nullptr };
    gguf_context * ctx_gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf) {
        std::cerr << "[Codec] Failed to open " << gguf_path << std::endl;
        return false;
    }

    if (!load_shared(nullptr, ctx_gguf, gguf_path, gpu_device, backend_type)) {
        gguf_free(ctx_gguf);
        return false;
    }

    if (!read_tensor_data(gguf_path, ctx_gguf)) {
        gguf_free(ctx_gguf);
        return false;
    }

    gguf_free(ctx_gguf);
    return true;
}

ggml_context * AudioCodec::weights_ctx() const {
    return impl_ ? impl_->ctx_w : nullptr;
}

bool AudioCodec::encode(const float * audio, int32_t n_samples, int32_t n_threads,
                         std::vector<int32_t> & codes_out, int32_t & n_frames_out) {

    const int32_t frame_length = (impl_->frame_length > 0) ? impl_->frame_length : 512;
    const int32_t padded = ((n_samples + frame_length - 1) / frame_length) * frame_length;
    std::vector<float> audio_padded(padded, 0.0f);
    std::copy(audio, audio + n_samples, audio_padded.begin());

    const size_t ctx_size = 128u * 1024u * 1024u;
    std::vector<uint8_t> ctx_buf(ctx_size);
    ggml_init_params p = { ctx_size, ctx_buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    if (!ctx) return false;

    transformer_inputs enc_inp;
    ggml_tensor * audio_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, padded);
    ggml_tensor * latent    = nullptr;
    try {

        ggml_tensor * x = causal_conv_1d(ctx,
            ggml_get_tensor(impl_->ctx_w, (impl_->tprefix + "encoder.block.0.conv.weight").c_str()),
            ggml_get_tensor(impl_->ctx_w, (impl_->tprefix + "encoder.block.0.conv.bias").c_str()),
            audio_in, 1, 1);

        for (size_t i = 0; i < impl_->encoder_rates.size(); ++i) {
            const std::string prefix = impl_->tprefix + "encoder.block." + std::to_string(i + 1) + ".block";
            const int32_t n_layers = (i < impl_->encoder_transformer_layers.size())
                                     ? impl_->encoder_transformer_layers[i] : 0;
            x = build_encoder_block(ctx, *impl_, prefix, x, impl_->encoder_rates[i], n_layers, enc_inp);
        }

        const int last = static_cast<int>(impl_->encoder_rates.size()) + 1;
        auto req = [&](const std::string & n) -> ggml_tensor * {
            ggml_tensor * t = ggml_get_tensor(impl_->ctx_w, n.c_str());
            if (!t) throw std::runtime_error("missing tensor: " + n);
            return t;
        };
        x = snake_activation(ctx, x, req(impl_->tprefix + "encoder.block." + std::to_string(last) + ".alpha"));
        x = causal_conv_1d(ctx,
            req(impl_->tprefix + "encoder.block." + std::to_string(last + 1) + ".conv.weight"),
            req(impl_->tprefix + "encoder.block." + std::to_string(last + 1) + ".conv.bias"),
            x, 1, 1);
        latent = ggml_cpy(ctx, x, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1]));
    } catch (const std::exception & e) {
        std::cerr << "[Codec::encode] encoder build failed: " << e.what() << std::endl;
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 131072, false);
    ggml_build_forward_expand(gf, latent);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
    if (!allocr || !ggml_gallocr_alloc_graph(allocr, gf)) {
        if (allocr) ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(audio_in, audio_padded.data(), 0, audio_padded.size() * sizeof(float));
    if (enc_inp.positions) {
        ggml_backend_tensor_set(enc_inp.positions, enc_inp.position_values.data(), 0,
                                enc_inp.position_values.size() * sizeof(int32_t));
    }
    if (enc_inp.mask) {
        ggml_backend_tensor_set(enc_inp.mask, enc_inp.mask_values.data(), 0,
                                enc_inp.mask_values.size() * sizeof(float));
    }

    if (ggml_backend_is_cpu(impl_->backend)) ggml_backend_cpu_set_n_threads(impl_->backend, resolve_n_threads(n_threads));
    if (ggml_backend_graph_compute(impl_->backend, gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "[Codec::encode] encoder compute failed." << std::endl;
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return false;
    }

    const int32_t latent_frames = static_cast<int32_t>(latent->ne[1]);
    std::vector<float> latent_out(static_cast<size_t>(latent->ne[0]) * latent_frames);
    ggml_backend_tensor_get(latent, latent_out.data(), 0, latent_out.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);

    {
        const size_t ctx2_size = 96u * 1024u * 1024u;
        std::vector<uint8_t> ctx2_buf(ctx2_size);
        ggml_init_params p2 = { ctx2_size, ctx2_buf.data(), true };
        ggml_context * ctx2 = ggml_init(p2);
        if (!ctx2) return false;

        transformer_inputs qenc_inp;
        ggml_tensor * latent_in = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, impl_->latent_dim, latent_frames);
        ggml_tensor * stage = nullptr;
        try {
            ggml_tensor * x2 = latent_in;
            auto req2 = [&](const std::string & n) -> ggml_tensor * {
                ggml_tensor * t = ggml_get_tensor(impl_->ctx_w, n.c_str());
                if (!t) throw std::runtime_error("missing tensor: " + n);
                return t;
            };
            for (size_t i = 0; i < impl_->quantizer_downsample_factor.size(); ++i) {
                const std::string pfx = impl_->tprefix + "quantizer.downsample." + std::to_string(i);
                x2 = causal_conv_1d(ctx2, req2(pfx + ".0.conv.weight"), req2(pfx + ".0.conv.bias"),
                                    x2, impl_->quantizer_downsample_factor[i], 1);
                x2 = build_convnext_block(ctx2, impl_->ctx_w, pfx + ".1", x2);
            }
            x2 = build_transformer(ctx2, impl_->ctx_w, impl_->tprefix + "quantizer.pre_module", x2,
                                    impl_->rvq_transformer_block_size,
                                    impl_->rvq_transformer_n_local_heads,
                                    impl_->rvq_transformer_head_dim,
                                    impl_->rvq_transformer_rope_base,
                                    impl_->rvq_transformer_norm_eps,
                                    impl_->rvq_transformer_window_size,
                                    backend_requires_explicit_causal_mask(impl_->backend),
                                    qenc_inp);
            stage = ggml_cpy(ctx2, x2, ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, x2->ne[0], x2->ne[1]));
        } catch (const std::exception & e) {
            std::cerr << "[Codec::encode] quantizer encode stage failed: " << e.what() << std::endl;
            ggml_free(ctx2);
            return false;
        }

        ggml_cgraph * gf2 = ggml_new_graph_custom(ctx2, 131072, false);
        ggml_build_forward_expand(gf2, stage);

        ggml_gallocr_t allocr2 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
        if (!allocr2 || !ggml_gallocr_alloc_graph(allocr2, gf2)) {
            if (allocr2) ggml_gallocr_free(allocr2);
            ggml_free(ctx2);
            return false;
        }

        ggml_backend_tensor_set(latent_in, latent_out.data(), 0, latent_out.size() * sizeof(float));
        if (qenc_inp.positions) {
            ggml_backend_tensor_set(qenc_inp.positions, qenc_inp.position_values.data(), 0,
                                    qenc_inp.position_values.size() * sizeof(int32_t));
        }
        if (qenc_inp.mask) {
            ggml_backend_tensor_set(qenc_inp.mask, qenc_inp.mask_values.data(), 0,
                                    qenc_inp.mask_values.size() * sizeof(float));
        }

        if (ggml_backend_is_cpu(impl_->backend)) ggml_backend_cpu_set_n_threads(impl_->backend, resolve_n_threads(n_threads));
        if (ggml_backend_graph_compute(impl_->backend, gf2) != GGML_STATUS_SUCCESS) {
            std::cerr << "[Codec::encode] quantizer stage compute failed." << std::endl;
            ggml_gallocr_free(allocr2);
            ggml_free(ctx2);
            return false;
        }

        const int32_t stage_frames = static_cast<int32_t>(stage->ne[1]);
        std::vector<float> stage_out(static_cast<size_t>(stage->ne[0]) * stage_frames);
        ggml_backend_tensor_get(stage, stage_out.data(), 0, stage_out.size() * sizeof(float));
        ggml_gallocr_free(allocr2);
        ggml_free(ctx2);

        std::vector<float> residual = stage_out;
        std::vector<int32_t> sem_codes;
        std::vector<float> sem_proj;
        quantize_with_vq(impl_->semantic_vq, residual, stage_frames, sem_codes, sem_proj);
        for (size_t i = 0; i < residual.size(); ++i) residual[i] -= sem_proj[i];

        const int32_t num_cb = impl_->quantizer_residual_codebooks + 1;
        n_frames_out = stage_frames;
        codes_out.resize(static_cast<size_t>(num_cb) * stage_frames);

        for (int32_t t = 0; t < stage_frames; ++t) codes_out[0 * stage_frames + t] = sem_codes[t];

        for (int32_t i = 0; i < impl_->quantizer_residual_codebooks; ++i) {
            std::vector<int32_t> rcodes;
            std::vector<float> rproj;
            quantize_with_vq(impl_->residual_vq[i], residual, stage_frames, rcodes, rproj);
            for (size_t j = 0; j < residual.size(); ++j) residual[j] -= rproj[j];
            for (int32_t t = 0; t < stage_frames; ++t) codes_out[(i + 1) * stage_frames + t] = rcodes[t];
        }
    }

    return true;
}

static int32_t clamp_decode_code(int32_t code, int32_t codebook_size) {
    if (code < 0) {
        return 0;
    }
    if (code >= codebook_size) {
        return codebook_size - 1;
    }
    return code;
}

static void sanitize_decode_codes(AudioCodec::Impl & impl, const int32_t * codes, int32_t n_frames,
                                  std::vector<int32_t> & sanitized_codes) {
    const int32_t num_codebooks = impl.quantizer_residual_codebooks + 1;
    sanitized_codes.resize(static_cast<size_t>(num_codebooks) * n_frames);

    for (int32_t t = 0; t < n_frames; ++t) {
        sanitized_codes[static_cast<size_t>(t)] =
            clamp_decode_code(codes[t], impl.quantizer_semantic_codebook_size);
    }
    for (int32_t cb = 0; cb < impl.quantizer_residual_codebooks; ++cb) {
        const size_t src_offset = static_cast<size_t>(cb + 1) * n_frames;
        const size_t dst_offset = static_cast<size_t>(cb + 1) * n_frames;
        for (int32_t t = 0; t < n_frames; ++t) {
            sanitized_codes[dst_offset + static_cast<size_t>(t)] =
                clamp_decode_code(codes[src_offset + static_cast<size_t>(t)],
                                  impl.quantizer_residual_codebook_size);
        }
    }
}

static bool build_cached_decode_graph(AudioCodec::Impl & impl, int32_t n_frames) {
    codec_decode_cache & cache = impl.decode_cache;
    reset_decode_cache(cache);

    cache.ctx_size = 256u * 1024u * 1024u;
    try {
        cache.ctx_buf.resize(cache.ctx_size);
    } catch (const std::bad_alloc &) {
        cache.failed_n_frames = n_frames;
        reset_decode_cache(cache);
        return false;
    }
    ggml_init_params p = { cache.ctx_size, cache.ctx_buf.data(), true };
    cache.ctx = ggml_init(p);
    if (!cache.ctx) {
        cache.failed_n_frames = n_frames;
        reset_decode_cache(cache);
        return false;
    }

    const int32_t num_codebooks = impl.quantizer_residual_codebooks + 1;
    cache.code_id_tensors.resize(static_cast<size_t>(num_codebooks));
    for (int32_t cb = 0; cb < num_codebooks; ++cb) {
        cache.code_id_tensors[static_cast<size_t>(cb)] =
            ggml_new_tensor_1d(cache.ctx, GGML_TYPE_I32, n_frames);
    }

    try {
        ggml_tensor * stage = build_decode_codes_stage_backend(cache.ctx, impl, cache.code_id_tensors);
        ggml_tensor * latent = build_quantizer_decode_stage(cache.ctx, impl, stage, cache.inp);
        ggml_tensor * audio = build_decoder(cache.ctx, impl, latent);
        cache.audio = ggml_cpy(cache.ctx, audio,
                               ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32,
                                                  audio->ne[0], audio->ne[1]));
    } catch (const std::exception &) {
        cache.failed_n_frames = n_frames;
        reset_decode_cache(cache);
        return false;
    }

    cache.graph = ggml_new_graph_custom(cache.ctx, 262144, false);
    if (!cache.graph) {
        cache.failed_n_frames = n_frames;
        reset_decode_cache(cache);
        return false;
    }
    ggml_build_forward_expand(cache.graph, cache.audio);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl.backend));
    if (!cache.allocr || !ggml_gallocr_alloc_graph(cache.allocr, cache.graph)) {
        cache.failed_n_frames = n_frames;
        reset_decode_cache(cache);
        return false;
    }

    cache.n_frames = n_frames;
    cache.n_samples = static_cast<int32_t>(ggml_nelements(cache.audio));
    return true;
}

static bool run_cached_decode_graph(AudioCodec::Impl & impl, const int32_t * codes, int32_t n_frames,
                                    int32_t n_threads, std::vector<float> & audio_out) {
    codec_decode_cache & cache = impl.decode_cache;
    if (cache.failed_n_frames == n_frames) {
        return false;
    }
    if (!cache.ctx || cache.n_frames != n_frames) {
        if (!build_cached_decode_graph(impl, n_frames)) {
            return false;
        }
    }

    sanitize_decode_codes(impl, codes, n_frames, cache.sanitized_codes);

    const int32_t num_codebooks = impl.quantizer_residual_codebooks + 1;
    for (int32_t cb = 0; cb < num_codebooks; ++cb) {
        ggml_backend_tensor_set(
            cache.code_id_tensors[static_cast<size_t>(cb)],
            cache.sanitized_codes.data() + static_cast<size_t>(cb) * n_frames,
            0,
            static_cast<size_t>(n_frames) * sizeof(int32_t));
    }
    if (cache.inp.positions) {
        ggml_backend_tensor_set(cache.inp.positions, cache.inp.position_values.data(), 0,
            cache.inp.position_values.size() * sizeof(int32_t));
    }
    if (cache.inp.mask) {
        ggml_backend_tensor_set(cache.inp.mask, cache.inp.mask_values.data(), 0,
            cache.inp.mask_values.size() * sizeof(float));
    }

    if (ggml_backend_is_cpu(impl.backend)) {
        ggml_backend_cpu_set_n_threads(impl.backend, resolve_n_threads(n_threads));
    }

    if (ggml_backend_graph_compute(impl.backend, cache.graph) != GGML_STATUS_SUCCESS) {
        cache.failed_n_frames = n_frames;
        reset_decode_cache(cache);
        return false;
    }

    audio_out.resize(static_cast<size_t>(cache.n_samples));
    ggml_backend_tensor_get(cache.audio, audio_out.data(), 0,
                            static_cast<size_t>(cache.n_samples) * sizeof(float));
    return true;
}

bool AudioCodec::decode(const int32_t * codes, int32_t n_frames, int32_t n_threads, 
                         std::vector<float> & audio_out) {
    if (n_frames <= 0) return false;
    if (!impl_ || !impl_->backend) return false;

    if (!ggml_backend_is_cpu(impl_->backend) &&
        run_cached_decode_graph(*impl_, codes, n_frames, n_threads, audio_out)) {
        return true;
    }

    int32_t latent_frames = 0;
    std::vector<float> latent_out;
    {
        const size_t ctx_size = 96u * 1024u * 1024u;
        std::vector<uint8_t> ctx_buf(ctx_size);
        ggml_init_params p = { ctx_size, ctx_buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        if (!ctx) return false;

        transformer_inputs inp;
        const int32_t num_codebooks = impl_->quantizer_residual_codebooks + 1;
        std::vector<ggml_tensor *> code_id_tensors(static_cast<size_t>(num_codebooks));
        for (int32_t cb = 0; cb < num_codebooks; ++cb) {
            code_id_tensors[static_cast<size_t>(cb)] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_frames);
        }
        ggml_tensor * latent   = nullptr;
        try {
            ggml_tensor * stage = build_decode_codes_stage_backend(ctx, *impl_, code_id_tensors);
            latent = build_quantizer_decode_stage(ctx, *impl_, stage, inp);
            latent = ggml_cpy(ctx, latent, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent->ne[0], latent->ne[1]));
        } catch (const std::exception & e) {
            std::cerr << "[Codec::decode] quantizer decode stage failed: " << e.what() << std::endl;
            ggml_free(ctx);
            return false;
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 131072, false);
        ggml_build_forward_expand(gf, latent);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
        if (!allocr || !ggml_gallocr_alloc_graph(allocr, gf)) {
            if (allocr) ggml_gallocr_free(allocr);
            ggml_free(ctx);
            return false;
        }

        std::vector<int32_t> sanitized_codes(static_cast<size_t>(num_codebooks) * n_frames);
        auto clamp_code = [](int32_t code, int32_t codebook_size) -> int32_t {
            if (code < 0) {
                return 0;
            }
            if (code >= codebook_size) {
                return codebook_size - 1;
            }
            return code;
            };

        for (int32_t t = 0; t < n_frames; ++t) {
            sanitized_codes[static_cast<size_t>(t)] =
                clamp_code(codes[t], impl_->quantizer_semantic_codebook_size);
        }
        for (int32_t cb = 0; cb < impl_->quantizer_residual_codebooks; ++cb) {
            const size_t src_offset = static_cast<size_t>(cb + 1) * n_frames;
            const size_t dst_offset = static_cast<size_t>(cb + 1) * n_frames;
            for (int32_t t = 0; t < n_frames; ++t) {
                sanitized_codes[dst_offset + static_cast<size_t>(t)] =
                    clamp_code(codes[src_offset + static_cast<size_t>(t)],
                        impl_->quantizer_residual_codebook_size);
            }
        }

        for (int32_t cb = 0; cb < num_codebooks; ++cb) {
            ggml_backend_tensor_set(
                code_id_tensors[static_cast<size_t>(cb)],
                sanitized_codes.data() + static_cast<size_t>(cb) * n_frames,
                0,
                static_cast<size_t>(n_frames) * sizeof(int32_t));
        }
        if (inp.positions) {
            ggml_backend_tensor_set(inp.positions, inp.position_values.data(), 0,
                inp.position_values.size() * sizeof(int32_t));
        }
        if (inp.mask) {
            ggml_backend_tensor_set(inp.mask, inp.mask_values.data(), 0,
                inp.mask_values.size() * sizeof(float));
        }

        if (ggml_backend_is_cpu(impl_->backend)) ggml_backend_cpu_set_n_threads(impl_->backend, resolve_n_threads(n_threads));

        if (ggml_backend_graph_compute(impl_->backend, gf) != GGML_STATUS_SUCCESS) {
            std::cerr << "[Codec::decode] quantizer decode compute failed." << std::endl;
            ggml_gallocr_free(allocr);
            ggml_free(ctx);
            return false;
        }

        latent_frames = static_cast<int32_t>(latent->ne[1]);
        latent_out.resize(static_cast<size_t>(latent->ne[0]) * latent_frames);
        ggml_backend_tensor_get(latent, latent_out.data(), 0, latent_out.size() * sizeof(float));
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
    }

    const size_t ctx_size = 128u * 1024u * 1024u;
    std::vector<uint8_t> ctx_buf(ctx_size);
    ggml_init_params p = { ctx_size, ctx_buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    if (!ctx) return false;

    ggml_tensor * latent_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, impl_->latent_dim, latent_frames);
    ggml_tensor * audio_t   = nullptr;
    try {
        audio_t = build_decoder(ctx, *impl_, latent_in);
        audio_t = ggml_cpy(ctx, audio_t, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio_t->ne[0], audio_t->ne[1]));
    } catch (const std::exception & e) {
        std::cerr << "[Codec::decode] decoder build failed: " << e.what() << std::endl;
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 131072, false);
    ggml_build_forward_expand(gf, audio_t);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
    if (!allocr || !ggml_gallocr_alloc_graph(allocr, gf)) {
            if (allocr) ggml_gallocr_free(allocr);
            ggml_free(ctx);
            return false;
    }

    ggml_backend_tensor_set(latent_in, latent_out.data(), 0, latent_out.size() * sizeof(float));

    if (ggml_backend_is_cpu(impl_->backend)) ggml_backend_cpu_set_n_threads(impl_->backend, resolve_n_threads(n_threads));

    if (ggml_backend_graph_compute(impl_->backend, gf) != GGML_STATUS_SUCCESS) {
            std::cerr << "[Codec::decode] decoder compute failed." << std::endl;
            ggml_gallocr_free(allocr);
            ggml_free(ctx);
            return false;
    }

    const int32_t n_samples = static_cast<int32_t>(ggml_nelements(audio_t));
    audio_out.resize(n_samples);
    ggml_backend_tensor_get(audio_t, audio_out.data(), 0, n_samples * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    
    return true;
}

}
