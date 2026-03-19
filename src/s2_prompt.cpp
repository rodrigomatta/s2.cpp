#include "../include/s2_prompt.h"
#include <iostream>
#include <cstring>

namespace s2 {

// ---------------------------------------------------------------------------
// build_prompt() — matches Python build_prompt_tensor() in ggml_pure.py
// ---------------------------------------------------------------------------
//
// Layout of the returned PromptTensor (rows x cols):
//   Row 0:        token IDs in vocabulary space.
//                 - Text positions: raw BPE token IDs.
//                 - VQ positions:   prompt_codes[0][t] + semantic_begin_id
//   Rows 1..num_cb: codebook values (0 for text positions).
//                 - VQ positions:   prompt_codes[cb][t]  (0-indexed, cb = 0..num_cb-1)
//
// When prompt_codes / prompt_text are absent (no reference audio), a simplified
// prompt is built without the VQ section.
// ---------------------------------------------------------------------------

PromptTensor build_prompt(
    const Tokenizer & tokenizer,
    const std::string & text,
    const std::string & prompt_text,
    const int32_t * prompt_codes,  // (num_codebooks, T_prompt) row-major, or nullptr
    int32_t num_codebooks,
    int32_t T_prompt
) {
    PromptTensor result;
    result.rows = num_codebooks + 1;

    const TokenizerConfig & cfg = tokenizer.config();
    const int32_t im_end_id     = cfg.im_end_id;
    const int32_t voice_id      = cfg.voice_id;
    const int32_t sem_begin     = cfg.semantic_begin_id;

    // NEWLINE token (byte 0x0A = 198 in Qwen vocabulary)
    const std::vector<int32_t> NEWLINE = { 198 };

    bool has_reference = (prompt_codes != nullptr && T_prompt > 0 && !prompt_text.empty());

    std::vector<int32_t> sys_pre;
    std::vector<int32_t> sys_post;

    if (has_reference) {
        // System section: introduce voice reference
        auto app = [&](std::vector<int32_t> & dst, const std::vector<int32_t> & src) {
            dst.insert(dst.end(), src.begin(), src.end());
        };

        app(sys_pre, tokenizer.encode("<|im_start|>system"));
        app(sys_pre, NEWLINE);
        app(sys_pre, tokenizer.encode("convert the provided text to speech reference to the following:\n\nText:\n"));
        app(sys_pre, tokenizer.encode("<|speaker:0|>"));
        app(sys_pre, tokenizer.encode(prompt_text));
        app(sys_pre, tokenizer.encode("\n\nSpeech:\n"));
        // VQ section goes here (T_prompt frames)

        app(sys_post, { im_end_id });
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode("<|im_start|>user"));
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode(text));
        app(sys_post, { im_end_id });
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode("<|im_start|>assistant"));
        app(sys_post, NEWLINE);
        app(sys_post, { voice_id });
    } else {
        // No reference audio: simpler prompt
        auto app = [&](std::vector<int32_t> & dst, const std::vector<int32_t> & src) {
            dst.insert(dst.end(), src.begin(), src.end());
        };

        // sys_pre is empty, everything goes into sys_post
        app(sys_post, tokenizer.encode("<|im_start|>system"));
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode("convert the provided text to speech."));
        app(sys_post, { im_end_id });
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode("<|im_start|>user"));
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode(text));
        app(sys_post, { im_end_id });
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode("<|im_start|>assistant"));
        app(sys_post, NEWLINE);
        app(sys_post, { voice_id });
    }

    // total columns
    int32_t total_len = (int32_t)sys_pre.size() + (has_reference ? T_prompt : 0) + (int32_t)sys_post.size();
    result.cols = total_len;

    // Allocate and zero-fill
    result.data.assign(static_cast<size_t>(result.rows) * total_len, 0);

    int32_t pos = 0;

    // Write sys_pre into row 0 (rows 1..num_cb remain 0)
    for (int32_t i = 0; i < (int32_t)sys_pre.size(); ++i) {
        result.data[0 * total_len + pos + i] = sys_pre[i];
    }
    pos += (int32_t)sys_pre.size();

    // Write VQ section
    if (has_reference && T_prompt > 0) {
        // Row 0: semantic token IDs = prompt_codes[0][t] + semantic_begin_id
        for (int32_t t = 0; t < T_prompt; ++t) {
            result.data[0 * total_len + pos + t] = prompt_codes[0 * T_prompt + t] + sem_begin;
        }
        // Rows 1..num_cb: prompt_codes[cb][t] (codebook-space 0-indexed)
        for (int32_t cb = 0; cb < num_codebooks; ++cb) {
            for (int32_t t = 0; t < T_prompt; ++t) {
                result.data[(cb + 1) * total_len + pos + t] = prompt_codes[cb * T_prompt + t];
            }
        }
        pos += T_prompt;
    }

    // Write sys_post into row 0
    for (int32_t i = 0; i < (int32_t)sys_post.size(); ++i) {
        result.data[0 * total_len + pos + i] = sys_post[i];
    }
    // rows 1..num_cb at sys_post positions remain 0

    return result;
}

} // namespace s2