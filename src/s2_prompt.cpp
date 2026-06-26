#include "../include/s2_prompt.h"

namespace s2 {

PromptTensor build_prompt(
    const Tokenizer & tokenizer,
    const std::string & text,
    const std::string & prompt_text,
    const int32_t * prompt_codes,
    int32_t num_codebooks,
    int32_t T_prompt,
    const std::string & instruction
) {
    PromptTensor result;
    result.rows = num_codebooks + 1;

    const TokenizerConfig & cfg = tokenizer.config();
    const int32_t im_end_id     = cfg.im_end_id;
    const int32_t voice_id      = cfg.voice_id;
    const int32_t sem_begin     = cfg.semantic_begin_id;

    const std::vector<int32_t> NEWLINE = { 198 };

    bool has_reference = (prompt_codes != nullptr && T_prompt > 0 && !prompt_text.empty());
    bool prompt_has_speaker_tag = prompt_text.find("<|speaker:") != std::string::npos;

    std::vector<int32_t> sys_pre;
    std::vector<int32_t> sys_post;

    if (has_reference) {
        auto app = [&](std::vector<int32_t> & dst, const std::vector<int32_t> & src) {
            dst.insert(dst.end(), src.begin(), src.end());
        };

        app(sys_pre, tokenizer.encode("<|im_start|>system"));
        app(sys_pre, NEWLINE);
        std::string sys_msg = "convert the provided text to speech reference to the following:\n\nText:\n";
        if (!instruction.empty()) {
            sys_msg = instruction + "\n\n" + sys_msg;
        }
        app(sys_pre, tokenizer.encode(sys_msg));
        if (!prompt_has_speaker_tag) {
            app(sys_pre, tokenizer.encode("<|speaker:0|>"));
        }
        app(sys_pre, tokenizer.encode(prompt_text));
        app(sys_pre, tokenizer.encode("\n\nSpeech:\n"));

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
        auto app = [&](std::vector<int32_t> & dst, const std::vector<int32_t> & src) {
            dst.insert(dst.end(), src.begin(), src.end());
        };

        app(sys_post, tokenizer.encode("<|im_start|>system"));
        app(sys_post, NEWLINE);
        std::string sys_msg = "convert the provided text to speech";
        if (!instruction.empty()) {
            sys_msg += ". " + instruction;
        }
        app(sys_post, tokenizer.encode(sys_msg));
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

    int32_t total_len = (int32_t)sys_pre.size() + (has_reference ? T_prompt : 0) + (int32_t)sys_post.size();
    result.cols = total_len;

    result.data.assign(static_cast<size_t>(result.rows) * total_len, 0);

    int32_t pos = 0;

    for (int32_t i = 0; i < (int32_t)sys_pre.size(); ++i) {
        result.data[0 * total_len + pos + i] = sys_pre[i];
    }
    pos += (int32_t)sys_pre.size();

    if (has_reference && T_prompt > 0) {
        for (int32_t t = 0; t < T_prompt; ++t) {
            result.data[0 * total_len + pos + t] = prompt_codes[0 * T_prompt + t] + sem_begin;
        }
        for (int32_t cb = 0; cb < num_codebooks; ++cb) {
            for (int32_t t = 0; t < T_prompt; ++t) {
                result.data[(cb + 1) * total_len + pos + t] = prompt_codes[cb * T_prompt + t];
            }
        }
        pos += T_prompt;
    }

    for (int32_t i = 0; i < (int32_t)sys_post.size(); ++i) {
        result.data[0 * total_len + pos + i] = sys_post[i];
    }

    return result;
}

}
