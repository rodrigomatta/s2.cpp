#include "../include/s2_generate.h"
#include "../include/s2_log.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace s2 {

GenerateResult generate(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params
) {
    const auto generate_t0 = std::chrono::steady_clock::now();
    GenerateResult out;
    out.num_codebooks = model.hparams().num_codebooks;
    if (out.num_codebooks <= 0) out.num_codebooks = 1;

    const int32_t vocab_size   = model.hparams().vocab_size;
    const int32_t sem_begin    = model.hparams().semantic_begin_id;
    const int32_t sem_end      = model.hparams().semantic_end_id;
    const int32_t codebook_size = model.hparams().codebook_size;
    const int32_t im_end_id    = config.im_end_id;
    const int32_t num_cb       = out.num_codebooks;

    std::vector<float> sem_mask(vocab_size, -std::numeric_limits<float>::infinity());
    for (int32_t i = sem_begin; i <= sem_end && i < vocab_size; ++i) {
        sem_mask[i] = 0.0f;
    }
    if (im_end_id >= 0 && im_end_id < vocab_size) {
        sem_mask[im_end_id] = 0.0f;
    }

    const int32_t rows = prompt.rows;
    const int32_t cols = prompt.cols;
    std::vector<int32_t> prompt_tm(static_cast<size_t>(rows) * cols);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t c = 0; c < cols; ++c) {
            prompt_tm[static_cast<size_t>(c) * rows + r] = prompt.data[static_cast<size_t>(r) * cols + c];
        }
    }

    StepResult state;
    if (params.verbose && log_enabled(LogLevel::Info)) {
        std::cout << "[Generate] Prefilling " << prompt.cols << " tokens..." << std::endl;
    }
    const auto prefill_t0 = std::chrono::steady_clock::now();
    if (!model.prefill(prompt_tm, prompt.cols, params.n_threads, state)) {
        std::cerr << "[Generate] Prefill failed." << std::endl;
        return out;
    }
    const auto prefill_t1 = std::chrono::steady_clock::now();

    auto apply_mask_and_sample = [&](const std::vector<float> & logits,
                                     bool block_im_end) -> int32_t {
        std::vector<float> biased(vocab_size);
        for (int32_t i = 0; i < vocab_size; ++i) {
            biased[i] = logits[i] + sem_mask[i];
        }
        if (block_im_end && im_end_id >= 0 && im_end_id < vocab_size) {
            biased[im_end_id] = -std::numeric_limits<float>::infinity();
        }
        SamplerParams sparams;
        sparams.temperature     = params.temperature;
        sparams.top_p           = params.top_p;
        sparams.top_k           = params.top_k;
        return sample_token(biased.data(), vocab_size, sparams);
    };

    bool block_end = (params.min_tokens_before_end > 0);
    int32_t main_token = apply_mask_and_sample(state.logits, block_end);

    out.codes.resize(static_cast<size_t>(num_cb) * params.max_new_tokens, 0);
    out.n_frames = 0;

    std::vector<float> fast_logits;

    SamplerParams sparams;
    sparams.temperature     = params.temperature;
    sparams.top_p           = params.top_p;
    sparams.top_k           = params.top_k;

    std::vector<int32_t> ras_window;
    const int32_t ras_window_size = 10;
    const float ras_high_temp     = 1.0f;
    const float ras_high_top_p    = 0.9f;

    if (params.verbose && log_enabled(LogLevel::Info)) {
        std::cout << "[Generate] Generating (max " << params.max_new_tokens << " tokens)..." << std::endl;
    }

    int32_t step = 0;
    const auto loop_t0 = std::chrono::steady_clock::now();
    while (main_token != im_end_id && step < params.max_new_tokens) {

        if (!ras_window.empty() &&
            std::find(ras_window.begin(), ras_window.end(), main_token) != ras_window.end() &&
            main_token >= sem_begin && main_token <= sem_end)
        {

            std::vector<float> biased(vocab_size);
            for (int32_t i = 0; i < vocab_size; ++i) {
                biased[i] = state.logits[i] + sem_mask[i];
            }
            if (step < params.min_tokens_before_end && im_end_id >= 0 && im_end_id < vocab_size) {
                biased[im_end_id] = -std::numeric_limits<float>::infinity();
            }
            SamplerParams ras_sparams;
            ras_sparams.temperature = ras_high_temp;
            ras_sparams.top_p       = ras_high_top_p;
            ras_sparams.top_k       = params.top_k;
            main_token = sample_token(biased.data(), vocab_size, ras_sparams);
        }

        ras_window.push_back(main_token);
        if ((int32_t)ras_window.size() > ras_window_size) {
            ras_window.erase(ras_window.begin());
        }

        int32_t sem_code = main_token - sem_begin;
        if (sem_code < 0)           sem_code = 0;
        if (sem_code >= codebook_size) sem_code = codebook_size - 1;

        std::vector<int32_t> codebooks_cb;
        codebooks_cb.reserve(num_cb);
        codebooks_cb.push_back(sem_code);

        for (int32_t cb_idx = 1; cb_idx < num_cb; ++cb_idx) {

            if (!model.fast_decode(state.hidden, codebooks_cb, params.n_threads, fast_logits)) {
                std::cerr << "[Generate] fast_decode failed at cb " << cb_idx << std::endl;

                for (int32_t r = cb_idx; r < num_cb; ++r) {
                    codebooks_cb.push_back(0);
                }
                break;
            }
            int32_t cb_token = sample_token(fast_logits.data(), (int32_t)fast_logits.size(), sparams);
            codebooks_cb.push_back(cb_token);
        }

        for (int32_t cb = 0; cb < num_cb; ++cb) {
            out.codes[static_cast<size_t>(cb) * params.max_new_tokens + step] = codebooks_cb[cb];
        }
        out.n_frames++;

        if (params.on_frame) {
            FrameCallbackData fcd;
            fcd.codes = codebooks_cb.data();
            fcd.frame_index = step;
            fcd.total_frames = out.n_frames;
            fcd.num_codebooks = num_cb;
            if (!params.on_frame(fcd)) {

                if (params.verbose && log_enabled(LogLevel::Info)) {
                    std::cout << "\n[Generate] Aborted by callback at frame " << step << std::endl;
                }
                break;
            }
        }

        std::vector<int32_t> step_input(num_cb + 1);
        step_input[0] = main_token;
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            step_input[cb + 1] = codebooks_cb[cb];
        }

        if (!model.step(step_input, params.n_threads, state)) {
            std::cerr << "[Generate] step() failed at step " << step << std::endl;
            break;
        }

        step++;
        if (params.verbose && log_enabled(LogLevel::Info) && step % 50 == 0) {
            std::cout << "\r[Generate] " << step << " / " << params.max_new_tokens << " tokens..." << std::flush;
        }

        bool block_next_end = (step < params.min_tokens_before_end);
        main_token = apply_mask_and_sample(state.logits, block_next_end);
    }

    if (params.verbose && log_enabled(LogLevel::Info)) {
        const auto loop_t1 = std::chrono::steady_clock::now();
        const auto generate_t1 = std::chrono::steady_clock::now();
        const double prefill_ms = std::chrono::duration<double, std::milli>(prefill_t1 - prefill_t0).count();
        const double loop_ms = std::chrono::duration<double, std::milli>(loop_t1 - loop_t0).count();
        const double total_ms = std::chrono::duration<double, std::milli>(generate_t1 - generate_t0).count();
        const double ms_per_frame = out.n_frames > 0 ? (loop_ms / out.n_frames) : 0.0;
        std::cout << std::endl;
        std::cout << "[Generate] Done: " << out.n_frames << " frames generated." << std::endl;
        std::cout << "[Metrics] Generate: prefill=" << prefill_ms
                  << " ms, loop=" << loop_ms
                  << " ms, total=" << total_ms
                  << " ms, avg=" << ms_per_frame
                  << " ms/frame" << std::endl;
    }

    const int32_t n_frames = out.n_frames;
    if (n_frames < params.max_new_tokens) {
        std::vector<int32_t> compacted(static_cast<size_t>(num_cb) * n_frames);
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            for (int32_t t = 0; t < n_frames; ++t) {
                compacted[static_cast<size_t>(cb) * n_frames + t] =
                    out.codes[static_cast<size_t>(cb) * params.max_new_tokens + t];
            }
        }
        out.codes = std::move(compacted);
    } else {
        out.codes.resize(static_cast<size_t>(num_cb) * n_frames);
    }

    return out;
}

}
