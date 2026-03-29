#include "../include/s2_pipeline.h"
#include <cstdio>
#include <cmath>

namespace s2 {

static void safe_print_ln(const std::string& msg) {
    fputs(msg.c_str(), stdout);
    fputc('\n', stdout);
}

static void safe_print_error_ln(const std::string& msg) {
    fputs(msg.c_str(), stderr);
    fputc('\n', stderr);
}

Pipeline::Pipeline() {}
Pipeline::~Pipeline() {}

bool Pipeline::init(const PipelineParams & params) {
    safe_print_ln("--- Pipeline Init ---");

    if (!tokenizer_.load(params.tokenizer_path)) {
        safe_print_error_ln("Pipeline error: could not load tokenizer from " + params.tokenizer_path);
        return false;
    }

    if (!model_.load(params.model_path, params.gpu_device, params.backend_type)) {
        safe_print_error_ln("Pipeline error: could not load model from " + params.model_path);
        return false;
    }

    if (!codec_.load(params.model_path, -1, -1)) {
        safe_print_error_ln("Pipeline error: could not load codec from " + params.model_path);
        return false;
    }

    {
        const ModelHParams & hp = model_.hparams();
        TokenizerConfig & tc    = tokenizer_.config();
        if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
        if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
        if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
        if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
        if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
    }

    initialized_ = true;
    return true;
}

bool Pipeline::synthesize(const PipelineParams & params) {
    std::vector<float> audio_out;
    AudioData ref_audio;

    if (!params.prompt_audio_path.empty() && params.prompt_text.empty()) {
        safe_print_error_ln("Pipeline error: prompt audio was provided without prompt text.");
        return false;
    }

    if (!params.prompt_audio_path.empty()) {
        safe_print_ln("Loading reference audio: " + params.prompt_audio_path);
        if (!load_audio(params.prompt_audio_path, ref_audio, codec_.sample_rate())) {
            safe_print_error_ln("Pipeline warning: load_audio failed, running without reference audio.");
        }
    }

    if (!this->synthesize_raw(params, ref_audio, audio_out)) {
        safe_print_error_ln("Pipeline error: synthesis failed.");
        return false;
    }

    if (params.trim_silence && !audio_out.empty()) {
        auto trimmed = audio_trim_trailing_silence(audio_out.data(), audio_out.size(), codec_.sample_rate());
        if (!trimmed.empty()) audio_out = std::move(trimmed);
    }

    if (params.normalize_dynamic && !audio_out.empty()) {
        audio_out = audio_normalize_dynamic(audio_out.data(), audio_out.size(), codec_.sample_rate());
    }

    if (!save_audio(params.output_path, audio_out, codec_.sample_rate(),
                    false, params.normalize_output)) {
        safe_print_error_ln("Pipeline error: save_audio failed to " + params.output_path);
        return false;
    }

    safe_print_ln("Saved audio to: " + params.output_path);
    return true;
}

bool Pipeline::synthesize_to_memory(const PipelineParams & params, void** ref_audio_buffer, size_t* ref_audio_size, void** wav_buffer, size_t* wav_size) {
    std::vector<float> audio_out;
    AudioData ref_audio;

    if (ref_audio_buffer && ref_audio_size && *ref_audio_buffer && *ref_audio_size > 0 && params.prompt_text.empty()) {
        safe_print_error_ln("Pipeline error: reference audio was provided without reference text.");
        return false;
    }

    if (ref_audio_buffer && ref_audio_size && *ref_audio_buffer && *ref_audio_size > 0) {
        safe_print_ln("Loading reference audio...");
        if (!load_audio_from_memory(*ref_audio_buffer, *ref_audio_size, ref_audio, codec_.sample_rate())) {
            safe_print_error_ln("Pipeline warning: load_audio failed, running without reference audio.");
        }
    }

    if (!this->synthesize_raw(params, ref_audio, audio_out)) {
        safe_print_error_ln("Pipeline error: synthesis failed.");
        return false;
    }

    if (params.trim_silence && !audio_out.empty()) {
        auto trimmed = audio_trim_trailing_silence(audio_out.data(), audio_out.size(), codec_.sample_rate());
        if (!trimmed.empty()) audio_out = std::move(trimmed);
    }

    if (params.normalize_dynamic && !audio_out.empty()) {
        audio_out = audio_normalize_dynamic(audio_out.data(), audio_out.size(), codec_.sample_rate());
    } else if (params.normalize_output && !audio_out.empty()) {
        float peak = 0.0f;
        for (float s : audio_out) { float a = std::fabs(s); if (a > peak) peak = a; }
        if (peak > 1e-6f) { float scale = 0.95f / peak; for (float & s : audio_out) s *= scale; }
    }

    if (!audio_write_memory_wav(wav_buffer, wav_size, audio_out.data(), audio_out.size(), codec_.sample_rate())) {
        safe_print_error_ln("Pipeline error: audio_write_memory_wav failed");
        return false;
    }

    safe_print_ln("Audio synthesized");
    return true;
}

bool Pipeline::synthesize_raw(const PipelineParams & params, AudioData & ref_audio, std::vector<float>& audio_out) {
    std::lock_guard<std::mutex> lock(synthesize_mutex_);

    if (!initialized_) {
        safe_print_error_ln("Pipeline not initialized.");
        return false;
    }

    model_.clear_kv_cache();

    safe_print_ln("--- Pipeline Synthesize ---");
    safe_print_ln("Text: " + params.text);

    const int32_t num_codebooks = model_.hparams().num_codebooks;

    std::vector<int32_t> ref_codes;
    int32_t T_prompt = 0;
    std::string effective_prompt_text = params.prompt_text;
    
    // Set voice storage directory
    voice_mgr_.set_storage_dir(params.voice_storage_dir);
    
    // 1. Encode reference audio if provided (takes precedence over voice_id)
    if (!ref_audio.samples.empty()) {
        safe_print_ln("Encoding reference audio...");
        if (!codec_.encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
                           params.gen.n_threads, ref_codes, T_prompt)) {
            safe_print_error_ln("Pipeline warning: encode failed, running without reference audio.");
            ref_codes.clear();
            T_prompt = 0;
        } else {
            safe_print_ln("Encoded reference audio: " + std::to_string(T_prompt) + " frames");
            // Save voice profile if requested
            if (params.save_voice && !params.voice_id.empty()) {
                if (!save_voice_profile(params.voice_id, ref_codes, T_prompt, effective_prompt_text, params)) {
                    safe_print_error_ln("Warning: failed to save voice profile.");
                }
            }
        }
    }
    // 2. Otherwise load existing voice profile if voice_id is provided
    else if (!params.voice_id.empty()) {
        safe_print_ln("Loading voice profile: " + params.voice_id);
        try {
            VoiceProfile profile = voice_mgr_.load(params.voice_id);
            if (!profile.is_compatible(num_codebooks, model_.hparams().codebook_size, codec_.sample_rate())) {
                safe_print_error_ln("Voice profile incompatible with current model/codec.");
                return false;
            }
            ref_codes = profile.codes;
            T_prompt = profile.T_prompt;
            effective_prompt_text = profile.transcript;
            safe_print_ln("Loaded voice profile: " + params.voice_id + " (" + std::to_string(T_prompt) + " frames)");
        } catch (const std::exception & e) {
            safe_print_error_ln("Failed to load voice profile " + params.voice_id + ": " + e.what());
            return false;
        }
    }

    PromptTensor prompt = build_prompt(
        tokenizer_, params.text, effective_prompt_text,
        ref_codes.empty() ? nullptr : ref_codes.data(),
        num_codebooks, T_prompt);

    int32_t max_seq_len = prompt.cols + params.gen.max_new_tokens;
    if (!model_.init_kv_cache(max_seq_len)) {
        safe_print_error_ln("Pipeline error: init_kv_cache failed.");
        return false;
    }

    GenerateResult res = generate(model_, tokenizer_.config(), prompt, params.gen);

    if (res.n_frames == 0) {
        safe_print_error_ln("Pipeline error: generation produced no frames.");
        return false;
    }

    if (!codec_.decode(res.codes.data(), res.n_frames, params.gen.n_threads, audio_out)) {
        safe_print_error_ln("Pipeline error: decode failed.");
        return false;
    }

    model_.clear_kv_cache();
    return true;
}

bool Pipeline::save_voice_profile(const std::string & voice_id, 
                                 const std::vector<int32_t> & codes, 
                                 int32_t T_prompt,
                                 const std::string & transcript,
                                 const PipelineParams & params) {
    if (voice_id.empty()) return false;
    
    voice_mgr_.set_storage_dir(params.voice_storage_dir);
    
    VoiceProfile profile;
    profile.transcript = transcript;
    profile.codes = codes;
    profile.num_codebooks = model_.hparams().num_codebooks;
    profile.T_prompt = T_prompt;
    profile.sample_rate = codec_.sample_rate();
    profile.codebook_size = model_.hparams().codebook_size;
    
    // Simple timestamp
    std::time_t now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    profile.timestamp = buf;
    
    if (voice_mgr_.save(voice_id, profile)) {
        safe_print_ln("Saved voice profile: " + voice_id);
        return true;
    } else {
        safe_print_error_ln("Failed to save voice profile: " + voice_id);
        return false;
    }
}

}
