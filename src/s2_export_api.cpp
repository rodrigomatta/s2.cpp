#include "../include/s2_export_api.h"
#include "../third_party/filesystem.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <string>
#include <vector>

namespace fs = ghc::filesystem;

namespace {

s2::BackendType to_backend_type(int32_t backend_type) {
    switch (backend_type) {
        case static_cast<int32_t>(s2::BackendType::Vulkan): return s2::BackendType::Vulkan;
        case static_cast<int32_t>(s2::BackendType::CUDA):   return s2::BackendType::CUDA;
        case static_cast<int32_t>(s2::BackendType::Metal):  return s2::BackendType::Metal;
        case static_cast<int32_t>(s2::BackendType::CPU):
        default: return s2::BackendType::CPU;
    }
}

bool has_value(const char* value) {
    return value != nullptr && value[0] != '\0';
}

void sync_tokenizer_config(s2::SlowARModel* model, s2::Tokenizer* tokenizer) {
    if (!model || !tokenizer) {
        return;
    }

    const s2::ModelHParams & hp = model->hparams();
    s2::TokenizerConfig & tc = tokenizer->config();
    if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
    if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
    if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
    if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
    if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
}

void apply_streaming_params(const S2StreamingParams* streaming_params,
                            s2::PipelineParams& pipeline_params) {
    if (!streaming_params) {
        return;
    }

    const bool explicit_stride = streaming_params->stream_decode_stride_frames > 0;
    const bool explicit_holdback = streaming_params->stream_holdback_frames >= 0;

    if (explicit_stride) {
        pipeline_params.stream_decode_stride_frames =
            streaming_params->stream_decode_stride_frames;
    }
    if (explicit_holdback) {
        pipeline_params.stream_holdback_frames =
            streaming_params->stream_holdback_frames;
    }
    if (streaming_params->codec_decode_context_frames >= 0) {
        pipeline_params.codec_decode_context_frames =
            streaming_params->codec_decode_context_frames;
    }

    if (streaming_params->low_latency != 0) {
        if (!explicit_stride && pipeline_params.stream_decode_stride_frames <= 0) {
            pipeline_params.stream_decode_stride_frames = 1;
        }
        if (!explicit_holdback && pipeline_params.stream_holdback_frames < 0) {
            pipeline_params.stream_holdback_frames = 0;
        }
    }
}

void apply_voice_selection(s2::PipelineParams& params,
                           const char* voice_value,
                           const char* voice_dir_value) {
    const std::string voice = voice_value ? voice_value : "";
    const std::string voice_dir = voice_dir_value ? voice_dir_value : "";

    if (voice.empty()) {
        if (!voice_dir.empty()) {
            params.voice_storage_dir = voice_dir;
        }
        return;
    }

    fs::path voice_path(voice);
    if (voice_path.extension() == ".s2voice") {
        if (!voice_path.parent_path().empty()) {
            params.voice_storage_dir = voice_path.parent_path().string();
        } else if (!voice_dir.empty()) {
            params.voice_storage_dir = voice_dir;
        }
        params.voice_id = voice_path.stem().string();
        return;
    }

    params.voice_id = voice;
    if (!voice_dir.empty()) {
        params.voice_storage_dir = voice_dir;
    }
}

std::string trim_copy(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }

    size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return value.substr(begin, end - begin);
}

std::vector<std::string> split_sentences_basic(const std::string& text) {
    std::vector<std::string> segments;
    std::string current;
    current.reserve(text.size());

    auto flush_current = [&]() {
        std::string trimmed = trim_copy(current);
        if (!trimmed.empty()) {
            segments.push_back(std::move(trimmed));
        }
        current.clear();
    };

    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        current.push_back(c);

        const bool hard_break =
            c == '.' || c == '!' || c == '?' || c == '\n';
        if (!hard_break) {
            continue;
        }

        while (i + 1 < text.size()) {
            const char next = text[i + 1];
            if (next == '.' || next == '!' || next == '?' ||
                next == '"' || next == '\'' || next == ')' || next == ']' ||
                std::isspace(static_cast<unsigned char>(next))) {
                current.push_back(next);
                ++i;
                if (!std::isspace(static_cast<unsigned char>(next))) {
                    continue;
                }
            }
            break;
        }

        flush_current();
    }

    flush_current();
    if (segments.empty()) {
        const std::string trimmed = trim_copy(text);
        if (!trimmed.empty()) {
            segments.push_back(trimmed);
        }
    }
    return segments;
}

std::vector<std::string> split_long_segment(const std::string& segment, size_t max_chars) {
    if (max_chars == 0 || segment.size() <= max_chars) {
        return {segment};
    }

    std::vector<std::string> parts;
    std::string remaining = trim_copy(segment);
    while (!remaining.empty() && remaining.size() > max_chars) {
        size_t split_pos = remaining.find_last_of(",;:", max_chars);
        if (split_pos == std::string::npos || split_pos < max_chars / 2) {
            split_pos = remaining.find_last_of(" \t", max_chars);
        }
        if (split_pos == std::string::npos || split_pos < max_chars / 2) {
            split_pos = max_chars;
        } else {
            split_pos += 1;
        }

        std::string head = trim_copy(remaining.substr(0, split_pos));
        if (!head.empty()) {
            parts.push_back(std::move(head));
        }
        remaining = trim_copy(remaining.substr(split_pos));
    }

    if (!remaining.empty()) {
        parts.push_back(std::move(remaining));
    }
    return parts;
}

std::vector<std::string> split_text_for_segmented_synthesis(const std::string& text,
                                                            size_t max_chars_per_segment) {
    std::vector<std::string> sentences = split_sentences_basic(text);
    if (max_chars_per_segment == 0) {
        return sentences;
    }

    std::vector<std::string> segments;
    for (const std::string& sentence : sentences) {
        std::vector<std::string> parts = split_long_segment(sentence, max_chars_per_segment);
        segments.insert(segments.end(),
                        std::make_move_iterator(parts.begin()),
                        std::make_move_iterator(parts.end()));
    }
    return segments;
}

bool synthesize_segmented_to_sink(s2::Pipeline& pipeline,
                                  const s2::PipelineParams& base_params,
                                  const int32_t* ref_codes,
                                  int32_t t_prompt,
                                  const std::vector<std::string>& segments,
                                  int32_t sentence_pause_ms,
                                  s2::StreamingSink& sink) {
    if (segments.empty()) {
        sink.on_error("No text segments to synthesize");
        return false;
    }

    uint8_t wav_header[44];
    s2::audio_write_streaming_wav_header(wav_header, pipeline.output_sample_rate(), 1, 16);
    if (!sink.on_header(wav_header, sizeof(wav_header))) {
        return false;
    }

    const int32_t sample_rate = pipeline.output_sample_rate();
    const size_t pause_samples = sentence_pause_ms > 0
        ? static_cast<size_t>((static_cast<int64_t>(sample_rate) * sentence_pause_ms) / 1000)
        : 0;
    std::vector<float> pause_pcm(pause_samples, 0.0f);

    for (size_t i = 0; i < segments.size(); ++i) {
        if (sink.is_cancelled()) {
            return false;
        }

        s2::PipelineParams segment_params = base_params;
        segment_params.text = segments[i];

        std::vector<float> audio_out;
        if (!pipeline.synthesize_with_prompt_codes(segment_params, ref_codes, t_prompt, audio_out)) {
            sink.on_error("Segment synthesis failed");
            return false;
        }

        if (!audio_out.empty()) {
            std::vector<float> trimmed =
                s2::audio_trim_trailing_silence(audio_out.data(), audio_out.size(), sample_rate);
            if (!trimmed.empty()) {
                audio_out = std::move(trimmed);
            }
            if (!sink.on_pcm_data(audio_out.data(), audio_out.size())) {
                return false;
            }
        }

        if (i + 1 < segments.size() && !pause_pcm.empty()) {
            if (!sink.on_pcm_data(pause_pcm.data(), pause_pcm.size())) {
                return false;
            }
        }
    }

    sink.on_done();
    return true;
}

class CallbackStreamingSink : public s2::StreamingSink {
public:
    explicit CallbackStreamingSink(const S2StreamingCallbacks* callbacks)
        : callbacks_(callbacks) {}

    bool on_pcm_data(const float* data, size_t n_samples) override {
        const std::vector<int16_t> pcm16 = s2::audio_to_pcm16(data, n_samples);
        return write_chunk(reinterpret_cast<const uint8_t*>(pcm16.data()),
                           pcm16.size() * sizeof(int16_t));
    }

    bool on_header(const uint8_t* header, size_t size) override {
        return write_chunk(header, size);
    }

    void on_done() override {
        done_ = true;
        if (callbacks_ && callbacks_->on_done) {
            callbacks_->on_done(callbacks_->user_data);
        }
    }

    void on_error(const std::string& message) override {
        last_error_ = message;
        if (callbacks_ && callbacks_->on_error) {
            callbacks_->on_error(last_error_.c_str(), callbacks_->user_data);
        }
    }

    bool is_cancelled() const override {
        if (callbacks_ && callbacks_->is_cancelled) {
            const int cancelled = callbacks_->is_cancelled(callbacks_->user_data);
            if (cancelled != 0) {
                cancelled_ = true;
                return true;
            }
        }
        return false;
    }

    bool aborted_by_callback() const {
        return aborted_by_callback_;
    }

    bool cancelled() const {
        return cancelled_;
    }

    bool done() const {
        return done_;
    }

private:
    bool write_chunk(const uint8_t* data, size_t size) {
        if (!callbacks_ || !callbacks_->on_wav_chunk) {
            on_error("Streaming callback on_wav_chunk is required");
            return false;
        }
#ifdef WIN32 //Fixes conflict of #max defines for MSVC builds.
        if (size > static_cast<size_t>(INT_MAX)) {
#else
        if (size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#endif
            on_error("Streaming chunk exceeds int32 size limit");
            return false;
        }

        const int ok = callbacks_->on_wav_chunk(
            data,
            static_cast<int32_t>(size),
            callbacks_->user_data);
        if (ok == 0) {
            aborted_by_callback_ = true;
            cancelled_ = is_cancelled();
            return false;
        }
        return true;
    }

    const S2StreamingCallbacks* callbacks_ = nullptr;
    mutable bool cancelled_ = false;
    bool aborted_by_callback_ = false;
    bool done_ = false;
    std::string last_error_;
};

}

s2::Pipeline* AllocS2Pipeline() {
    return new s2::Pipeline();
}

void ReleaseS2Pipeline(s2::Pipeline* Pipeline) {
    delete Pipeline;
}

void SetS2LogLevel(int32_t LogLevel) {
    s2::set_log_level(static_cast<s2::LogLevel>(LogLevel));
}

int32_t GetS2LogLevel() {
    return static_cast<int32_t>(s2::get_log_level());
}

void SyncS2TokenizerConfigFromS2Model(s2::SlowARModel* Model, s2::Tokenizer* Tokenizer) {
    sync_tokenizer_config(Model, Tokenizer);
}

int InitializeS2Pipeline(s2::Pipeline* Pipeline, s2::Tokenizer* Tokenizer,
                         s2::SlowARModel* Model, s2::AudioCodec* AudioCodec) {
    if (!Pipeline || !Tokenizer || !Model || !AudioCodec) {
        return 0;
    }

    return Pipeline->init_from_components(Tokenizer, Model, AudioCodec) ? 1 : 0;
}

int InitializeS2PipelineFromFiles(s2::Pipeline* Pipeline, const char* gguf_path,
                                  const char* tokenizer_path, int32_t gpu_device,
                                  int32_t backend_type, int32_t n_gpu_layers,
                                  int codec_follow_backend) {
    if (!Pipeline || !has_value(gguf_path) || !has_value(tokenizer_path)) {
        return 0;
    }

    s2::PipelineParams params;
    params.model_path = gguf_path;
    params.tokenizer_path = tokenizer_path;
    params.gpu_device = gpu_device;
    params.backend_type = to_backend_type(backend_type);
    params.n_gpu_layers = n_gpu_layers;
    params.codec_follow_backend = codec_follow_backend != 0;
    return Pipeline->init(params) ? 1 : 0;
}

s2::GenerateParams* AllocS2GenerateParams() {
    return new s2::GenerateParams();
}

void ReleaseS2GenerateParams(s2::GenerateParams* GenerateParams) {
    delete GenerateParams;
}

int InitializeS2GenerateParams(s2::GenerateParams* GenerateParams,
                               int32_t max_new_tokens, float temperature,
                               float top_p, int32_t top_k,
                               int32_t min_tokens_before_end,
                               int32_t n_threads, int verbose) {
    if (!GenerateParams) {
        return 0;
    }

    if (max_new_tokens >= 0) GenerateParams->max_new_tokens = max_new_tokens;
    if (temperature >= 0.0f) GenerateParams->temperature = temperature;
    if (top_p >= 0.0f) GenerateParams->top_p = top_p;
    if (top_k >= 0) GenerateParams->top_k = top_k;
    if (min_tokens_before_end >= 0) GenerateParams->min_tokens_before_end = min_tokens_before_end;
    if (n_threads >= 0) GenerateParams->n_threads = n_threads;
    if (verbose >= 0) GenerateParams->verbose = verbose != 0;
    return 1;
}

s2::SlowARModel* AllocS2Model() {
    return new s2::SlowARModel();
}

void ReleaseS2Model(s2::SlowARModel* Model) {
    delete Model;
}

int InitializeS2Model(s2::SlowARModel* Model, const char* gguf_path,
                      int32_t gpu_device, int32_t backend_type) {
    return InitializeS2ModelWithGpuLayers(Model, gguf_path, gpu_device, backend_type, -1);
}

int InitializeS2ModelWithGpuLayers(s2::SlowARModel* Model, const char* gguf_path,
                                   int32_t gpu_device, int32_t backend_type,
                                   int32_t n_gpu_layers) {
    if (!Model || !has_value(gguf_path)) {
        return 0;
    }

    return Model->load(std::string(gguf_path), gpu_device, to_backend_type(backend_type),
                       n_gpu_layers) ? 1 : 0;
}

s2::Tokenizer* AllocS2Tokenizer() {
    return new s2::Tokenizer();
}

void ReleaseS2Tokenizer(s2::Tokenizer* Tokenizer) {
    delete Tokenizer;
}

int InitializeS2Tokenizer(s2::Tokenizer* Tokenizer, const char* path) {
    if (!Tokenizer || !has_value(path)) {
        return 0;
    }

    return Tokenizer->load(std::string(path)) ? 1 : 0;
}

s2::AudioCodec* AllocS2AudioCodec() {
    return new s2::AudioCodec();
}

void ReleaseS2AudioCodec(s2::AudioCodec* AudioCodec) {
    delete AudioCodec;
}

int InitializeS2AudioCodec(s2::AudioCodec* AudioCodec, const char* gguf_path,
                           int32_t gpu_device, int32_t backend_type) {
    if (!AudioCodec || !has_value(gguf_path)) {
        return 0;
    }

    return AudioCodec->load(std::string(gguf_path), gpu_device, to_backend_type(backend_type)) ? 1 : 0;
}

std::vector<int32_t>* AllocS2AudioPromptCodes() {
    return new std::vector<int32_t>();
}

void ReleaseS2AudioPromptCodes(std::vector<int32_t>* AudioPromptCodes) {
    delete AudioPromptCodes;
}

int InitializeAudioPromptCodes(s2::Pipeline* Pipeline, int32_t ThreadCount,
                               const char* ReferenceAudioPath,
                               std::vector<int32_t>* AudioPromptCodes,
                               int32_t* TPrompt) {
    if (!Pipeline || !Pipeline->is_initialized() || !AudioPromptCodes || !TPrompt) {
        return 0;
    }

    if (!AudioPromptCodes->empty()) {
        return 1;
    }

    if (!has_value(ReferenceAudioPath)) {
        *TPrompt = 0;
        return 1;
    }

    return Pipeline->encode_prompt_audio(ReferenceAudioPath, ThreadCount,
                                         *AudioPromptCodes, *TPrompt) ? 1 : -1;
}

std::vector<float>* AllocS2AudioBuffer(int InitialSize) {
    return InitialSize > 0 ? new std::vector<float>(static_cast<size_t>(InitialSize))
                           : new std::vector<float>();
}

void ReleaseS2AudioBuffer(std::vector<float>* AudioBuffer) {
    delete AudioBuffer;
}

float* GetS2AudioBufferDataPointer(std::vector<float>* AudioBuffer) {
    if (!AudioBuffer || AudioBuffer->empty()) {
        return nullptr;
    }
    return AudioBuffer->data();
}

int S2Synthesize(s2::Pipeline* Pipeline, const s2::GenerateParams* GenerateParams,
                 std::vector<float>* AudioBuffer,
                 std::vector<int32_t>* ReferenceAudioPromptCodes,
                 int32_t* ReferenceAudioTPrompt,
                 const char* ReferenceAudioPath,
                 const char* ReferenceAudioTranscript,
                 const char* TextToInfer,
                 const char* OutputAudioPath,
                 int32_t* AudioBufferOutputLength) {
    if (!Pipeline || !Pipeline->is_initialized() || !GenerateParams) {
        return 0;
    }

    s2::PipelineParams params;
    params.text = TextToInfer ? TextToInfer : "";
    params.prompt_text = ReferenceAudioTranscript ? ReferenceAudioTranscript : "";
    params.gen = *GenerateParams;

    std::vector<int32_t> local_prompt_codes;
    int32_t local_T_prompt = 0;
    std::vector<int32_t>* prompt_codes =
        ReferenceAudioPromptCodes ? ReferenceAudioPromptCodes : &local_prompt_codes;
    int32_t* T_prompt = ReferenceAudioTPrompt ? ReferenceAudioTPrompt : &local_T_prompt;

    int return_code = 1;
    const bool has_reference_path = has_value(ReferenceAudioPath);
    const bool has_precomputed_reference =
        ReferenceAudioPromptCodes && !ReferenceAudioPromptCodes->empty();

    if ((has_reference_path || has_precomputed_reference) && params.prompt_text.empty()) {
        return -7;
    }
    if (has_precomputed_reference && !ReferenceAudioTPrompt) {
        return -8;
    }

    if (has_reference_path && prompt_codes->empty()) {
        if (!Pipeline->encode_prompt_audio(ReferenceAudioPath, params.gen.n_threads,
                                           *prompt_codes, *T_prompt)) {
            return_code = -1;
        }
    }

    const int32_t usable_T_prompt =
        (!prompt_codes->empty() && T_prompt) ? std::max<int32_t>(0, *T_prompt) : 0;
    const int32_t* prompt_data =
        (usable_T_prompt > 0 && !prompt_codes->empty()) ? prompt_codes->data() : nullptr;

    std::vector<float> owned_audio_buffer;
    std::vector<float>* audio_out = AudioBuffer ? AudioBuffer : &owned_audio_buffer;
    audio_out->clear();

    if (!Pipeline->synthesize_with_prompt_codes(params, prompt_data, usable_T_prompt, *audio_out)) {
        return -4;
    }

    if (AudioBufferOutputLength) {
        *AudioBufferOutputLength = static_cast<int32_t>(audio_out->size());
    }

    if (has_value(OutputAudioPath)) {
        if (!s2::save_audio(std::string(OutputAudioPath), *audio_out,
                            Pipeline->output_sample_rate())) {
            return -6;
        }
    }

    return return_code;
}

int S2SynthesizeStreaming(s2::Pipeline* Pipeline,
                          const s2::GenerateParams* GenerateParams,
                          const S2StreamingCallbacks* StreamingCallbacks,
                          std::vector<int32_t>* ReferenceAudioPromptCodes,
                          int32_t* ReferenceAudioTPrompt,
                          const char* ReferenceAudioPath,
                          const char* ReferenceAudioTranscript,
                          const char* TextToInfer,
                          int32_t StreamDecodeStrideFrames) {
    S2StreamingParams streaming_params;
    streaming_params.stream_decode_stride_frames = StreamDecodeStrideFrames;
    return S2SynthesizeStreamingEx(Pipeline,
                                   GenerateParams,
                                   StreamingCallbacks,
                                   ReferenceAudioPromptCodes,
                                   ReferenceAudioTPrompt,
                                   ReferenceAudioPath,
                                   ReferenceAudioTranscript,
                                   TextToInfer,
                                   &streaming_params);
}

int S2SynthesizeStreamingEx(s2::Pipeline* Pipeline,
                            const s2::GenerateParams* GenerateParams,
                            const S2StreamingCallbacks* StreamingCallbacks,
                            std::vector<int32_t>* ReferenceAudioPromptCodes,
                            int32_t* ReferenceAudioTPrompt,
                            const char* ReferenceAudioPath,
                            const char* ReferenceAudioTranscript,
                            const char* TextToInfer,
                            const S2StreamingParams* StreamingParams) {
    if (!Pipeline || !Pipeline->is_initialized() || !GenerateParams) {
        return 0;
    }
    if (!StreamingCallbacks || !StreamingCallbacks->on_wav_chunk) {
        return -9;
    }

    s2::PipelineParams params;
    params.text = TextToInfer ? TextToInfer : "";
    params.prompt_text = ReferenceAudioTranscript ? ReferenceAudioTranscript : "";
    params.gen = *GenerateParams;
    apply_streaming_params(StreamingParams, params);
    if (StreamingParams) {
        apply_voice_selection(params, StreamingParams->voice, StreamingParams->voice_dir);
    }

    std::vector<int32_t> local_prompt_codes;
    int32_t local_T_prompt = 0;
    std::vector<int32_t>* prompt_codes =
        ReferenceAudioPromptCodes ? ReferenceAudioPromptCodes : &local_prompt_codes;
    int32_t* T_prompt = ReferenceAudioTPrompt ? ReferenceAudioTPrompt : &local_T_prompt;
    std::vector<int32_t> resolved_voice_codes;
    int32_t resolved_voice_t_prompt = 0;
    std::string effective_prompt_text = params.prompt_text;

    int return_code = 1;
    const bool has_reference_path = has_value(ReferenceAudioPath);
    const bool has_precomputed_reference =
        ReferenceAudioPromptCodes && !ReferenceAudioPromptCodes->empty();
    const bool has_voice_id = !params.voice_id.empty();

    if ((has_reference_path || has_precomputed_reference) && params.prompt_text.empty()) {
        return -7;
    }
    if (has_precomputed_reference && !ReferenceAudioTPrompt) {
        return -8;
    }

    if (has_reference_path && prompt_codes->empty()) {
        if (!Pipeline->encode_prompt_audio(ReferenceAudioPath, params.gen.n_threads,
                                           *prompt_codes, *T_prompt)) {
            return_code = -1;
        }
    }

    if (!has_reference_path && !has_precomputed_reference && has_voice_id) {
        s2::AudioData empty_ref_audio;
        double ref_encode_ms = 0.0;
        if (!Pipeline->resolve_prompt_reference(params,
                                               empty_ref_audio,
                                               resolved_voice_codes,
                                               resolved_voice_t_prompt,
                                               effective_prompt_text,
                                               ref_encode_ms)) {
            CallbackStreamingSink sink(StreamingCallbacks);
            sink.on_error("Failed to load saved voice profile");
            return -4;
        }
        prompt_codes = &resolved_voice_codes;
        T_prompt = &resolved_voice_t_prompt;
    }

    params.prompt_text = effective_prompt_text;

    const int32_t usable_T_prompt =
        (!prompt_codes->empty() && T_prompt) ? std::max<int32_t>(0, *T_prompt) : 0;
    const int32_t* prompt_data =
        (usable_T_prompt > 0 && !prompt_codes->empty()) ? prompt_codes->data() : nullptr;

    CallbackStreamingSink sink(StreamingCallbacks);
    bool ok = false;
    if (StreamingParams && StreamingParams->segment_sentences != 0) {
        const std::vector<std::string> segments = split_text_for_segmented_synthesis(
            params.text,
            StreamingParams->segment_max_chars > 0
                ? static_cast<size_t>(StreamingParams->segment_max_chars)
                : 0u);
        ok = synthesize_segmented_to_sink(*Pipeline,
                                          params,
                                          prompt_data,
                                          usable_T_prompt,
                                          segments,
                                          StreamingParams->sentence_pause_ms,
                                          sink);
    } else {
        ok = Pipeline->synthesize_streaming_with_prompt_codes(params, prompt_data,
                                                              usable_T_prompt, sink);
    }

    if (!ok) {
        if (sink.cancelled() || sink.aborted_by_callback()) {
            return -10;
        }
        return -4;
    }

    return return_code;
}
