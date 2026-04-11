#pragma once

#include "s2_audio.h"
#include "s2_backend.h"
#include "s2_codec.h"
#include "s2_generate.h"
#include "s2_model.h"
#include "s2_tokenizer.h"
#include "s2_voice.h"

#include <cstdint>
#include <string>
#include <mutex>

namespace s2 {

class StreamingSink {
public:
    virtual ~StreamingSink() = default;
    virtual bool on_pcm_data(const float* data, size_t n_samples) = 0;
    virtual bool on_header(const uint8_t* header, size_t size) = 0;
    virtual void on_done() = 0;
    virtual void on_error(const std::string& message) = 0;
    virtual bool is_cancelled() const { return false; }
};

struct PipelineParams {
    std::string model_path;
    std::string tokenizer_path;
    std::string text;
    std::string prompt_text;
    std::string prompt_audio_path;
    std::string output_path;
    GenerateParams gen;
    int32_t gpu_device = -1;
    BackendType backend_type = BackendType::CPU;
    int32_t n_gpu_layers = -1;
    bool codec_auto_backend = true;
    bool codec_follow_backend = true;
    int32_t stream_decode_stride_frames = 0;
    int32_t stream_holdback_frames = -1;
    int32_t codec_decode_context_frames = -1;
    bool trim_silence = false;
    bool normalize_output = false;
    bool normalize_dynamic = false;
    std::string voice_id;
    bool save_voice = false;
    std::string voice_storage_dir = "./voices";
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    bool init(const PipelineParams & params);
    bool init_from_components(Tokenizer* tokenizer, SlowARModel* model, AudioCodec* codec);
    bool is_initialized() const { return initialized_; }
    bool synthesize(const PipelineParams & params);

    bool synthesize_to_memory(const PipelineParams & params, const void * ref_audio_buffer, size_t ref_audio_size, void** wav_buffer, size_t* wav_size);
    bool synthesize_raw(const PipelineParams & params, AudioData & ref_audio, std::vector<float> & audio_out);
    bool synthesize_with_prompt_codes(const PipelineParams & params, const int32_t* ref_codes,
                                      int32_t T_prompt, std::vector<float> & audio_out);
    bool resolve_prompt_reference(const PipelineParams & params, AudioData & ref_audio,
                                  std::vector<int32_t> & ref_codes, int32_t & T_prompt,
                                  std::string & effective_prompt_text, double & ref_encode_ms);
    bool encode_prompt_audio(const std::string & audio_path, int32_t n_threads,
                             std::vector<int32_t> & codes_out, int32_t & n_frames_out);
    bool encode_prompt_audio_data(const AudioData & ref_audio, int32_t n_threads,
                                  std::vector<int32_t> & codes_out, int32_t & n_frames_out);
    int32_t output_sample_rate() const;

    bool synthesize_streaming_raw(const PipelineParams & params, AudioData & ref_audio,
                                   StreamingSink & sink);
    bool synthesize_streaming_with_prompt_codes(const PipelineParams & params, const int32_t* ref_codes,
                                                int32_t T_prompt, StreamingSink & sink);
    bool synthesize_streaming_file(const PipelineParams & params);

private:
    Tokenizer& tokenizer() { return *tokenizer_ref_; }
    const Tokenizer& tokenizer() const { return *tokenizer_ref_; }
    SlowARModel& model() { return *model_ref_; }
    const SlowARModel& model() const { return *model_ref_; }
    AudioCodec& codec() { return *codec_ref_; }
    const AudioCodec& codec() const { return *codec_ref_; }

    bool resolve_reference_prompt_locked(const PipelineParams & params, AudioData & ref_audio,
                                         std::vector<int32_t> & ref_codes, int32_t & T_prompt,
                                         std::string & effective_prompt_text,
                                         double & ref_encode_ms);
    bool save_voice_profile_locked(const std::string & voice_id,
                                   const std::vector<int32_t> & codes,
                                   int32_t T_prompt,
                                   const std::string & transcript,
                                   const PipelineParams & params);
    bool synthesize_prompt_codes_locked(const PipelineParams & params, const int32_t* ref_codes,
                                        int32_t T_prompt, std::vector<float> & audio_out,
                                        double ref_encode_ms);
    bool synthesize_streaming_prompt_codes_locked(const PipelineParams & params, const int32_t* ref_codes,
                                                  int32_t T_prompt, StreamingSink & sink,
                                                  double ref_encode_ms);

    Tokenizer   owned_tokenizer_;
    SlowARModel owned_model_;
    AudioCodec  owned_codec_;
    VoiceProfileManager voice_mgr_;
    Tokenizer* tokenizer_ref_ = &owned_tokenizer_;
    SlowARModel* model_ref_ = &owned_model_;
    AudioCodec* codec_ref_ = &owned_codec_;
    mutable std::mutex synthesize_mutex_;
    bool initialized_ = false;
};

}
