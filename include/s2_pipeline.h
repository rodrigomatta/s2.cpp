#pragma once

#include "s2_audio.h"
#include "s2_codec.h"
#include "s2_generate.h"
#include "s2_model.h"
#include "s2_tokenizer.h"

#include <cstdint>
#include <string>
#include <mutex>

namespace s2 {

struct PipelineParams {
    std::string model_path;
    std::string tokenizer_path;
    std::string text;
    std::string prompt_text;
    std::string prompt_audio_path;
    std::string output_path;
    GenerateParams gen;
    int32_t gpu_device = -1;   // -1 = CPU only
    int32_t backend_type = -1; //0 = Vulkan; 1 = Cuda; 2 = Metal;
    bool trim_silence = false;
    bool normalize_output = false;
    bool normalize_dynamic = false;
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    bool init(const PipelineParams & params);
    bool synthesize(const PipelineParams & params);

    bool synthesize_to_memory(const PipelineParams & params, void** ref_audio_buffer, size_t* ref_audio_size, void** wav_buffer, size_t* wav_size);
    bool synthesize_raw(const PipelineParams & params, AudioData & ref_audio, std::vector<float> & audio_out, int32_t* audio_out_length);

//private:
    Tokenizer   tokenizer_;
    SlowARModel model_;
    AudioCodec  codec_;
    mutable std::mutex synthesize_mutex_;
    bool initialized_ = false;

    // Cached reference voice (pre-encoded from -pa / -pt at startup)
    std::vector<int32_t> ref_codes_cache_;
    int32_t              T_prompt_cache_ = 0;
    std::string          prompt_text_cache_;
};

}
