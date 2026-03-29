#pragma once

#include "s2_audio.h"
#include "s2_codec.h"
#include "s2_generate.h"
#include "s2_model.h"
#include "s2_tokenizer.h"
#include "s2_voice.h"

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
    int32_t backend_type = -1; //0 = Vulkan; 1 = Cuda;
    bool trim_silence = false;
    bool normalize_output = false;
    bool normalize_dynamic = false;
    
    // Voice persistence
    std::string voice_id;           // load saved voice profile
    bool save_voice = false;        // save encoded voice profile after cloning
    std::string voice_storage_dir = "./voices"; // where profiles are stored
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    bool init(const PipelineParams & params);
    bool synthesize(const PipelineParams & params);

    bool synthesize_to_memory(const PipelineParams & params, void** ref_audio_buffer, size_t* ref_audio_size, void** wav_buffer, size_t* wav_size);
    bool synthesize_raw(const PipelineParams & params, AudioData & ref_audio, std::vector<float> & audio_out);

private:
    Tokenizer   tokenizer_;
    SlowARModel model_;
    AudioCodec  codec_;
    VoiceProfileManager voice_mgr_;
    mutable std::mutex synthesize_mutex_;
    bool initialized_ = false;
    
    // Save voice profile from encoded codes and transcript
    bool save_voice_profile(const std::string & voice_id, 
                           const std::vector<int32_t> & codes, 
                           int32_t T_prompt,
                           const std::string & transcript,
                           const PipelineParams & params);
};

}
