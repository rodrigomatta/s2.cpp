#pragma once
// s2_pipeline.h — End-to-end TTS pipeline
//
// Orchestrates: tokenize → encode reference → build prompt → generate → decode → WAV

#include "s2_audio.h"
#include "s2_codec.h"
#include "s2_generate.h"
#include "s2_model.h"
#include "s2_tokenizer.h"

#include <cstdint>
#include <string>

namespace s2 {

struct PipelineParams {
    // Paths
    std::string model_path;       // unified GGUF
    std::string tokenizer_path;   // tokenizer.json

    // Input
    std::string text;
    std::string prompt_text;
    std::string prompt_audio_path;
    std::string output_path;

    // Generation
    GenerateParams gen;

    // Backend
    int32_t gpu_device = -1;   // -1 = CPU only
    int32_t backend_type = -1; //0 = Vulkan; 1 = Cuda;
    bool split_sentences = true; // Always true, kept for compatibility
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    // Load model + tokenizer + codec
    bool init(const PipelineParams & params);

    // Run synthesis: text (+ optional reference audio) → WAV
    bool synthesize(const PipelineParams & params);

private:
    Tokenizer   tokenizer_;
    SlowARModel model_;
    AudioCodec  codec_;
    bool initialized_ = false;

    std::vector<std::string> split_text_into_chunks(const std::string & text, int32_t max_new_tokens, bool split_sentences);
};

} // namespace s2
