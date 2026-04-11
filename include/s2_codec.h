#pragma once

#include "s2_backend.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace s2 {

class AudioCodec {
public:
    AudioCodec();
    ~AudioCodec();

    bool load(const std::string & gguf_path, int32_t gpu_device = -1, BackendType backend_type = BackendType::CPU);

    bool load_shared(gguf_context * gguf_ctx, const std::string & gguf_path, int32_t gpu_device = -1, BackendType backend_type = BackendType::CPU);

    bool read_tensor_data(const std::string & gguf_path, gguf_context * gguf_ctx);

    bool refresh_host_caches();

    ggml_context * weights_ctx() const;

    bool encode(const float * audio, int32_t n_samples, int32_t n_threads,
                std::vector<int32_t> & codes_out, int32_t & n_frames_out);

    bool decode(const int32_t * codes, int32_t n_frames, int32_t n_threads,
                std::vector<float> & audio_out);

    int32_t sample_rate()     const { return sample_rate_; }
    int32_t hop_length()      const { return hop_length_; }
    int32_t num_codebooks()   const { return num_codebooks_; }
    int32_t samples_per_code_frame() const { return samples_per_code_frame_; }
    int32_t streaming_history_frames() const { return streaming_history_frames_; }
    const char * backend_name() const;

    struct Impl;
    Impl * impl_ = nullptr;

private:

    int32_t sample_rate_    = 44100;
    int32_t hop_length_     = 512;
    int32_t num_codebooks_  = 10;
    int32_t samples_per_code_frame_ = 2048;
    int32_t streaming_history_frames_ = 160;
};

}
