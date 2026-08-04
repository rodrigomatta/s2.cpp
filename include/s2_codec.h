#pragma once

#include "s2_backend.h"
#include "s2_mapped_file.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "s2_model.h"

namespace s2 {

class AudioCodec {
public:
    AudioCodec();
    ~AudioCodec();

    bool load(const std::string & gguf_path, int32_t gpu_device = -1, BackendType backend_type = BackendType::CPU);

    bool load_shared(SlowARModel* Model, gguf_context * gguf_ctx, const std::string & gguf_path, int32_t gpu_device = -1, BackendType backend_type = BackendType::CPU);

    ggml_context * weights_ctx() const;

    bool encode(const float * audio, int32_t n_samples, int32_t n_threads,
                std::vector<int32_t> & codes_out, int32_t & n_frames_out);

    bool decode(const int32_t * codes, int32_t n_frames, int32_t n_threads,
        std::vector<float> & audio_out);

    void clear_decode_cache();

    MappedFile& mapped_file();

    bool restore_weights_to_gpu();

    bool free_gpu_weights();

    bool free_encoder_weights();
    bool restore_encoder_weights();
    bool free_decoder_weights();
    bool restore_decoder_weights();
    bool is_encoder_on_gpu() const;
    bool is_decoder_on_gpu() const;
    size_t get_encoder_gpu_bytes() const;
    size_t get_decoder_gpu_bytes() const;

    bool is_weights_on_gpu() const;

    bool refresh_host_caches_from_mmap();

    bool ensure_weights_loaded();

    size_t get_gpu_memory_usage_bytes() const;

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
