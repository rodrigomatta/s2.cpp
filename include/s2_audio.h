#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../third_party/dr_wav.h"

namespace s2 {

struct AudioData {
    std::vector<float> samples;
    int32_t            sample_rate = 0;
};

bool audio_read(const std::string & path, AudioData & out);
bool audio_read_from_memory(const void * in_data, size_t in_data_size, AudioData & out);

bool audio_write_wav(const std::string & path, const float * data, size_t n_samples, int32_t sample_rate);
bool audio_write_memory_wav(void ** pWavData, size_t * pWavSize, const float * data, size_t n_samples, int32_t sample_rate);
void audio_free_memory_wav(void** pWavData, size_t* pWavSize, const drwav_allocation_callbacks* pAllocationCallbacks);

bool audio_write_streaming_wav_header(uint8_t* out, int32_t sample_rate,
                                       int32_t channels = 1, int32_t bits_per_sample = 16);

std::vector<int16_t> audio_to_pcm16(const float * data, size_t n_samples);

std::vector<float> audio_resample(const float * data, size_t n_samples, int32_t src_rate, int32_t dst_rate);
std::vector<float> audio_normalize_dynamic(const float * data, size_t n_samples, int32_t sample_rate,
                                           float window_sec = 1.0f, float target_rms = 0.0f);
std::vector<float> audio_trim_trailing_silence(const float * data, size_t n_samples,
                                              int32_t sample_rate,
                                              float threshold = 0.01f,
                                              float min_silence_duration = 0.08f);

bool load_audio(const std::string & path, AudioData & out, int32_t target_sample_rate = 0);
bool load_audio_from_memory(const void * data, size_t bytes, AudioData & out, int32_t target_sample_rate = 0);
bool save_audio(const std::string & path,
                const std::vector<float> & data,
                int32_t sample_rate,
                bool trim_silence = true,
                bool normalize_peak = true);

}
