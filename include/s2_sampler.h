#pragma once
// s2_sampler.h — Top-k / top-p / temperature sampling with RAS
//
// Pure C++ port of the numpy sampler from ggml_pure.py.

#include <cstdint>
#include <vector>
#include <random>

namespace s2 {

struct SamplerParams {
    float   temperature = 0.7f;
    float   top_p       = 0.7f;
    int32_t top_k       = 30;
};

// Sample a single token from logits using top-k + top-p + temperature.
// always_include_id: if >= 0 and has a finite logit, this token is guaranteed
// to survive both top-k and top-p truncation (used to ensure EOS is always
// reachable regardless of GPU numerical precision differences).
int32_t sample_token(const float * logits, int32_t vocab_size, const SamplerParams & params,
                     std::mt19937 & gen, int32_t always_include_id = -1);

// Repetition Aware Sampling (RAS):
// Tracks a window of recent tokens, resamples with high temp if repeating.
class RASSampler {
public:
    RASSampler(int32_t window_size = 10,
               float high_temp = 1.0f,
               float high_top_p = 0.9f);

    // Sample with RAS check. sem_begin/sem_end define the semantic token range.
    int32_t sample(const float * logits, int32_t vocab_size,
                   const SamplerParams & params,
                   std::mt19937 & gen,
                   int32_t sem_begin, int32_t sem_end);

    void reset();

private:
    int32_t window_size_;
    float   high_temp_;
    float   high_top_p_;
    std::vector<int32_t> window_;
};

} // namespace s2
