#pragma once
// s2_generate.h — Autoregressive generation loop
//
// Port of generate() from ggml_pure.py.
// Combines Slow-AR prefill + step-by-step generation + Fast-AR decode.

#include "s2_model.h"
#include "s2_sampler.h"
#include "s2_tokenizer.h"
#include "s2_prompt.h"

#include <cstdint>
#include <vector>
#include <functional>

namespace s2 {

struct GenerateParams {
    int32_t max_new_tokens          = 2048;
    float   temperature             = 0.7f;
    float   top_p                   = 0.7f;
    int32_t top_k                   = 30;
    int32_t min_tokens_before_end   = 64;
    int32_t n_threads               = 4;
    int32_t seed                    = -1; // -1 = random
    bool    verbose                 = true;
};

// Generate VQ codes autoregressively.
// Returns flattened (num_codebooks, T_generated) codes in row-major order.
struct GenerateResult {
    std::vector<int32_t> codes;
    int32_t num_codebooks = 0;
    int32_t n_frames      = 0;
};

GenerateResult generate(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params
);

} // namespace s2
