#pragma once

#include "s2_tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace s2 {

struct PromptTensor {
    std::vector<int32_t> data;
    int32_t rows    = 0;
    int32_t cols    = 0;
};

PromptTensor build_prompt(
    const Tokenizer & tokenizer,
    const std::string & text,
    const std::string & prompt_text,
    const int32_t * prompt_codes,
    int32_t num_codebooks,
    int32_t T_prompt
);

}
