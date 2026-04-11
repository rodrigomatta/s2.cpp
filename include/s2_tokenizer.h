#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

namespace s2 {

struct TokenizerConfig {
    int32_t im_start_id       = 0;
    int32_t im_end_id         = 0;
    int32_t voice_id          = 0;
    int32_t semantic_begin_id = 0;
    int32_t semantic_end_id   = 0;
    int32_t num_codebooks     = 10;
    int32_t codebook_size     = 4096;
    int32_t vocab_size        = 155776;
};

class Tokenizer {
public:
    Tokenizer() = default;

    bool load(const std::string & path);

    std::vector<int32_t> encode(const std::string & text) const;

    int32_t token_to_id(const std::string & token) const;

    bool is_loaded() const { return loaded_; }

    const TokenizerConfig & config() const { return config_; }
    TokenizerConfig & config() { return config_; }

private:
    bool loaded_ = false;
    TokenizerConfig config_;

    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int32_t> merge_rank_;

    std::vector<std::pair<std::string, int32_t>> special_tokens_;

    std::vector<int32_t> bpe_encode_word(const std::string & word) const;
};

}
