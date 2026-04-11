
#include "../include/s2_tokenizer.h"
#include "../third_party/json.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>

using json = nlohmann::json;

namespace s2 {

static const uint32_t * byte_to_cp_table() {
    static uint32_t table[256] = {};
    static bool initialized = false;
    if (!initialized) {
        initialized = true;

        for (int b = 0x21; b <= 0x7E; ++b) table[b] = (uint32_t)b;
        for (int b = 0xA1; b <= 0xAC; ++b) table[b] = (uint32_t)b;
        for (int b = 0xAE; b <= 0xFF; ++b) table[b] = (uint32_t)b;

        uint32_t extra = 0x0100;
        for (int b = 0x00; b <= 0x20; ++b) table[b] = extra++;
        for (int b = 0x7F; b <= 0xA0; ++b) table[b] = extra++;
        table[0xAD] = extra;
    }
    return table;
}

static std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

static std::string to_byte_level(const std::string & raw) {
    const uint32_t * tbl = byte_to_cp_table();
    std::string result;
    for (unsigned char b : raw) {
        result += cp_to_utf8(tbl[b]);
    }
    return result;
}

static std::vector<std::string> utf8_chars(const std::string & text) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = 1;
        unsigned char c = static_cast<unsigned char>(text[i]);
        if      ((c & 0xF8) == 0xF0) len = 4;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xE0) == 0xC0) len = 2;
        if (i + len > text.size()) len = 1;
        chars.push_back(text.substr(i, len));
        i += len;
    }
    return chars;
}

static std::vector<std::string> split_on_specials(
    const std::string & text,
    const std::vector<std::pair<std::string, int32_t>> & specials,
    std::vector<int32_t> & special_ids
) {
    struct Match { size_t pos; size_t len; int32_t id; };

    std::vector<Match> matches;
    for (const auto & sp : specials) {
        size_t pos = 0;
        while ((pos = text.find(sp.first, pos)) != std::string::npos) {
            matches.push_back({pos, sp.first.size(), sp.second});
            pos += sp.first.size();
        }
    }

    std::sort(matches.begin(), matches.end(), [](const Match & a, const Match & b) {
        return a.pos < b.pos || (a.pos == b.pos && a.len > b.len);
    });

    std::vector<Match> filtered;
    size_t last_end = 0;
    for (const auto & m : matches) {
        if (m.pos >= last_end) {
            filtered.push_back(m);
            last_end = m.pos + m.len;
        }
    }

    std::vector<std::string> segments;
    special_ids.clear();
    size_t pos = 0;
    for (const auto & m : filtered) {
        if (m.pos > pos) {
            segments.push_back(text.substr(pos, m.pos - pos));
            special_ids.push_back(-1);
        }
        segments.push_back(text.substr(m.pos, m.len));
        special_ids.push_back(m.id);
        pos = m.pos + m.len;
    }
    if (pos < text.size()) {
        segments.push_back(text.substr(pos));
        special_ids.push_back(-1);
    }

    return segments;
}

static std::vector<std::string> pre_tokenize(const std::string & text) {
    std::vector<std::string> words;
    std::string current;
    auto chars = utf8_chars(text);
    for (const auto & ch : chars) {
        if (ch.size() == 1 && (ch[0] == ' ' || ch[0] == '\t' || ch[0] == '\n' || ch[0] == '\r')) {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            current = ch;
        } else {
            current += ch;
        }
    }
    if (!current.empty()) {
        words.push_back(current);
    }
    return words;
}

std::vector<int32_t> Tokenizer::bpe_encode_word(const std::string & word) const {
    if (word.empty()) return {};

    const std::string bl = to_byte_level(word);

    auto it = vocab_.find(bl);
    if (it != vocab_.end()) {
        return {it->second};
    }

    std::vector<std::string> symbols = utf8_chars(bl);

    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_pos = std::string::npos;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string pair = symbols[i] + symbols[i + 1];
            auto it2 = merge_rank_.find(pair);
            if (it2 != merge_rank_.end() && it2->second < best_rank) {
                best_rank = it2->second;
                best_pos = i;
            }
        }

        if (best_pos == std::string::npos) break;

        symbols[best_pos] += symbols[best_pos + 1];
        symbols.erase(symbols.begin() + static_cast<long>(best_pos) + 1);
    }

    std::vector<int32_t> ids;
    for (const auto & sym : symbols) {
        auto it2 = vocab_.find(sym);
        if (it2 != vocab_.end()) {
            ids.push_back(it2->second);
        }
    }
    return ids;
}

bool Tokenizer::load(const std::string & path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[s2_tokenizer] failed to open: %s\n", path.c_str());
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const json::parse_error & e) {
        std::fprintf(stderr, "[s2_tokenizer] JSON parse error: %s\n", e.what());
        return false;
    }

    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto & tok : j["added_tokens"]) {
            std::string content = tok.value("content", "");
            int32_t id = tok.value("id", -1);
            if (!content.empty() && id >= 0) {
                vocab_[content] = id;
                if (tok.value("special", false)) {
                    special_tokens_.push_back({content, id});
                }
            }
        }
    }

    if (j.contains("model") && j["model"].contains("vocab")) {
        for (auto it2 = j["model"]["vocab"].begin(); it2 != j["model"]["vocab"].end(); ++it2) {
            int32_t id = it2.value().get<int32_t>();
            const std::string & key = it2.key();
            if (vocab_.find(key) == vocab_.end()) {
                vocab_[key] = id;
            }
        }
    }

    if (j.contains("model") && j["model"].contains("merges")) {
        int32_t rank = 0;
        for (const auto & merge_item : j["model"]["merges"]) {
            if (merge_item.is_string()) {
                std::string m = merge_item.get<std::string>();
                size_t sp = m.find(' ');
                if (sp != std::string::npos) {
                    std::string a = m.substr(0, sp);
                    std::string b = m.substr(sp + 1);
                    merge_rank_[a + b] = rank++;
                }
            } else if (merge_item.is_array() && merge_item.size() >= 2) {
                std::string a = merge_item[0].get<std::string>();
                std::string b = merge_item[1].get<std::string>();
                merge_rank_[a + b] = rank++;
            }
        }
    }

    std::sort(special_tokens_.begin(), special_tokens_.end(),
        [](const std::pair<std::string, int32_t> & a, const std::pair<std::string, int32_t> & b) {
            return a.first.size() > b.first.size();
        }
    );

    config_.im_start_id       = token_to_id("<|im_start|>");
    config_.im_end_id         = token_to_id("<|im_end|>");
    config_.voice_id          = token_to_id("<|voice|>");

    return true;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    if (text.empty()) return {};

    std::vector<int32_t> special_ids;
    std::vector<std::string> segments = split_on_specials(text, special_tokens_, special_ids);

    std::vector<int32_t> ids;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (special_ids[i] != -1) {
            ids.push_back(special_ids[i]);
        } else if (!segments[i].empty()) {
            std::vector<std::string> words = pre_tokenize(segments[i]);
            for (const auto & w : words) {
                std::vector<int32_t> w_ids = bpe_encode_word(w);
                ids.insert(ids.end(), w_ids.begin(), w_ids.end());
            }
        }
    }
    return ids;
}

int32_t Tokenizer::token_to_id(const std::string & token) const {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) {
        return it->second;
    }
    return -1;
}

}
