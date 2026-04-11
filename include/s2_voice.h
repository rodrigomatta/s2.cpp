#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace s2 {

struct VoiceProfile {
    std::string transcript;
    std::vector<int32_t> codes;
    int32_t num_codebooks = 0;
    int32_t T_prompt = 0;
    int32_t sample_rate = 44100;
    int32_t codebook_size = 4096;

    bool save(const std::string & path) const;
    static VoiceProfile load(const std::string & path);
    bool is_compatible(int32_t expected_num_codebooks, int32_t expected_codebook_size,
                       int32_t expected_sample_rate = 44100) const;
};

class VoiceProfileManager {
public:
    VoiceProfileManager() = default;

    void set_storage_dir(const std::string & dir);
    bool save(const std::string & voice_id, const VoiceProfile & profile);
    VoiceProfile load(const std::string & voice_id);
    bool remove(const std::string & voice_id);
    std::vector<std::string> list() const;

private:
    std::string storage_dir_ = "./voices";

    std::string get_path(const std::string & voice_id) const;
};

}
