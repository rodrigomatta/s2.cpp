#include "../include/s2_voice.h"
#include "../third_party/filesystem.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace fs = ghc::filesystem;

namespace s2 {

static const char MAGIC[8] = {'S', '2', 'V', 'O', 'I', 'C', 'E', '\0'};
static const uint32_t VERSION = 1;

bool VoiceProfile::save(const std::string & path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    out.write(MAGIC, sizeof(MAGIC));

    uint32_t version = VERSION;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&num_codebooks), sizeof(num_codebooks));
    out.write(reinterpret_cast<const char*>(&T_prompt), sizeof(T_prompt));
    out.write(reinterpret_cast<const char*>(&sample_rate), sizeof(sample_rate));
    out.write(reinterpret_cast<const char*>(&codebook_size), sizeof(codebook_size));

    const uint64_t transcript_len = static_cast<uint64_t>(transcript.size() + 1);
    out.write(reinterpret_cast<const char*>(&transcript_len), sizeof(transcript_len));

    const uint64_t codes_size = static_cast<uint64_t>(codes.size()) * sizeof(int32_t);
    out.write(reinterpret_cast<const char*>(&codes_size), sizeof(codes_size));

    out.write(transcript.c_str(), static_cast<std::streamsize>(transcript_len));
    out.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes_size));

    return out.good();
}

VoiceProfile VoiceProfile::load(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open voice profile: " + path);
    }

    char magic[8];
    in.read(magic, sizeof(magic));
    if (std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) {
        throw std::runtime_error("invalid voice profile magic");
    }

    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != VERSION) {
        throw std::runtime_error("unsupported voice profile version");
    }

    VoiceProfile profile;
    in.read(reinterpret_cast<char*>(&profile.num_codebooks), sizeof(profile.num_codebooks));
    in.read(reinterpret_cast<char*>(&profile.T_prompt), sizeof(profile.T_prompt));
    in.read(reinterpret_cast<char*>(&profile.sample_rate), sizeof(profile.sample_rate));
    in.read(reinterpret_cast<char*>(&profile.codebook_size), sizeof(profile.codebook_size));

    uint64_t transcript_len = 0;
    in.read(reinterpret_cast<char*>(&transcript_len), sizeof(transcript_len));

    uint64_t codes_size = 0;
    in.read(reinterpret_cast<char*>(&codes_size), sizeof(codes_size));

    if (transcript_len == 0) {
        throw std::runtime_error("invalid voice profile transcript length");
    }

    std::vector<char> transcript_buf(static_cast<size_t>(transcript_len));
    in.read(transcript_buf.data(), static_cast<std::streamsize>(transcript_len));
    if (transcript_buf.back() != '\0') {
        throw std::runtime_error("transcript not null-terminated");
    }
    profile.transcript = transcript_buf.data();

    const size_t n_codes = static_cast<size_t>(codes_size / sizeof(int32_t));
    profile.codes.resize(n_codes);
    in.read(reinterpret_cast<char*>(profile.codes.data()), static_cast<std::streamsize>(codes_size));

    if (!in) {
        throw std::runtime_error("truncated voice profile");
    }

    return profile;
}

bool VoiceProfile::is_compatible(int32_t expected_num_codebooks, int32_t expected_codebook_size,
                                 int32_t expected_sample_rate) const {
    return num_codebooks == expected_num_codebooks &&
           codebook_size == expected_codebook_size &&
           sample_rate == expected_sample_rate;
}

void VoiceProfileManager::set_storage_dir(const std::string & dir) {
    storage_dir_ = dir;
}

std::string VoiceProfileManager::get_path(const std::string & voice_id) const {
    fs::path dir(storage_dir_);
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
    return (dir / (voice_id + ".s2voice")).string();
}

bool VoiceProfileManager::save(const std::string & voice_id, const VoiceProfile & profile) {
    return profile.save(get_path(voice_id));
}

VoiceProfile VoiceProfileManager::load(const std::string & voice_id) {
    return VoiceProfile::load(get_path(voice_id));
}

bool VoiceProfileManager::remove(const std::string & voice_id) {
    const std::string path = get_path(voice_id);
    if (!fs::exists(path)) {
        return false;
    }
    return fs::remove(path);
}

std::vector<std::string> VoiceProfileManager::list() const {
    std::vector<std::string> result;
    if (!fs::exists(storage_dir_)) {
        return result;
    }

    for (const auto & entry : fs::directory_iterator(storage_dir_)) {
        if (entry.path().extension() == ".s2voice") {
            result.push_back(entry.path().stem().string());
        }
    }
    return result;
}

}
