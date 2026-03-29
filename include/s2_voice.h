#pragma once
// s2_voice.h — Voice profile persistence for S2 voice cloning
//
// Stores encoded reference codes and transcript, allowing reuse of a
// cloned voice without re‑encoding the reference audio each time.
//
// Binary file format (.s2voice) — little‑endian, portable across x86/ARM:
//   Offset  Size       Content
//   0       8 bytes    Magic header: 'S','2','V','O','I','C','E','\0'
//   8       4 bytes    Version (uint32_t, currently 1)
//   12      4 bytes    num_codebooks (int32_t)
//   16      4 bytes    T_prompt (int32_t)
//   20      4 bytes    sample_rate (int32_t)
//   24      4 bytes    codebook_size (int32_t)
//   28      8 bytes    transcript_len (uint64_t, includes null terminator)
//   36      8 bytes    codes_size (uint64_t, bytes = codes.size() * sizeof(int32_t))
//   44      transcript_len bytes   Transcript (null‑terminated C‑string)
//   …       codes_size bytes      Code data (row‑major int32_t array)
//
// Compatibility: a profile must match the current model's num_codebooks,
// codebook_size, and sample_rate.

#include <cstdint>
#include <string>
#include <vector>

namespace s2 {

struct VoiceProfile {
    std::string transcript;
    std::vector<int32_t> codes;        // row‑major: (num_codebooks, T_prompt)
    int32_t num_codebooks = 0;
    int32_t T_prompt = 0;
    int32_t sample_rate = 44100;
    int32_t codebook_size = 4096;
    
    // Metadata
    std::string model_hash;            // optional identifier of the source model
    std::string timestamp;
    
    // Save to file
    bool save(const std::string & path) const;
    
    // Load from file
    static VoiceProfile load(const std::string & path);
    
    // Check compatibility with current codec
    bool is_compatible(int32_t expected_num_codebooks, int32_t expected_codebook_size,
                       int32_t expected_sample_rate = 44100) const;
};

class VoiceProfileManager {
public:
    VoiceProfileManager() = default;
    
    // Set storage directory (default: "./voices")
    void set_storage_dir(const std::string & dir);
    
    // Save profile with given ID
    bool save(const std::string & voice_id, const VoiceProfile & profile);
    
    // Load profile by ID (searches storage directory)
    VoiceProfile load(const std::string & voice_id);
    
    // Delete profile
    bool remove(const std::string & voice_id);
    
    // List available voice IDs
    std::vector<std::string> list() const;
    
private:
    std::string storage_dir_ = "./voices";
    
    std::string get_path(const std::string & voice_id) const;
};

} // namespace s2