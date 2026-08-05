#pragma once

#include <string>
#include <cstddef>
#include <cstdint>

namespace s2 {

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile() { close(); }
    
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    bool open(const std::string& path);
    void close();
    void drop_page_cache();
    void warm_page_cache();
    
    bool is_open() const { return data_ != nullptr; }
    const uint8_t* data() const { return static_cast<const uint8_t*>(data_); }
    size_t size() const { return size_; }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
    
#ifdef _WIN32
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}
