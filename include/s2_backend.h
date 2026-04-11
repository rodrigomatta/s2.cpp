#pragma once

#include <cstdint>

namespace s2 {

enum class BackendType : int32_t {
    CPU = -1,
    Vulkan = 0,
    CUDA = 1,
    Metal = 2,
};

}
