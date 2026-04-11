#include "../include/s2_log.h"

#include <algorithm>

namespace s2 {
namespace {

std::atomic<int32_t> g_log_level{static_cast<int32_t>(LogLevel::Info)};

}

void set_log_level(LogLevel level) {
    const int32_t clamped = std::max(
        static_cast<int32_t>(LogLevel::Error),
        std::min(static_cast<int32_t>(LogLevel::Debug), static_cast<int32_t>(level)));
    g_log_level.store(clamped, std::memory_order_relaxed);
}

LogLevel get_log_level() {
    return static_cast<LogLevel>(g_log_level.load(std::memory_order_relaxed));
}

bool log_enabled(LogLevel level) {
    return static_cast<int32_t>(level) <= g_log_level.load(std::memory_order_relaxed);
}

}
