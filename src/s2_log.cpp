#include "../include/s2_log.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <mutex>

namespace s2 {
namespace {

std::atomic<int32_t> g_log_level{static_cast<int32_t>(LogLevel::Info)};

// Forward ggml's own logging through the s2 level filter. ggml's default
// callback ignores the level and prints everything, which floods stderr with
// per-frame debug noise (e.g. "CUDA Graph id N reused"). This hides ggml DEBUG
// messages unless the log level is raised to Debug.
void ggml_log_forward(enum ggml_log_level level, const char * text, void *) {
    LogLevel gate = LogLevel::Info;
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: gate = LogLevel::Debug;   break;
        case GGML_LOG_LEVEL_INFO:  gate = LogLevel::Info;    break;
        case GGML_LOG_LEVEL_WARN:  gate = LogLevel::Warning; break;
        case GGML_LOG_LEVEL_ERROR: gate = LogLevel::Error;   break;
        default:                   gate = LogLevel::Info;    break;
    }
    if (log_enabled(gate)) {
        std::fputs(text, stderr);
        std::fflush(stderr);
    }
}

void install_ggml_log_forwarding() {
    static std::once_flag once;
    std::call_once(once, []() { ggml_log_set(ggml_log_forward, nullptr); });
}

}

void set_log_level(LogLevel level) {
    const int32_t clamped = std::max(
        static_cast<int32_t>(LogLevel::Error),
        std::min(static_cast<int32_t>(LogLevel::Debug), static_cast<int32_t>(level)));
    g_log_level.store(clamped, std::memory_order_relaxed);
    install_ggml_log_forwarding();
}

LogLevel get_log_level() {
    return static_cast<LogLevel>(g_log_level.load(std::memory_order_relaxed));
}

bool log_enabled(LogLevel level) {
    return static_cast<int32_t>(level) <= g_log_level.load(std::memory_order_relaxed);
}

}
