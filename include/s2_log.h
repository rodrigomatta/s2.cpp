#pragma once

#include <atomic>
#include <cstdint>
#include <iostream>
#include <string>

namespace s2 {

enum class LogLevel : int32_t {
    Error = 0,
    Warning = 1,
    Info = 2,
    Debug = 3,
};

void set_log_level(LogLevel level);
LogLevel get_log_level();
bool log_enabled(LogLevel level);

}

#define S2_LOG_INFO_STREAM(expr) \
    do { if (::s2::log_enabled(::s2::LogLevel::Info)) { std::cout << expr; } } while (false)

#define S2_LOG_WARN_STREAM(expr) \
    do { if (::s2::log_enabled(::s2::LogLevel::Warning)) { std::cerr << expr; } } while (false)
