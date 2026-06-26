#pragma once

#include "ggml.h"

#include <cstdio>
#include <cstdint>
#include <thread>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#elif defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <vector>
#elif defined(__linux__)
#  include <cstdio>
#  include <set>
#  include <utility>
#endif

namespace s2 {

// Number of physical "performance" cores available for compute.
//
// ggml's CPU backend scales poorly when oversubscribed onto hyperthreads or,
// on Apple Silicon, onto the slow efficiency cores: every thread in a fork/join
// matmul waits on the slowest one, so the whole step runs at efficiency-core
// speed. Measured on an M4 Pro (10P + 4E), defaulting to all 14 logical cores
// was ~2.9x slower end-to-end than capping at the 10 performance cores.
//
// This returns a sensible default core count; callers fall back to it only when
// the user did not pass an explicit --threads value.
inline int32_t detect_compute_threads() {
#if defined(__APPLE__)
    // Performance-core count (perflevel0). Falls back to all physical cores.
    int32_t n = 0;
    size_t  sz = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &sz, nullptr, 0) == 0 && n > 0) {
        return n;
    }
    n = 0; sz = sizeof(n);
    if (sysctlbyname("hw.physicalcpu", &n, &sz, nullptr, 0) == 0 && n > 0) {
        return n;
    }
#elif defined(_WIN32)
    // Count physical cores via processor-core relationships.
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    if (len > 0) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
            len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (GetLogicalProcessorInformation(buf.data(), &len)) {
            int32_t cores = 0;
            for (const auto & info : buf) {
                if (info.Relationship == RelationProcessorCore) cores++;
            }
            if (cores > 0) return cores;
        }
    }
#elif defined(__linux__)
    // Count distinct (physical_id, core_id) pairs from /proc/cpuinfo to skip
    // SMT siblings.
    if (FILE * f = std::fopen("/proc/cpuinfo", "r")) {
        std::set<std::pair<int, int>> cores;
        int phys = -1, core = -1;
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            int v;
            if (std::sscanf(line, "physical id : %d", &v) == 1) phys = v;
            else if (std::sscanf(line, "core id : %d", &v) == 1) core = v;
            if (line[0] == '\n') { // blank line separates logical CPUs
                if (phys >= 0 && core >= 0) cores.insert({phys, core});
                phys = core = -1;
            }
        }
        if (phys >= 0 && core >= 0) cores.insert({phys, core});
        std::fclose(f);
        if (!cores.empty()) return static_cast<int32_t>(cores.size());
    }
#endif
    // Generic fallback: assume SMT and halve the logical-core count, but never
    // go below 1.
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) return 4;
    return static_cast<int32_t>(hw >= 2 ? hw / 2 : hw);
}

// Resolve a caller-supplied thread count: honor explicit positive values,
// otherwise pick a good default for this machine.
inline int32_t resolve_n_threads(int32_t n_threads) {
    if (n_threads > 0) return n_threads;
    return detect_compute_threads();
}

inline ggml_tensor * repeat_checked(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                                    const char * label = "repeat") {
    if (!ggml_can_repeat(a, b)) {
        std::fprintf(stderr, "%s a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
            label,
            (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
            (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
        std::fflush(stderr);
    }
    return ggml_repeat(ctx, a, b);
}

inline ggml_tensor * mul_mat_checked(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                                     const char * label = "mul_mat") {
    const bool can_mul =
        a->ne[0] == b->ne[0] &&
        (b->ne[2] % a->ne[2] == 0) &&
        (b->ne[3] % a->ne[3] == 0);
    if (!can_mul || ggml_is_transposed(a)) {
        std::fprintf(stderr,
            "%s transposed=%d a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
            label, ggml_is_transposed(a) ? 1 : 0,
            (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
            (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
        std::fflush(stderr);
    }
    return ggml_mul_mat(ctx, a, b);
}

}
