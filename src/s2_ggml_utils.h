#pragma once

#include "ggml.h"

#include <cstdio>

namespace s2 {

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
