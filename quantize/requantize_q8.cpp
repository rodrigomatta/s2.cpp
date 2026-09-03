// requantize_q8.cpp
//
// One-off utility: re-quantize every quantized tensor in a GGUF to q8_0,
// leaving floating-point / integer tensors untouched.
//
// Rationale: the S2 Pro Slow-AR FFN dominates tiny-batch (seq_len=1) inference
// and is bound by the q5_K dequantization in the DP4A MMVQ kernel. q8_0 has a
// much simpler dequant (int8 -> f32 * scale), which may net out faster despite
// reading ~1.45x more bytes per weight. This tool produces a q8_0 variant to
// benchmark against the shipped q5_k_m model.
//
// Build (VS2022 x64 host tools):
//   cl /std:c++17 /O2 /EHsc /I <ggml>/include requantize_q8.cpp \
//      <build>/ggml/src/ggml-base.lib /Fe:requantize_q8.exe
//
// Run (ensure ggml*.dll are on PATH):
//   requantize_q8.exe in.gguf out.gguf

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <input.gguf> <output.gguf>\n", argv[0]);
        return 1;
    }
    const char * in_fname  = argv[1];
    const char * out_fname = argv[2];

    // Load input: metadata + full tensor data into a ggml context.
    struct ggml_context * ctx_data = nullptr;
    struct gguf_init_params params = { /*no_alloc=*/false, /*ctx=*/&ctx_data };
    struct gguf_context * in = gguf_init_from_file(in_fname, params);
    if (!in) {
        std::fprintf(stderr, "failed to load %s\n", in_fname);
        return 1;
    }

    // New output context; copy all metadata KV pairs verbatim.
    struct gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);

    const int64_t n_tensors = gguf_get_n_tensors(in);

    // Keep re-quantized buffers alive until the write completes (gguf_set_tensor_data
    // stores a pointer; the writer copies from it only during gguf_write_to_file).
    std::vector<std::vector<uint8_t>> q8_buffers(n_tensors);

    size_t n_converted = 0;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(in, i);
        const enum ggml_type type = gguf_get_tensor_type(in, i);
        const struct ggml_tensor * t = ggml_get_tensor(ctx_data, name);
        if (!t) {
            std::fprintf(stderr, "tensor '%s' not found in ggml context\n", name);
            return 1;
        }

        const struct ggml_type_traits * traits = ggml_get_type_traits(type);

        if (traits->is_quantized && type != GGML_TYPE_Q8_0) {
            const int64_t nelements = ggml_nelements(t);

            // Dequantize to f32, then re-quantize to q8_0.
            std::vector<float> f32(nelements);
            traits->to_float(t->data, f32.data(), nelements);

            const size_t q8_nbytes = ggml_row_size(GGML_TYPE_Q8_0, nelements);
            q8_buffers[i].resize(q8_nbytes);
            ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(f32.data(), q8_buffers[i].data(), nelements);

            gguf_add_tensor(out, t);
            gguf_set_tensor_type(out, name, GGML_TYPE_Q8_0);
            gguf_set_tensor_data(out, name, q8_buffers[i].data());
            n_converted++;
        } else {
            // Keep as-is; t->data points into the loaded blob, still valid at write time.
            gguf_add_tensor(out, t);
            gguf_set_tensor_data(out, name, t->data);
        }
    }

    std::fprintf(stderr, "converted %zu / %lld tensors to q8_0\n", n_converted, (long long) n_tensors);

    if (!gguf_write_to_file(out, out_fname, false)) {
        std::fprintf(stderr, "failed to write %s\n", out_fname);
        return 1;
    }

    gguf_free(out);
    gguf_free(in);
    ggml_free(ctx_data);
    return 0;
}
