# AGENTS.md

## Build Commands

### Basic Build (CPU only)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```
For debug builds, use `-DCMAKE_BUILD_TYPE=Debug`.

### With Vulkan GPU support
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_VULKAN=ON
cmake --build build --parallel $(nproc)
```

### With CUDA GPU support
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_CUDA=ON
cmake --build build --parallel $(nproc)
```

### With Metal GPU support (macOS)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_METAL=ON
cmake --build build --parallel $(nproc)
```

### Clean build
```bash
rm -rf build
```

### Submodule initialization
If you cloned without `--recurse-submodules`, initialize and update submodules:
```bash
git submodule update --init --recursive
```

### Run the executable
```bash
./build/s2 -m model.gguf -t tokenizer.json -text "Hello world" -o output.wav
```

## Linting and Formatting

The project uses `clang-format` for code formatting. The ggml submodule provides a `.clang-format` configuration. To format all source files:

```bash
find src include -name '*.cpp' -o -name '*.h' | xargs clang-format -i
```

Alternatively, use the ggml's formatting script if available.

## Testing

No unit tests are currently defined in the main s2.cpp project. The ggml submodule has its own test suite, but it's disabled by default (`GGML_BUILD_TESTS OFF`). To enable and run ggml tests:

1. Set `GGML_BUILD_TESTS ON` in `CMakeLists.txt` (line 22) or via CMake command line:
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_TESTS=ON
   ```
2. Rebuild: `cmake --build build --parallel $(nproc)`
3. Run the ggml test executable: `./build/ggml/tests/test-*` (specific test binary) or use ctest if configured.

Alternatively, you can run the test suite via `ctest` after building:
```bash
cd build && ctest --output-on-failure
```

## Project Architecture Overview

s2.cpp implements a **Dual‑Autoregressive (Dual‑AR) text‑to‑speech inference engine** for Fish Audio's S2 Pro model. It is a pure C++17 GGML‑based pipeline that runs locally with CPU, Vulkan, or CUDA backends (no Python required).

### Core Pipeline
```
Text → Tokenizer → Prompt Builder → Slow‑AR Transformer → Fast‑AR Decoder → Audio Codec → WAV
```

### Key Components

1. **Tokenizer** (`s2_tokenizer.cpp`):  
   - BPE tokenizer reading HuggingFace `tokenizer.json` (Qwen3 with Byte‑Level pre‑tokenization).  
   - Handles special tokens (`<|im_start|>`, `<|semantic:N|>`, `<|voice|>`, etc.).

2. **Prompt Builder** (`s2_prompt.cpp`):  
   - Constructs `(num_codebooks + 1) × T` integer tensor combining text tokens and optional reference‑audio codes for voice cloning.

3. **Slow‑AR Model** (`s2_model.cpp`):  
   - 36‑layer Qwen3‑based transformer (4.13B params) with GQA, RoPE, QK‑norm, and KV cache.  
   - Processes semantic tokens; outputs hidden state and logits for next semantic token.  
   - **Operations**: `load()`, `init_kv_cache()`, `prefill()`, `step()`.

4. **Fast‑AR Decoder** (`s2_model.cpp`):  
   - 4‑layer transformer (0.42B params) that takes Slow‑AR hidden state plus prefix codebook tokens.  
   - Autoregressively predicts remaining acoustic codebook tokens (10 codebooks total).  
   - **Operation**: `fast_decode()`.

5. **Audio Codec** (`s2_codec.cpp`):  
   - Convolutional encoder/decoder with RVQ (10 codebooks × 4096 entries).  
   - Encodes reference audio to codes; decodes generated codes to 44.1 kHz mono waveform.  
   - Always runs on CPU (tiny workload).

6. **Generation Loop** (`s2_generate.cpp`):  
   - Manages the autoregressive loop: prefill → while not EOS → sample semantic token → fast‑decode codebooks → store frame → step.  
   - Implements **Repetition‑Aware Sampling (RAS)** and semantic‑mask enforcement.  
   - Uses top‑k + top‑p + temperature sampling matching Fish‑Speech.

7. **Pipeline** (`s2_pipeline.cpp`):  
   - Top‑level orchestrator: initializes tokenizer, model, codec; handles voice‑cloning flow; applies post‑processing (normalization, silence trimming).  
   - **HTTP server** (`s2_server.cpp`) exposes a `/generate` endpoint for remote synthesis.

### Dual‑AR Design Rationale
- **Slow‑AR**: models long‑range linguistic dependencies (one semantic token per ~21.5 ms frame).  
- **Fast‑AR**: models local acoustic correlations (10 codebook tokens per frame).  
- This separation drastically reduces per‑step FLOPs compared to a monolithic AR model over all codebooks.

### Memory & Execution Model
- Uses **GGML** tensors and allocators.  
- Separate allocators for: KV‑cache (persistent), Slow‑AR compute buffer, Fast‑AR compute buffer, prefill temporary buffer.  
- GPU backends run the transformer; codec stays on CPU.  
- **posix_fadvise(DONTNEED)** on Linux to drop GGUF file from page cache after loading weights.

### File Structure
```
include/                 # Headers (one per component)
src/                     # Implementations
├── s2_model.cpp        # Slow‑AR + Fast‑AR
├── s2_codec.cpp        # Audio codec
├── s2_tokenizer.cpp    # Tokenizer
├── s2_generate.cpp     # Generation loop
├── s2_prompt.cpp       # Prompt builder
├── s2_pipeline.cpp     # Top‑level pipeline
├── s2_sampler.cpp      # Sampling utilities
├── s2_audio.cpp        # WAV I/O & audio processing
├── s2_server.cpp       # HTTP server
└── main.cpp            # CLI entry‑point
```

### Important Data Structures
- `ModelHParams`: model hyper‑parameters (context length, vocab size, codebook size, etc.).  
- `PromptTensor`: `(num_codebooks+1, T)` integer matrix for model input.  
- `StepResult`: hidden state + logits from a Slow‑AR step.  
- `GenerateParams`: generation settings (temperature, top‑p, top‑k, max tokens, etc.).  

### Voice‑Cloning Flow
1. Load reference audio (WAV/MP3) → encode to codes via codec.  
2. Build prompt: `<|im_start|> <|voice|> transcript <|im_end|> reference‑codes <|im_start|> text <|im_end|>`.  
3. Model learns speaker’s voice from reference codes and transcript.  
4. **Voice profile persistence** (optional): encoded codes + transcript can be saved to a `.s2voice` binary file and reused later via `--voice <id>`. Profiles are stored in `./voices/` and checked for compatibility (codebook size, sample rate, num_codebooks).

### When Modifying
- The **GGUF file** contains both transformer weights and codec tensors (`c.*` prefix).  
- Adding new source files requires updating `CMakeLists.txt` `S2_SOURCES`.  
- Follow existing patterns for error handling (`bool` returns, `std::runtime_error` for fatal errors).  
- Use GPU backend guards (`#ifdef GGML_USE_VULKAN`, `GGML_USE_CUDA`).

## Code Style Guidelines

### Language Standard
- C++17
- Use standard library facilities where possible.

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `SlowARModel`, `Tokenizer`)
- **Structs**: `PascalCase` (e.g., `ModelHParams`, `StepResult`)
- **Functions**: `snake_case` (e.g., `load`, `prefill`, `fast_decode`)
- **Variables**: `snake_case` (e.g., `hparams_`, `max_seq_len_`)
- **Member variables**: suffix with underscore `_` (e.g., `backend_`, `ctx_kv_`)
- **Constants**: `snake_case` with `k` prefix? (not observed; seems to use `snake_case` for static constants)
- **Namespaces**: lowercase (e.g., `s2`)

### File Organization
- Header files in `include/` with `.h` extension.
- Source files in `src/` with `.cpp` extension.
- Each header should have `#pragma once` guard.
- Include order:
  1. Corresponding header (if in source file)
  2. Third-party library includes (e.g., `"../third_party/json.hpp"`)
  3. System includes (`<iostream>`, `<vector>`, etc.)
  4. Project includes (`"../include/..."`)

### Indentation and Formatting
- Indent with 4 spaces (no tabs).
- Opening braces on the same line as function/class/struct definition.
- Use spaces around operators.
- Line length: aim for 80-100 characters, but not strictly enforced.
- Use `//` for single-line comments, `/* */` for multi-line.

### Error Handling
- Use `bool` return values for operations that can fail (e.g., `load`, `prefill`).
- For unrecoverable errors (e.g., missing tensors), throw `std::runtime_error` with a descriptive message.
- Log errors to `stderr` using `std::fprintf(stderr, ...)` or `std::cerr`.

### Memory Management
- Use RAII; avoid raw `new`/`delete`.
- The project uses ggml's allocators (`ggml_context`, `ggml_backend_buffer_t`). Ensure proper cleanup in destructors.
- Use `std::vector` for dynamic arrays.

### Types
- Prefer `int32_t`, `uint32_t`, etc. from `<cstdint>` for fixed-width integers.
- Use `size_t` for sizes and indices.
- Use `float` for floating-point computations (ggml uses float).

### Includes
- Minimize includes in headers; forward declare where possible.
- Use C++ versions of C headers (e.g., `<cstdio>` instead of `<stdio.h>`).

### Example Code Snippet
```cpp
#include "../include/s2_model.h"
#include <cstdio>
#include <vector>

namespace s2 {

bool SlowARModel::load(const std::string & gguf_path, int32_t gpu_device, int32_t backend_type) {
    // implementation
    if (!success) {
        std::fprintf(stderr, "Failed to load model from %s\n", gguf_path.c_str());
        return false;
    }
    return true;
}

} // namespace s2
```

## Additional Notes

- The project depends on ggml as a submodule; do not modify ggml source files.
- When adding new source files, update `CMakeLists.txt` `S2_SOURCES` list.
- The codebase is cross-platform (Linux, Windows, macOS). Use preprocessor guards for platform-specific code (`#ifdef __linux__`, `#ifdef _WIN32`).
- For GPU backends, use `#ifdef GGML_USE_VULKAN` / `GGML_USE_CUDA` guards.

## Commit Guidelines

- Write concise commit messages in imperative mood.
- Ensure the build passes before committing.
- Format code with clang-format before committing.