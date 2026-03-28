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