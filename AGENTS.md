# AGENTS.md

## Scope

This repository contains the current `s2.cpp` C++17 / GGML implementation of
Fish Audio S2 Pro inference, including:

- CLI synthesis (`src/main.cpp`)
- HTTP server mode with multipart `/generate` (`src/s2_server.cpp`)
- Optional exported C ABI / shared library (`src/s2_export_api.cpp`)
- Voice profile persistence via `.s2voice` (`src/s2_voice.cpp`)

Use this file as the repo-local engineering reference for build, verification,
and architecture.

## Build Commands

### Clone with submodules

```bash
git clone --recurse-submodules https://github.com/rodrigomatta/s2.cpp.git
cd s2.cpp
```

If the repo was cloned without submodules:

```bash
git submodule update --init --recursive
```

### CPU-only build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel "$(nproc)"
```

For debug builds:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel "$(nproc)"
```

### Vulkan build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DS2_VULKAN=ON
cmake --build build --parallel "$(nproc)"
```

### CUDA build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DS2_CUDA=ON
cmake --build build --parallel "$(nproc)"
```

### Metal build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DS2_METAL=ON
cmake --build build --parallel "$(sysctl -n hw.ncpu)"
```

### Build shared/static library targets

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DS2_BUILD_SHARED_LIBRARIES=ON
cmake --build build --parallel "$(nproc)"
```

This builds:

- `s2` executable
- `s2_shared` with output name `s2`
- `s2_static` with output name `s2_static`

### Disable local ggml patch application

By default, configure/build applies local patches from `patches/*.patch` through
`cmake/apply_local_patches.cmake`.

To disable that behavior:

```bash
cmake -S . -B build -DS2_AUTO_APPLY_LOCAL_PATCHES=OFF
```

### Clean build

```bash
rm -rf build
```

### Basic run

```bash
./build/s2 \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --text "Hello world" \
  --output output.wav
```

### HTTP server mode

```bash
./build/s2 --model model.gguf --server
```

### Shared-library smoke check

If `S2_BUILD_SHARED_LIBRARIES=ON` is enabled:

```bash
python3 examples/python/ctypes_export_api.py --smoke-only --library build/libs2.so
```

Adjust the library path for macOS / Windows or custom build directories.

## Verification

There is no first-party automated test suite in the main `s2.cpp` target today.

For most changes, verify with:

1. A full configure + build of the relevant target set
2. `./build/s2 --help`
3. A real synthesis run if a local GGUF + tokenizer are available
4. If touching HTTP/OpenAPI docs, validate YAML loads:

```bash
python3 -c 'import yaml; yaml.safe_load(open("openapi/s2-openapi.yaml")); print("yaml-ok")'
```

5. If touching the exported ABI, run at least the Python ctypes smoke example

### About ggml tests

Top-level `CMakeLists.txt` currently forces:

- `GGML_BUILD_TESTS=OFF`
- `GGML_BUILD_EXAMPLES=OFF`

So ggml tests are not enabled through this repo's normal configure flow. If you
need ggml tests, patch the top-level CMake configuration or build ggml
separately.

## Formatting and Style

There is no repo-wide `.clang-format` checked in at the top level today.
Preserve the existing local style and keep diffs small.

Observed conventions in the current codebase:

- C++17
- 4-space indentation
- `PascalCase` for classes/structs
- `snake_case` for functions and variables
- member fields commonly use trailing `_`
- headers use `#pragma once`
- project code uses fixed-width integer types from `<cstdint>`
- error handling mixes `bool` returns for recoverable operations and
  `std::runtime_error` for hard failures

Use `apply_patch`-style minimal edits and avoid large formatting-only churn.

## Current Architecture

### High-level pipeline

```text
Text
  -> tokenizer
  -> prompt builder
  -> Slow-AR transformer
  -> Fast-AR acoustic/codebook decode
  -> audio codec decode
  -> WAV / PCM streaming / HTTP response
```

### Top-level surfaces

- CLI entry point: `src/main.cpp`
- HTTP server: `src/s2_server.cpp`
- Exported C ABI: `include/s2_export_api.h`, `src/s2_export_api.cpp`
- OpenAPI description: `openapi/s2-openapi.yaml`
- Language examples: `examples/python`, `examples/csharp`, `examples/golang`

### Key components

#### Tokenizer

Files:

- `include/s2_tokenizer.h`
- `src/s2_tokenizer.cpp`

Responsibilities:

- Load Hugging Face `tokenizer.json`
- Handle S2/Fish-style special tokens
- Provide tokenizer config used by prompting and generation

#### Prompt builder

Files:

- `include/s2_prompt.h`
- `src/s2_prompt.cpp`

Responsibilities:

- Build prompt tensors from target text
- Inject cloned/reference voice transcript and semantic prompt codes
- Support saved voice profiles and direct reference-audio cloning flows

#### Model

Files:

- `include/s2_model.h`
- `src/s2_model.cpp`

Responsibilities:

- Load transformer weights from GGUF
- Manage KV cache and prefill/step execution
- Run Slow-AR and Fast-AR parts of the model
- Support CPU, Vulkan, CUDA, and Metal backends through ggml

#### Generation loop

Files:

- `include/s2_generate.h`
- `src/s2_generate.cpp`
- `include/s2_sampler.h`
- `src/s2_sampler.cpp`

Responsibilities:

- Run the autoregressive loop
- Apply sampling controls: temperature, top-p, top-k, EOS floor
- Emit codebook frames for offline and streaming decode paths

#### Audio codec

Files:

- `include/s2_codec.h`
- `src/s2_codec.cpp`

Responsibilities:

- Encode reference audio into prompt codes
- Decode generated code frames back to waveform
- Load codec tensors from the same GGUF file
- Run on CPU by default, or optionally follow / benchmark GPU backends

Important current behavior:

- The codec is no longer "always CPU"
- `PipelineParams.codec_auto_backend` and `codec_follow_backend` control whether
  the codec stays on CPU, follows the selected backend, or benchmarks CPU vs GPU
  for best throughput
- If GPU codec init/allocation fails, the runtime falls back to CPU

#### Pipeline orchestration

Files:

- `include/s2_pipeline.h`
- `src/s2_pipeline.cpp`

Responsibilities:

- Own or bind tokenizer/model/codec components
- Load shared GGUF state across model + codec
- Select codec backend
- Run offline synthesis, streaming synthesis, and in-memory synthesis
- Handle voice profile save/load compatibility checks
- Apply post-processing: trim, normalize, dynamic normalize
- On Linux, call `posix_fadvise(..., POSIX_FADV_DONTNEED)` after loading weights

#### Voice profiles

Files:

- `include/s2_voice.h`
- `src/s2_voice.cpp`

Responsibilities:

- Persist reusable `.s2voice` files
- Store transcript, prompt codes, codebook count, prompt length, sample rate,
  and codebook size
- Load saved profiles and verify compatibility with the current model/codec

Current user-facing flows:

- `--voice <id-or-path.s2voice>`
- `--save-voice`
- `--voice-dir`
- `--list-voices`

Note: internal remove support exists in `VoiceProfileManager`, but there is no
user-facing CLI command for profile deletion today.

#### HTTP server

Files:

- `include/s2_server.h`
- `src/s2_server.cpp`
- `openapi/s2-openapi.yaml`

Responsibilities:

- Serve `POST /generate`
- Accept multipart form fields for text, reference audio, saved voice ids, and
  JSON params
- Support one-shot WAV, finalized streaming WAV, chunked live WAV, and raw
  `pcm_s16le` transport
- Support sentence-segmented synthesis and low-latency streaming presets

Important current behavior:

- Only one synthesis request is processed at a time
- Concurrent requests return HTTP `503`
- Request validation errors return HTTP `400`
- Streaming responses may terminate early if synthesis fails after the body has
  already started

#### Exported C ABI

Files:

- `include/s2_export_api.h`
- `src/s2_export_api.cpp`

Responsibilities:

- Expose pipeline/model/tokenizer/codec allocation and initialization
- Provide one-shot synthesis and callback-based streaming APIs
- Expose `S2StreamingParams` with low-latency, sentence segmentation, and voice
  selection support

Use the examples under `examples/` as the current reference clients.

## Important Data Structures

- `s2::PipelineParams`: top-level runtime configuration for CLI/server/library
- `s2::GenerateParams`: sampling and generation parameters
- `s2::VoiceProfile`: serialized saved voice profile payload
- `S2StreamingParams`: exported C ABI streaming controls
- `s2::ServerParams`: HTTP server bind + pipeline defaults

## File Layout

```text
include/              Public/internal headers
src/                  Core implementation
openapi/              OpenAPI 3.1 description and notes for HTTP mode
examples/             Python, C#, and Go library/API examples
cmake/                Build helpers
patches/              Local patches applied to ggml during build
ggml/                 Bundled submodule dependency
```

## When Modifying

- If you add a new core source file, update `CMakeLists.txt`
  (`S2_CORE_SOURCES` and, if exported, `S2_LIBRARY_SOURCES`)
- Keep OpenAPI + README in sync with `src/s2_server.cpp`
- Keep exported ABI docs/examples in sync with `include/s2_export_api.h`
- Do not silently change `.s2voice` binary compatibility without updating the
  versioned format in `src/s2_voice.cpp`
- Avoid editing `ggml/` directly unless the change is intended for the submodule
  or represented as a local patch under `patches/`

## Practical Change Guidance

If you touch:

- CLI flags: update `src/main.cpp` help text and `README.md`
- HTTP request/response behavior: update `src/s2_server.cpp`, `openapi/`, and
  `README.md`
- voice profile format: update `include/s2_voice.h`, `src/s2_voice.cpp`,
  loading compatibility checks, and docs
- exported ABI: update `include/s2_export_api.h`, `src/s2_export_api.cpp`, and
  at least the Python example

## Commit Guidance

- Keep commit messages short and imperative
- Prefer commits that leave the repo building
- Verify the concrete surface you changed instead of relying on docs-only claims
