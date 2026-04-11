# Changelog

## 2026-04-11

This entry separates:

- branch-comparison findings for `kolulu23`, `felipeov1`, and `subspecs`
- later changes observed in the current `maio` branch state

### kolulu23 - Voice Cloning Persistence

Main feature:

- **Voice cloning persistence** - Save, reuse, and manage cloned voice profiles
  through `.s2voice` files.

Technical changes:

- **`include/s2_voice.h`** - Adds `VoiceProfile` and `VoiceProfileManager` for
  voice profile storage and management.
- **`src/s2_voice.cpp`** - Implements:
  - Voice profile serialization
  - Loading saved profiles
  - Listing available voices
  - Internal support for removing `.s2voice` files

CLI additions:

- `--voice` - Select a saved voice profile
- `--save-voice` - Save a new voice profile
- `--voice-dir` - Set a custom directory for stored voices
- `--list-voices` - List available saved voices

Documentation:

- **`AGENTS.md`** - Build instructions, architecture notes, and project
  conventions
- **`openapi/`** - OpenAPI 3.1 specification for the HTTP server
- **`README.md`** - Updated with the new voice profile workflows

Reported scope from branch comparison:

- **3 commits**
- **11 files changed** (8 modified + 3 new)
- **824 lines added** (6 removed)

Summary:

- Introduced a complete cloned voice persistence and reuse workflow, expanded
  the CLI, and documented the API surface around it.

### felipeov1 - Apple Metal GPU Backend

Main feature:

- **Apple Metal GPU backend** - Enables Apple Silicon GPUs (M1, M2, M3, and
  newer) on macOS for accelerated inference.

Technical changes:

- **`CMakeLists.txt`** - Adds the `S2_METAL` build flag and propagates Metal
  support into the build.
- **`include/s2_model.h`** - Adds conditional `ggml-metal.h` support.
- **`include/s2_pipeline.h`** - Expands backend handling to include Metal.
- **`src/main.cpp`** - Adds:
  - `-M` / `--metal` - Enable Metal
  - Separate CLI options for Vulkan and CUDA instead of a combined path
- **`src/s2_model.cpp`** - Adds Metal backend initialization with CPU fallback.

Attention/mask compatibility changes:

- Replaces the older multi-step attention masking flow with the more compatible
  `ggml_soft_max_ext(...)` path.
- Affects both:
  - `SlowARModel::eval_cached`
  - `SlowARModel::fast_decode`

Reported scope from branch comparison:

- **1 commit**
- **5 files changed**
- **42 lines added** (8 removed)

Summary:

- Added full Apple Silicon GPU acceleration support and improved attention-mask
  handling for Metal compatibility.

### subspecs - Original Export API Foundation

Main feature:

- A full **export API foundation** was added so `s2.cpp` could be used as a
  library instead of only as a standalone executable.

Original API capabilities attributed to this work:

- TTS with or without reference voice audio
- Dual output modes:
  - Save to `.wav`
  - Return `float` samples directly in memory
- Modular loading through:
  - `AllocS2Model()`
  - `AllocS2Tokenizer()`
  - `AllocS2AudioCodec()`
  - `AllocS2Pipeline()`
- Reusable cached reference prompt codes through:
  - `AllocS2AudioPromptCodes()`
- A reusable C-facing wrapper surface around the core engine

Related additions:

- **`s2_export_api.cpp` / `s2_export_api.h`** - Exported C wrapper surface
- **`s2_config.h`** - Export/import configuration macros for shared-library use
- Runtime log control for library hosts via:
  - `SetS2LogLevel()`
  - `GetS2LogLevel()`
- Example integrations in:
  - `examples/python`
  - `examples/csharp`
  - `examples/golang`
- External companion wrapper by `@subspecs`:
  - **FishS2Sharp** - A .NET Standard 2.1 wrapper targeting C# / Unity-style
    integration

Notes:

- Branch-comparison summaries around this work referenced a
  `SuppressNonEssentialVerbosity` variable for library embedding scenarios.
- In the current repository state, shared-library visibility is handled by
  `s2_config.h`, and runtime log suppression for host applications is exposed
  through `SetS2LogLevel()` / `GetS2LogLevel()`.
- Not every export/streaming feature present in the current repository should be
  attributed to this original contribution. Later work expanded the exported
  surface substantially, especially around streaming controls and examples.

Summary:

- The project gained the original reusable native library interface,
  in-memory audio output, reusable model/tokenizer/codec instances, cached
  reference prompt codes, configurable host-side logging, and an external C# /
  Unity wrapper ecosystem around that API.

### rodrigomatta / staging branch - Later Streaming, Integration, and Documentation Work

The items below are **not** attributed to `kolulu23` or `felipeov1`.
They are later additions visible in the current `maio` branch state.

Main improvements present in the current `maio` branch state:

- **HTTP server mode** with `POST /generate`
- **OpenAPI 3.1** description for the HTTP server
- **Streaming audio delivery** in both WAV and raw `pcm_s16le`
- **Low-latency streaming controls**, including:
  - `low_latency`
  - `stream_holdback_frames`
  - `stream_start_buffer_ms`
- **Sentence-aware synthesis** controls, including:
  - `segment_sentences`
  - `sentence_pause_ms`
  - `segment_max_chars`
- **Later extensions on top of the original exported API** via
  `S2StreamingParams` and `S2SynthesizeStreamingEx()`
- **Codec backend auto-selection** with performance benchmarking and explicit
  CPU fallback when GPU codec initialization or allocation fails
- **Step-graph reuse and chunked prefill** to improve memory behavior and
  throughput in the autoregressive path
- **Multi-language example coverage** across Python, C#, and Go
- **Documentation alignment** across `README.md`, `openapi/`, and `AGENTS.md`

Summary:

- Expanded the project from a CLI-focused TTS engine into a broader platform
  with streaming support, HTTP serving, performance improvements, stronger
  integration surfaces, and better-aligned documentation built on top of the
  original reusable API foundation.
