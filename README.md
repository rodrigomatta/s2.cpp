# s2.cpp

> **ALPHA — EXPERIMENTAL SOFTWARE**
> This is an early-stage, community-built inference engine. Expect rough edges, missing features, and breaking changes. Not production-ready.

**s2.cpp** — Fish Audio's S2 Pro Dual-AR text-to-speech model running locally via a pure C++/GGML inference engine with CPU, Vulkan, CUDA, and Metal GPU backends. No Python runtime required after build.

> **Built on Fish Audio S2 Pro**
> The model weights are licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.
> See [LICENSE.md](LICENSE.md) for full terms. Commercial use requires a separate license from Fish Audio — contact [business@fish.audio](mailto:business@fish.audio).

---

## Contributors

- [@felipeov1](https://github.com/felipeov1)
- [@ivanodintsov](https://github.com/ivanodintsov)
- [@kolulu23](https://github.com/kolulu23)
- [@rodrigomatta](https://github.com/rodrigomatta)
- [@subspecs](https://github.com/subspecs)

---

## What this is

This repository contains:

- **`s2.cpp`** — a self-contained C++17 inference engine built on [ggml](https://github.com/ggml-org/ggml), handling tokenization, Dual-AR generation, audio codec encode/decode, and WAV output with no Python dependency
- **`tokenizer.json`** — Qwen3 BPE tokenizer with ByteLevel pre-tokenization
- GGUF model files are **not included** here — see [Model variants](#model-variants) below

The engine runs the full pipeline: text → tokens → Slow-AR transformer (with KV cache) → Fast-AR codebook decoder → audio codec → WAV file.

---

## Model variants

GGUF files are available at [rodrigomt/s2-pro-gguf](https://huggingface.co/rodrigomt/s2-pro-gguf) on Hugging Face.
VRAM notes in this table were re-tested after fixing the duplicate weight load path that had previously inflated GPU memory usage.
The `~5 GB VRAM` note below refers to model-weight memory after that fix; full runtime usage is higher once the KV cache, codec, and backend overhead are included.

| File | Size | Notes |
|---|---|---|
| `s2-pro-f16.gguf` | 9.9 GB | Full precision — reference quality |
| `s2-pro-q8_0.gguf` | 5.6 GB | Near-lossless — model weights use ~5 GB VRAM after the duplicate-load fix; full runtime usage is higher |
| `s2-pro-q6_k.gguf` | 4.5 GB | Good quality/size balance — recommended for 6+ GB VRAM |
| `s2-pro-q5_k_m.gguf` | 4.0 GB | Smaller with still-good quality |
| `s2-pro-q4_k_m.gguf` | 3.6 GB | Best compact variant so far in quick RU validation |
| `s2-pro-q3_k.gguf` | 3.0 GB | Usable, but starts stretching short words |
| `s2-pro-q2_k.gguf` | 2.6 GB | Lowest-size experimental variant |

All variants include both the transformer weights and the audio codec in a single file.
The quantized variants above were regenerated with the codec tensors (`c.*`) kept in `F16`, so only the AR transformer is quantized.

On CUDA builds, `s2.cpp` keeps the embedding tables on CPU in the hybrid offload path. This avoids unsupported CUDA `get_rows` cases for K-quant embedding tensors (`q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`) and avoids the unstable voice-cloning prefill behavior seen when those lookups were pushed to CUDA. Transformer layers, Fast-AR, and the KV cache can still be offloaded according to `--gpu-layers`. On Vulkan and Metal, full-model offload keeps the entire model on GPU and automatically splits the weights across multiple backend buffers when the driver imposes a per-buffer allocation limit.

---

## Requirements

### Build dependencies

- CMake ≥ 3.14
- C++17 compiler (GCC ≥ 10, Clang ≥ 11, MSVC 2019+)
- For Vulkan GPU support: Vulkan SDK and `glslc`
- For CUDA/NVIDIA GPU support: CUDA Toolkit ≥ 12.4
- For Metal/macOS GPU support: Xcode Command Line Tools
  - **MSVC 2019+ note:** MSVC 2019 and later require CUDA ≥ 12.4 when building GGML. Older CUDA versions will produce compiler compatibility errors; upgrade to 12.4+ to resolve them.

```bash
# Ubuntu / Debian
sudo apt install cmake build-essential

# Vulkan (optional, for AMD/Intel GPU acceleration)
sudo apt install vulkan-tools libvulkan-dev glslc

# CUDA (optional, for NVIDIA GPU acceleration)
# Install from https://developer.nvidia.com/cuda-downloads

# macOS / Metal (optional, for Apple GPU acceleration)
xcode-select --install
```

### Runtime

No Python or PyTorch required. The executable and optional library targets link against the local `ggml` target built from the bundled submodule. Whether those artifacts are static or shared depends on the platform and CMake configuration.

---

## Building

Clone with submodules (ggml is a submodule):

```bash
git clone --recurse-submodules https://github.com/rodrigomatta/s2.cpp.git
cd s2.cpp
```

### CPU only

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```

### With Vulkan GPU support (AMD/Intel)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_VULKAN=ON
cmake --build build --parallel $(nproc)
```

### With CUDA GPU support (NVIDIA)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_CUDA=ON
cmake --build build --parallel $(nproc)
```

### With Metal GPU support (macOS / Apple GPU)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_METAL=ON
cmake --build build --parallel $(sysctl -n hw.ncpu)
```

The binary is produced at `build/s2`.
During configure, `CMakeLists.txt` also applies any optional local patches found under `patches/` to the bundled `ggml` submodule.

### Shared/static library targets

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_BUILD_SHARED_LIBRARIES=ON
cmake --build build --parallel $(nproc)
```

This also builds the shared library and the static library with stable public names:

- Linux: `build/libs2.so` and `build/libs2_static.a`
- macOS: `build/libs2.dylib` and `build/libs2_static.a`
- Windows: `build/s2.dll` and `build/s2_static.lib`

The exported C-linkage C++ API is declared in `include/s2_export_api.h`.

#### Memory management

All opaque objects follow an `Alloc` / `Release` pattern:

| Function | Purpose |
|---|---|
| `AllocS2Pipeline()` / `ReleaseS2Pipeline()` | Pipeline handle |
| `AllocS2Model()` / `ReleaseS2Model()` | Slow-AR model handle |
| `AllocS2Tokenizer()` / `ReleaseS2Tokenizer()` | Tokenizer handle |
| `AllocS2AudioCodec()` / `ReleaseS2AudioCodec()` | Audio codec handle |
| `AllocS2GenerateParams()` / `ReleaseS2GenerateParams()` | Generation parameters handle |
| `AllocS2AudioBuffer(size)` / `ReleaseS2AudioBuffer()` | Float sample buffer |
| `AllocS2AudioPromptCodes()` / `ReleaseS2AudioPromptCodes()` | Pre-encoded reference codes |

#### Initialization

| Function | Purpose |
|---|---|
| `InitializeS2Pipeline(pipeline, tokenizer, model, codec)` | Bind pre-initialized components as non-owning references — those objects must outlive the pipeline |
| `InitializeS2PipelineFromFiles(pipeline, gguf_path, tokenizer_path, gpu_device, backend_type, n_gpu_layers, codec_follow_backend)` | Pipeline owns its components directly, loaded from files |
| `InitializeS2Model(model, gguf_path, gpu_device, backend_type)` | Load model with default GPU layers |
| `InitializeS2ModelWithGpuLayers(model, gguf_path, gpu_device, backend_type, n_gpu_layers)` | Load model with explicit GPU layer count |
| `InitializeS2Tokenizer(tokenizer, path)` | Load tokenizer from file |
| `InitializeS2AudioCodec(codec, gguf_path, gpu_device, backend_type)` | Load audio codec from GGUF |
| `InitializeS2GenerateParams(params, ...)` | Set generation parameters (max_new_tokens, temperature, top_p, top_k, min_tokens_before_end, n_threads, verbose) |
| `InitializeAudioPromptCodes(pipeline, threads, ref_audio_path, codes_out, t_prompt_out)` | Pre-encode reference audio into reusable codes |
| `SyncS2TokenizerConfigFromS2Model(model, tokenizer)` | Sync tokenizer configuration from a loaded model |

#### Synthesis

| Function | Purpose |
|---|---|
| `S2Synthesize(pipeline, params, audio_buffer, ref_codes, t_prompt, ref_audio_path, ref_transcript, text, output_path, out_length)` | One-shot synthesis to float buffer and/or WAV file |
| `S2SynthesizeStreaming(pipeline, params, callbacks, ref_codes, t_prompt, ref_audio_path, ref_transcript, text, stride)` | Callback-based PCM16 streaming with fixed stride |
| `S2SynthesizeStreamingEx(pipeline, params, callbacks, ref_codes, t_prompt, ref_audio_path, ref_transcript, text, streaming_params)` | Extended streaming with full `S2StreamingParams` control |
| `GetS2AudioBufferDataPointer(buffer)` | Get raw float pointer from audio buffer |

#### S2StreamingParams fields

| Field | Default | Description |
|---|---|---|
| `stream_decode_stride_frames` | `0` (auto) | Decode cadence in frames |
| `stream_holdback_frames` | `-1` (auto) | Trailing frames buffered before emission |
| `codec_decode_context_frames` | `-1` (auto) | Codec decode history window; lower uses less VRAM |
| `low_latency` | `0` | Aggressive live preset (`stride=1`, `holdback=0`) |
| `segment_sentences` | `0` | Sentence-by-sentence synthesis mode |
| `sentence_pause_ms` | `180` | Silence gap between sentences |
| `segment_max_chars` | `0` | Break long sentences into clauses |
| `voice` | `nullptr` | Saved voice id or `.s2voice` path |
| `voice_dir` | `nullptr` | Directory for id-based voice lookup |

Runtime logging is controlled globally with `SetS2LogLevel(level)` / `GetS2LogLevel()`, where `0 = error`, `1 = warning`, `2 = info`, and `3 = debug`. The default is `info`, matching the CLI behavior. Library hosts that need quiet embedding should call `SetS2LogLevel(0)` before loading models; errors still go to `stderr`.

Direct Python `ctypes` example: `python3 examples/python/ctypes_export_api.py --smoke-only` validates that the shared library can be loaded and that the exported symbols are callable. Pass `--model <model.gguf>` to run a short synthesis through the library API, add `--streaming` to exercise the callback-based PCM16 path, add `--low-latency` to mirror the HTTP API's realtime defaults, add `--segment-sentences` to mirror the HTTP API's sentence-aware segmentation, pass `--voice <voice-id-or-path.s2voice>` to use a saved voice profile, or add `--play-only` on Linux to send streamed PCM16 directly to `aplay` without saving a WAV.

Example: realtime Python playback on Linux through the exported library API, using Vulkan and a saved `.s2voice` profile:

```bash
LD_LIBRARY_PATH=build-vulkan:build-vulkan/ggml/src \
python3 examples/python/ctypes_export_api.py \
  --library build-vulkan/libs2.so \
  --model /path/to/ggufs-s2/s2-pro-q2_K.gguf \
  --tokenizer /path/to/ggufs-s2/tokenizer.json \
  --backend vulkan \
  --gpu-device 0 \
  --threads 8 \
  --max-tokens 64 \
  --text "First sentence. A longer second sentence to confirm the pause." \
  --streaming \
  --segment-sentences \
  --sentence-pause-ms 180 \
  --voice /path/to/voices/hope.s2voice \
  --play-only \
  --log-level 2
```

You can also load the same voice by id instead of by path:

```bash
--voice hope --voice-dir /path/to/voices
```

Updated native C# P/Invoke example: [`examples/csharp`](examples/csharp) contains a current sample project that binds every exported symbol, including `InitializeS2PipelineFromFiles()`, `S2SynthesizeStreaming()`, and `S2SynthesizeStreamingEx()` with low-latency, sentence segmentation, and saved `.s2voice` selection. Its local README also compares the current API surface against the original modular design proposed by SubSpecs.

Go CGo example: [`examples/golang`](examples/golang) mirrors the C# example with the same five flows (smoke, from-files, modular, legacy-stream, stream-ex) using runtime `dlopen` loading — no build-time dependency on `libs2`. Streaming callbacks use `//export` Go functions with `sync.Map` for session state, equivalent to the C# `GCHandle` pattern.

---

## Community Made Wrappers/Ports:

| Repo. Name | Language | Maintainer |
|---|---|:---:|
| [FishS2Sharp](https://github.com/subspecs/FishS2Sharp) | `C#` | <a href="https://github.com/subspecs" target="_blank"><center><img src="https://images.weserv.nl/?url=avatars.githubusercontent.com/u/45248469?v=4&h=35&w=35&fit=cover&mask=circle&maxage=7d" alt="SubSpecs"></center></a> |

---

## Usage

### Basic synthesis (CPU)

```bash
./build/s2 \
  --model s2-pro-q6_k.gguf \
  --tokenizer tokenizer.json \
  --text "The quick brown fox jumps over the lazy dog." \
  --output output.wav
```

`tokenizer.json` is searched automatically in the same directory as the model file, then the parent directory. If not found in either, it falls back to `tokenizer.json` in the current working directory.

### Voice cloning with a reference audio

Provide a short reference clip (5–30 seconds, WAV or MP3) and a transcript of it:

```bash
./build/s2 \
  --model s2-pro-q6_k.gguf \
  --tokenizer tokenizer.json \
  --prompt-audio reference.wav \
  --prompt-text "Transcript of what the reference speaker says." \
  --text "Now synthesize this text in that voice." \
  --output output.wav
```

Reference audio is decoded to mono and resampled internally to the codec sample rate, so it does not need to be recorded at 44.1 kHz. WAV and MP3 references are supported in both CLI and HTTP server paths.

By default, the engine uses fish-speech-aligned sampling defaults: `--min-tokens-before-end 0`, no trailing-silence trim, no peak normalization, and no dynamic loudness normalization. All of these behaviors are optional and can be enabled from the CLI.

### Voice profile persistence

You can persist encoded reference codes as reusable `.s2voice` profiles so repeated cloning requests do not need to re-encode the same reference audio.

Save a profile while cloning:

```bash
./build/s2 \
  --model s2-pro-q6_k.gguf \
  --tokenizer tokenizer.json \
  --prompt-audio reference.wav \
  --prompt-text "Transcript of what the reference speaker says." \
  --voice alice \
  --save-voice \
  --text "Now synthesize this text in that voice." \
  --output output.wav
```

Reuse a saved profile later without passing reference audio again:

```bash
./build/s2 \
  --model s2-pro-q6_k.gguf \
  --tokenizer tokenizer.json \
  --voice alice \
  --text "Another sentence in the same voice." \
  --output output.wav
```

List the saved profiles:

```bash
./build/s2 --list-voices
```

Profiles are stored in `./voices` by default. Override that location with `--voice-dir`.
For example, `--voice alice --save-voice` writes `./voices/alice.s2voice` by default.

### GPU inference via Vulkan (AMD/Intel)

```bash
./build/s2 \
  --model s2-pro-q6_k.gguf \
  --tokenizer tokenizer.json \
  --text "Text to synthesize." \
  --vulkan 0 \
  --output output.wav
```

`--vulkan 0` selects the first Vulkan device.
When `--gpu-layers` covers all 36 transformer layers, Vulkan keeps the full model on GPU. If the Vulkan driver rejects a single monolithic allocation, `s2.cpp` splits the model weights across multiple GPU buffers automatically. This avoids per-buffer allocation failures, but it does not reduce the total VRAM requirement.

### GPU inference via CUDA (NVIDIA)

```bash
./build/s2 \
  --model s2-pro-q8_0.gguf \
  --tokenizer tokenizer.json \
  --text "Text to synthesize." \
  --cuda 0 \
  --output output.wav
```

`--cuda 0` selects the first CUDA device. By default, both the transformer and the audio codec follow the selected backend. If codec GPU initialization or weight allocation fails, `s2.cpp` logs the failure and falls back to CPU for the codec. Use `--codec-cpu` if you want to keep the codec on CPU explicitly.
For stability, embedding lookups remain on CPU in the current CUDA hybrid path, while the transformer layers, Fast-AR path, and KV cache still follow the selected `--gpu-layers` split.

### GPU inference via Metal (macOS / Apple GPU)

```bash
./build/s2 \
  --model s2-pro-q8_0.gguf \
  --tokenizer tokenizer.json \
  --text "Text to synthesize." \
  --metal \
  --output output.wav
```

`--metal` selects the default Metal device on macOS. The CLI treats Metal as a GPU backend for `--gpu-layers` and `--codec-cpu`.

### All options

| Flag | Default | Description |
|---|---|---|
| `-m`, `--model` | `model.gguf` | Path to GGUF model file |
| `-t`, `--tokenizer` | `tokenizer.json` | Path to tokenizer.json |
| `--text`, `-text` | `"Hello world"` | Text to synthesize |
| `-pa`, `--prompt-audio` | — | Reference audio file for voice cloning (WAV/MP3) |
| `-pt`, `--prompt-text` | — | Transcript of the reference audio |
| `--voice` | — | Load a saved `.s2voice` profile |
| `--save-voice` | off | Save the encoded reference audio as the profile named by `--voice` |
| `--voice-dir` | `./voices` | Directory used to load and save voice profiles |
| `--list-voices` | — | List available saved voice profiles and exit |
| `-o`, `--output` | `out.wav` | Output WAV file path |
| `-v`, `--vulkan` | `-1` (CPU) | Vulkan device index (`-1` = CPU only) |
| `-c`, `--cuda` | `-1` (CPU) | CUDA device index (`-1` = CPU only) |
| `-M`, `--metal` | off | Use the default Metal device on macOS |
| `-ngl`, `--gpu-layers` | `-1` (auto) | Number of transformer layers to offload to GPU (`-1` = all layers when a GPU backend is selected, otherwise `0`; `0` = CPU only) |
| `--threads N`, `-threads N` | `0` (auto) | Number of CPU threads (`0` = hardware concurrency when available, else `4`) |
| `--max-tokens N`, `-max-tokens N` | `1024` | Max tokens to generate |
| `--min-tokens-before-end N` | `0` | Minimum generated tokens before `EOS` is allowed; `0` matches fish-speech default behavior |
| `--temperature F`, `--temp F`, `-temp F` | `0.8` | Sampling temperature |
| `--top-p F`, `-top-p F` | `0.8` | Top-p nucleus sampling |
| `--top-k N`, `-top-k N` | `30` | Top-k sampling |
| `--dynamic-normalize` / `--no-dynamic-normalize` | `disabled` | Enable or disable dynamic RMS normalization |
| `--trim-silence` / `--no-trim-silence` | `trim` disabled | Enable or disable trailing silence trimming on the saved WAV |
| `--normalize` / `--no-normalize` | `normalize` disabled | Enable or disable peak normalization to `0.95` on the saved WAV |
| `--codec-auto` / `--codec-follow-backend` / `--codec-cpu` | `--codec-auto` | `--codec-auto` benchmarks codec backends and keeps the fastest; `--codec-follow-backend` lets the codec follow the selected GPU backend; `--codec-cpu` keeps codec on CPU. If codec GPU init or allocation fails, the runtime falls back to CPU |
| `--codec-context-frames <n>` | `auto` | Override codec decode history window. Lower values use less VRAM but may reduce quality |
| `--stream-file` | — | Write the WAV through the exact streaming path instead of the final one-shot save |
| `--stream-decode-stride N` | `0` | Decode cadence in frames. `0` = auto (`4` for server streaming, `16` for `--stream-file` and offline synthesis) |
| `--log-level LEVEL` | `info` | Runtime log verbosity: `error`, `warn`, `info`, or `debug` |
| `--server` | — | Start HTTP server instead of CLI synthesis |
| `-H`, `--host` | `127.0.0.1` | Server bind address |
| `-P`, `--port` | `3030` | Server port |

Setting `--min-tokens-before-end 0` matches the upstream fish-speech behavior. Non-zero values deliberately bias the model away from early `EOS`.

---

### HTTP server mode

Start the server:

```bash
./build/s2 --model s2-pro-q6_k.gguf --server
# or with custom host/port:
./build/s2 --model s2-pro-q6_k.gguf --server --host 0.0.0.0 --port 8080
```

An OpenAPI 3.1 description of the server lives at `openapi/s2-openapi.yaml`.

**`POST /generate`** — synthesize audio (multipart/form-data)

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Text to synthesize |
| `reference` | file | no | Reference audio file for voice cloning (WAV or MP3). Aliases: `reference_audio`, `prompt_audio`, `ref_audio` |
| `reference_text` | string | if reference audio is provided | Transcript of the reference audio. Aliases: `ref_text`, `prompt_text` |
| `voice` | string | no | Saved voice profile id such as `hope`, or a `.s2voice` path such as `voices/hope.s2voice`. Aliases: `voice_id`, `voice_profile` |
| `voice_dir` | string | no | Directory used when `voice` is given as an id instead of a full `.s2voice` path |
| `params` | JSON string | no | Generation params: `max_new_tokens`, `temperature`, `top_p`, `top_k`, `min_tokens_before_end`, `n_threads`, `codec_follow_backend`, `codec_auto_backend`, `codec_decode_context_frames` (alias: `codec_context_frames`), `voice`, `voice_id`, `voice_dir`, `stream`, `chunked`, `realtime`, `stream_decode_stride_frames`, `stream_decode_stride`, `stream_holdback_frames`, `stream_start_buffer_ms`, `segment_sentences`, `sentence_pause_ms`, `segment_max_chars`, `low_latency`, `output_format`, `verbose` |

Returns a finalized `audio/wav` download by default.

- Default request: one-shot synthesis, finalized WAV response.
- `params.stream=true`: uses the exact streaming synthesis path, but still returns a finalized WAV that saves cleanly to disk.
- `params.stream=true` plus `params.chunked=true` (or `params.realtime=true`): keeps the real-time chunked PCM16 WAV transport for clients that want bytes as soon as they are generated. The WAV header uses large provisional RIFF/data sizes because the server cannot seek back over a live HTTP stream. If you save that response directly to disk, readers should still decode it cleanly, but tools may estimate duration from bitrate because the header is not fully finalized; use `params.stream=true` without `chunked` when you need a clean finalized WAV header.
- `voice=hope` reuses `./voices/hope.s2voice` by default. You can also pass `voice=voices/hope.s2voice` to target a specific file directly.
- `params.output_format="pcm_s16le"` with `params.stream=true`: skips the WAV header entirely and returns raw PCM16 mono bytes. This is the preferred transport for a browser/app that wants to maintain a dynamic playback buffer instead of saving a file. The server includes `X-Audio-Sample-Rate`, `X-Audio-Channels`, and `X-Audio-Encoding` headers for convenience; clients should use those headers instead of assuming a fixed sample rate.
- `params.low_latency=true`: applies an aggressive live preset for weak machines by defaulting to `stream_decode_stride_frames=1` and `stream_holdback_frames=0`, so playback can start as soon as samples exist.
- `params.stream_holdback_frames=N`: controls how many trailing codec frames stay buffered before they are emitted. Lower values reduce startup latency; higher values keep chunk boundaries more stable. `0` is the lowest-latency option, while the default auto mode uses the codec's full history window.
- `params.stream_start_buffer_ms=N`: for chunked HTTP streaming, waits until roughly `N` milliseconds of PCM are queued before the server starts sending bytes. This is the best option when you want natural playback and can tolerate a short startup delay.
- `params.segment_sentences=true`: switches the server from exact frame streaming to sentence-by-sentence synthesis. This is usually the best mode for slower machines because each sentence is generated as a whole, then queued for playback.
- `params.sentence_pause_ms=N`: inserts a fixed silence gap between synthesized sentences when `segment_sentences=true`.
- `params.segment_max_chars=N`: optionally breaks very long sentences into smaller clauses before synthesis.

If the machine synthesizes slower than real time, uninterrupted live playback is only possible when the client keeps a dynamic buffer. In practice, stream `pcm_s16le`, wait until a few seconds of PCM are queued before starting playback, and pause/resume or change the target buffer depth when the queue drains too far.

The HTTP server currently runs one synthesis at a time. If another `POST /generate` request arrives while generation is active, the server returns HTTP `503` immediately instead of queuing behind the active request. If the client disconnects or cancels the download, cancellation is propagated into the generation loop so the synthesis aborts early and releases the busy state for the next request. Request validation failures, JSON parse failures, and invalid `params` value types return HTTP `400`; synthesis failures detected before a response body starts return HTTP `500`. Real-time chunked responses may instead terminate early if generation fails after streaming has already begun.

```bash
# Basic
curl -X POST http://127.0.0.1:3030/generate \
  --form "text=Hello world" \
  --form 'params={"max_new_tokens":512,"temperature":0.58,"top_p":0.88,"top_k":40}' \
  -o output.wav

# With voice cloning
curl -X POST http://127.0.0.1:3030/generate \
  --form "reference=@reference.wav" \
  --form "reference_text=Transcript of the reference." \
  --form "text=Text to synthesize in that voice." \
  --form 'params={"max_new_tokens":512,"temperature":0.58,"top_p":0.88,"top_k":40}' \
  -o output.wav

# The ffplay examples below assume the current 44.1 kHz models.
# Real clients should prefer the X-Audio-Sample-Rate response header.

# With a saved .s2voice profile
curl -X POST http://127.0.0.1:3030/generate \
  --form "voice=hope" \
  --form "text=English text using the Hope voice." \
  --form 'params={"stream":true,"chunked":true,"output_format":"pcm_s16le","segment_sentences":true,"stream_start_buffer_ms":6000,"max_new_tokens":512}' \
| ffplay -autoexit -nodisp -infbuf -f s16le -ar 44100 -ac 1 -

# Use the streaming synthesis path, but still save a finalized WAV
curl -X POST http://127.0.0.1:3030/generate \
  --form "text=Hello from the streaming path" \
  --form 'params={"stream":true,"max_new_tokens":512}' \
  -o output-streaming.wav

# Keep the live real-time chunked transport
curl -X POST http://127.0.0.1:3030/generate \
  --form "text=Hello from the live chunked transport" \
  --form 'params={"stream":true,"chunked":true,"max_new_tokens":512}' \
  -o output-live.wav

# Live low-latency PCM stream for a browser/app player with adaptive buffering
curl -X POST http://127.0.0.1:3030/generate \
  --form "text=Hello from the live PCM transport" \
  --form 'params={"stream":true,"chunked":true,"output_format":"pcm_s16le","low_latency":true,"max_new_tokens":512}'

# Natural live PCM playback: higher holdback + startup buffer before the stream begins
curl -sN -X POST http://127.0.0.1:3030/generate \
  --form "text=Hello from the buffered live PCM transport" \
  --form 'params={"stream":true,"chunked":true,"output_format":"pcm_s16le","stream_decode_stride_frames":4,"stream_holdback_frames":16,"stream_start_buffer_ms":4000,"max_new_tokens":512}' \
| ffplay -autoexit -nodisp -infbuf -f s16le -ar 44100 -ac 1 -

# Natural live playback on slower machines: synthesize sentence by sentence
curl -sN -X POST http://127.0.0.1:3030/generate \
  --form "text=Sereias são criaturas mitológicas híbridas, tradicionalmente representadas com busto feminino humano e cauda de peixe. Originárias de lendas gregas como seres metade mulher e metade pássaro, transfiguraram-se na Idade Média para a forma marinha. Representam perigos do mar, mistério e beleza, com equivalentes em diversas culturas." \
  --form 'params={"stream":true,"chunked":true,"output_format":"pcm_s16le","segment_sentences":true,"sentence_pause_ms":180,"stream_start_buffer_ms":6000,"max_new_tokens":512}' \
| ffplay -autoexit -nodisp -infbuf -f s16le -ar 44100 -ac 1 -

# Same request using the accepted aliases
curl -X POST http://127.0.0.1:3030/generate \
  --form "reference_audio=@reference.wav" \
  --form "ref_text=Transcript of the reference." \
  --form "text=Text to synthesize in that voice." \
  -o output.wav
```

---

## Choosing a model

| VRAM available | Recommended model | Notes |
|---|---|---|
| ≥ 10 GB | `q8_0` — near-lossless quality | All 36 layers + codec on GPU |
| 8–9 GB | `q6_k` — good quality/size balance | All 36 layers on GPU, codec on GPU |
| 6–7 GB | `q4_k_m` — best compact variant | Use `--gpu-layers 18` to split layers between GPU and CPU; or `--codec-cpu` to keep codec off GPU |
| < 6 GB | `q4_k_m` or smaller | Requires partial offload: `--gpu-layers 10` or lower + `--codec-cpu` |

### Understanding VRAM usage

Runtime VRAM usage is significantly higher than the model file size. For example, `s2-pro-q4_k_m.gguf` (3.6 GB on disk) uses approximately **7.2 GB VRAM** with all layers on GPU. The breakdown:

| Component | Approx. size | Details |
|---|---|---|
| AR transformer weights | ~2.8 GB | Quantized (Q4_K_M) |
| Codec weights | ~0.4 GB | Kept in F16 |
| KV cache (K + V) | ~2.5 GB | F16, pre-allocated for 32 768 tokens × 36 layers × 8 KV heads |
| Compute buffers + overhead | ~1.5 GB | Graph workspaces, alignment, temporary contexts |

The **KV cache** is the largest overhead beyond the model weights. It is always fully allocated on GPU when any layer is offloaded (`--gpu-layers > 0`), regardless of how many layers you choose.
The Vulkan multi-buffer loader only removes single-allocation limits; cards with 6 GB VRAM still generally need partial offload and/or `--codec-cpu`.

### Reducing VRAM usage

If VRAM is tight, use `--gpu-layers` to offload only some of the 36 transformer layers to GPU — the remaining layers run on CPU:

```bash
# Example: 18 layers on GPU, 18 on CPU (fits in ~5 GB VRAM with q4_k_m)
./build/s2 --model s2-pro-q4_k_m.gguf --cuda 0 --gpu-layers 18 --text "Hello" --output out.wav

# Example: very limited VRAM — 10 layers on GPU, codec on CPU
./build/s2 --model s2-pro-q4_k_m.gguf --cuda 0 --gpu-layers 10 --codec-cpu --text "Hello" --output out.wav
```

Fewer GPU layers means more work on CPU (slower), but dramatically lower VRAM usage. The KV cache (~2.5 GB) still goes to GPU as long as `--gpu-layers > 0`.

---

## Architecture notes

S2 Pro uses a **Dual-AR** architecture:

- **Slow-AR** — a 36-layer Qwen3-based transformer (4.13B params) that processes the full token sequence with GQA (32 heads, 8 KV heads), RoPE at 1M base, QK norm, and a persistent KV cache
- **Fast-AR** — a 4-layer transformer (0.42B params) that autoregressively generates 10 acoustic codebook tokens from the Slow-AR hidden state for each semantic step
- **Audio codec** — a convolutional encoder/decoder with residual vector quantization (RVQ, 10 codebooks × 4096 entries) that converts between audio waveforms and discrete codes

Total: ~4.56B parameters.

---

## Implementation notes

The C++ engine (`src/`) is built on a bundled [ggml](https://github.com/ggml-org/ggml) submodule. If the `patches/` directory contains local fixes, `CMakeLists.txt` applies them at configure time. Key design decisions:

- **Scheduler-based execution for Slow-AR and Fast-AR** — the model keeps dedicated `ggml_backend_sched_t` instances for the two transformer paths instead of the older persistent `gallocr` design
- **Cached single-token step graph plus adaptive chunked prefill** — the autoregressive loop reuses the step graph, while prompt prefills build larger graphs and may be chunked on GPU backends to keep cloning scratch usage under control
- **CUDA-only hybrid embedding path** — CUDA keeps embedding gathers on CPU for stability, while Vulkan/Metal full offload can keep the whole model on GPU
- **Automatic multi-buffer weight placement** — when a backend exposes a per-buffer allocation limit, `s2.cpp` splits large weight sets across multiple backend buffers instead of requiring one monolithic allocation
- **Codec follows the selected backend by default** — the default `--codec-auto` mode benchmarks available backends and keeps the fastest. Use `--codec-follow-backend` to let the codec follow the selected GPU backend, or `--codec-cpu` to keep it on CPU. If codec GPU init or weight allocation fails, it falls back to CPU explicitly
- **posix_fadvise(DONTNEED)** after loading the weights *(Linux only)* — advises the kernel to drop the GGUF file from page cache once the tensors are already in the backend buffer, reducing duplicate RAM use
- **Correct ByteLevel tokenization** — the GPT-2 byte-to-unicode table is applied before BPE, producing token IDs identical to the HuggingFace reference tokenizer

---

## Tips

### Long outputs

Voice quality and amplitude tend to degrade after ~800 tokens (~37 s of audio). For longer texts, split into sentences and concatenate the resulting WAV files. Optional post-processing flags such as `--dynamic-normalize`, `--normalize`, and `--trim-silence` can help clean up the result, but splitting remains the most reliable approach.

---

## Known limitations (alpha)

- Exact streaming still re-decodes the confirmed prefix, so streaming latency and CPU time remain structurally high until the codec becomes incremental/stateful. `stream_holdback_frames` can reduce perceived startup delay, but too-low values may add audible boundary artifacts
- No batch inference
- No HTTP request queue: the server accepts one active `/generate` synthesis and returns `503` for concurrent requests
- Voice cloning quality depends heavily on reference audio length and SNR
- Windows: CUDA and Vulkan backends are supported; when using MSVC 2019+, ensure CUDA ≥ 12.4 is installed before building
- macOS is untested

---

## License

The model weights and associated materials are licensed under the **Fish Audio Research License**. Key points:

- **Research and non-commercial use:** free, under the terms of this Agreement
- **Commercial use:** requires a separate written license from Fish Audio
- When distributing, you must include a copy of the license and the attribution notice
- Attribution: *"This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved."*

Full license: [LICENSE.md](LICENSE.md)

Commercial licensing: [https://fish.audio](https://fish.audio) · [business@fish.audio](mailto:business@fish.audio)

The inference engine source code (`src/`) is a Derivative Work of the Fish Audio Materials as defined in the Agreement and is distributed under the same Fish Audio Research License terms.
