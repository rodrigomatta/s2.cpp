# s2.cpp

> **ALPHA — EXPERIMENTAL SOFTWARE**
> This is an early-stage, community-built inference engine. Expect rough edges, missing features, and breaking changes. Not production-ready.

**s2.cpp** — Fish Audio's S2 Pro Dual-AR text-to-speech model running locally via a pure C++/GGML inference engine with CPU, Vulkan, and CUDA GPU backends. No Python runtime required after build.

> **Built on Fish Audio S2 Pro**
> The model weights are licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.
> See [LICENSE.md](LICENSE.md) for full terms. Commercial use requires a separate license from Fish Audio — contact [business@fish.audio](mailto:business@fish.audio).

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

| File | Size | Notes |
|---|---|---|
| `s2-pro-f16.gguf` | 9.9 GB | Full precision — reference quality |
| `s2-pro-q8_0.gguf` | 5.6 GB | Near-lossless — recommended for 8+ GB VRAM |
| `s2-pro-q6_k.gguf` | 4.5 GB | Good quality/size balance — recommended for 6+ GB VRAM |
| `s2-pro-q5_k_m.gguf` | 4.0 GB | Smaller with still-good quality |
| `s2-pro-q4_k_m.gguf` | 3.6 GB | Best compact variant so far in quick RU validation |
| `s2-pro-q3_k.gguf` | 3.0 GB | Usable, but starts stretching short words |
| `s2-pro-q2_k.gguf` | 2.6 GB | Lowest-size experimental variant |

All variants include both the transformer weights and the audio codec in a single file.
The quantized variants above were regenerated with the codec tensors (`c.*`) kept in `F16`, so only the AR transformer is quantized.

---

## Requirements

### Build dependencies

- CMake ≥ 3.14
- C++17 compiler (GCC ≥ 10, Clang ≥ 11, MSVC 2019+)
- For Vulkan GPU support: Vulkan SDK and `glslc`
- For CUDA/NVIDIA GPU support: CUDA Toolkit ≥ 12.4
  - **MSVC 2019+ note:** MSVC 2019 and later require CUDA ≥ 12.4 when building GGML. Older CUDA versions will produce compiler compatibility errors; upgrade to 12.4+ to resolve them.

```bash
# Ubuntu / Debian
sudo apt install cmake build-essential

# Vulkan (optional, for AMD/Intel GPU acceleration)
sudo apt install vulkan-tools libvulkan-dev glslc

# CUDA (optional, for NVIDIA GPU acceleration)
# Install from https://developer.nvidia.com/cuda-downloads
```

### Runtime

No Python or PyTorch required. The binary links only against the ggml shared libraries built alongside it.

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

The binary is produced at `build/s2`.

---

## Usage

### Basic synthesis (CPU)

```bash
./build/s2 \
  -m s2-pro-q6_k.gguf \
  -t tokenizer.json \
  -text "The quick brown fox jumps over the lazy dog." \
  -o output.wav
```

`tokenizer.json` is searched automatically in the same directory as the model file, then the parent directory. If not found in either, it falls back to `tokenizer.json` in the current working directory.

### Voice cloning with a reference audio

Provide a short reference clip (5–30 seconds, WAV or MP3) and a transcript of it:

```bash
./build/s2 \
  -m s2-pro-q6_k.gguf \
  -t tokenizer.json \
  -pa reference.wav \
  -pt "Transcript of what the reference speaker says." \
  -text "Now synthesize this text in that voice." \
  -o output.wav
```

By default, the engine uses fish-speech-aligned sampling defaults: `--min-tokens-before-end 0`, no trailing-silence trim, no peak normalization, and no dynamic loudness normalization. All of these behaviors are optional and can be enabled from the CLI.

### Voice profile persistence

Encoded voice profiles can be saved and reused, eliminating the need to re‑encode the reference audio and transcript each time.

**Save a voice profile** (clones voice and saves profile):
```bash
./build/s2 \
  -m s2-pro-q6_k.gguf \
  -t tokenizer.json \
  -pa reference.wav \
  -pt "Transcript of what the reference speaker says." \
  --voice alice \
  --save-voice \
  -text "Now synthesize this text in that voice." \
  -o output.wav
```

**Reuse a saved voice profile** (no reference audio needed):
```bash
./build/s2 \
  -m s2-pro-q6_k.gguf \
  -t tokenizer.json \
  --voice alice \
  -text "Another sentence in the same voice." \
  -o output.wav
```

**List saved profiles:**
```bash
./build/s2 --list-voices
```

Profiles are stored as `.s2voice` binary files in `./voices` (customizable with `--voice-dir`). They contain the encoded reference codes, transcript, and metadata (codebook size, sample rate, etc.). The format is **little‑endian** and portable across Windows, Linux, and macOS on x86/ARM.

### GPU inference via Vulkan (AMD/Intel)

```bash
./build/s2 \
  -m s2-pro-q6_k.gguf \
  -t tokenizer.json \
  -text "Text to synthesize." \
  -v 0 \
  -o output.wav
```

`-v 0` selects the first Vulkan device.

### GPU inference via CUDA (NVIDIA)

```bash
./build/s2 \
  -m s2-pro-q6_k.gguf \
  -t tokenizer.json \
  -text "Text to synthesize." \
  -c 0 \
  -o output.wav
```

`-c 0` selects the first CUDA device. The transformer runs on GPU; the audio codec always runs on CPU (executes only twice per synthesis).

### All options

| Flag | Default | Description |
|---|---|---|
| `-m`, `--model` | `model.gguf` | Path to GGUF model file |
| `-t`, `--tokenizer` | `tokenizer.json` | Path to tokenizer.json |
| `-text` | `"Hello world"` | Text to synthesize |
| `-pa`, `--prompt-audio` | — | Reference audio file for voice cloning (WAV/MP3) |
| `-pt`, `--prompt-text` | — | Transcript of the reference audio |
| `--voice` | — | Use saved voice profile (instead of -pa/-pt) |
| `--save-voice` | `false` | Save encoded voice profile after cloning (requires --voice and -pa/-pt) |
| `--voice-dir` | `"./voices"` | Directory for voice profiles |
| `--list-voices` | — | List saved voice profiles and exit |
| `-o`, `--output` | `out.wav` | Output WAV file path |
| `-v`, `--vulkan` | `-1` (CPU) | Vulkan device index (`-1` = CPU only) |
| `-c`, `--cuda` | `-1` (CPU) | CUDA device index (`-1` = CPU only) |
| `-threads N` | `4` | Number of CPU threads |
| `-max-tokens N` | `1024` | Max tokens to generate |
| `--min-tokens-before-end N` | `0` | Minimum generated tokens before `EOS` is allowed; `0` matches fish-speech default behavior |
| `-temp F` | `0.8` | Sampling temperature |
| `-top-p F` | `0.8` | Top-p nucleus sampling |
| `-top-k N` | `30` | Top-k sampling |
| `--dynamic-normalize` / `--no-dynamic-normalize` | `disabled` | Enable or disable dynamic RMS normalization |
| `--trim-silence` / `--no-trim-silence` | `trim` disabled | Enable or disable trailing silence trimming on the saved WAV |
| `--normalize` / `--no-normalize` | `normalize` disabled | Enable or disable peak normalization to `0.95` on the saved WAV |
| `--server` | — | Start HTTP server instead of CLI synthesis |
| `-H`, `--host` | `127.0.0.1` | Server bind address |
| `-P`, `--port` | `3030` | Server port |

Setting `--min-tokens-before-end 0` matches the upstream fish-speech behavior. Non-zero values deliberately bias the model away from early `EOS`.

---

### HTTP server mode

Start the server:

```bash
./build/s2 -m s2-pro-q6_k.gguf --server
# or with custom host/port:
./build/s2 -m s2-pro-q6_k.gguf --server -H 0.0.0.0 -P 8080
```

**`POST /generate`** — synthesize audio (multipart/form-data)

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Text to synthesize |
| `reference` | file | no | Reference audio file for voice cloning (WAV or MP3). Aliases: `reference_audio`, `prompt_audio`, `ref_audio` |
| `reference_text` | string | if reference audio is provided | Transcript of the reference audio. Aliases: `ref_text`, `prompt_text` |
| `params` | JSON string | no | Generation params: `max_new_tokens`, `temperature`, `top_p`, `top_k`, `min_tokens_before_end`, `n_threads`, `verbose` |

Returns `audio/wav`.

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

# Same request using the accepted aliases
curl -X POST http://127.0.0.1:3030/generate \
  --form "reference_audio=@reference.wav" \
  --form "ref_text=Transcript of the reference." \
  --form "text=Text to synthesize in that voice." \
  -o output.wav
```

---

## Choosing a model

| VRAM available | Recommended model |
|---|---|
| ≥ 10 GB | `q8_0` — near-lossless quality |
| 6–9 GB | `q6_k` — good quality/size balance |
| 5–7 GB | `q4_k_m` — best compact variant in current quick validation |
| < 5 GB | `q3_k` or `q2_k` — experimental, quality drops faster |

VRAM usage at runtime is roughly on the order of the model size, but actual usage depends on backend buffers, KV cache length, and allocator overhead. The audio codec executes on CPU during inference.

---

## Architecture notes

S2 Pro uses a **Dual-AR** architecture:

- **Slow-AR** — a 36-layer Qwen3-based transformer (4.13B params) that processes the full token sequence with GQA (32 heads, 8 KV heads), RoPE at 1M base, QK norm, and a persistent KV cache
- **Fast-AR** — a 4-layer transformer (0.42B params) that autoregressively generates 10 acoustic codebook tokens from the Slow-AR hidden state for each semantic step
- **Audio codec** — a convolutional encoder/decoder with residual vector quantization (RVQ, 10 codebooks × 4096 entries) that converts between audio waveforms and discrete codes

Total: ~4.56B parameters.

---

## Implementation notes

The C++ engine (`src/`) is built entirely on [ggml](https://github.com/ggml-org/ggml) (unmodified, pinned as a submodule). Key design decisions:

- **Separate persistent `gallocr` allocators** for Slow-AR and Fast-AR — each path keeps its own compute buffer, avoiding memory re-planning per token
- **Temporary prefill allocator** — freed immediately after prefill, so the large compute buffer does not persist into the generation loop
- **Codec on CPU** — the audio codec executes once per synthesis (decode only) or twice when a reference audio is provided (encode reference + decode output), so running it on CPU has zero impact on generation throughput
- **posix_fadvise(DONTNEED)** after loading the weights *(Linux only)* — advises the kernel to drop the GGUF file from page cache once the tensors are already in the backend buffer, reducing duplicate RAM use
- **Correct ByteLevel tokenization** — the GPT-2 byte-to-unicode table is applied before BPE, producing token IDs identical to the HuggingFace reference tokenizer

---

## Tips

### Long outputs

Voice quality and amplitude tend to degrade after ~800 tokens (~37 s of audio). For longer texts, split into sentences and concatenate the resulting WAV files. Optional post-processing flags such as `--dynamic-normalize`, `--normalize`, and `--trim-silence` can help clean up the result, but splitting remains the most reliable approach.

---

## Known limitations (alpha)

- No streaming output — WAV is written only after full generation completes
- No batch inference
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