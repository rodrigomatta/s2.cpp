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

| File               | Size   | Notes                                                  |
| ------------------ | ------ | ------------------------------------------------------ |
| `s2-pro-f16.gguf`  | 9.3 GB | Full precision — reference quality                     |
| `s2-pro-q8_0.gguf` | 5.7 GB | Near-lossless — recommended for 8+ GB VRAM             |
| `s2-pro-q6_k.gguf` | 4.8 GB | Good quality/size balance — recommended for 6+ GB VRAM |

All variants include both the transformer weights and the audio codec in a single file.

---

## Requirements

### Build dependencies

- CMake ≥ 3.14
- C++17 compiler (GCC ≥ 10, Clang ≥ 11, MSVC 2019+)
- For Vulkan GPU support: Vulkan SDK and `glslc`
- For CUDA GPU support: CUDA Toolkit ≥ 12.4
    - **MSVC 2019+ note:** MSVC 2019 and later require CUDA ≥ 12.4 when building GGML. Older CUDA versions will produce compiler compatibility errors; upgrade to 12.4+ to resolve them.

```bash
# Ubuntu / Debian
sudo apt install cmake build-essential

# Vulkan (optional, recommended for GPU acceleration on AMD/Intel/NVIDIA)
sudo apt install vulkan-tools libvulkan-dev glslc

# CUDA (optional, recommended for NVIDIA GPUs)
# Install the CUDA Toolkit ≥ 12.4 from https://developer.nvidia.com/cuda-downloads
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

### With both Vulkan and CUDA (select backend at runtime)

You can compile with both backends enabled simultaneously and choose which one to use at runtime via `--vulkan` or `--cuda`:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_VULKAN=ON -DS2_CUDA=ON
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

`tokenizer.json` is searched automatically in the same directory as the model file, then the parent directory, then the working directory.

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

### GPU inference via Vulkan

```bash
./build/s2 \
  -m s2-pro-q6_k.gguf \
  -t tokenizer.json \
  -text "Text to synthesize." \
  --vulkan 0 \
  -o output.wav
```

`--vulkan 0` selects the first Vulkan device. The transformer runs on GPU; the audio codec always runs on CPU (executes only twice per synthesis).

### GPU inference via CUDA

```bash
./build/s2 \
  -m s2-pro-q6_k.gguf \
  -t tokenizer.json \
  -text "Text to synthesize." \
  --cuda 0 \
  -o output.wav
```

`--cuda 0` selects the first CUDA device. As with Vulkan, the transformer runs on GPU and the audio codec on CPU.

### All options

| Flag                    | Default          | Description                                            |
| ----------------------- | ---------------- | ------------------------------------------------------ |
| `-m`, `--model`         | `model.gguf`     | Path to GGUF model file                                |
| `-t`, `--tokenizer`     | `tokenizer.json` | Path to tokenizer.json                                 |
| `-text`                 | `"Hello world"`  | Text to synthesize                                     |
| `-pa`, `--prompt-audio` | —                | Reference audio file for voice cloning (WAV/MP3)       |
| `-pt`, `--prompt-text`  | —                | Transcript of the reference audio                      |
| `-o`, `--output`        | `out.wav`        | Output WAV file path                                   |
| `--vulkan N`            | —                | Use Vulkan backend, device index N (e.g. `--vulkan 0`) |
| `--cuda N`              | —                | Use CUDA backend, device index N (e.g. `--cuda 0`)     |
| `-threads N`            | `4`              | Number of CPU threads                                  |
| `-max-tokens N`         | `512`            | Max tokens to generate (~21s of audio per 440 tokens)  |
| `-temp F`               | `0.7`            | Sampling temperature                                   |
| `-top-p F`              | `0.7`            | Top-p nucleus sampling                                 |
| `-top-k N`              | `30`             | Top-k sampling                                         |

---

## Choosing a model

| VRAM available | Recommended model                                                                 |
| -------------- | --------------------------------------------------------------------------------- |
| ≥ 10 GB        | `q8_0` — near-lossless quality                                                    |
| 6–9 GB         | `q6_k` — good quality/size balance                                                |
| < 6 GB         | `f16` on CPU (slow) — no GPU variant at this quality level is currently available |

VRAM usage at runtime is approximately equal to the file size (transformer weights only; codec runs on CPU).

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
- **Codec on CPU** — the audio codec executes exactly twice per synthesis (encode reference + decode output), so running it on CPU has zero impact on generation throughput
- **posix_fadvise(DONTNEED)** after mmap — releases the GGUF file from kernel page cache after weights are loaded to VRAM, preventing RAM duplication equal to the model file size
- **Correct ByteLevel tokenization** — the GPT-2 byte-to-unicode table is applied before BPE, producing token IDs identical to the HuggingFace reference tokenizer

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
- Attribution: _"This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved."_

Full license: [LICENSE.md](LICENSE.md)

Commercial licensing: [https://fish.audio](https://fish.audio) · [business@fish.audio](mailto:business@fish.audio)

The inference engine source code (`src/`) is a Derivative Work of the Fish Audio Materials as defined in the Agreement and is distributed under the same Fish Audio Research License terms.

---

## WebUI

A graphical interface is available via `app.py`.

### Install Dependencies

```bash
pip install gradio
```

### Run the WebUI

```bash
python app.py
```

or run start_webui.bat

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### Features

- **Easy Synthesis**: Type text and click "Synthesize"
- **Voice Cloning**: Upload a reference audio clip and provide its transcript in the "Voice Cloning" accordion
- **Backend Selection**: Switch between CPU, CUDA, and Vulkan
- **Adjustable Tokens**: Control the chunk size for long-form generation

> Ensure you have your `.gguf` model file in the `models` folder
