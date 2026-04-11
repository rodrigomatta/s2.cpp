# Python Export API Example

This directory contains a Python `ctypes` example for the exported `s2.cpp` library API.

## Files

- `ctypes_export_api.py`: standalone Python script that binds every current exported function and demonstrates smoke testing, batch synthesis, streaming, extended streaming, and real-time playback

## Requirements

- Python 3.10+
- Linux with `aplay` (from `alsa-utils`) for `--play` / `--play-only` playback
- A built `libs2.so` / `libs2.dylib` / `s2.dll` and a model GGUF

## Modes

The example supports two main modes:

- `--smoke-only`: validates that the exported symbols can be loaded and basic allocators work
- **Full synthesis**: initializes the pipeline with `InitializeS2PipelineFromFiles()` and runs either batch synthesis via `S2Synthesize()` or streaming via `S2SynthesizeStreaming()` / `S2SynthesizeStreamingEx()`

## Example Commands

### Smoke test

```bash
LD_LIBRARY_PATH=build-vulkan:build-vulkan/ggml/src \
python3 examples/python/ctypes_export_api.py \
  --library build-vulkan/libs2.so \
  --smoke-only
```

### Batch synthesis (Vulkan)

```bash
LD_LIBRARY_PATH=build-vulkan:build-vulkan/ggml/src \
python3 examples/python/ctypes_export_api.py \
  --library build-vulkan/libs2.so \
  --model /path/to/s2-pro-q2_K.gguf \
  --tokenizer /path/to/tokenizer.json \
  --backend vulkan \
  --gpu-device 0 \
  --threads 8 \
  --max-tokens 64 \
  --text "Hello from the Python export API." \
  --output /tmp/s2_python.wav \
  --log-level 2
```

### Streaming with low-latency defaults

```bash
LD_LIBRARY_PATH=build-vulkan:build-vulkan/ggml/src \
python3 examples/python/ctypes_export_api.py \
  --library build-vulkan/libs2.so \
  --model /path/to/s2-pro-q2_K.gguf \
  --tokenizer /path/to/tokenizer.json \
  --backend vulkan \
  --gpu-device 0 \
  --threads 8 \
  --max-tokens 32 \
  --text "Hello in low-latency mode." \
  --streaming \
  --low-latency \
  --output /tmp/s2_python_stream.wav \
  --log-level 2
```

### Sentence-segmented streaming with a saved `.s2voice` profile

```bash
LD_LIBRARY_PATH=build-vulkan:build-vulkan/ggml/src \
python3 examples/python/ctypes_export_api.py \
  --library build-vulkan/libs2.so \
  --model /path/to/s2-pro-q2_K.gguf \
  --tokenizer /path/to/tokenizer.json \
  --backend vulkan \
  --gpu-device 0 \
  --threads 8 \
  --max-tokens 64 \
  --text "First sentence. Second, longer sentence to confirm the pause." \
  --streaming \
  --segment-sentences \
  --sentence-pause-ms 180 \
  --voice /path/to/voices/hope.s2voice \
  --output /tmp/s2_python_voice.wav \
  --log-level 2
```

### Real-time playback on Linux (no WAV saved)

```bash
LD_LIBRARY_PATH=build-vulkan:build-vulkan/ggml/src \
python3 examples/python/ctypes_export_api.py \
  --library build-vulkan/libs2.so \
  --model /path/to/s2-pro-q2_K.gguf \
  --tokenizer /path/to/tokenizer.json \
  --backend vulkan \
  --gpu-device 0 \
  --threads 8 \
  --max-tokens 64 \
  --text "First sentence. Second, longer sentence to confirm the pause." \
  --streaming \
  --segment-sentences \
  --sentence-pause-ms 180 \
  --voice /path/to/voices/hope.s2voice \
  --play-only \
  --log-level 2
```

### Load voice by id instead of by path

Replace the `--voice` path with the voice id and add `--voice-dir`:

```bash
LD_LIBRARY_PATH=build-vulkan:build-vulkan/ggml/src \
python3 examples/python/ctypes_export_api.py \
  --library build-vulkan/libs2.so \
  --model /path/to/s2-pro-q2_K.gguf \
  --tokenizer /path/to/tokenizer.json \
  --backend vulkan \
  --gpu-device 0 \
  --threads 8 \
  --max-tokens 64 \
  --text "Hello from a saved voice profile." \
  --streaming \
  --segment-sentences \
  --sentence-pause-ms 180 \
  --voice hope \
  --voice-dir /path/to/voices \
  --play-only \
  --log-level 2
```

## Extended Streaming API

The example uses `S2SynthesizeStreamingEx()` when any of these flags are present:

| Flag | Description |
|---|---|
| `--low-latency` | Sets `stream_decode_stride_frames=1` and `stream_holdback_frames=0` unless overridden |
| `--segment-sentences` | Splits text on sentence-ending punctuation/newlines before synthesis |
| `--sentence-pause-ms <ms>` | Pause inserted between segmented sentences (default: 180) |
| `--segment-max-chars <n>` | Further splits long segments near punctuation/whitespace |
| `--voice <id-or-path>` | Saved voice id or direct `.s2voice` path |
| `--voice-dir <dir>` | Voice storage directory for id-based lookup |
| `--stream-holdback-frames <n>` | Override trailing frame holdback |
| `--codec-context-frames <n>` | Override codec streaming context window |

When none of these flags are present, the example falls back to `S2SynthesizeStreaming()`.

## Playback

`--play` saves the WAV file while streaming PCM16 to `aplay` in real time. `--play-only` streams directly to `aplay` without saving a file. Both require `aplay` from `alsa-utils` and are Linux-only.
