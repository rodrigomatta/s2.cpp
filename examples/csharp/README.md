# C# Export API Example

This directory contains a modernized C# example for the current exported `s2.cpp` library API.

## Files

- `S2ExportApiExample.csproj`: minimal .NET console project
- `Program.cs`: P/Invoke bindings for every current exported function plus sample flows

## Modes

The sample supports five modes:

- `smoke`: validates that the exported symbols can be loaded and basic allocators work
- `from-files`: initializes the pipeline directly with `InitializeS2PipelineFromFiles()` and runs one-shot synthesis
- `modular`: mirrors the original SubSpecs design by creating tokenizer/model/codec separately, syncing tokenizer config from the model, binding them into a reusable pipeline, optionally precomputing prompt codes, and synthesizing
- `legacy-stream`: demonstrates `S2SynthesizeStreaming()`
- `stream-ex`: demonstrates `S2SynthesizeStreamingEx()` with the current extended options such as low latency, sentence segmentation, and `.s2voice`

## Example Commands

One-shot modular flow close to the original idea:

```bash
dotnet run --project examples/csharp/S2ExportApiExample.csproj -- \
  --mode modular \
  --library build-vulkan/libs2.so \
  --model /path/to/s2-pro-q2_K.gguf \
  --tokenizer /path/to/tokenizer.json \
  --backend vulkan \
  --gpu-device 0 \
  --threads 8 \
  --reference-audio /path/to/reference.wav \
  --reference-text "Transcript of the reference audio." \
  --text "Hello from the modular C# example." \
  --output /tmp/s2_csharp_modular.wav
```

Streaming with the current extended API, using a saved `.s2voice` profile:

```bash
dotnet run --project examples/csharp/S2ExportApiExample.csproj -- \
  --mode stream-ex \
  --library build-vulkan/libs2.so \
  --model /path/to/s2-pro-q2_K.gguf \
  --tokenizer /path/to/tokenizer.json \
  --backend vulkan \
  --gpu-device 0 \
  --threads 8 \
  --text "First sentence. Second, longer sentence to confirm the pause." \
  --segment-sentences \
  --sentence-pause-ms 180 \
  --voice /path/to/voices/hope.s2voice \
  --output /tmp/s2_csharp_stream.wav \
  --log-level info
```

## Comparison To The Original SubSpecs Idea

What stayed the same:

- The example still treats `Pipeline`, `Tokenizer`, `Model`, `AudioCodec`, prompt codes, and audio buffers as opaque handles owned by the native library.
- The modular flow is still first-class: load the components separately, synchronize tokenizer config from the model, bind them into a pipeline, and reuse precomputed prompt codes.
- The C# side still maps well to `DllImport` and pointer-style FFI usage rather than requiring a pure-C header redesign.

What changed in the updated example:

- The sample now binds every current export, not just the original one-shot subset.
- Error handling reflects the current return codes instead of the older `-2/-3/-5` assumptions from the early sample.
- A runtime library-path resolver replaced the original hardcoded DLL path, so the example can load `libs2.so`, `s2.dll`, or `libs2.dylib` from a command-line argument.
- `InitializeS2PipelineFromFiles()` is covered as a convenience path in addition to the original modular path.
- The new example covers `S2SynthesizeStreaming()` and `S2SynthesizeStreamingEx()`, including low-latency streaming, sentence segmentation, and saved `.s2voice` loading.

What is still intentionally not "pure C":

- Handles are still modeled as opaque native objects owned by the C++ library, which matches the original intent.
- The example is designed for practical C# / `DllImport` interop, not for compiling the public header as strict C.
