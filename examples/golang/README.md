# Go Export API Example

Go example for the current exported `s2.cpp` library API, mirroring the C# example at `../csharp/`.

## Files

- `go.mod`: minimal Go module definition
- `main.go`: CGo + dlopen bindings for every current exported function plus sample flows

## Approach

The example uses CGo with `dlopen`/`dlsym` for **runtime** library loading. There is no build-time dependency on `libs2.so` — the shared library path is passed via `--library` at invocation, exactly like the C# example's `NativeLibrary.SetDllImportResolver`.

Streaming callbacks use `//export` Go functions with a `sync.Map` for session state lookup (equivalent to the C# `GCHandle` pattern).

## Modes

The sample supports five modes, identical to the C# example:

- `smoke`: validates that the exported symbols can be loaded and basic allocators work
- `from-files`: initializes the pipeline directly with `InitializeS2PipelineFromFiles()` and runs one-shot synthesis
- `modular`: creates tokenizer/model/codec separately, syncs tokenizer config from the model, binds them into a reusable pipeline, optionally precomputes prompt codes, and synthesizes
- `legacy-stream`: demonstrates `S2SynthesizeStreaming()`
- `stream-ex`: demonstrates `S2SynthesizeStreamingEx()` with low latency, sentence segmentation, and `.s2voice`

## Build & Run

```bash
# Build
cd examples/golang
go build -o s2-example .

# Smoke test (just loads the library and checks symbols)
./s2-example --mode smoke --library /path/to/libs2.so

# One-shot modular synthesis
./s2-example \
  --mode modular \
  --library /path/to/libs2.so \
  --model /path/to/model.gguf \
  --tokenizer /path/to/tokenizer.json \
  --backend vulkan \
  --gpu-device 0 \
  --threads 8 \
  --reference-audio /path/to/ref.wav \
  --reference-text "Reference transcript." \
  --text "Hello from the Go example." \
  --output /tmp/s2_go_modular.wav

# Streaming with extended API and .s2voice
./s2-example \
  --mode stream-ex \
  --library /path/to/libs2.so \
  --model /path/to/model.gguf \
  --tokenizer /path/to/tokenizer.json \
  --backend vulkan \
  --gpu-device 0 \
  --threads 8 \
  --text "First sentence. Second sentence to confirm the pause." \
  --segment-sentences \
  --sentence-pause-ms 180 \
  --voice /path/to/hope.s2voice \
  --output /tmp/s2_go_stream.wav \
  --log-level info
```

## Platform Support

The dlopen approach works on **Linux** and **macOS** out of the box. For Windows, replace `dlopen`/`dlsym` with `LoadLibraryA`/`GetProcAddress` in the CGo preamble and use `#cgo windows LDFLAGS: -lkernel32`.

## Comparison To The C# Example

| Aspect | C# | Go |
|--------|----|----|
| FFI mechanism | P/Invoke (`DllImport`) | CGo + dlopen |
| Runtime loading | `NativeLibrary.SetDllImportResolver` | `dlopen` at program start |
| Callbacks | `delegate` + `GCHandle` | `//export` + `sync.Map` |
| Opaque handles | `nint` (`IntPtr`) | `unsafe.Pointer` |
| WAV header patch | `FileStream.Position` | `os.File.Seek` |
| String interop | `MarshalAs(UnmanagedType.LPUTF8Str)` | `C.CString` / `C.GoString` |
| Error handling | exceptions + `ExpectSuccess` | `fatal` / `expectSuccess` |
| CLI parsing | manual `for` loop | manual `for` loop (same `--flag value` syntax) |
