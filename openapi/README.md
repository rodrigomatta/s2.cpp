# OpenAPI specification for s2.cpp HTTP server

This directory contains the OpenAPI 3.1 description for the built-in `--server`
mode.

## Files

- `s2-openapi.yaml` - machine-readable API description for `POST /generate`

## Scope

The current server exposes a single endpoint:

- `POST /generate` - synthesize speech from text, with optional reference audio
  for voice cloning

By default the response is a finalized `audio/wav` file. Real-time chunked
PCM16 WAV transport is used only when `params.stream=true` and
`params.chunked=true` (or `params.realtime=true`). That live transport starts
with a provisional WAV header because the server cannot seek back over an open
HTTP stream; saved files should still decode, but the header is not fully
finalized. `params.output_format="pcm_s16le"` skips the WAV container and
returns raw mono PCM16 bytes instead, which is easier for clients that keep a
dynamic playback buffer. The server handles one active synthesis at a time and
returns HTTP `503` when another request is already in progress.

## Request fields

Canonical multipart fields:

- `text`
- `reference`
- `reference_text`
- `voice`
- `voice_dir`
- `params`

Accepted aliases:

- `reference`: `reference_audio`, `prompt_audio`, `ref_audio`
- `reference_text`: `ref_text`, `prompt_text`
- `voice`: `voice_id`, `voice_profile`

The `params` field is a JSON string. The OpenAPI schema documents the supported
keys currently parsed by the server:

- `max_new_tokens`
- `temperature`
- `top_p`
- `top_k`
- `min_tokens_before_end`
- `n_threads`
- `codec_follow_backend`
- `codec_auto_backend`
- `codec_decode_context_frames`
- `codec_context_frames` (alias)
- `voice`
- `voice_id`
- `voice_dir`
- `stream`
- `chunked`
- `realtime`
- `stream_decode_stride_frames`
- `stream_decode_stride`
- `stream_holdback_frames`
- `stream_start_buffer_ms`
- `segment_sentences`
- `sentence_pause_ms`
- `segment_max_chars`
- `low_latency`
- `output_format`
- `verbose`

Notes:

- `codec_follow_backend=true` asks the codec to follow the selected GPU backend
  when possible; if codec GPU initialization or allocation fails, the runtime
  falls back to CPU.
- `codec_auto_backend=true` lets the runtime benchmark/select the codec backend
  automatically. When disabled, `codec_follow_backend` is applied directly.
- `voice=hope` resolves to `./voices/hope.s2voice` by default.
- `voice=voices/hope.s2voice` uses that exact file by inferring `voice_dir` and
  `voice_id` from the path.
- `stream_decode_stride_frames=0` keeps the server auto cadence, currently `4`
  frames.
- `stream_decode_stride` is an alias for `stream_decode_stride_frames`.
- `stream_holdback_frames=0` emits audio immediately after decode instead of
  waiting for the codec's full stability window. That is the lowest-latency
  option, but it can make chunk boundaries less stable.
- `stream_start_buffer_ms=4000` is a good starting point for natural playback
  with `ffplay` or similar pipe readers: the server waits for about four
  seconds of PCM before it begins the chunked response.
- `segment_sentences=true` switches from exact frame streaming to
  sentence-by-sentence synthesis, which is usually the most natural option on
  machines that cannot keep the exact streaming path under real time.
- `sentence_pause_ms=180` adds a short fixed pause between segmented sentences.
- `segment_max_chars` can force very long sentences to be broken into smaller
  clauses before synthesis.
- `low_latency=true` is a convenience preset for live playback: it defaults to
  stride `1` and holdback `0` unless you already overrode them explicitly.
- `output_format="pcm_s16le"` returns raw PCM16 bytes and adds
  `X-Audio-Sample-Rate`, `X-Audio-Channels`, and `X-Audio-Encoding` headers so
  the client can configure playback without parsing a WAV header. Clients
  should use those headers instead of assuming a fixed sample rate.

## Quick example

```bash
curl -X POST http://127.0.0.1:3030/generate \
  --form "text=Hello world" \
  --form 'params={"temperature":0.9,"top_p":0.9}' \
  -o output.wav
```

Reference audio must be accompanied by its transcript via `reference_text` or
one of its accepted aliases. Request validation failures, JSON parse failures,
and invalid `params` value types return `400`; synthesis failures detected
before the response body starts return `500`. Real-time chunked responses may
instead terminate early if generation fails after streaming has already begun.
