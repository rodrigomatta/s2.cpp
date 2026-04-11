#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = REPO_ROOT / "tokenizer.json"
DEFAULT_MODEL = REPO_ROOT / "s2-pro-q4_k_m.gguf"

if sys.platform == "win32":
    DEFAULT_LIBRARY = REPO_ROOT / "build" / "s2.dll"
elif sys.platform == "darwin":
    DEFAULT_LIBRARY = REPO_ROOT / "build" / "libs2.dylib"
else:
    DEFAULT_LIBRARY = REPO_ROOT / "build" / "libs2.so"

BACKENDS = {
    "cpu": -1,
    "vulkan": 0,
    "cuda": 1,
    "metal": 2,
}

LOG_LEVELS = {
    "error": 0,
    "warning": 1,
    "warn": 1,
    "info": 2,
    "debug": 3,
}

ERROR_CODES = {
    0: "pipeline not initialized or invalid parameters",
    -1: "reference audio encode failed; synthesis continued without reference",
    -4: "synthesis failed or produced no frames",
    -6: "failed to save output audio",
    -7: "reference audio/codes require a non-empty transcript",
    -8: "precomputed prompt codes require ReferenceAudioTPrompt",
    -9: "streaming callback configuration is invalid",
    -10: "streaming was aborted or cancelled by the client callback",
}

STREAM_PLAYBACK_SAMPLE_RATE = 44100


STREAM_WRITE_CB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int32,
    ctypes.c_void_p,
)
STREAM_DONE_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
STREAM_ERROR_CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)
STREAM_CANCEL_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class S2StreamingCallbacks(ctypes.Structure):
    _fields_ = [
        ("on_wav_chunk", STREAM_WRITE_CB),
        ("on_done", STREAM_DONE_CB),
        ("on_error", STREAM_ERROR_CB),
        ("is_cancelled", STREAM_CANCEL_CB),
        ("user_data", ctypes.c_void_p),
    ]


class S2StreamingParams(ctypes.Structure):
    _fields_ = [
        ("stream_decode_stride_frames", ctypes.c_int32),
        ("stream_holdback_frames", ctypes.c_int32),
        ("codec_decode_context_frames", ctypes.c_int32),
        ("low_latency", ctypes.c_int),
        ("segment_sentences", ctypes.c_int),
        ("sentence_pause_ms", ctypes.c_int32),
        ("segment_max_chars", ctypes.c_int32),
        ("voice", ctypes.c_char_p),
        ("voice_dir", ctypes.c_char_p),
    ]


def parse_log_level(value: str) -> int:
    normalized = value.lower()
    if normalized in LOG_LEVELS:
        return LOG_LEVELS[normalized]
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use error, warn, info, debug, or 0..3") from exc
    if parsed < 0 or parsed > 3:
        raise argparse.ArgumentTypeError("log level must be between 0 and 3")
    return parsed


def encode_optional(value: str | os.PathLike[str] | None) -> bytes | None:
    if value is None:
        return None
    value_str = os.fspath(value)
    if value_str == "":
        return None
    return value_str.encode("utf-8")


def require_pointer(name: str, pointer: int | None) -> int:
    if not pointer:
        raise RuntimeError(f"{name} returned NULL")
    return pointer


def bind_api(lib: ctypes.CDLL) -> ctypes.CDLL:
    c_void_p = ctypes.c_void_p
    c_char_p = ctypes.c_char_p
    c_int = ctypes.c_int
    c_int32 = ctypes.c_int32
    c_float = ctypes.c_float

    lib.SetS2LogLevel.argtypes = [c_int32]
    lib.SetS2LogLevel.restype = None
    lib.GetS2LogLevel.argtypes = []
    lib.GetS2LogLevel.restype = c_int32

    lib.AllocS2Pipeline.argtypes = []
    lib.AllocS2Pipeline.restype = c_void_p
    lib.ReleaseS2Pipeline.argtypes = [c_void_p]
    lib.ReleaseS2Pipeline.restype = None
    lib.InitializeS2PipelineFromFiles.argtypes = [
        c_void_p,
        c_char_p,
        c_char_p,
        c_int32,
        c_int32,
        c_int32,
        c_int,
    ]
    lib.InitializeS2PipelineFromFiles.restype = c_int

    lib.AllocS2GenerateParams.argtypes = []
    lib.AllocS2GenerateParams.restype = c_void_p
    lib.ReleaseS2GenerateParams.argtypes = [c_void_p]
    lib.ReleaseS2GenerateParams.restype = None
    lib.InitializeS2GenerateParams.argtypes = [
        c_void_p,
        c_int32,
        c_float,
        c_float,
        c_int32,
        c_int32,
        c_int32,
        c_int,
    ]
    lib.InitializeS2GenerateParams.restype = c_int

    lib.AllocS2Model.argtypes = []
    lib.AllocS2Model.restype = c_void_p
    lib.ReleaseS2Model.argtypes = [c_void_p]
    lib.ReleaseS2Model.restype = None
    lib.AllocS2Tokenizer.argtypes = []
    lib.AllocS2Tokenizer.restype = c_void_p
    lib.ReleaseS2Tokenizer.argtypes = [c_void_p]
    lib.ReleaseS2Tokenizer.restype = None
    lib.AllocS2AudioCodec.argtypes = []
    lib.AllocS2AudioCodec.restype = c_void_p
    lib.ReleaseS2AudioCodec.argtypes = [c_void_p]
    lib.ReleaseS2AudioCodec.restype = None

    lib.AllocS2AudioPromptCodes.argtypes = []
    lib.AllocS2AudioPromptCodes.restype = c_void_p
    lib.ReleaseS2AudioPromptCodes.argtypes = [c_void_p]
    lib.ReleaseS2AudioPromptCodes.restype = None

    lib.AllocS2AudioBuffer.argtypes = [c_int]
    lib.AllocS2AudioBuffer.restype = c_void_p
    lib.ReleaseS2AudioBuffer.argtypes = [c_void_p]
    lib.ReleaseS2AudioBuffer.restype = None
    lib.GetS2AudioBufferDataPointer.argtypes = [c_void_p]
    lib.GetS2AudioBufferDataPointer.restype = ctypes.POINTER(c_float)

    lib.S2Synthesize.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        ctypes.POINTER(c_int32),
        c_char_p,
        c_char_p,
        c_char_p,
        c_char_p,
        ctypes.POINTER(c_int32),
    ]
    lib.S2Synthesize.restype = c_int

    lib.S2SynthesizeStreaming.argtypes = [
        c_void_p,
        c_void_p,
        ctypes.POINTER(S2StreamingCallbacks),
        c_void_p,
        ctypes.POINTER(c_int32),
        c_char_p,
        c_char_p,
        c_char_p,
        c_int32,
    ]
    lib.S2SynthesizeStreaming.restype = c_int

    try:
        lib.S2SynthesizeStreamingEx.argtypes = [
            c_void_p,
            c_void_p,
            ctypes.POINTER(S2StreamingCallbacks),
            c_void_p,
            ctypes.POINTER(c_int32),
            c_char_p,
            c_char_p,
            c_char_p,
            ctypes.POINTER(S2StreamingParams),
        ]
        lib.S2SynthesizeStreamingEx.restype = c_int
        lib._has_streaming_ex = True
    except AttributeError:
        lib._has_streaming_ex = False

    return lib


def load_library(path: Path) -> ctypes.CDLL:
    try:
        return bind_api(ctypes.CDLL(str(path)))
    except OSError as exc:
        raise RuntimeError(
            f"failed to load {path}. If ggml dependencies are not discoverable, "
            "run from the build tree or set LD_LIBRARY_PATH to the build ggml dirs."
        ) from exc


def patch_wav_header(path: Path, total_size: int) -> None:
    if total_size < 44 or total_size > 0xFFFFFFFF:
        return
    riff_size = total_size - 8
    data_size = total_size - 44
    with path.open("r+b") as output_file:
        output_file.seek(4)
        output_file.write(struct.pack("<I", riff_size))
        output_file.seek(40)
        output_file.write(struct.pack("<I", data_size))


def run_smoke(lib: ctypes.CDLL, log_level: int) -> None:
    lib.SetS2LogLevel(log_level)

    pipeline = params = audio_buffer = prompt_codes = None
    model = tokenizer = codec = None
    try:
        pipeline = require_pointer("AllocS2Pipeline", lib.AllocS2Pipeline())
        params = require_pointer("AllocS2GenerateParams", lib.AllocS2GenerateParams())
        audio_buffer = require_pointer("AllocS2AudioBuffer", lib.AllocS2AudioBuffer(-1))
        prompt_codes = require_pointer("AllocS2AudioPromptCodes", lib.AllocS2AudioPromptCodes())
        model = require_pointer("AllocS2Model", lib.AllocS2Model())
        tokenizer = require_pointer("AllocS2Tokenizer", lib.AllocS2Tokenizer())
        codec = require_pointer("AllocS2AudioCodec", lib.AllocS2AudioCodec())

        ok = lib.InitializeS2GenerateParams(params, -1, -1.0, -1.0, -1, -1, -1, 0)
        if ok != 1:
            raise RuntimeError(f"InitializeS2GenerateParams failed with {ok}")

        empty_data = lib.GetS2AudioBufferDataPointer(audio_buffer)
        if bool(empty_data):
            raise RuntimeError("new audio buffer should not expose sample data before synthesis")

        print(f"loaded library: log_level={lib.GetS2LogLevel()}")
        print("smoke test: exported symbols, allocators, params, and audio buffer OK")
    finally:
        if codec:
            lib.ReleaseS2AudioCodec(codec)
        if tokenizer:
            lib.ReleaseS2Tokenizer(tokenizer)
        if model:
            lib.ReleaseS2Model(model)
        if prompt_codes:
            lib.ReleaseS2AudioPromptCodes(prompt_codes)
        if audio_buffer:
            lib.ReleaseS2AudioBuffer(audio_buffer)
        if params:
            lib.ReleaseS2GenerateParams(params)
        if pipeline:
            lib.ReleaseS2Pipeline(pipeline)


class StreamingSession:

    def __init__(self, output_path: Path | None, play: bool, play_only: bool, play_device: str = ""):
        self.output_path = output_path
        self.play = play or play_only
        self.play_only = play_only
        self.play_device = play_device
        self.chunk_count = 0
        self.byte_count = 0
        self.done_called = False
        self.header_skipped = not self.play
        self.errors: list[str] = []
        self.callback_errors: list[str] = []
        self.output_file = None
        self.player_proc = None

    def start(self) -> None:
        if not self.play_only and self.output_path:
            self.output_file = self.output_path.open("wb")

        if self.play:
            aplay_path = shutil.which("aplay")
            if not aplay_path:
                raise RuntimeError("aplay not found; install alsa-utils or disable --play")

            player_cmd = [
                aplay_path,
                "-q",
                "-f", "S16_LE",
                "-r", str(STREAM_PLAYBACK_SAMPLE_RATE),
                "-c", "1",
            ]
            if self.play_device:
                player_cmd.extend(["-D", self.play_device])
            self.player_proc = subprocess.Popen(player_cmd, stdin=subprocess.PIPE)

    def build_callbacks(self) -> S2StreamingCallbacks:
        session = self

        @STREAM_WRITE_CB
        def on_wav_chunk(data_ptr, size, _user_data):
            try:
                chunk = ctypes.string_at(data_ptr, size)
                if session.output_file is not None:
                    session.output_file.write(chunk)
                    session.output_file.flush()

                if session.player_proc is not None and session.player_proc.stdin is not None:
                    pcm_chunk = chunk
                    if not session.header_skipped:
                        pcm_chunk = pcm_chunk[44:] if len(pcm_chunk) >= 44 else b""
                        session.header_skipped = True
                    if pcm_chunk:
                        session.player_proc.stdin.write(pcm_chunk)
                        session.player_proc.stdin.flush()
            except Exception as exc:
                session.callback_errors.append(str(exc))
                return 0
            session.chunk_count += 1
            session.byte_count += int(size)
            return 1

        @STREAM_DONE_CB
        def on_done(_user_data):
            session.done_called = True

        @STREAM_ERROR_CB
        def on_error(message, _user_data):
            session.errors.append(message.decode("utf-8") if message else "")

        @STREAM_CANCEL_CB
        def is_cancelled(_user_data):
            return 0

        return S2StreamingCallbacks(
            on_wav_chunk=on_wav_chunk,
            on_done=on_done,
            on_error=on_error,
            is_cancelled=is_cancelled,
            user_data=None,
        )

    def finalize_output(self) -> None:
        if self.output_file is not None:
            self.output_file.close()
        if self.player_proc is not None:
            if self.player_proc.stdin is not None:
                self.player_proc.stdin.close()
            self.player_proc.wait(timeout=10)

        if not self.play_only and self.done_called and self.byte_count >= 44 and self.output_path:
            patch_wav_header(self.output_path, self.byte_count)

    def print_result(self, function_name: str, code: int) -> None:
        message = ERROR_CODES.get(code, "success" if code > 0 else "unknown error")
        print(f"{function_name} returned {code}: {message}")
        print(
            f"chunks={self.chunk_count} bytes={self.byte_count} "
            f"output={self.output_path if not self.play_only else 'disabled'} "
            f"done={self.done_called}"
        )
        if self.errors:
            print(f"stream_errors={self.errors}")
        if self.callback_errors:
            print(f"callback_errors={self.callback_errors}")


def run_synthesis(lib: ctypes.CDLL, args: argparse.Namespace) -> None:
    lib.SetS2LogLevel(args.log_level)

    pipeline = params = audio_buffer = prompt_codes = None
    try:
        pipeline = require_pointer("AllocS2Pipeline", lib.AllocS2Pipeline())
        params = require_pointer("AllocS2GenerateParams", lib.AllocS2GenerateParams())
        audio_buffer = require_pointer("AllocS2AudioBuffer", lib.AllocS2AudioBuffer(-1))
        prompt_codes = require_pointer("AllocS2AudioPromptCodes", lib.AllocS2AudioPromptCodes())

        ok = lib.InitializeS2GenerateParams(
            params,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.min_tokens_before_end,
            args.threads,
            0,
        )
        if ok != 1:
            raise RuntimeError(f"InitializeS2GenerateParams failed with {ok}")

        ok = lib.InitializeS2PipelineFromFiles(
            pipeline,
            encode_optional(args.model),
            encode_optional(args.tokenizer),
            args.gpu_device,
            BACKENDS[args.backend],
            args.gpu_layers,
            0 if args.codec_cpu else 1,
        )
        if ok != 1:
            raise RuntimeError(f"InitializeS2PipelineFromFiles failed with {ok}")

        t_prompt = ctypes.c_int32(0)
        if args.streaming:
            session = StreamingSession(
                output_path=args.output if not args.play_only else None,
                play=args.play,
                play_only=args.play_only,
                play_device=args.play_device,
            )
            session.start()
            callbacks = session.build_callbacks()

            wants_extended_streaming = (
                bool(args.low_latency)
                or args.stream_holdback_frames is not None
                or args.codec_context_frames is not None
                or args.segment_sentences
                or bool(args.voice)
                or bool(args.voice_dir)
            )

            if wants_extended_streaming and not getattr(lib, "_has_streaming_ex", False):
                session.finalize_output()
                raise RuntimeError(
                    "library does not export S2SynthesizeStreamingEx; "
                    "rebuild s2.cpp or avoid --low-latency/holdback/context options"
                )

            if getattr(lib, "_has_streaming_ex", False):
                streaming_params = S2StreamingParams(
                    stream_decode_stride_frames=args.stream_decode_stride,
                    stream_holdback_frames=(
                        args.stream_holdback_frames
                        if args.stream_holdback_frames is not None
                        else -1
                    ),
                    codec_decode_context_frames=(
                        args.codec_context_frames
                        if args.codec_context_frames is not None
                        else -1
                    ),
                    low_latency=1 if args.low_latency else 0,
                    segment_sentences=1 if args.segment_sentences else 0,
                    sentence_pause_ms=args.sentence_pause_ms,
                    segment_max_chars=args.segment_max_chars,
                    voice=encode_optional(args.voice),
                    voice_dir=encode_optional(args.voice_dir),
                )
                code = lib.S2SynthesizeStreamingEx(
                    pipeline,
                    params,
                    ctypes.byref(callbacks),
                    prompt_codes,
                    ctypes.byref(t_prompt),
                    encode_optional(args.reference_audio),
                    encode_optional(args.reference_text),
                    encode_optional(args.text),
                    ctypes.byref(streaming_params),
                )
            else:
                code = lib.S2SynthesizeStreaming(
                    pipeline,
                    params,
                    ctypes.byref(callbacks),
                    prompt_codes,
                    ctypes.byref(t_prompt),
                    encode_optional(args.reference_audio),
                    encode_optional(args.reference_text),
                    encode_optional(args.text),
                    args.stream_decode_stride,
                )

            session.finalize_output()
            session.print_result("S2SynthesizeStreaming", code)

            if code <= -4 or code in (0, -7, -8, -9, -10):
                raise SystemExit(1)
        else:
            sample_count = ctypes.c_int32(0)
            code = lib.S2Synthesize(
                pipeline,
                params,
                audio_buffer,
                prompt_codes,
                ctypes.byref(t_prompt),
                encode_optional(args.reference_audio),
                encode_optional(args.reference_text),
                encode_optional(args.text),
                encode_optional(args.output),
                ctypes.byref(sample_count),
            )

            message = ERROR_CODES.get(code, "success" if code > 0 else "unknown error")
            print(f"S2Synthesize returned {code}: {message}")
            print(f"samples={sample_count.value} output={args.output}")

            data = lib.GetS2AudioBufferDataPointer(audio_buffer)
            if sample_count.value > 0 and bool(data):
                preview_count = min(5, sample_count.value)
                preview = [round(float(data[index]), 6) for index in range(preview_count)]
                print(f"first_samples={preview}")

            if code <= -4 or code in (0, -6, -7, -8):
                raise SystemExit(1)
    finally:
        if prompt_codes:
            lib.ReleaseS2AudioPromptCodes(prompt_codes)
        if audio_buffer:
            lib.ReleaseS2AudioBuffer(audio_buffer)
        if params:
            lib.ReleaseS2GenerateParams(params)
        if pipeline:
            lib.ReleaseS2Pipeline(pipeline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load s2.cpp's exported shared library from Python with ctypes.",
        epilog="""
Examples:
  python3 %(prog)s --smoke-only
  python3 %(prog)s --model model.gguf --text "Hello world"
  python3 %(prog)s --model model.gguf --streaming --low-latency
  python3 %(prog)s --model model.gguf --streaming --segment-sentences --voice hope
  python3 %(prog)s --model model.gguf --streaming --voice path.s2voice --play-only
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY,
                        help="Path to libs2.so, s2.dll, or libs2.dylib")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Validate import/symbols without loading a model")

    parser.add_argument("--model", type=Path,
                        default=DEFAULT_MODEL if DEFAULT_MODEL.exists() else None,
                        help="Path to the model GGUF file")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER,
                        help="Path to tokenizer.json")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="cpu",
                        help="Compute backend: cpu, vulkan, cuda, or metal")
    parser.add_argument("--gpu-device", type=int, default=-1,
                        help="GPU device index (default: -1 = auto)")
    parser.add_argument("--gpu-layers", type=int, default=-1,
                        help="Number of model layers to offload to GPU (default: -1 = all)")
    parser.add_argument("--codec-cpu", action="store_true",
                        help="Keep codec on CPU when using a GPU backend")

    parser.add_argument("--text", default="This is a short Python library test.",
                        help="Text to synthesize")
    parser.add_argument("--output", type=Path, default=Path("/tmp/s2_python_example.wav"),
                        help="Output WAV file path")
    parser.add_argument("--reference-audio", type=Path,
                        help="Path to reference audio for voice cloning")
    parser.add_argument("--reference-text", default="",
                        help="Transcript for the reference audio")
    parser.add_argument("--voice", default="",
                        help="Saved voice id or direct path to a .s2voice profile")
    parser.add_argument("--voice-dir", default="",
                        help="Voice storage directory used with --voice when passing an id")
    parser.add_argument("--max-tokens", type=int, default=-1,
                        help="Maximum tokens to generate (default: -1 = model default)")
    parser.add_argument("--temperature", type=float, default=-1.0,
                        help="Sampling temperature (default: -1 = model default)")
    parser.add_argument("--top-p", type=float, default=-1.0,
                        help="Top-p sampling (default: -1 = model default)")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Top-k sampling (default: -1 = model default)")
    parser.add_argument("--min-tokens-before-end", type=int, default=-1,
                        help="Minimum tokens before end token (default: -1 = model default)")
    parser.add_argument("--threads", type=int, default=-1,
                        help="Number of CPU threads (default: -1 = auto)")

    parser.add_argument("--streaming", action="store_true",
                        help="Use callback-based streaming synthesis")
    parser.add_argument("--play", action="store_true",
                        help="Play streamed PCM16 in real time with aplay while also saving the WAV")
    parser.add_argument("--play-only", action="store_true",
                        help="Play streamed PCM16 in real time with aplay without saving the WAV")
    parser.add_argument("--play-device", default="",
                        help="Optional ALSA device name for aplay, e.g. default or hw:0,0")
    parser.add_argument("--stream-decode-stride", type=int, default=0,
                        help="Streaming decode cadence in frames (0 = library default)")
    parser.add_argument("--stream-holdback-frames", type=int, default=None,
                        help="Keep this many trailing frames buffered before callback delivery")
    parser.add_argument("--codec-context-frames", type=int, default=None,
                        help="Override codec streaming context window")
    parser.add_argument("--low-latency", action="store_true",
                        help="Match the HTTP API low_latency behavior (stride=1, holdback=0 unless overridden)")
    parser.add_argument("--segment-sentences", action="store_true",
                        help="Split synthesis on sentence-ending punctuation/newlines like HTTP API segmented mode")
    parser.add_argument("--sentence-pause-ms", type=int, default=180,
                        help="Pause inserted between segmented sentences (ms)")
    parser.add_argument("--segment-max-chars", type=int, default=0,
                        help="Further split long segmented sentences near punctuation/whitespace")

    parser.add_argument("--log-level", type=parse_log_level, default=0,
                        help="Log level: error, warn, info, debug, or 0-3")

    args = parser.parse_args()

    if args.play and args.play_only:
        parser.error("use either --play or --play-only, not both")

    args.library = args.library.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.output = args.output.resolve()
    if args.model is not None:
        args.model = args.model.resolve()
    if args.reference_audio is not None:
        args.reference_audio = args.reference_audio.resolve()
    if args.voice:
        args.voice = os.fspath(args.voice)
    if args.voice_dir:
        args.voice_dir = os.fspath(args.voice_dir)

    return args


def main() -> None:
    args = parse_args()

    if not args.library.exists():
        raise SystemExit(f"library not found: {args.library}\n"
                         f"Build s2.cpp first or pass --library <path/to/libs2.so>")

    lib = load_library(args.library)

    if args.smoke_only:
        run_smoke(lib, args.log_level)
        return

    if args.model is None or not args.model.exists():
        raise SystemExit(
            f"model not found: {args.model}\n"
            f"Pass --model <path.gguf> or run with --smoke-only"
        )
    if not args.tokenizer.exists():
        raise SystemExit(f"tokenizer not found: {args.tokenizer}\n"
                         f"Pass --tokenizer <path/to/tokenizer.json>")

    if args.reference_audio is not None and not args.reference_audio.exists():
        raise SystemExit(f"reference audio not found: {args.reference_audio}")

    if not args.play_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    run_synthesis(lib, args)


if __name__ == "__main__":
    main()
