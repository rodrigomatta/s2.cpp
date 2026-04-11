using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;

namespace S2ExportApiExample;

internal static unsafe class Program
{
    private static int Main(string[] args)
    {
        try
        {
            Options options = Options.Parse(args);
            NativeLibraryBootstrap.Configure(options.LibraryPath);
            NativeMethods.SetS2LogLevel((int) options.LogLevel);

            return options.Mode switch
            {
                RunMode.Smoke        => RunSmoke(),
                RunMode.FromFiles    => RunFromFiles(options),
                RunMode.Modular      => RunModular(options),
                RunMode.LegacyStream => RunLegacyStreaming(options),
                RunMode.StreamEx     => RunStreamingEx(options),
                _                    => throw new ArgumentOutOfRangeException()
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
    }

    private static int RunSmoke()
    {
        nint pipeline = 0;
        nint generateParams = 0;
        nint audioBuffer = 0;
        nint promptCodes = 0;
        nint model = 0;
        nint tokenizer = 0;
        nint codec = 0;

        try
        {
            pipeline = RequireHandle(nameof(NativeMethods.AllocS2Pipeline), NativeMethods.AllocS2Pipeline());
            generateParams = RequireHandle(nameof(NativeMethods.AllocS2GenerateParams), NativeMethods.AllocS2GenerateParams());
            audioBuffer = RequireHandle(nameof(NativeMethods.AllocS2AudioBuffer), NativeMethods.AllocS2AudioBuffer(-1));
            promptCodes = RequireHandle(nameof(NativeMethods.AllocS2AudioPromptCodes), NativeMethods.AllocS2AudioPromptCodes());
            model = RequireHandle(nameof(NativeMethods.AllocS2Model), NativeMethods.AllocS2Model());
            tokenizer = RequireHandle(nameof(NativeMethods.AllocS2Tokenizer), NativeMethods.AllocS2Tokenizer());
            codec = RequireHandle(nameof(NativeMethods.AllocS2AudioCodec), NativeMethods.AllocS2AudioCodec());

            ExpectSuccess(NativeMethods.InitializeS2GenerateParams(generateParams, -1, -1, -1, -1, -1, -1, 0),
                "InitializeS2GenerateParams failed");

            Console.WriteLine($"loaded library: log_level={NativeMethods.GetS2LogLevel()}");
            Console.WriteLine("smoke test: exported symbols, allocators, params, and audio buffer OK");
            return 0;
        }
        finally
        {
            ReleaseHandle(codec, NativeMethods.ReleaseS2AudioCodec);
            ReleaseHandle(tokenizer, NativeMethods.ReleaseS2Tokenizer);
            ReleaseHandle(model, NativeMethods.ReleaseS2Model);
            ReleaseHandle(promptCodes, NativeMethods.ReleaseS2AudioPromptCodes);
            ReleaseHandle(audioBuffer, NativeMethods.ReleaseS2AudioBuffer);
            ReleaseHandle(generateParams, NativeMethods.ReleaseS2GenerateParams);
            ReleaseHandle(pipeline, NativeMethods.ReleaseS2Pipeline);
        }
    }

    private static int RunFromFiles(Options options)
    {
        options.ValidateForSynthesis();

        nint pipeline = 0;
        nint generateParams = 0;
        nint audioBuffer = 0;
        nint promptCodes = 0;

        try
        {
            pipeline = RequireHandle(nameof(NativeMethods.AllocS2Pipeline), NativeMethods.AllocS2Pipeline());
            generateParams = RequireHandle(nameof(NativeMethods.AllocS2GenerateParams), NativeMethods.AllocS2GenerateParams());
            audioBuffer = RequireHandle(nameof(NativeMethods.AllocS2AudioBuffer), NativeMethods.AllocS2AudioBuffer(-1));
            promptCodes = RequireHandle(nameof(NativeMethods.AllocS2AudioPromptCodes), NativeMethods.AllocS2AudioPromptCodes());

            InitializeGenerateParams(generateParams, options);
            ExpectSuccess(
                NativeMethods.InitializeS2PipelineFromFiles(
                    pipeline,
                    options.ModelPath,
                    options.TokenizerPath,
                    options.GpuDevice,
                    (int) options.Backend,
                    options.GpuLayers,
                    options.CodecFollowBackend ? 1 : 0),
                "InitializeS2PipelineFromFiles failed");

            int tPrompt = 0;
            int sampleCount = 0;
            int code = NativeMethods.S2Synthesize(
                pipeline,
                generateParams,
                audioBuffer,
                promptCodes,
                &tPrompt,
                options.ReferenceAudioPath,
                options.ReferenceText,
                options.Text,
                options.OutputPath,
                &sampleCount);

            PrintSynthesizeResult("S2Synthesize", code, sampleCount, options.OutputPath);
            PrintAudioPreview(audioBuffer, sampleCount);
            return IsSynthesizeFailure(code) ? 1 : 0;
        }
        finally
        {
            ReleaseHandle(promptCodes, NativeMethods.ReleaseS2AudioPromptCodes);
            ReleaseHandle(audioBuffer, NativeMethods.ReleaseS2AudioBuffer);
            ReleaseHandle(generateParams, NativeMethods.ReleaseS2GenerateParams);
            ReleaseHandle(pipeline, NativeMethods.ReleaseS2Pipeline);
        }
    }

    private static int RunModular(Options options)
    {
        options.ValidateForSynthesis();

        nint pipeline = 0;
        nint generateParams = 0;
        nint audioBuffer = 0;
        nint promptCodes = 0;
        nint model = 0;
        nint tokenizer = 0;
        nint codec = 0;

        try
        {
            pipeline = RequireHandle(nameof(NativeMethods.AllocS2Pipeline), NativeMethods.AllocS2Pipeline());
            generateParams = RequireHandle(nameof(NativeMethods.AllocS2GenerateParams), NativeMethods.AllocS2GenerateParams());
            audioBuffer = RequireHandle(nameof(NativeMethods.AllocS2AudioBuffer), NativeMethods.AllocS2AudioBuffer(-1));
            promptCodes = RequireHandle(nameof(NativeMethods.AllocS2AudioPromptCodes), NativeMethods.AllocS2AudioPromptCodes());
            model = RequireHandle(nameof(NativeMethods.AllocS2Model), NativeMethods.AllocS2Model());
            tokenizer = RequireHandle(nameof(NativeMethods.AllocS2Tokenizer), NativeMethods.AllocS2Tokenizer());
            codec = RequireHandle(nameof(NativeMethods.AllocS2AudioCodec), NativeMethods.AllocS2AudioCodec());

            ExpectSuccess(NativeMethods.InitializeS2Tokenizer(tokenizer, options.TokenizerPath),
                "InitializeS2Tokenizer failed");
            int modelInit = options.GpuLayers >= 0
                ? NativeMethods.InitializeS2ModelWithGpuLayers(
                    model,
                    options.ModelPath,
                    options.GpuDevice,
                    (int) options.Backend,
                    options.GpuLayers)
                : NativeMethods.InitializeS2Model(
                    model,
                    options.ModelPath,
                    options.GpuDevice,
                    (int) options.Backend);
            ExpectSuccess(modelInit,
                options.GpuLayers >= 0 ? "InitializeS2ModelWithGpuLayers failed" : "InitializeS2Model failed");
            ExpectSuccess(NativeMethods.InitializeS2AudioCodec(codec, options.ModelPath,
                    options.CodecFollowBackend ? options.GpuDevice : -1,
                    options.CodecFollowBackend ? (int) options.Backend : (int) BackendType.CPU),
                "InitializeS2AudioCodec failed");
            ExpectSuccess(NativeMethods.InitializeS2GenerateParams(generateParams,
                    options.MaxTokens, options.Temperature, options.TopP, options.TopK,
                    options.MinTokensBeforeEnd, options.Threads, options.Verbose ? 1 : 0),
                "InitializeS2GenerateParams failed");

            NativeMethods.SyncS2TokenizerConfigFromS2Model(model, tokenizer);
            ExpectSuccess(NativeMethods.InitializeS2Pipeline(pipeline, tokenizer, model, codec),
                "InitializeS2Pipeline failed");

            int tPrompt = 0;
            if (!string.IsNullOrWhiteSpace(options.ReferenceAudioPath))
            {
                int promptInit = NativeMethods.InitializeAudioPromptCodes(
                    pipeline,
                    options.Threads > 0 ? options.Threads : Environment.ProcessorCount,
                    options.ReferenceAudioPath,
                    promptCodes,
                    &tPrompt);

                if (promptInit == -1)
                {
                    Console.WriteLine("InitializeAudioPromptCodes returned -1: reference encode failed, continuing without prompt codes");
                    tPrompt = 0;
                }
                else
                {
                    ExpectSuccess(promptInit, "InitializeAudioPromptCodes failed");
                }
            }

            int sampleCount = 0;
            int code = NativeMethods.S2Synthesize(
                pipeline,
                generateParams,
                audioBuffer,
                promptCodes,
                &tPrompt,
                null,
                options.ReferenceText,
                options.Text,
                options.OutputPath,
                &sampleCount);

            PrintSynthesizeResult("S2Synthesize", code, sampleCount, options.OutputPath);
            PrintAudioPreview(audioBuffer, sampleCount);
            return IsSynthesizeFailure(code) ? 1 : 0;
        }
        finally
        {
            ReleaseHandle(codec, NativeMethods.ReleaseS2AudioCodec);
            ReleaseHandle(tokenizer, NativeMethods.ReleaseS2Tokenizer);
            ReleaseHandle(model, NativeMethods.ReleaseS2Model);
            ReleaseHandle(promptCodes, NativeMethods.ReleaseS2AudioPromptCodes);
            ReleaseHandle(audioBuffer, NativeMethods.ReleaseS2AudioBuffer);
            ReleaseHandle(generateParams, NativeMethods.ReleaseS2GenerateParams);
            ReleaseHandle(pipeline, NativeMethods.ReleaseS2Pipeline);
        }
    }

    private static int RunLegacyStreaming(Options options)
    {
        options.ValidateForSynthesis();

        nint pipeline = 0;
        nint generateParams = 0;
        nint promptCodes = 0;

        try
        {
            pipeline = RequireHandle(nameof(NativeMethods.AllocS2Pipeline), NativeMethods.AllocS2Pipeline());
            generateParams = RequireHandle(nameof(NativeMethods.AllocS2GenerateParams), NativeMethods.AllocS2GenerateParams());
            promptCodes = RequireHandle(nameof(NativeMethods.AllocS2AudioPromptCodes), NativeMethods.AllocS2AudioPromptCodes());

            InitializeGenerateParams(generateParams, options);
            ExpectSuccess(
                NativeMethods.InitializeS2PipelineFromFiles(
                    pipeline,
                    options.ModelPath,
                    options.TokenizerPath,
                    options.GpuDevice,
                    (int) options.Backend,
                    options.GpuLayers,
                    options.CodecFollowBackend ? 1 : 0),
                "InitializeS2PipelineFromFiles failed");

            int tPrompt = 0;
            using StreamingSession session = new(options.OutputPath, options.Play);
            NativeMethods.S2StreamingCallbacks callbacks = session.BuildCallbacks();

            int code = NativeMethods.S2SynthesizeStreaming(
                pipeline,
                generateParams,
                ref callbacks,
                promptCodes,
                &tPrompt,
                options.ReferenceAudioPath,
                options.ReferenceText,
                options.Text,
                options.StreamDecodeStrideFrames);

            session.FinalizeOutput();
            PrintStreamingResult("S2SynthesizeStreaming", code, session);
            return IsStreamingFailure(code) ? 1 : 0;
        }
        finally
        {
            ReleaseHandle(promptCodes, NativeMethods.ReleaseS2AudioPromptCodes);
            ReleaseHandle(generateParams, NativeMethods.ReleaseS2GenerateParams);
            ReleaseHandle(pipeline, NativeMethods.ReleaseS2Pipeline);
        }
    }

    private static int RunStreamingEx(Options options)
    {
        options.ValidateForSynthesis();

        nint pipeline = 0;
        nint generateParams = 0;
        nint promptCodes = 0;

        try
        {
            pipeline = RequireHandle(nameof(NativeMethods.AllocS2Pipeline), NativeMethods.AllocS2Pipeline());
            generateParams = RequireHandle(nameof(NativeMethods.AllocS2GenerateParams), NativeMethods.AllocS2GenerateParams());
            promptCodes = RequireHandle(nameof(NativeMethods.AllocS2AudioPromptCodes), NativeMethods.AllocS2AudioPromptCodes());

            InitializeGenerateParams(generateParams, options);
            ExpectSuccess(
                NativeMethods.InitializeS2PipelineFromFiles(
                    pipeline,
                    options.ModelPath,
                    options.TokenizerPath,
                    options.GpuDevice,
                    (int) options.Backend,
                    options.GpuLayers,
                    options.CodecFollowBackend ? 1 : 0),
                "InitializeS2PipelineFromFiles failed");

            int tPrompt = 0;
            using StreamingSession session = new(options.OutputPath, options.Play);
            NativeMethods.S2StreamingCallbacks callbacks = session.BuildCallbacks();
            NativeMethods.S2StreamingParams streamingParams = new()
            {
                stream_decode_stride_frames = options.StreamDecodeStrideFrames,
                stream_holdback_frames = options.StreamHoldbackFrames,
                codec_decode_context_frames = options.CodecContextFrames,
                low_latency = options.LowLatency ? 1 : 0,
                segment_sentences = options.SegmentSentences ? 1 : 0,
                sentence_pause_ms = options.SentencePauseMs,
                segment_max_chars = options.SegmentMaxChars,
                voice = options.Voice,
                voice_dir = options.VoiceDir
            };

            int code = NativeMethods.S2SynthesizeStreamingEx(
                pipeline,
                generateParams,
                ref callbacks,
                promptCodes,
                &tPrompt,
                options.ReferenceAudioPath,
                options.ReferenceText,
                options.Text,
                ref streamingParams);

            session.FinalizeOutput();
            PrintStreamingResult("S2SynthesizeStreamingEx", code, session);
            return IsStreamingFailure(code) ? 1 : 0;
        }
        finally
        {
            ReleaseHandle(promptCodes, NativeMethods.ReleaseS2AudioPromptCodes);
            ReleaseHandle(generateParams, NativeMethods.ReleaseS2GenerateParams);
            ReleaseHandle(pipeline, NativeMethods.ReleaseS2Pipeline);
        }
    }

    private static void InitializeGenerateParams(nint generateParams, Options options)
    {
        ExpectSuccess(NativeMethods.InitializeS2GenerateParams(
                generateParams,
                options.MaxTokens,
                options.Temperature,
                options.TopP,
                options.TopK,
                options.MinTokensBeforeEnd,
                options.Threads,
                options.Verbose ? 1 : 0),
            "InitializeS2GenerateParams failed");
    }

    private static void PrintSynthesizeResult(string functionName, int code, int sampleCount, string? outputPath)
    {
        Console.WriteLine($"{functionName} returned {code}: {DescribeSynthesizeCode(code)}");
        Console.WriteLine($"samples={sampleCount} output={outputPath ?? "(none)"}");
    }

    private static void PrintStreamingResult(string functionName, int code, StreamingSession session)
    {
        Console.WriteLine($"{functionName} returned {code}: {DescribeStreamingCode(code)}");
        Console.WriteLine($"chunks={session.ChunkCount} bytes={session.ByteCount} output={session.OutputPath ?? "(none)"} done={session.Done}");
        if (session.Errors.Count > 0)
        {
            Console.WriteLine($"stream_errors=[{string.Join(", ", session.Errors)}]");
        }
    }

    private static bool IsSynthesizeFailure(int code) => code <= -4 || code is 0 or -6 or -7 or -8;
    private static bool IsStreamingFailure(int code) => code <= -4 || code is 0 or -7 or -8 or -9 or -10;

    private static string DescribeSynthesizeCode(int code) => code switch
    {
        0 => "pipeline not initialized or invalid parameters",
        -1 => "reference audio encode failed; synthesis continued without reference",
        -4 => "synthesis failed or produced no frames",
        -6 => "failed to save output audio",
        -7 => "reference audio/codes require a non-empty transcript",
        -8 => "precomputed prompt codes require ReferenceAudioTPrompt",
        _ when code > 0 => "success",
        _ => "unknown error"
    };

    private static string DescribeStreamingCode(int code) => code switch
    {
        0 => "pipeline not initialized or invalid parameters",
        -1 => "reference audio encode failed; synthesis continued without reference",
        -4 => "streaming synthesis failed or produced no frames",
        -7 => "reference audio/codes require a non-empty transcript",
        -8 => "precomputed prompt codes require ReferenceAudioTPrompt",
        -9 => "streaming callback configuration is invalid",
        -10 => "streaming was aborted or cancelled by the client callback",
        _ when code > 0 => "success",
        _ => "unknown error"
    };

    private static nint RequireHandle(string name, nint handle)
    {
        if (handle == 0)
        {
            throw new InvalidOperationException($"{name} returned NULL");
        }

        return handle;
    }

    private static void ExpectSuccess(int result, string message)
    {
        if (result != 1)
        {
            throw new InvalidOperationException($"{message}: return_code={result}");
        }
    }

    private static void ReleaseHandle(nint handle, Action<nint> release)
    {
        if (handle != 0)
        {
            release(handle);
        }
    }

    private static void PrintAudioPreview(nint audioBuffer, int sampleCount)
    {
        if (audioBuffer == 0 || sampleCount <= 0)
        {
            return;
        }

        float* data = NativeMethods.GetS2AudioBufferDataPointer(audioBuffer);
        if (data == null)
        {
            return;
        }

        int previewCount = Math.Min(5, sampleCount);
        string[] preview = new string[previewCount];
        for (int i = 0; i < previewCount; ++i)
        {
            preview[i] = data[i].ToString("0.000000", CultureInfo.InvariantCulture);
        }

        Console.WriteLine($"first_samples=[{string.Join(", ", preview)}]");
    }
}

internal static class NativeLibraryBootstrap
{
    private static bool _configured;

    public static void Configure(string? libraryPath)
    {
        if (_configured)
        {
            return;
        }

        _configured = true;
        if (string.IsNullOrWhiteSpace(libraryPath))
        {
            return;
        }

        string fullPath = Path.GetFullPath(libraryPath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("s2 library not found", fullPath);
        }

        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, (libraryName, _, _) =>
        {
            if (!string.Equals(libraryName, NativeMethods.LibraryName, StringComparison.Ordinal))
            {
                return IntPtr.Zero;
            }

            return NativeLibrary.Load(fullPath);
        });
    }
}

internal enum RunMode
{
    Smoke,
    FromFiles,
    Modular,
    LegacyStream,
    StreamEx
}

internal enum BackendType
{
    CPU = -1,
    Vulkan = 0,
    CUDA = 1,
    Metal = 2
}

internal enum LogLevel
{
    Error = 0,
    Warning = 1,
    Info = 2,
    Debug = 3
}

internal sealed class Options
{
    public string? LibraryPath { get; private set; }
    public RunMode Mode { get; private set; } = RunMode.Modular;
    public string? ModelPath { get; private set; }
    public string? TokenizerPath { get; private set; }
    public string Text { get; private set; } = "This is a short C# export API test.";
    public string? OutputPath { get; private set; }
    public string? ReferenceAudioPath { get; private set; }
    public string? ReferenceText { get; private set; } = string.Empty;
    public string? Voice { get; private set; }
    public string? VoiceDir { get; private set; }
    public BackendType Backend { get; private set; } = BackendType.CPU;
    public int GpuDevice { get; private set; } = -1;
    public int GpuLayers { get; private set; } = -1;
    public bool CodecFollowBackend { get; private set; } = true;
    public int MaxTokens { get; private set; } = -1;
    public float Temperature { get; private set; } = -1;
    public float TopP { get; private set; } = -1;
    public int TopK { get; private set; } = -1;
    public int MinTokensBeforeEnd { get; private set; } = -1;
    public int Threads { get; private set; } = -1;
    public bool Verbose { get; private set; }
    public LogLevel LogLevel { get; private set; } = LogLevel.Error;
    public int StreamDecodeStrideFrames { get; private set; }
    public int StreamHoldbackFrames { get; private set; } = -1;
    public int CodecContextFrames { get; private set; } = -1;
    public bool LowLatency { get; private set; }
    public bool SegmentSentences { get; private set; }
    public int SentencePauseMs { get; private set; } = 180;
    public int SegmentMaxChars { get; private set; }
    public bool Play { get; private set; }

    public static Options Parse(string[] args)
    {
        Options options = new();

        for (int i = 0; i < args.Length; ++i)
        {
            string arg = args[i];
            switch (arg)
            {
                case "--library":
                    options.LibraryPath = ReadValue(args, ref i, arg);
                    break;
                case "--mode":
                    options.Mode = ParseMode(ReadValue(args, ref i, arg));
                    break;
                case "--model":
                    options.ModelPath = Path.GetFullPath(ReadValue(args, ref i, arg));
                    break;
                case "--tokenizer":
                    options.TokenizerPath = Path.GetFullPath(ReadValue(args, ref i, arg));
                    break;
                case "--text":
                    options.Text = ReadValue(args, ref i, arg);
                    break;
                case "--output":
                    options.OutputPath = Path.GetFullPath(ReadValue(args, ref i, arg));
                    break;
                case "--reference-audio":
                    options.ReferenceAudioPath = Path.GetFullPath(ReadValue(args, ref i, arg));
                    break;
                case "--reference-text":
                    options.ReferenceText = ReadValue(args, ref i, arg);
                    break;
                case "--voice":
                    options.Voice = ReadValue(args, ref i, arg);
                    break;
                case "--voice-dir":
                    options.VoiceDir = Path.GetFullPath(ReadValue(args, ref i, arg));
                    break;
                case "--backend":
                    options.Backend = ParseBackend(ReadValue(args, ref i, arg));
                    break;
                case "--gpu-device":
                    options.GpuDevice = int.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--gpu-layers":
                    options.GpuLayers = int.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--codec-cpu":
                    options.CodecFollowBackend = false;
                    break;
                case "--max-tokens":
                    options.MaxTokens = int.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--temperature":
                    options.Temperature = float.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--top-p":
                    options.TopP = float.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--top-k":
                    options.TopK = int.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--min-tokens-before-end":
                    options.MinTokensBeforeEnd = int.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--threads":
                    options.Threads = int.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--verbose":
                    options.Verbose = true;
                    break;
                case "--log-level":
                    options.LogLevel = ParseLogLevel(ReadValue(args, ref i, arg));
                    break;
                case "--stream-decode-stride":
                    options.StreamDecodeStrideFrames = int.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--stream-holdback-frames":
                    options.StreamHoldbackFrames = int.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--codec-context-frames":
                    options.CodecContextFrames = int.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--low-latency":
                    options.LowLatency = true;
                    break;
                case "--segment-sentences":
                    options.SegmentSentences = true;
                    break;
                case "--sentence-pause-ms":
                    options.SentencePauseMs = int.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--segment-max-chars":
                    options.SegmentMaxChars = int.Parse(ReadValue(args, ref i, arg), CultureInfo.InvariantCulture);
                    break;
                case "--play":
                    options.Play = true;
                    break;
                case "--help":
                case "-h":
                    PrintHelpAndExit();
                    break;
                default:
                    throw new ArgumentException($"Unknown argument: {arg}");
            }
        }

        return options;
    }

    public void ValidateForSynthesis()
    {
        if (string.IsNullOrWhiteSpace(ModelPath))
        {
            throw new ArgumentException("--model is required");
        }

        if (string.IsNullOrWhiteSpace(TokenizerPath))
        {
            throw new ArgumentException("--tokenizer is required");
        }

        if (!string.IsNullOrWhiteSpace(OutputPath))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(OutputPath)!);
        }
    }

    private static string ReadValue(string[] args, ref int index, string flag)
    {
        if (index + 1 >= args.Length)
        {
            throw new ArgumentException($"Missing value for {flag}");
        }

        return args[++index];
    }

    private static RunMode ParseMode(string value) => value.ToLowerInvariant() switch
    {
        "smoke" => RunMode.Smoke,
        "from-files" => RunMode.FromFiles,
        "modular" => RunMode.Modular,
        "legacy-stream" => RunMode.LegacyStream,
        "stream-ex" => RunMode.StreamEx,
        _ => throw new ArgumentException($"Unknown mode: {value}")
    };

    private static BackendType ParseBackend(string value) => value.ToLowerInvariant() switch
    {
        "cpu" => BackendType.CPU,
        "vulkan" => BackendType.Vulkan,
        "cuda" => BackendType.CUDA,
        "metal" => BackendType.Metal,
        _ => throw new ArgumentException($"Unknown backend: {value}")
    };

    private static LogLevel ParseLogLevel(string value) => value.ToLowerInvariant() switch
    {
        "0" or "error" => LogLevel.Error,
        "1" or "warn" or "warning" => LogLevel.Warning,
        "2" or "info" => LogLevel.Info,
        "3" or "debug" => LogLevel.Debug,
        _ => throw new ArgumentException($"Unknown log level: {value}")
    };

    private static void PrintHelpAndExit()
    {
        Console.WriteLine("""
Usage:
  dotnet run --project examples/csharp/S2ExportApiExample.csproj -- \
    --mode <smoke|from-files|modular|legacy-stream|stream-ex> \
    --library <path/to/libs2.so|s2.dll|libs2.dylib> \
    [other options]

Common options:
  --model <model.gguf>
  --tokenizer <tokenizer.json>
  --text "Text to synthesize"
  --output <path.wav>
  --reference-audio <ref.wav|ref.mp3>
  --reference-text "Transcript for the reference audio"
  --voice <voice-id-or-path.s2voice>
  --voice-dir <voice profile directory>
  --backend <cpu|vulkan|cuda|metal>
  --gpu-device <index>
  --gpu-layers <count>
  --codec-cpu
  --threads <count>
  --max-tokens <count>
  --log-level <error|warning|info|debug>

Streaming-only options:
  --stream-decode-stride <frames>
  --stream-holdback-frames <frames>
  --codec-context-frames <frames>
  --low-latency
  --segment-sentences
  --sentence-pause-ms <milliseconds>
  --segment-max-chars <chars>
  --play   play audio in realtime via aplay (no output file needed)
""");
        Environment.Exit(0);
    }
}

internal sealed unsafe class StreamingSession : IDisposable
{
    private readonly FileStream? _output;
    private readonly GCHandle _selfHandle;
    private bool _headerSkipped;
    private readonly bool _play;
    private System.Diagnostics.Process? _player;
    private Stream? _playerStdin;

    private readonly NativeMethods.S2StreamingWriteCallback _writeCallback;
    private readonly NativeMethods.S2StreamingDoneCallback _doneCallback;
    private readonly NativeMethods.S2StreamingErrorCallback _errorCallback;
    private readonly NativeMethods.S2StreamingCancelCallback _cancelCallback;

    public StreamingSession(string? outputPath, bool play)
    {
        OutputPath = outputPath;
        _play = play;
        if (!string.IsNullOrWhiteSpace(outputPath))
        {
            _output = File.Create(outputPath);
        }

        if (play)
        {
            StartPlayer();
        }

        _selfHandle = GCHandle.Alloc(this);
        _writeCallback = HandleWrite;
        _doneCallback = HandleDone;
        _errorCallback = HandleError;
        _cancelCallback = HandleCancel;
    }

    public string? OutputPath { get; }
    public int ChunkCount { get; private set; }
    public int ByteCount { get; private set; }
    public bool Done { get; private set; }
    public List<string> Errors { get; } = [];

    private void StartPlayer()
    {
        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = "aplay",
            ArgumentList = { "-q", "-f", "S16_LE", "-r", "44100", "-c", "1" },
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        _player = System.Diagnostics.Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start aplay (install alsa-utils)");
        _playerStdin = _player.StandardInput.BaseStream;
    }

    public NativeMethods.S2StreamingCallbacks BuildCallbacks() => new()
    {
        on_wav_chunk = Marshal.GetFunctionPointerForDelegate(_writeCallback),
        on_done = Marshal.GetFunctionPointerForDelegate(_doneCallback),
        on_error = Marshal.GetFunctionPointerForDelegate(_errorCallback),
        is_cancelled = Marshal.GetFunctionPointerForDelegate(_cancelCallback),
        user_data = GCHandle.ToIntPtr(_selfHandle)
    };

    public void FinalizeOutput()
    {
        if (_output is null)
        {
            return;
        }

        _output.Flush();
        if (Done && ByteCount >= 44)
        {
            PatchWavHeader(_output, ByteCount);
        }
    }

    public void Dispose()
    {
        _playerStdin?.Dispose();
        if (_player is not null && !_player.HasExited)
        {
            _player.WaitForExit(5000);
        }
        if (_player is not null && !_player.HasExited)
        {
            _player.Kill();
        }
        _player?.Dispose();
        _output?.Dispose();
        if (_selfHandle.IsAllocated)
        {
            _selfHandle.Free();
        }
    }

    private static StreamingSession FromUserData(nint userData)
    {
        GCHandle handle = GCHandle.FromIntPtr(userData);
        return (StreamingSession) handle.Target!;
    }

    private static int HandleWrite(byte* data, int size, nint userData)
    {
        StreamingSession session = FromUserData(userData);
        session.ChunkCount++;
        session.ByteCount += size;

        ReadOnlySpan<byte> span = new(data, size);

        if (session._output is not null)
        {
            session._output.Write(span);
            session._output.Flush();
        }

        if (session._playerStdin is not null)
        {
            ReadOnlySpan<byte> pcm = span;
            if (!session._headerSkipped)
            {
                if (pcm.Length <= 44)
                {
                    session._headerSkipped = true;
                    return 1;
                }
                pcm = pcm[44..];
                session._headerSkipped = true;
            }
            if (pcm.Length > 0)
            {
                session._playerStdin.Write(pcm);
                session._playerStdin.Flush();
            }
        }

        return 1;
    }

    private static void HandleDone(nint userData)
    {
        StreamingSession session = FromUserData(userData);
        session.Done = true;
    }

    private static void HandleError(byte* message, nint userData)
    {
        StreamingSession session = FromUserData(userData);
        string text = Marshal.PtrToStringUTF8((nint) message) ?? string.Empty;
        session.Errors.Add(text);
    }

    private static int HandleCancel(nint userData)
    {
        _ = FromUserData(userData);
        return 0;
    }

    private static void PatchWavHeader(FileStream file, int totalSize)
    {
        if (totalSize is < 44 or > int.MaxValue)
        {
            return;
        }

        uint riffSize = (uint) (totalSize - 8);
        uint dataSize = (uint) (totalSize - 44);

        long cursor = file.Position;
        file.Position = 4;
        file.Write(BitConverter.GetBytes(riffSize));
        file.Position = 40;
        file.Write(BitConverter.GetBytes(dataSize));
        file.Position = cursor;
        file.Flush();
    }
}

internal static unsafe class NativeMethods
{
    public const string LibraryName = "s2";

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate int S2StreamingWriteCallback(byte* data, int size, nint userData);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void S2StreamingDoneCallback(nint userData);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void S2StreamingErrorCallback(byte* message, nint userData);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate int S2StreamingCancelCallback(nint userData);

    [StructLayout(LayoutKind.Sequential)]
    public struct S2StreamingCallbacks
    {
        public nint on_wav_chunk;
        public nint on_done;
        public nint on_error;
        public nint is_cancelled;
        public nint user_data;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct S2StreamingParams
    {
        public int stream_decode_stride_frames;
        public int stream_holdback_frames;
        public int codec_decode_context_frames;
        public int low_latency;
        public int segment_sentences;
        public int sentence_pause_ms;
        public int segment_max_chars;
        [MarshalAs(UnmanagedType.LPUTF8Str)] public string? voice;
        [MarshalAs(UnmanagedType.LPUTF8Str)] public string? voice_dir;
    }

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nint AllocS2Pipeline();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ReleaseS2Pipeline(nint pipeline);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void SetS2LogLevel(int logLevel);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int GetS2LogLevel();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void SyncS2TokenizerConfigFromS2Model(nint model, nint tokenizer);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int InitializeS2Pipeline(nint pipeline, nint tokenizer, nint model, nint audioCodec);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int InitializeS2PipelineFromFiles(
        nint pipeline,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string tokenizerPath,
        int gpuDevice,
        int backendType,
        int nGpuLayers,
        int codecFollowBackend);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nint AllocS2GenerateParams();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ReleaseS2GenerateParams(nint generateParams);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int InitializeS2GenerateParams(
        nint generateParams,
        int maxNewTokens,
        float temperature,
        float topP,
        int topK,
        int minTokensBeforeEnd,
        int nThreads,
        int verbose);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nint AllocS2Model();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ReleaseS2Model(nint model);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int InitializeS2Model(
        nint model,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
        int gpuDevice,
        int backendType);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int InitializeS2ModelWithGpuLayers(
        nint model,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
        int gpuDevice,
        int backendType,
        int nGpuLayers);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nint AllocS2Tokenizer();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ReleaseS2Tokenizer(nint tokenizer);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int InitializeS2Tokenizer(
        nint tokenizer,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nint AllocS2AudioCodec();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ReleaseS2AudioCodec(nint audioCodec);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int InitializeS2AudioCodec(
        nint audioCodec,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
        int gpuDevice,
        int backendType);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nint AllocS2AudioPromptCodes();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ReleaseS2AudioPromptCodes(nint audioPromptCodes);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int InitializeAudioPromptCodes(
        nint pipeline,
        int threadCount,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? referenceAudioPath,
        nint audioPromptCodes,
        int* tPrompt);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nint AllocS2AudioBuffer(int initialSize);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ReleaseS2AudioBuffer(nint audioBuffer);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern float* GetS2AudioBufferDataPointer(nint audioBuffer);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int S2Synthesize(
        nint pipeline,
        nint generateParams,
        nint audioBuffer,
        nint referenceAudioPromptCodes,
        int* referenceAudioTPrompt,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? referenceAudioPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? referenceAudioTranscript,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? textToInfer,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? outputAudioPath,
        int* audioBufferOutputLength);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int S2SynthesizeStreaming(
        nint pipeline,
        nint generateParams,
        ref S2StreamingCallbacks streamingCallbacks,
        nint referenceAudioPromptCodes,
        int* referenceAudioTPrompt,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? referenceAudioPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? referenceAudioTranscript,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? textToInfer,
        int streamDecodeStrideFrames);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int S2SynthesizeStreamingEx(
        nint pipeline,
        nint generateParams,
        ref S2StreamingCallbacks streamingCallbacks,
        nint referenceAudioPromptCodes,
        int* referenceAudioTPrompt,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? referenceAudioPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? referenceAudioTranscript,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? textToInfer,
        ref S2StreamingParams streamingParams);
}
