#include "../third_party/filesystem.hpp"
namespace fs = ghc::filesystem;
#include "s2_log.h"
#include "s2_pipeline.h"
#include "s2_server.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <io.h>
#include <fcntl.h>
#else
#include <clocale>
#endif

static void safe_print(const char* msg) {
    fputs(msg, stdout);
}

static void safe_print(const std::string& msg) {
    fputs(msg.c_str(), stdout);
}

static void safe_print_error(const char* msg) {
    fputs(msg, stderr);
}

static void safe_print_error(const std::string& msg) {
    fputs(msg.c_str(), stderr);
}

static bool parse_log_level(const std::string& value, s2::LogLevel& out) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "error" || lower == "errors" || lower == "0") {
        out = s2::LogLevel::Error;
        return true;
    }
    if (lower == "warn" || lower == "warning" || lower == "warnings" || lower == "1") {
        out = s2::LogLevel::Warning;
        return true;
    }
    if (lower == "info" || lower == "2") {
        out = s2::LogLevel::Info;
        return true;
    }
    if (lower == "debug" || lower == "3") {
        out = s2::LogLevel::Debug;
        return true;
    }
    return false;
}

void print_uso() {
    safe_print("Usage: s2 [options]\n");
    safe_print("Options:\n");
    safe_print("  -m, --model        <path>   Path to GGUF model\n");
    safe_print("  -t, --tokenizer    <path>   Path to tokenizer.json\n");
    safe_print("  -text, --text      <text>   Text to synthesize\n");
    safe_print("  -pa, --prompt-audio <path>  Path to reference audio for cloning\n");
    safe_print("  -pt, --prompt-text <text>   Text of the reference audio\n");
    safe_print("  --voice            <id>     Load a saved voice profile\n");
    safe_print("  --save-voice                Save the encoded reference as a voice profile\n");
    safe_print("  --voice-dir        <path>   Directory used for saved voice profiles\n");
    safe_print("  --list-voices               List available saved voice profiles and exit\n");
    safe_print("  -o, --output       <path>   Output WAV path\n");
    safe_print("  -v, --vulkan     <id>   Vulkan device index (-1 = CPU)\n");
    safe_print("  -c, --cuda       <id>   CUDA device index (-1 = CPU)\n");
    safe_print("  -M, --metal             Use Metal backend (macOS only)\n");
    safe_print("  -ngl, --gpu-layers  <n>     Transformer layers on GPU (-1 = auto, 0 = CPU only)\n");
    safe_print("  -threads, --threads <n>     CPU threads (0 = auto)\n");
    safe_print("  -max-tokens, --max-tokens <n>  Max tokens to generate\n");
    safe_print("  --min-tokens-before-end <n> Minimum tokens before EOS is allowed\n");
    safe_print("  -temp, --temp, --temperature <f>  Temperature\n");
    safe_print("  -top-p, --top-p    <f>      Top-p sampling\n");
    safe_print("  -top-k, --top-k    <n>      Top-k sampling\n");
    safe_print("  --dynamic-normalize         Apply dynamic RMS normalization\n");
    safe_print("  --no-dynamic-normalize      Disable dynamic RMS normalization\n");
    safe_print("  --no-trim-silence           Keep trailing silence in output WAV\n");
    safe_print("  --trim-silence              Trim trailing silence in output WAV\n");
    safe_print("  --no-normalize              Keep original output peak level\n");
    safe_print("  --normalize                 Peak-normalize output WAV to 0.95\n");
    safe_print("  --codec-auto                Benchmark codec backends and keep the fastest (default)\n");
    safe_print("  --codec-follow-backend      Force codec to follow the selected GPU backend\n");
    safe_print("  --codec-cpu                 Force codec on CPU even when model uses GPU\n");
    safe_print("  --stream-file               Write output WAV through the streaming path\n");
    safe_print("  --stream-decode-stride <n>  Decode cadence in frames (0 = auto: server 4, file/offline 16)\n");
    safe_print("  --codec-context-frames <n>  Override codec decode history (lower uses less VRAM, default: auto)\n");
    safe_print("  --log-level <level>         error, warn, info, or debug (default: info)\n");
    safe_print("  --server                    Start HTTP server\n");
    safe_print("  -H, --host         <host>   Server host (default: 127.0.0.1)\n");
    safe_print("  -P, --port         <port>   Server port (default: 3030)\n");
    safe_print("  -h, --help                  Show this help\n");
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    fflush(stdout);
    fflush(stderr);
    if (!_isatty(_fileno(stdout))) _setmode(_fileno(stdout), _O_BINARY);
    if (!_isatty(_fileno(stderr))) _setmode(_fileno(stderr), _O_BINARY);

    int argc_w;
    LPWSTR* argv_w = CommandLineToArgvW(GetCommandLineW(), &argc_w);
    static std::vector<std::string> args_utf8;
    static std::vector<char*> new_argv;
    if (argv_w != NULL) {
        for (int i = 0; i < argc_w; i++) {
            int size = WideCharToMultiByte(CP_UTF8, 0, argv_w[i], -1, NULL, 0, NULL, NULL);
            std::vector<char> arg(size);
            WideCharToMultiByte(CP_UTF8, 0, argv_w[i], -1, arg.data(), size, NULL, NULL);
            args_utf8.emplace_back(arg.data());
        }
        LocalFree(argv_w);
        for (auto& s : args_utf8) new_argv.push_back(s.data());
        argv = new_argv.data();
        argc = argc_w;
    }
#endif

#ifndef _WIN32
    if (!std::setlocale(LC_ALL, "")) {
        std::setlocale(LC_ALL, "C.UTF-8");
    }
#endif

    if (argc < 2) {
        print_uso();
        return 1;
    }

    fs::u8arguments u8guard(argc, argv);

    s2::PipelineParams params;
    params.model_path = "model.gguf";
    params.tokenizer_path = "tokenizer.json";
    params.output_path = "out.wav";
    params.text = u8"Hello world";
    params.gpu_device = -1;
    params.backend_type = s2::BackendType::CPU;

    bool use_server = false;
    bool use_stream_file = false;
    bool list_voices = false;
    s2::ServerParams serverParams;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "-m"  || arg == "--model")        { if (i+1 < argc) params.model_path        = argv[++i]; }
        else if (arg == "-t"  || arg == "--tokenizer")    { if (i+1 < argc) params.tokenizer_path    = argv[++i]; }
        else if (arg == "-text" || arg == "--text")       { if (i+1 < argc) params.text               = argv[++i]; }
        else if (arg == "-pa" || arg == "--prompt-audio") { if (i+1 < argc) params.prompt_audio_path = argv[++i]; }
        else if (arg == "-pt" || arg == "--prompt-text")  { if (i+1 < argc) params.prompt_text        = argv[++i]; }
        else if (arg == "--voice")                        { if (i+1 < argc) params.voice_id           = argv[++i]; }
        else if (arg == "--save-voice")                   { params.save_voice = true; }
        else if (arg == "--voice-dir")                    { if (i+1 < argc) params.voice_storage_dir  = argv[++i]; }
        else if (arg == "--list-voices")                  { list_voices = true; }
        else if (arg == "-o"  || arg == "--output")       { if (i+1 < argc) params.output_path        = argv[++i]; }
        else if (arg == "-v"  || arg == "--vulkan")       { if (i+1 < argc) { try { params.gpu_device = std::stoi(argv[++i]); } catch(...) {} params.backend_type = s2::BackendType::Vulkan; } }
        else if (arg == "-c"  || arg == "--cuda")         { if (i+1 < argc) { try { params.gpu_device = std::stoi(argv[++i]); } catch(...) {} params.backend_type = s2::BackendType::CUDA; } }
        else if (arg == "-M"  || arg == "--metal")        { params.gpu_device = 0; params.backend_type = s2::BackendType::Metal; }
        else if (arg == "-ngl" || arg == "--gpu-layers")  { if (i+1 < argc) { try { params.n_gpu_layers = std::stoi(argv[++i]); } catch(...) {} } }
        else if (arg == "-threads" || arg == "--threads") { if (i+1 < argc) { try { params.gen.n_threads      = std::stoi(argv[++i]); } catch(...) {} } }
        else if (arg == "-max-tokens" || arg == "--max-tokens") {
            if (i+1 < argc) {
                try { params.gen.max_new_tokens = std::stoi(argv[++i]); } catch(...) {}
            }
        }
        else if (arg == "--min-tokens-before-end")        {
            if (i+1 < argc) {
                try {
                    params.gen.min_tokens_before_end = std::stoi(argv[++i]);
                    if (params.gen.min_tokens_before_end < 0) params.gen.min_tokens_before_end = 0;
                } catch(...) {}
            }
        }
        else if (arg == "-temp" || arg == "--temp" || arg == "--temperature") { if (i+1 < argc) { try { params.gen.temperature = std::stof(argv[++i]); } catch(...) {} } }
        else if (arg == "-top-p" || arg == "--top-p")      { if (i+1 < argc) { try { params.gen.top_p       = std::stof(argv[++i]); } catch(...) {} } }
        else if (arg == "-top-k" || arg == "--top-k")      { if (i+1 < argc) { try { params.gen.top_k       = std::stoi(argv[++i]); } catch(...) {} } }
        else if (arg == "--dynamic-normalize")     { params.normalize_dynamic = true;  }
        else if (arg == "--no-dynamic-normalize")  { params.normalize_dynamic = false; }
        else if (arg == "--no-trim-silence")  { params.trim_silence     = false; }
        else if (arg == "--trim-silence")     { params.trim_silence     = true;  }
        else if (arg == "--no-normalize")     { params.normalize_output = false; }
        else if (arg == "--normalize")        { params.normalize_output = true;  }
        else if (arg == "--codec-auto")       { params.codec_auto_backend = true;  params.codec_follow_backend = true;  }
        else if (arg == "--codec-follow-backend") { params.codec_auto_backend = false; params.codec_follow_backend = true; }
        else if (arg == "--codec-cpu")        { params.codec_auto_backend = false; params.codec_follow_backend = false; }
        else if (arg == "--stream-file")      { use_stream_file = true; }
        else if (arg == "--stream-decode-stride") {
            if (i+1 < argc) {
                try { params.stream_decode_stride_frames = std::stoi(argv[++i]); } catch(...) {}
            }
        }
        else if (arg == "--codec-context-frames") {
            if (i+1 < argc) {
#ifdef WIN32
                try { params.codec_decode_context_frames = max(0, std::stoi(argv[++i])); } catch(...) {}
#else
                try { params.codec_decode_context_frames = std::max(0, std::stoi(argv[++i])); } catch(...) {}
#endif
            }
        }
        else if (arg == "--log-level") {
            if (i+1 < argc) {
                s2::LogLevel log_level;
                if (parse_log_level(argv[++i], log_level)) {
                    s2::set_log_level(log_level);
                } else {
                    safe_print_error("Warning: unknown --log-level value; expected error, warn, info, or debug.\n");
                }
            }
        }
        else if (arg == "--server")           { use_server = true; }
        else if (arg == "-H" || arg == "--host") { if (i+1 < argc) serverParams.host = argv[++i]; }
        else if (arg == "-P" || arg == "--port") { if (i+1 < argc) { try { serverParams.port = std::stoi(argv[++i]); } catch(...) {} } }
        else if (arg == "-h" || arg == "--help") { print_uso(); return 0; }
    }

    if (list_voices) {
        std::vector<std::string> voice_ids;
        const fs::path dir(params.voice_storage_dir);
        if (fs::exists(dir)) {
            for (const auto & entry : fs::directory_iterator(dir)) {
                if (entry.path().extension() == ".s2voice") {
                    voice_ids.push_back(entry.path().stem().string());
                }
            }
        }

        std::sort(voice_ids.begin(), voice_ids.end());
        if (voice_ids.empty()) {
            safe_print("No saved voice profiles found.\n");
        } else {
            safe_print("Saved voice profiles:\n");
            for (const std::string & voice_id : voice_ids) {
                safe_print("  " + voice_id + "\n");
            }
        }
        return 0;
    }

    const bool has_prompt_audio = !params.prompt_audio_path.empty();
    const bool has_prompt_text = !params.prompt_text.empty();
    const bool has_voice_id = !params.voice_id.empty();

    if (params.save_voice) {
        if (!has_voice_id) {
            safe_print_error("Error: --save-voice requires --voice <id>.\n");
            return 1;
        }
        if (!has_prompt_audio || !has_prompt_text) {
            safe_print_error("Error: --save-voice requires both --prompt-audio and --prompt-text.\n");
            return 1;
        }
    }

    if (has_voice_id && has_prompt_audio && !params.save_voice) {
        safe_print_error("Warning: prompt audio takes precedence over --voice for this request.\n");
    }

    if (has_voice_id && !has_prompt_audio && has_prompt_text) {
        safe_print_error("Warning: --prompt-text is ignored when loading a saved --voice profile.\n");
    }

    if (params.n_gpu_layers > 0 &&
        params.gpu_device < 0 &&
        params.backend_type != s2::BackendType::Metal) {
        safe_print_error("Error: --gpu-layers requires --cuda, --vulkan, or --metal to select a GPU backend.\n");
        return 1;
    }

    if (params.tokenizer_path == "tokenizer.json") {
        std::string model_path = params.model_path;
        size_t slash = model_path.find_last_of("/\\");
        if (slash != std::string::npos) {
            std::string model_dir = model_path.substr(0, slash + 1);
            std::string candidate = model_dir + "tokenizer.json";
            if (FILE * f = std::fopen(candidate.c_str(), "r")) {
                std::fclose(f);
                params.tokenizer_path = candidate;
            } else {
                size_t parent_slash = model_dir.find_last_of("/\\", slash - 1);
                if (parent_slash != std::string::npos) {
                    candidate = model_dir.substr(0, parent_slash + 1) + "tokenizer.json";
                    if (FILE * f2 = std::fopen(candidate.c_str(), "r")) {
                        std::fclose(f2);
                        params.tokenizer_path = candidate;
                    }
                }
            }
        }
    }

    if (params.gen.max_new_tokens > 1024) {
        if (s2::log_enabled(s2::LogLevel::Warning)) {
            safe_print_error("Warning: --max-tokens > 1024 may cause voice quality degradation. Consider splitting long texts.\n");
        }
    }

    if (use_server) {
        serverParams.pipeline = params;
        s2::Server server;
        if (!server.serve(serverParams)) {
            safe_print_error("Server initialization failed.\n");
            return 1;
        }
        return 0;
    }

    s2::Pipeline pipeline;
    if (!pipeline.init(params)) {
        safe_print_error("Pipeline initialization failed.\n");
        return 1;
    }

    if (use_stream_file) {
        if (!pipeline.synthesize_streaming_file(params)) {
            safe_print_error("Synthesis failed.\n");
            return 1;
        }
        return 0;
    }

    if (!pipeline.synthesize(params)) {
        safe_print_error("Synthesis failed.\n");
        return 1;
    }

    return 0;
}
