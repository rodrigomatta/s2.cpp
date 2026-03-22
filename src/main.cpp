#include "s2_pipeline.h"
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>

static std::string utf16_to_utf8(const std::wstring & wstr) {
    if (wstr.empty()) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}
#endif

void print_uso() {
    std::cout << "Usage: s2 [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model <path>                  Path to unified GGUF model\n";
    std::cout << "  -t, --tokenizer <path>              Path to tokenizer.json\n";
    std::cout << "  -text <string>                      Text to synthesize\n";
    std::cout << "  -pa, --prompt-audio <p>             Path to reference audio for cloning\n";
    std::cout << "  -pt, --prompt-text <s>              Text of the reference audio for cloning\n";
    std::cout << "  -o, --output <path>                 Output WAV path\n";
    std::cout << "  -v, -c, --vulkan, --cuda <device>   Vulkan/Cuda device index (-1 for CPU)\n";
    std::cout << "  -threads N                          Number of CPU threads for inference (default: 4)\n";
    std::cout << "  -max-tokens N                       Max tokens to generate (default: 2048, ~96s audio)\n";
    std::cout << "  -temp F                             Sampling temperature (default: 0.7)\n";
    std::cout << "  -top-p F                            Top-p sampling (default: 0.7)\n";
    std::cout << "  -top-k N                            Top-k sampling (default: 30)\n";
    std::cout << "  --no-split-sentences                Disable sentence splitting\n";
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    // On Windows, the command line is passed in the local code page in argv.
    // We must use GetCommandLineW and convert to UTF-8 for proper Unicode support.
    int argc_w;
    LPWSTR* argv_w = CommandLineToArgvW(GetCommandLineW(), &argc_w);
    std::vector<std::string> args_utf8;
    std::vector<char*> argv_utf8;
    if (argv_w) {
        for (int i = 0; i < argc_w; ++i) {
            args_utf8.push_back(utf16_to_utf8(argv_w[i]));
        }
        LocalFree(argv_w);
        for (auto & s : args_utf8) {
            argv_utf8.push_back(&s[0]);
        }
        argc = argc_w;
        argv = argv_utf8.data();
    }
#endif

    if (argc < 2) {
        print_uso();
        return 1;
    }

    s2::PipelineParams params;
    // Default paths
    params.model_path = "model.gguf";
    params.tokenizer_path = "tokenizer.json";
    params.output_path = "out.wav";
    params.text = "Hello world";
    params.gpu_device = -1;
    params.backend_type = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) params.model_path = argv[++i];
        } else if (arg == "-t" || arg == "--tokenizer") {
            if (i + 1 < argc) params.tokenizer_path = argv[++i];
        } else if (arg == "-text") {
            if (i + 1 < argc) params.text = argv[++i];
        } else if (arg == "-pa" || arg == "--prompt-audio") {
            if (i + 1 < argc) params.prompt_audio_path = argv[++i];
        } else if (arg == "-pt" || arg == "--prompt-text") {
            if (i + 1 < argc) params.prompt_text = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) params.output_path = argv[++i];
        } else if (arg == "-v" || arg == "--vulkan") {
            if (i + 1 < argc) params.gpu_device = std::stoi(argv[++i]); params.backend_type = 0; //Vulkan.
        } else if (arg == "-c" || arg == "--cuda") {
            if (i + 1 < argc) params.gpu_device = std::stoi(argv[++i]); params.backend_type = 1; //Cuda.
        } else if (arg == "-threads") {
            if (i + 1 < argc) params.gen.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-max-tokens") {
            if (i + 1 < argc) params.gen.max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "-temp") {
            if (i + 1 < argc) params.gen.temperature = std::stof(argv[++i]);
        } else if (arg == "-top-p") {
            if (i + 1 < argc) params.gen.top_p = std::stof(argv[++i]);
        } else if (arg == "-top-k") {
            if (i + 1 < argc) params.gen.top_k = std::stoi(argv[++i]);
        } else if (arg == "--no-split-sentences") {
            params.split_sentences = false;
        } else if (arg == "-h" || arg == "--help") {
            print_uso();
            return 0;
        }
    }
    
    // Fallback: if model file doesn't exist at the given path, check in "models/" folder
    {
        FILE * f = std::fopen(params.model_path.c_str(), "r");
        if (f) {
            std::fclose(f);
        } else {
            // Check models/ prefix
            std::string candidate = "models/" + params.model_path;
            if (FILE * f2 = std::fopen(candidate.c_str(), "r")) {
                std::fclose(f2);
                params.model_path = candidate;
            }
        }
    }

    // If tokenizer path was not explicitly set, search for tokenizer.json in:
    //   1. Same directory as the model file
    //   2. Parent directory of the model file
    //   3. Working directory (default fallback)
    if (params.tokenizer_path == "tokenizer.json") {
        std::string model_path = params.model_path;
        size_t slash = model_path.find_last_of("/\\");
        if (slash != std::string::npos) {
            std::string model_dir = model_path.substr(0, slash + 1);
            // Check same dir as model
            std::string candidate = model_dir + "tokenizer.json";
            if (FILE * f = std::fopen(candidate.c_str(), "r")) {
                std::fclose(f);
                params.tokenizer_path = candidate;
            } else {
                // Check parent dir
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

    s2::Pipeline pipeline;
    if (!pipeline.init(params)) {
        std::cerr << "Pipeline initialization failed." << std::endl;
        return 1;
    }

    if (!pipeline.synthesize(params)) {
        std::cerr << "Synthesis failed." << std::endl;
        return 1;
    }

    return 0;
}
