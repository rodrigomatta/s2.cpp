#include "../include/s2_pipeline.h"
#include "../include/s2_log.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <limits>
#include <thread>
#include <utility>
#include <unordered_set>

#ifdef __linux__
#include <sys/resource.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

class FileStreamingSink : public s2::StreamingSink {
public:
    FileStreamingSink(const std::string& path) : path_(path), file_(nullptr), bytes_written_(0) {}

    ~FileStreamingSink() {
        if (file_) {
            std::fflush(file_);
            std::fclose(file_);
        }
    }

    bool on_header(const uint8_t* header, size_t size) override {
        file_ = std::fopen(path_.c_str(), "wb");
        if (!file_) {
            std::fprintf(stderr, "[FileStreamingSink] Failed to open %s for writing\n", path_.c_str());
            return false;
        }
        size_t written = std::fwrite(header, 1, size, file_);
        if (written != size) {
            std::fprintf(stderr, "[FileStreamingSink] Failed to write WAV header\n");
            return false;
        }
        bytes_written_ += written;
        return true;
    }

    bool on_pcm_data(const float* data, size_t n_samples) override {
        if (!file_) return false;
        const std::vector<int16_t> pcm16 = s2::audio_to_pcm16(data, n_samples);
        const size_t byte_count = pcm16.size() * sizeof(int16_t);
        const size_t written = std::fwrite(pcm16.data(), 1, byte_count, file_);
        if (written != byte_count) {
            std::fprintf(stderr, "[FileStreamingSink] Write incomplete\n");
            return false;
        }
        bytes_written_ += written;

        std::fflush(file_);
        return true;
    }

    void on_done() override {
        if (file_) {
            const uint32_t riff_size = bytes_written_ >= 8
                ? static_cast<uint32_t>(bytes_written_ - 8)
                : 0;
            const uint32_t data_size = bytes_written_ >= 44
                ? static_cast<uint32_t>(bytes_written_ - 44)
                : 0;
            std::fseek(file_, 4, SEEK_SET);
            std::fwrite(&riff_size, 1, sizeof(riff_size), file_);
            std::fseek(file_, 40, SEEK_SET);
            std::fwrite(&data_size, 1, sizeof(data_size), file_);
            std::fseek(file_, 0, SEEK_END);
            std::fflush(file_);

        }
        if (s2::log_enabled(s2::LogLevel::Info)) {
            std::fprintf(stdout, "[FileStreamingSink] Wrote %zu bytes to %s\n", bytes_written_, path_.c_str());
        }
    }

    void on_error(const std::string& message) override {
        std::fprintf(stderr, "[FileStreamingSink] Error: %s\n", message.c_str());

    }

private:
    std::string path_;
    FILE* file_;
    size_t bytes_written_;
};

}

namespace s2 {

static const char * backend_type_name(BackendType backend_type) {
    switch (backend_type) {
        case BackendType::CPU:    return "CPU";
        case BackendType::Vulkan: return "Vulkan";
        case BackendType::CUDA:   return "CUDA";
        case BackendType::Metal:  return "Metal";
    }
    return "Unknown";
}

struct CodecDecodeCacheScope {
    explicit CodecDecodeCacheScope(AudioCodec & codec) : codec_(codec) {
        codec_.clear_decode_cache();
    }

    ~CodecDecodeCacheScope() {
        codec_.clear_decode_cache();
    }

    AudioCodec & codec_;
};

static void safe_print_ln(const std::string& msg) {
    if (!log_enabled(LogLevel::Info)) return;
    fputs(msg.c_str(), stdout);
    fputc('\n', stdout);
}

static void safe_print_error_ln(const std::string& msg) {
    fputs(msg.c_str(), stderr);
    fputc('\n', stderr);
}

static void safe_print_warn_ln(const std::string& msg) {
    if (!log_enabled(LogLevel::Warning)) return;
    fputs(msg.c_str(), stderr);
    fputc('\n', stderr);
}

static double get_max_rss_mb() {
#ifdef __linux__
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss / 1024.0;
    }
#endif
    return 0.0;
}

struct CodecBenchmarkResult {
    bool ok = false;
    BackendType backend_type = BackendType::CPU;
    int32_t gpu_device = -1;
    std::string backend_name = "unavailable";
    double decode_ms = std::numeric_limits<double>::infinity();
    std::string error;
};

static CodecBenchmarkResult benchmark_codec_backend(const PipelineParams & params,
                                                    BackendType backend_type,
                                                    int32_t gpu_device) {
    CodecBenchmarkResult result;
    result.backend_type = backend_type;
    result.gpu_device = gpu_device;

    AudioCodec codec;
    if (!codec.load(params.model_path, gpu_device, backend_type)) {
        result.error = "load failed";
        return result;
    }

    result.backend_name = codec.backend_name();
    const int32_t benchmark_frames = 8;
    const int32_t benchmark_threads = params.gen.n_threads > 0
        ? params.gen.n_threads
        : static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency()));
    std::vector<int32_t> codes(static_cast<size_t>(codec.num_codebooks()) * benchmark_frames, 0);
    std::vector<float> audio_out;

    const auto t0 = std::chrono::steady_clock::now();
    if (!codec.decode(codes.data(), benchmark_frames, benchmark_threads, audio_out)) {
        result.error = "decode failed";
        return result;
    }
    const auto t1 = std::chrono::steady_clock::now();

    result.ok = !audio_out.empty();
    result.decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!result.ok) {
        result.error = "decode produced no audio";
    }
    return result;
}

static bool should_auto_select_codec_backend(const PipelineParams & params) {
    return params.codec_auto_backend &&
           params.codec_follow_backend &&
           params.backend_type != BackendType::CPU &&
           (params.gpu_device >= 0 || params.backend_type == BackendType::Metal);
}

static bool decode_codes_windowed(AudioCodec & codec, const int32_t * codes,
                                  int32_t total_frames, int32_t num_codebooks,
                                  int32_t n_threads, int32_t stride_frames,
                                  int32_t context_frames_override,
                                  std::vector<float> & audio_out,
                                  double * decode_ms_out = nullptr,
                                  int32_t * decode_batches_out = nullptr) {
    if (total_frames <= 0 || num_codebooks <= 0) {
        return false;
    }

    const int32_t decode_stride_frames = std::max(1, stride_frames > 0 ? stride_frames : 16);
    const int32_t history_frames = context_frames_override >= 0
        ? context_frames_override
        : std::max(0, codec.streaming_history_frames());
    const size_t samples_per_frame = static_cast<size_t>(std::max(1, codec.samples_per_code_frame()));

    std::vector<int32_t> decode_codes;
    std::vector<float> pcm;
    int32_t committed_frames = 0;
    double decode_ms = 0.0;
    int32_t decode_batches = 0;

    audio_out.clear();
    audio_out.reserve(static_cast<size_t>(total_frames) * samples_per_frame);

    auto decode_window = [&](int32_t frames_available, bool finalize) -> bool {
        if (frames_available <= 0 || frames_available <= committed_frames) {
            return true;
        }

        const int32_t stable_frames = finalize
            ? frames_available
            : std::max(0, frames_available - history_frames);
        if (stable_frames <= committed_frames && !finalize) {
            return true;
        }

        const int32_t window_start_frame = std::max(0, committed_frames - history_frames);
        const int32_t window_frames = frames_available - window_start_frame;
        if (window_frames <= 0) {
            return true;
        }

        decode_codes.resize(static_cast<size_t>(num_codebooks) * window_frames);
        for (int32_t cb = 0; cb < num_codebooks; ++cb) {
            const int32_t * src = codes + static_cast<size_t>(cb) * total_frames + window_start_frame;
            std::copy(src, src + window_frames,
                      decode_codes.begin() + static_cast<size_t>(cb) * window_frames);
        }

        const auto decode_t0 = std::chrono::steady_clock::now();
        if (!codec.decode(decode_codes.data(), window_frames, n_threads, pcm)) {
            return false;
        }
        const auto decode_t1 = std::chrono::steady_clock::now();
        decode_ms += std::chrono::duration<double, std::milli>(decode_t1 - decode_t0).count();
        decode_batches++;

        const size_t emit_begin_samples =
            static_cast<size_t>(std::max(0, committed_frames - window_start_frame)) * samples_per_frame;
        const size_t emit_end_samples = finalize
            ? pcm.size()
            : std::min(
                pcm.size(),
                static_cast<size_t>(std::max(0, stable_frames - window_start_frame)) * samples_per_frame);
        if (emit_end_samples > emit_begin_samples) {
            audio_out.insert(audio_out.end(), pcm.begin() + emit_begin_samples, pcm.begin() + emit_end_samples);
        }

        committed_frames = finalize ? frames_available : stable_frames;
        return true;
    };

    for (int32_t frames_available = std::min(total_frames, decode_stride_frames);
         frames_available < total_frames;
         frames_available += decode_stride_frames) {
        if (!decode_window(frames_available, false)) {
            return false;
        }
    }

    if (!decode_window(total_frames, true)) {
        return false;
    }

    if (decode_ms_out) {
        *decode_ms_out = decode_ms;
    }
    if (decode_batches_out) {
        *decode_batches_out = decode_batches;
    }
    return true;
}

static void sync_tokenizer_config_from_model(Tokenizer& tokenizer, const SlowARModel& model) {
    const ModelHParams & hp = model.hparams();
    TokenizerConfig & tc = tokenizer.config();
    if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
    if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
    if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
    if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
    if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
}

Pipeline::Pipeline() {}
Pipeline::~Pipeline() {}

static bool read_all_tensor_data(
    const std::string & gguf_path,
    gguf_context * gguf_ctx,
    s2::SlowARModel & model,
    s2::AudioCodec & codec)
{
    const size_t data_offset = gguf_get_data_offset(gguf_ctx);
    const int64_t n_tensors  = gguf_get_n_tensors(gguf_ctx);

    std::FILE * f = std::fopen(gguf_path.c_str(), "rb");
    if (!f) {
        std::cerr << "[Pipeline] Cannot reopen " << gguf_path << " for data loading." << std::endl;
        return false;
    }

    const auto & model_weights = model.weight_tensor_set();
    ggml_context * codec_ctx = codec.weights_ctx();
    std::vector<uint8_t> tmp;

    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * tname = gguf_get_tensor_name(gguf_ctx, ti);
        const size_t toff  = data_offset + gguf_get_tensor_offset(gguf_ctx, ti);

        ggml_tensor * t = ggml_get_tensor(model.weights_ctx(), tname);
        if (t && model_weights.find(t) != model_weights.end()) {
            const size_t tsize = ggml_nbytes(t);
            if (tmp.size() < tsize) tmp.resize(tsize);
#ifdef _WIN32
            _fseeki64(f, (int64_t)toff, SEEK_SET);
#else
            fseeko(f, (off_t)toff, SEEK_SET);
#endif
            if (std::fread(tmp.data(), 1, tsize, f) != tsize) {
                std::cerr << "[Pipeline] Failed to read tensor: " << tname << std::endl;
                std::fclose(f);
                return false;
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, tsize);
            continue;
        }

        if (codec_ctx) {
            t = ggml_get_tensor(codec_ctx, tname);
            if (t) {
                const size_t tsize = ggml_nbytes(t);
                if (tmp.size() < tsize) tmp.resize(tsize);
#ifdef _WIN32
                _fseeki64(f, (int64_t)toff, SEEK_SET);
#else
                fseeko(f, (off_t)toff, SEEK_SET);
#endif
                if (std::fread(tmp.data(), 1, tsize, f) != tsize) {
                    std::cerr << "[Pipeline] Failed to read tensor: " << tname << std::endl;
                    std::fclose(f);
                    return false;
                }
                ggml_backend_tensor_set(t, tmp.data(), 0, tsize);
                continue;
            }
        }

    }
    tmp.clear();
    tmp.shrink_to_fit();
    std::fclose(f);

    if (!codec.refresh_host_caches()) {
        std::cerr << "[Pipeline] Failed to refresh codec host caches after weight load." << std::endl;
        return false;
    }

#ifdef __linux__
    {
        int fd = ::open(gguf_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
        }
    }
#endif

    S2_LOG_INFO_STREAM("[Model] Weights loaded. Total tensors: " << n_tensors << std::endl);
    return true;
}

bool Pipeline::init(const PipelineParams & params) {
    tokenizer_ref_ = &owned_tokenizer_;
    model_ref_ = &owned_model_;
    codec_ref_ = &owned_codec_;
    initialized_ = false;

    const auto init_t0 = std::chrono::steady_clock::now();
    safe_print_ln("--- Pipeline Init ---");

    const auto tok_t0 = std::chrono::steady_clock::now();
    if (!tokenizer().load(params.tokenizer_path)) {
        safe_print_error_ln("Pipeline error: could not load tokenizer from " + params.tokenizer_path);
        return false;
    }
    const auto tok_t1 = std::chrono::steady_clock::now();

    struct gguf_init_params gguf_params = { true, nullptr };
    gguf_context * shared_gguf = gguf_init_from_file(params.model_path.c_str(), gguf_params);
    if (!shared_gguf) {
        safe_print_error_ln("Pipeline error: failed to open GGUF from " + params.model_path);
        return false;
    }

    const auto model_t0 = std::chrono::steady_clock::now();
    if (!model().load_shared(shared_gguf, params.model_path, params.gpu_device, params.backend_type, params.n_gpu_layers)) {
        safe_print_error_ln("Pipeline error: could not load model from " + params.model_path);
        gguf_free(shared_gguf);
        return false;
    }
    const auto model_t1 = std::chrono::steady_clock::now();

    const auto codec_t0 = std::chrono::steady_clock::now();
    bool codec_loaded = false;
    BackendType codec_backend_type = BackendType::CPU;
    int32_t codec_gpu_device = -1;

    const bool auto_select_codec_backend = should_auto_select_codec_backend(params);
    const bool prefer_gpu_codec =
        !auto_select_codec_backend &&
        params.codec_follow_backend &&
        (params.gpu_device >= 0 || params.backend_type == BackendType::Metal) &&
        params.backend_type != BackendType::CPU;

    if (auto_select_codec_backend) {
        safe_print_ln("Pipeline: benchmarking codec backends (performance-first)...");
        CodecBenchmarkResult cpu_result = benchmark_codec_backend(params, BackendType::CPU, -1);
        CodecBenchmarkResult gpu_result = benchmark_codec_backend(params, params.backend_type, params.gpu_device);

        auto format_result = [](const CodecBenchmarkResult & r) -> std::string {
            if (!r.ok) {
                return r.backend_name + " unavailable (" + r.error + ")";
            }
            return r.backend_name + "=" + std::to_string(r.decode_ms) + " ms";
        };

        safe_print_ln(
            "Pipeline: codec benchmark results: CPU " + format_result(cpu_result) +
            ", backend " + format_result(gpu_result));

        const bool choose_gpu =
            gpu_result.ok &&
            (!cpu_result.ok || gpu_result.decode_ms < cpu_result.decode_ms * 0.90);

        if (choose_gpu) {
            codec_backend_type = params.backend_type;
            codec_gpu_device = params.gpu_device;
            safe_print_ln("Pipeline: selected codec backend " + gpu_result.backend_name + " for best throughput.");
        } else {
            codec_backend_type = BackendType::CPU;
            codec_gpu_device = -1;
            safe_print_ln("Pipeline: selected codec backend CPU for best throughput.");
        }
    } else if (prefer_gpu_codec) {
        codec_backend_type = params.backend_type;
        codec_gpu_device = params.gpu_device;
    }

    const bool use_gpu_codec =
        codec_backend_type != BackendType::CPU &&
        (codec_gpu_device >= 0 || codec_backend_type == BackendType::Metal);

    if (use_gpu_codec) {
        std::string backend_label = std::string(backend_type_name(codec_backend_type));
        if (codec_backend_type == BackendType::Metal) {
            safe_print_ln("Pipeline: loading codec on " + backend_label + "...");
        } else {
            safe_print_ln("Pipeline: loading codec on " + backend_label +
                          " device " + std::to_string(codec_gpu_device) + "...");
        }
        codec_loaded = codec().load_shared(&model(), shared_gguf, params.model_path, codec_gpu_device, codec_backend_type);
        if (!codec_loaded) {
            if (codec_backend_type == BackendType::Metal) {
                safe_print_warn_ln(
                    "Pipeline warning: codec " + backend_label +
                    " load failed, falling back to CPU.");
            } else {
                safe_print_warn_ln(
                    "Pipeline warning: codec " + backend_label +
                    " load failed on device " + std::to_string(codec_gpu_device) +
                    ", falling back to CPU.");
            }
        }
    }
    if (!codec_loaded) {
        if (!use_gpu_codec) {
            safe_print_ln("Pipeline: loading codec on CPU.");
        }
        codec_loaded = codec().load_shared(&model(), shared_gguf, params.model_path, -1, BackendType::CPU);
    }
    if (!codec_loaded) {
        safe_print_error_ln("Pipeline error: could not load codec from " + params.model_path);
        gguf_free(shared_gguf);
        return false;
    }

    if (!read_all_tensor_data(params.model_path, shared_gguf, model(), codec())) {
        safe_print_error_ln("Pipeline error: failed to read tensor data from " + params.model_path);
        gguf_free(shared_gguf);
        return false;
    }

    gguf_free(shared_gguf);

    const auto codec_t1 = std::chrono::steady_clock::now();

    sync_tokenizer_config_from_model(tokenizer(), model());

    initialized_ = true;
    const auto init_t1 = std::chrono::steady_clock::now();
    safe_print_ln(
        "[Metrics] Init: tokenizer=" +
        std::to_string(std::chrono::duration<double, std::milli>(tok_t1 - tok_t0).count()) +
        " ms, model=" +
        std::to_string(std::chrono::duration<double, std::milli>(model_t1 - model_t0).count()) +
        " ms, codec=" +
        std::to_string(std::chrono::duration<double, std::milli>(codec_t1 - codec_t0).count()) +
        " ms (" + codec().backend_name() + "), total=" +
        std::to_string(std::chrono::duration<double, std::milli>(init_t1 - init_t0).count()) +
        " ms, max_rss=" +
        std::to_string(get_max_rss_mb()) + " MB");
    return true;
}

bool Pipeline::init_from_components(Tokenizer* tokenizer, SlowARModel* model, AudioCodec* codec) {
    std::lock_guard<std::mutex> lock(synthesize_mutex_);
    if (!tokenizer || !model || !codec || initialized_) {
        return false;
    }

    tokenizer_ref_ = tokenizer;
    model_ref_ = model;
    codec_ref_ = codec;
    sync_tokenizer_config_from_model(*tokenizer, *model);
    initialized_ = true;
    return true;
}

int32_t Pipeline::output_sample_rate() const {
    return codec().sample_rate();
}

bool Pipeline::save_voice_profile_locked(const std::string & voice_id,
                                         const std::vector<int32_t> & codes,
                                         int32_t T_prompt,
                                         const std::string & transcript,
                                         const PipelineParams & params) {
    if (voice_id.empty() || codes.empty() || T_prompt <= 0) {
        return false;
    }
    if (transcript.empty()) {
        safe_print_error_ln("Pipeline error: cannot save voice profile without prompt text.");
        return false;
    }

    voice_mgr_.set_storage_dir(params.voice_storage_dir);

    VoiceProfile profile;
    profile.transcript = transcript;
    profile.codes = codes;
    profile.num_codebooks = model().hparams().num_codebooks;
    profile.T_prompt = T_prompt;
    profile.sample_rate = codec().sample_rate();
    profile.codebook_size = model().hparams().codebook_size;

    if (!voice_mgr_.save(voice_id, profile)) {
        safe_print_error_ln("Pipeline error: failed to save voice profile: " + voice_id);
        return false;
    }

    safe_print_ln("Saved voice profile: " + voice_id);
    return true;
}

bool Pipeline::resolve_reference_prompt_locked(const PipelineParams & params, AudioData & ref_audio,
                                               std::vector<int32_t> & ref_codes, int32_t & T_prompt,
                                               std::string & effective_prompt_text,
                                               double & ref_encode_ms) {
    ref_codes.clear();
    T_prompt = 0;
    ref_encode_ms = 0.0;
    effective_prompt_text = params.prompt_text;

    voice_mgr_.set_storage_dir(params.voice_storage_dir);

    if (!ref_audio.samples.empty()) {
        const auto ref_t0 = std::chrono::steady_clock::now();
        if (!codec().encode(ref_audio.samples.data(), static_cast<int32_t>(ref_audio.samples.size()),
                            params.gen.n_threads, ref_codes, T_prompt)) {
            safe_print_warn_ln("Pipeline warning: encode failed, running without reference audio.");
            ref_codes.clear();
            T_prompt = 0;
        }
        const auto ref_t1 = std::chrono::steady_clock::now();
        ref_encode_ms = std::chrono::duration<double, std::milli>(ref_t1 - ref_t0).count();

        if (!ref_codes.empty() && params.save_voice && !params.voice_id.empty()) {
            save_voice_profile_locked(params.voice_id, ref_codes, T_prompt,
                                      effective_prompt_text, params);
        }
        return true;
    }

    if (params.voice_id.empty()) {
        return true;
    }

    safe_print_ln("Loading voice profile: " + params.voice_id);
    try {
        VoiceProfile profile = voice_mgr_.load(params.voice_id);
        if (!profile.is_compatible(model().hparams().num_codebooks,
                                   model().hparams().codebook_size,
                                   codec().sample_rate())) {
            safe_print_error_ln("Pipeline error: voice profile incompatible with current model/codec.");
            return false;
        }

        ref_codes = std::move(profile.codes);
        T_prompt = profile.T_prompt;
        effective_prompt_text = std::move(profile.transcript);
        safe_print_ln("Loaded voice profile: " + params.voice_id +
                      " (" + std::to_string(T_prompt) + " frames)");
    } catch (const std::exception & e) {
        safe_print_error_ln("Pipeline error: failed to load voice profile " +
                            params.voice_id + ": " + e.what());
        return false;
    }

    return true;
}

bool Pipeline::synthesize(const PipelineParams & params) {
    std::vector<float> audio_out;
    AudioData ref_audio;

    if (!params.prompt_audio_path.empty() && params.prompt_text.empty()) {
        safe_print_error_ln("Pipeline error: prompt audio was provided without prompt text.");
        return false;
    }

    if (!params.prompt_audio_path.empty()) {
        safe_print_ln("Loading reference audio: " + params.prompt_audio_path);
        if (!load_audio(params.prompt_audio_path, ref_audio, codec().sample_rate())) {
            safe_print_warn_ln("Pipeline warning: load_audio failed, running without reference audio.");
        }
    }

    if (!this->synthesize_raw(params, ref_audio, audio_out)) {
        safe_print_error_ln("Pipeline error: synthesis failed.");
        return false;
    }

    if (params.trim_silence && !audio_out.empty()) {
        auto trimmed = audio_trim_trailing_silence(audio_out.data(), audio_out.size(), codec().sample_rate());
        if (!trimmed.empty()) audio_out = std::move(trimmed);
    }

    if (params.normalize_dynamic && !audio_out.empty()) {
        audio_out = audio_normalize_dynamic(audio_out.data(), audio_out.size(), codec().sample_rate());
    }

    if (!save_audio(params.output_path, audio_out, codec().sample_rate(),
                    false, params.normalize_output)) {
        safe_print_error_ln("Pipeline error: save_audio failed to " + params.output_path);
        return false;
    }

    safe_print_ln("Saved audio to: " + params.output_path);
    return true;
}

bool Pipeline::synthesize_to_memory(const PipelineParams & params, const void * ref_audio_buffer, size_t ref_audio_size, void** wav_buffer, size_t* wav_size) {
    std::vector<float> audio_out;
    AudioData ref_audio;

    if (ref_audio_buffer != nullptr && ref_audio_size > 0 && params.prompt_text.empty()) {
        safe_print_error_ln("Pipeline error: reference audio was provided without reference text.");
        return false;
    }

    if (ref_audio_buffer != nullptr && ref_audio_size > 0) {
        safe_print_ln("Loading reference audio...");
        if (!load_audio_from_memory(ref_audio_buffer, ref_audio_size, ref_audio, codec().sample_rate())) {
            safe_print_warn_ln("Pipeline warning: load_audio failed, running without reference audio.");
        }
    }

    if (!this->synthesize_raw(params, ref_audio, audio_out)) {
        safe_print_error_ln("Pipeline error: synthesis failed.");
        return false;
    }

    if (params.trim_silence && !audio_out.empty()) {
        auto trimmed = audio_trim_trailing_silence(audio_out.data(), audio_out.size(), codec().sample_rate());
        if (!trimmed.empty()) audio_out = std::move(trimmed);
    }

    if (params.normalize_dynamic && !audio_out.empty()) {
        audio_out = audio_normalize_dynamic(audio_out.data(), audio_out.size(), codec().sample_rate());
    } else if (params.normalize_output && !audio_out.empty()) {
        float peak = 0.0f;
        for (float s : audio_out) { float a = std::fabs(s); if (a > peak) peak = a; }
        if (peak > 1e-6f) { float scale = 0.95f / peak; for (float & s : audio_out) s *= scale; }
    }

    if (!audio_write_memory_wav(wav_buffer, wav_size, audio_out.data(), audio_out.size(), codec().sample_rate())) {
        safe_print_error_ln("Pipeline error: audio_write_memory_wav failed");
        return false;
    }

    safe_print_ln("Audio synthesized");
    return true;
}

bool Pipeline::encode_prompt_audio(const std::string & audio_path, int32_t n_threads,
                                   std::vector<int32_t> & codes_out, int32_t & n_frames_out) {
    std::lock_guard<std::mutex> lock(synthesize_mutex_);
    if (!initialized_) {
        safe_print_error_ln("Pipeline not initialized.");
        return false;
    }

    codes_out.clear();
    n_frames_out = 0;
    if (audio_path.empty()) {
        return false;
    }

    AudioData ref_audio;
    if (!load_audio(audio_path, ref_audio, codec().sample_rate())) {
        safe_print_warn_ln("Pipeline warning: load_audio failed, running without reference audio.");
        return false;
    }

    if (!codec().encode(ref_audio.samples.data(), static_cast<int32_t>(ref_audio.samples.size()),
                        n_threads, codes_out, n_frames_out)) {
        safe_print_warn_ln("Pipeline warning: encode failed, running without reference audio.");
        codes_out.clear();
        n_frames_out = 0;
        return false;
    }

    return true;
}

bool Pipeline::encode_prompt_audio_data(const AudioData & ref_audio, int32_t n_threads,
                                        std::vector<int32_t> & codes_out, int32_t & n_frames_out) {
    std::lock_guard<std::mutex> lock(synthesize_mutex_);
    if (!initialized_) {
        safe_print_error_ln("Pipeline not initialized.");
        return false;
    }

    codes_out.clear();
    n_frames_out = 0;
    if (ref_audio.samples.empty()) {
        return false;
    }

    if (!codec().encode(ref_audio.samples.data(), static_cast<int32_t>(ref_audio.samples.size()),
                        n_threads, codes_out, n_frames_out)) {
        safe_print_warn_ln("Pipeline warning: encode failed, running without reference audio.");
        codes_out.clear();
        n_frames_out = 0;
        return false;
    }

    return true;
}

bool Pipeline::synthesize_raw(const PipelineParams & params, AudioData & ref_audio, std::vector<float>& audio_out) {
    std::lock_guard<std::mutex> lock(synthesize_mutex_);
    std::vector<int32_t> ref_codes;
    int32_t T_prompt = 0;
    double ref_encode_ms = 0.0;
    std::string effective_prompt_text = params.prompt_text;

    if (!initialized_) {
        safe_print_error_ln("Pipeline not initialized.");
        return false;
    }

    if (!resolve_reference_prompt_locked(params, ref_audio, ref_codes, T_prompt,
                                         effective_prompt_text, ref_encode_ms)) {
        return false;
    }

    PipelineParams effective_params = params;
    effective_params.prompt_text = std::move(effective_prompt_text);

    return synthesize_prompt_codes_locked(
        effective_params,
        ref_codes.empty() ? nullptr : ref_codes.data(),
        T_prompt,
        audio_out,
        ref_encode_ms);
}

bool Pipeline::synthesize_with_prompt_codes(const PipelineParams & params, const int32_t* ref_codes,
                                            int32_t T_prompt, std::vector<float> & audio_out) {
    std::lock_guard<std::mutex> lock(synthesize_mutex_);
    return synthesize_prompt_codes_locked(params, ref_codes, T_prompt, audio_out, 0.0);
}

bool Pipeline::resolve_prompt_reference(const PipelineParams & params, AudioData & ref_audio,
                                        std::vector<int32_t> & ref_codes, int32_t & T_prompt,
                                        std::string & effective_prompt_text, double & ref_encode_ms) {
    std::lock_guard<std::mutex> lock(synthesize_mutex_);
    if (!initialized_) {
        safe_print_error_ln("Pipeline not initialized.");
        return false;
    }
    return resolve_reference_prompt_locked(params, ref_audio, ref_codes, T_prompt,
                                           effective_prompt_text, ref_encode_ms);
}

bool Pipeline::synthesize_prompt_codes_locked(const PipelineParams & params, const int32_t* ref_codes,
                                              int32_t T_prompt, std::vector<float> & audio_out,
                                              double ref_encode_ms) {
    const auto synth_t0 = std::chrono::steady_clock::now();

    if (!initialized_) {
        safe_print_error_ln("Pipeline not initialized.");
        return false;
    }

    CodecDecodeCacheScope codec_decode_cache_scope(codec());
    model().clear_kv_cache();

    safe_print_ln("--- Pipeline Synthesize ---");
    safe_print_ln("Text: " + params.text);

    const int32_t num_codebooks = model().hparams().num_codebooks;

    PromptTensor prompt = build_prompt(
        tokenizer(), params.text, params.prompt_text,
        ref_codes,
        num_codebooks, T_prompt);

    int32_t max_seq_len = prompt.cols + params.gen.max_new_tokens;
    const auto kv_t0 = std::chrono::steady_clock::now();
    if (!model().init_kv_cache(max_seq_len)) {
        safe_print_error_ln("Pipeline error: init_kv_cache failed.");
        return false;
    }
    const auto kv_t1 = std::chrono::steady_clock::now();

    const auto gen_t0 = std::chrono::steady_clock::now();
    GenerateResult res = generate(model(), tokenizer().config(), prompt, params.gen);
    const auto gen_t1 = std::chrono::steady_clock::now();

    if (res.n_frames == 0) {
        safe_print_error_ln("Pipeline error: generation produced no frames.");
        return false;
    }

    const int32_t offline_decode_stride_frames =
        params.stream_decode_stride_frames > 0 ? params.stream_decode_stride_frames : 16;
    double decode_ms = 0.0;
    int32_t decode_batches = 0;
    const auto decode_t0 = std::chrono::steady_clock::now();
    if (!decode_codes_windowed(codec(), res.codes.data(), res.n_frames, num_codebooks,
                               params.gen.n_threads, offline_decode_stride_frames,
                               params.codec_decode_context_frames,
                               audio_out, &decode_ms, &decode_batches)) {
        safe_print_error_ln("Pipeline error: decode failed.");
        return false;
    }
    const auto decode_t1 = std::chrono::steady_clock::now();

    model().clear_kv_cache();
    const auto synth_t1 = std::chrono::steady_clock::now();

    const double kv_ms = std::chrono::duration<double, std::milli>(kv_t1 - kv_t0).count();
    const double gen_ms = std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count();
    const double decode_wall_ms = std::chrono::duration<double, std::milli>(decode_t1 - decode_t0).count();
    const double total_ms = std::chrono::duration<double, std::milli>(synth_t1 - synth_t0).count();
    const double audio_seconds = codec().sample_rate() > 0
        ? (static_cast<double>(audio_out.size()) / codec().sample_rate())
        : 0.0;
    const double gen_ms_per_frame = res.n_frames > 0 ? (gen_ms / res.n_frames) : 0.0;
    const double total_ms_per_frame = res.n_frames > 0 ? (total_ms / res.n_frames) : 0.0;
    const double gen_rtf = audio_seconds > 0.0 ? ((gen_ms / 1000.0) / audio_seconds) : 0.0;
    const double total_rtf = audio_seconds > 0.0 ? ((total_ms / 1000.0) / audio_seconds) : 0.0;

    safe_print_ln(
        "[Metrics] Synthesis: frames=" + std::to_string(res.n_frames) +
        ", audio_s=" + std::to_string(audio_seconds) +
        ", ref_encode=" + std::to_string(ref_encode_ms) +
        " ms, kv_init=" + std::to_string(kv_ms) +
        " ms, generate=" + std::to_string(gen_ms) +
        " ms, decode=" + std::to_string(decode_ms) +
        " ms, decode_wall=" + std::to_string(decode_wall_ms) +
        " ms, decode_batches=" + std::to_string(decode_batches) +
        ", decode_stride=" + std::to_string(offline_decode_stride_frames) +
        " frames, total=" + std::to_string(total_ms) +
        " ms, gen_avg=" + std::to_string(gen_ms_per_frame) +
        " ms/frame, total_avg=" + std::to_string(total_ms_per_frame) +
        " ms/frame, gen_rtf=" + std::to_string(gen_rtf) +
        ", total_rtf=" + std::to_string(total_rtf) +
        ", max_rss=" + std::to_string(get_max_rss_mb()) + " MB");
    return true;
}

bool Pipeline::synthesize_streaming_raw(const PipelineParams & params, AudioData & ref_audio,
                                        StreamingSink & sink) {
    std::lock_guard<std::mutex> lock(synthesize_mutex_);
    std::vector<int32_t> ref_codes;
    int32_t T_prompt = 0;
    double ref_encode_ms = 0.0;
    std::string effective_prompt_text = params.prompt_text;

    if (!initialized_) {
        safe_print_error_ln("Pipeline not initialized.");
        sink.on_error("Pipeline not initialized");
        return false;
    }

    if (!resolve_reference_prompt_locked(params, ref_audio, ref_codes, T_prompt,
                                         effective_prompt_text, ref_encode_ms)) {
        sink.on_error("Failed to resolve reference prompt");
        return false;
    }

    PipelineParams effective_params = params;
    effective_params.prompt_text = std::move(effective_prompt_text);

    return synthesize_streaming_prompt_codes_locked(
        effective_params,
        ref_codes.empty() ? nullptr : ref_codes.data(),
        T_prompt,
        sink,
        ref_encode_ms);
}

bool Pipeline::synthesize_streaming_with_prompt_codes(const PipelineParams & params,
                                                      const int32_t* ref_codes,
                                                      int32_t T_prompt,
                                                      StreamingSink & sink) {
    std::lock_guard<std::mutex> lock(synthesize_mutex_);
    return synthesize_streaming_prompt_codes_locked(params, ref_codes, T_prompt, sink, 0.0);
}

bool Pipeline::synthesize_streaming_prompt_codes_locked(const PipelineParams & params,
                                                        const int32_t* ref_codes,
                                                        int32_t T_prompt,
                                                        StreamingSink & sink,
                                                        double ref_encode_ms) {
    const auto synth_t0 = std::chrono::steady_clock::now();

    if (!initialized_) {
        safe_print_error_ln("Pipeline not initialized.");
        sink.on_error("Pipeline not initialized");
        return false;
    }

    CodecDecodeCacheScope codec_decode_cache_scope(codec());
    model().clear_kv_cache();

    safe_print_ln("--- Pipeline Streaming Synthesize ---");
    safe_print_ln("Text: " + params.text);

    const int32_t num_codebooks = model().hparams().num_codebooks;
    const int32_t sample_rate = codec().sample_rate();

    uint8_t wav_header[44];
    audio_write_streaming_wav_header(wav_header, sample_rate, 1, 16);
    if (!sink.on_header(wav_header, 44)) {
        model().clear_kv_cache();
        return false;
    }

    PromptTensor prompt = build_prompt(
        tokenizer(), params.text, params.prompt_text,
        ref_codes,
        num_codebooks, T_prompt);

    int32_t max_seq_len = prompt.cols + params.gen.max_new_tokens;
    const auto kv_t0 = std::chrono::steady_clock::now();
    if (!model().init_kv_cache(max_seq_len)) {
        safe_print_error_ln("Pipeline error: init_kv_cache failed.");
        sink.on_error("init_kv_cache failed");
        model().clear_kv_cache();
        return false;
    }
    const auto kv_t1 = std::chrono::steady_clock::now();

    std::vector<std::vector<int32_t>> accumulated_codes_by_cb(num_codebooks);
    for (auto & row : accumulated_codes_by_cb) {
        row.reserve(static_cast<size_t>(params.gen.max_new_tokens));
    }
    std::vector<int32_t> decode_codes;
    size_t emitted_samples = 0;
    int32_t last_decoded_frames = 0;
    int32_t committed_frames = 0;
    double stream_decode_ms = 0.0;
    int32_t stream_decode_batches = 0;
    bool stream_aborted = false;
    bool stream_failed = false;
    const int32_t stream_decode_stride_frames =
        params.stream_decode_stride_frames > 0 ? params.stream_decode_stride_frames : 4;
    const int32_t codec_context_frames = params.codec_decode_context_frames >= 0
        ? params.codec_decode_context_frames
        : std::max(0, codec().streaming_history_frames());
    const int32_t stream_holdback_frames =
        params.stream_holdback_frames >= 0 ? params.stream_holdback_frames : codec_context_frames;
    const size_t samples_per_frame = static_cast<size_t>(std::max(1, codec().samples_per_code_frame()));

    auto emit_pcm_range = [&](const std::vector<float> & pcm,
                              size_t begin_samples,
                              size_t end_samples) -> bool {
        begin_samples = std::min(begin_samples, pcm.size());
        end_samples = std::min(end_samples, pcm.size());
        if (end_samples <= begin_samples) {
            return true;
        }
        const size_t delta_count = end_samples - begin_samples;
        if (!sink.on_pcm_data(pcm.data() + begin_samples, delta_count)) {
            stream_aborted = true;
            return false;
        }
        emitted_samples += delta_count;
        return true;
    };

    auto decode_window_and_emit = [&](int32_t total_frames, bool finalize) -> bool {
        if (total_frames <= 0 || total_frames <= committed_frames) {
            return true;
        }

        const int32_t stable_frames = finalize
            ? total_frames
            : std::max(0, total_frames - stream_holdback_frames);
        if (stable_frames <= committed_frames && !finalize) {
            return true;
        }

        const int32_t window_start_frame = std::max(0, committed_frames - codec_context_frames);
        const int32_t window_frames = total_frames - window_start_frame;
        if (window_frames <= 0) {
            return true;
        }

        decode_codes.resize(static_cast<size_t>(num_codebooks) * window_frames);
        for (int32_t cb = 0; cb < num_codebooks; ++cb) {
            if (static_cast<int32_t>(accumulated_codes_by_cb[cb].size()) < total_frames) {
                safe_print_error_ln("Pipeline streaming: internal codebook accumulation underflow.");
                sink.on_error("Internal streaming codebook accumulation error");
                stream_failed = true;
                return false;
            }
            std::copy(
                accumulated_codes_by_cb[cb].begin() + window_start_frame,
                accumulated_codes_by_cb[cb].begin() + total_frames,
                decode_codes.begin() + static_cast<size_t>(cb) * window_frames
            );
        }

        std::vector<float> pcm;
        const auto decode_t0 = std::chrono::steady_clock::now();
        if (!codec().decode(decode_codes.data(), window_frames,
                           params.gen.n_threads, pcm)) {
            safe_print_error_ln("Pipeline streaming: decode failed at frame batch ending " +
                                std::to_string(total_frames - 1));
            sink.on_error("Codec decode failed");
            stream_failed = true;
            return false;
        }
        const auto decode_t1 = std::chrono::steady_clock::now();
        stream_decode_ms += std::chrono::duration<double, std::milli>(decode_t1 - decode_t0).count();
        stream_decode_batches++;

        const size_t emit_begin_samples =
            static_cast<size_t>(std::max(0, committed_frames - window_start_frame)) * samples_per_frame;
        const size_t emit_end_samples =
            finalize
                ? pcm.size()
                : std::min(
                    pcm.size(),
                    static_cast<size_t>(std::max(0, stable_frames - window_start_frame)) * samples_per_frame
                );
        if (!emit_pcm_range(pcm, emit_begin_samples, emit_end_samples)) {
            return false;
        }

        committed_frames = finalize ? total_frames : stable_frames;
        last_decoded_frames = total_frames;
        return true;
    };

    GenerateParams gen_params = params.gen;
    gen_params.on_frame = [&](const FrameCallbackData & fcd) -> bool {
        if (sink.is_cancelled()) {
            stream_aborted = true;
            return false;
        }

        for (int32_t cb = 0; cb < fcd.num_codebooks; ++cb) {
            accumulated_codes_by_cb[cb].push_back(fcd.codes[cb]);
        }

        if ((fcd.total_frames - last_decoded_frames) < stream_decode_stride_frames) {
            return true;
        }
        return decode_window_and_emit(fcd.total_frames, false);
    };

    const auto gen_t0 = std::chrono::steady_clock::now();
    GenerateResult res = generate(model(), tokenizer().config(), prompt, gen_params);
    const auto gen_t1 = std::chrono::steady_clock::now();

    if (res.n_frames == 0) {
        if (stream_aborted) {
            model().clear_kv_cache();
            return false;
        }
        safe_print_error_ln("Pipeline error: generation produced no frames.");
        sink.on_error("Generation produced no frames");
        model().clear_kv_cache();
        return false;
    }

    if (stream_failed || stream_aborted) {
        model().clear_kv_cache();
        return false;
    }

    if (res.n_frames > last_decoded_frames || committed_frames < res.n_frames) {
        if (!decode_window_and_emit(res.n_frames, true)) {
            model().clear_kv_cache();
            return false;
        }
    }

    sink.on_done();
    safe_print_ln("Streaming synthesis complete: " + std::to_string(res.n_frames) + " frames");
    model().clear_kv_cache();
    const auto synth_t1 = std::chrono::steady_clock::now();

    const double kv_ms = std::chrono::duration<double, std::milli>(kv_t1 - kv_t0).count();
    const double gen_ms = std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count();
    const double total_ms = std::chrono::duration<double, std::milli>(synth_t1 - synth_t0).count();
    const double audio_seconds = sample_rate > 0
        ? (static_cast<double>(emitted_samples) / sample_rate)
        : 0.0;
    const double total_ms_per_frame = res.n_frames > 0 ? (total_ms / res.n_frames) : 0.0;
    const double decode_ms_per_frame = res.n_frames > 0 ? (stream_decode_ms / res.n_frames) : 0.0;
    const double ar_ms = std::max(0.0, gen_ms - stream_decode_ms);
    const double ar_ms_per_frame = res.n_frames > 0 ? (ar_ms / res.n_frames) : 0.0;
    const double total_rtf = audio_seconds > 0.0 ? ((total_ms / 1000.0) / audio_seconds) : 0.0;

    safe_print_ln(
        "[Metrics] Streaming: frames=" + std::to_string(res.n_frames) +
        ", audio_s=" + std::to_string(audio_seconds) +
        ", ref_encode=" + std::to_string(ref_encode_ms) +
        " ms, kv_init=" + std::to_string(kv_ms) +
        " ms, stride=" + std::to_string(stream_decode_stride_frames) +
        " frames, holdback=" + std::to_string(stream_holdback_frames) +
        " frames, decode_context=" + std::to_string(codec_context_frames) +
        " frames, generate=" + std::to_string(gen_ms) +
        " ms, stream_decode=" + std::to_string(stream_decode_ms) +
        " ms, stream_batches=" + std::to_string(stream_decode_batches) +
        ", ar_only=" + std::to_string(ar_ms) +
        " ms, total=" + std::to_string(total_ms) +
        " ms, total_avg=" + std::to_string(total_ms_per_frame) +
        " ms/frame, decode_avg=" + std::to_string(decode_ms_per_frame) +
        " ms/frame, ar_avg=" + std::to_string(ar_ms_per_frame) +
        " ms/frame, total_rtf=" + std::to_string(total_rtf) +
        ", max_rss=" + std::to_string(get_max_rss_mb()) + " MB");
    return true;
}

bool Pipeline::synthesize_streaming_file(const PipelineParams & params) {
    AudioData ref_audio;

    if (!params.prompt_audio_path.empty() && params.prompt_text.empty()) {
        safe_print_error_ln("Pipeline error: prompt audio was provided without prompt text.");
        return false;
    }

    if (!params.prompt_audio_path.empty()) {
        safe_print_ln("Loading reference audio: " + params.prompt_audio_path);
        if (!load_audio(params.prompt_audio_path, ref_audio, codec().sample_rate())) {
            safe_print_warn_ln("Pipeline warning: load_audio failed, running without reference audio.");
        }
    }

    FileStreamingSink sink(params.output_path);
    PipelineParams file_stream_params = params;
    if (file_stream_params.stream_decode_stride_frames <= 0) {
        file_stream_params.stream_decode_stride_frames = 16;
    }
    return synthesize_streaming_raw(file_stream_params, ref_audio, sink);
}

}
