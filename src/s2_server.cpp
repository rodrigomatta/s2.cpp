#include "../third_party/httplib.h"
#include "../third_party/json.hpp"
#include "../third_party/filesystem.hpp"

#include "../include/s2_log.h"
#include "../include/s2_server.h"
#include <iostream>
#include <thread>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <iterator>

using json = nlohmann::json;
namespace fs = ghc::filesystem;

namespace {

enum class StreamAudioFormat {
    Wav,
    PcmS16LE,
};

static StreamAudioFormat parse_stream_audio_format(const std::string & value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "pcm" || lower == "pcm16" || lower == "pcm_s16le" || lower == "s16le") {
        return StreamAudioFormat::PcmS16LE;
    }
    return StreamAudioFormat::Wav;
}

static std::string stream_audio_content_type(StreamAudioFormat format, int32_t sample_rate) {
    switch (format) {
        case StreamAudioFormat::PcmS16LE:
            return "audio/L16; rate=" + std::to_string(sample_rate) + "; channels=1";
        case StreamAudioFormat::Wav:
            return "audio/wav";
    }
    return "application/octet-stream";
}

static const char * stream_audio_filename(StreamAudioFormat format) {
    switch (format) {
        case StreamAudioFormat::PcmS16LE: return "generated_audio.pcm";
        case StreamAudioFormat::Wav:      return "generated_audio.wav";
    }
    return "generated_audio.bin";
}

static std::string trim_copy(const std::string & value) {
    size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }

    size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return value.substr(begin, end - begin);
}

static std::vector<std::string> split_sentences_basic(const std::string & text) {
    std::vector<std::string> segments;
    std::string current;
    current.reserve(text.size());

    auto flush_current = [&]() {
        std::string trimmed = trim_copy(current);
        if (!trimmed.empty()) {
            segments.push_back(std::move(trimmed));
        }
        current.clear();
    };

    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        current.push_back(c);

        const bool hard_break =
            c == '.' || c == '!' || c == '?' || c == '\n';
        if (!hard_break) {
            continue;
        }

        while (i + 1 < text.size()) {
            const char next = text[i + 1];
            if (next == '.' || next == '!' || next == '?' ||
                next == '"' || next == '\'' || next == ')' || next == ']' ||
                std::isspace(static_cast<unsigned char>(next))) {
                current.push_back(next);
                ++i;
                if (!std::isspace(static_cast<unsigned char>(next))) {
                    continue;
                }
            }
            break;
        }

        flush_current();
    }

    flush_current();
    if (segments.empty()) {
        const std::string trimmed = trim_copy(text);
        if (!trimmed.empty()) {
            segments.push_back(trimmed);
        }
    }
    return segments;
}

static std::vector<std::string> split_long_segment(const std::string & segment, size_t max_chars) {
    if (max_chars == 0 || segment.size() <= max_chars) {
        return {segment};
    }

    std::vector<std::string> parts;
    std::string remaining = trim_copy(segment);
    while (!remaining.empty() && remaining.size() > max_chars) {
        size_t split_pos = remaining.find_last_of(",;:", max_chars);
        if (split_pos == std::string::npos || split_pos < max_chars / 2) {
            split_pos = remaining.find_last_of(" \t", max_chars);
        }
        if (split_pos == std::string::npos || split_pos < max_chars / 2) {
            split_pos = max_chars;
        } else {
            split_pos += 1;
        }

        std::string head = trim_copy(remaining.substr(0, split_pos));
        if (!head.empty()) {
            parts.push_back(std::move(head));
        }
        remaining = trim_copy(remaining.substr(split_pos));
    }

    if (!remaining.empty()) {
        parts.push_back(std::move(remaining));
    }
    return parts;
}

static std::vector<std::string> split_text_for_segmented_synthesis(const std::string & text,
                                                                   size_t max_chars_per_segment) {
    std::vector<std::string> sentences = split_sentences_basic(text);
    if (max_chars_per_segment == 0) {
        return sentences;
    }

    std::vector<std::string> segments;
    for (const std::string & sentence : sentences) {
        std::vector<std::string> parts = split_long_segment(sentence, max_chars_per_segment);
        segments.insert(segments.end(),
                        std::make_move_iterator(parts.begin()),
                        std::make_move_iterator(parts.end()));
    }
    return segments;
}

static void apply_voice_selection(s2::PipelineParams & params,
                                  const std::string & voice_value,
                                  const std::string & voice_dir_value) {
    if (voice_value.empty()) {
        if (!voice_dir_value.empty()) {
            params.voice_storage_dir = voice_dir_value;
        }
        return;
    }

    fs::path voice_path(voice_value);
    if (voice_path.extension() == ".s2voice") {
        if (!voice_path.parent_path().empty()) {
            params.voice_storage_dir = voice_path.parent_path().string();
        } else if (!voice_dir_value.empty()) {
            params.voice_storage_dir = voice_dir_value;
        }
        params.voice_id = voice_path.stem().string();
        return;
    }

    params.voice_id = voice_value;
    if (!voice_dir_value.empty()) {
        params.voice_storage_dir = voice_dir_value;
    }
}

static bool synthesize_segmented_to_sink(s2::Pipeline & pipeline,
                                         const s2::PipelineParams & base_params,
                                         const s2::AudioData & ref_audio,
                                         const std::vector<std::string> & segments,
                                         int32_t sentence_pause_ms,
                                         s2::StreamingSink & sink) {
    if (segments.empty()) {
        sink.on_error("No text segments to synthesize");
        return false;
    }

    s2::AudioData ref_audio_copy = ref_audio;
    std::vector<int32_t> ref_codes;
    int32_t t_prompt = 0;
    std::string effective_prompt_text = base_params.prompt_text;
    double ref_encode_ms = 0.0;
    if (!pipeline.resolve_prompt_reference(base_params, ref_audio_copy, ref_codes, t_prompt,
                                          effective_prompt_text, ref_encode_ms)) {
        sink.on_error("Reference prompt resolution failed");
        return false;
    }

    uint8_t wav_header[44];
    s2::audio_write_streaming_wav_header(wav_header, pipeline.output_sample_rate(), 1, 16);
    if (!sink.on_header(wav_header, sizeof(wav_header))) {
        return false;
    }

    const int32_t sample_rate = pipeline.output_sample_rate();
    const size_t pause_samples = sentence_pause_ms > 0
        ? static_cast<size_t>((static_cast<int64_t>(sample_rate) * sentence_pause_ms) / 1000)
        : 0;
    std::vector<float> pause_pcm(pause_samples, 0.0f);

    for (size_t i = 0; i < segments.size(); ++i) {
        if (sink.is_cancelled()) {
            return false;
        }

        s2::PipelineParams segment_params = base_params;
        segment_params.text = segments[i];
        segment_params.prompt_text = effective_prompt_text;

        std::vector<float> audio_out;
        const bool ok = pipeline.synthesize_with_prompt_codes(
            segment_params,
            ref_codes.empty() ? nullptr : ref_codes.data(),
            t_prompt,
            audio_out);
        if (!ok) {
            sink.on_error("Segment synthesis failed");
            return false;
        }

        if (!audio_out.empty()) {
            std::vector<float> trimmed =
                s2::audio_trim_trailing_silence(audio_out.data(), audio_out.size(), sample_rate);
            if (!trimmed.empty()) {
                audio_out = std::move(trimmed);
            }

            if (!sink.on_pcm_data(audio_out.data(), audio_out.size())) {
                return false;
            }
        }

        if (i + 1 < segments.size() && !pause_pcm.empty()) {
            if (!sink.on_pcm_data(pause_pcm.data(), pause_pcm.size())) {
                return false;
            }
        }
    }

    sink.on_done();
    return true;
}

static bool patch_streaming_wav_header(std::vector<uint8_t> & wav_bytes) {
    if (wav_bytes.size() < 44 || wav_bytes.size() > 0xFFFFFFFFu) {
        return false;
    }

    const uint32_t riff_size = wav_bytes.size() >= 8
        ? static_cast<uint32_t>(wav_bytes.size() - 8)
        : 0;
    const uint32_t data_size = wav_bytes.size() >= 44
        ? static_cast<uint32_t>(wav_bytes.size() - 44)
        : 0;

    std::memcpy(wav_bytes.data() + 4, &riff_size, sizeof(riff_size));
    std::memcpy(wav_bytes.data() + 40, &data_size, sizeof(data_size));
    return true;
}

struct StreamContext {
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<std::vector<uint8_t>> chunks;
    size_t queued_bytes = 0;
    size_t startup_buffer_bytes = 0;
    bool started = false;
    bool done = false;
    bool error = false;
    std::string error_message;
    std::atomic<bool> cancelled{false};
};

class BufferedAudioSink : public s2::StreamingSink {
public:
    explicit BufferedAudioSink(StreamAudioFormat format) : format_(format) {}

    bool on_header(const uint8_t* header, size_t size) override {
        bytes_.clear();
        if (format_ == StreamAudioFormat::Wav) {
            bytes_.assign(header, header + size);
        }
        error_message_.clear();
        return true;
    }

    bool on_pcm_data(const float* data, size_t n_samples) override {
        if (!error_message_.empty()) {
            return false;
        }

        const std::vector<int16_t> pcm16 = s2::audio_to_pcm16(data, n_samples);
        const size_t old_size = bytes_.size();
        const size_t byte_count = pcm16.size() * sizeof(int16_t);
        bytes_.resize(old_size + byte_count);
        std::memcpy(bytes_.data() + old_size, pcm16.data(), byte_count);
        return true;
    }

    void on_done() override {
        if (format_ == StreamAudioFormat::Wav && !patch_streaming_wav_header(bytes_)) {
            error_message_ = "Failed to finalize streamed WAV header.";
        }
    }

    void on_error(const std::string& message) override {
        error_message_ = message;
    }

    bool ok() const {
        if (!error_message_.empty()) {
            return false;
        }
        if (format_ == StreamAudioFormat::Wav) {
            return bytes_.size() >= 44;
        }
        return !bytes_.empty();
    }

    const std::string & error_message() const {
        return error_message_;
    }

    const std::vector<uint8_t> & bytes() const {
        return bytes_;
    }

private:
    StreamAudioFormat format_;
    std::vector<uint8_t> bytes_;
    std::string error_message_;
};

class HttpStreamSink : public s2::StreamingSink {
public:
    HttpStreamSink(std::shared_ptr<StreamContext> ctx, StreamAudioFormat format)
        : ctx_(std::move(ctx)), format_(format) {}

    bool on_header(const uint8_t* header, size_t size) override {
        if (format_ != StreamAudioFormat::Wav) {
            return true;
        }
        std::vector<uint8_t> chunk(header, header + size);
        {
            std::lock_guard<std::mutex> lock(ctx_->mtx);
            ctx_->queued_bytes += chunk.size();
            ctx_->chunks.push_back(std::move(chunk));
        }
        ctx_->cv.notify_one();
        return true;
    }

    bool on_pcm_data(const float* data, size_t n_samples) override {
        if (ctx_->cancelled.load()) return false;
        const std::vector<int16_t> pcm16 = s2::audio_to_pcm16(data, n_samples);
        const size_t byte_count = pcm16.size() * sizeof(int16_t);
        std::vector<uint8_t> chunk(reinterpret_cast<const uint8_t*>(pcm16.data()),
                                    reinterpret_cast<const uint8_t*>(pcm16.data()) + byte_count);
        {
            std::lock_guard<std::mutex> lock(ctx_->mtx);
            ctx_->queued_bytes += chunk.size();
            ctx_->chunks.push_back(std::move(chunk));
        }
        ctx_->cv.notify_one();
        return true;
    }

    void on_done() override {
        std::lock_guard<std::mutex> lock(ctx_->mtx);
        ctx_->done = true;
        ctx_->cv.notify_one();
    }

    void on_error(const std::string& message) override {
        std::lock_guard<std::mutex> lock(ctx_->mtx);
        ctx_->error = true;
        ctx_->error_message = message;
        ctx_->cv.notify_one();
    }

    bool is_cancelled() const override {
        return ctx_->cancelled.load();
    }

private:
    std::shared_ptr<StreamContext> ctx_;
    StreamAudioFormat format_;
};

}

namespace s2
{
    static std::string get_first_form_field(const httplib::MultipartFormData& form,
                                            const std::initializer_list<const char*>& keys) {
        for (const char* key : keys) {
            if (form.has_field(key)) {
                return form.get_field(key);
            }
        }
        return {};
    }

    static bool get_first_form_file(const httplib::MultipartFormData& form,
                                    const std::initializer_list<const char*>& keys,
                                    httplib::FormData& out) {
        for (const char* key : keys) {
            if (form.has_file(key)) {
                out = form.get_file(key);
                return true;
            }
        }
        return false;
    }

    Server::Server() {}
    Server::~Server() {}

    bool Server::serve(const ServerParams& params)
    {
        httplib::Server svr;
        auto server_busy = std::make_shared<std::atomic<bool>>(false);

        auto pipeline = std::make_shared<s2::Pipeline>();
        if (!pipeline->init(params.pipeline))
        {
            std::cerr << "Pipeline initialization failed." << std::endl;
            return 0;
        }

        std::mutex active_threads_mtx;
        std::vector<std::thread> active_threads;

        svr.set_pre_routing_handler([](const auto& req, auto& res) -> httplib::Server::HandlerResponse {
            auto start = std::chrono::high_resolution_clock::now();

            S2_LOG_INFO_STREAM("[START] " << req.method << " " << req.path << std::endl);

            res.set_header("X-Request-Start",
                std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                    start.time_since_epoch()).count()));

            return httplib::Server::HandlerResponse::Unhandled;
            });

        svr.set_logger([](const auto& req, const auto& res)
            {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end.time_since_epoch()).count() -
                    std::stoll(res.get_header_value("X-Request-Start", "0"));

                S2_LOG_INFO_STREAM("[END] " << req.method << " " << req.path
                    << " -> " << res.status
                    << " (" << duration << "ms)" << std::endl); });

        svr.Post("/generate", [pipeline, server_busy, &active_threads_mtx, &active_threads, &params](const httplib::Request& req, httplib::Response& res)
            {

                PipelineParams pipelineParams = params.pipeline;
                bool stream_response = false;
                bool chunked_response = false;
                bool low_latency_mode = false;
                bool explicit_stream_stride = false;
                bool explicit_stream_holdback = false;
                bool explicit_start_buffer = false;
                int32_t stream_start_buffer_ms = 0;
                bool segment_sentences = false;
                int32_t sentence_pause_ms = 180;
                int32_t segment_max_chars = 0;
                StreamAudioFormat stream_audio_format = StreamAudioFormat::Wav;

                if (!req.form.has_field("text"))
                {
                    json err = { {"error", "No text field in multipart form"} };
                    res.set_content(err.dump(), "application/json");
                    res.status = 400;
                    return;
                }

                pipelineParams.text = req.form.get_field("text");

                pipelineParams.prompt_text = get_first_form_field(
                    req.form, {"reference_text", "ref_text", "prompt_text"});
                std::string voice_value = get_first_form_field(
                    req.form, {"voice", "voice_id", "voice_profile"});
                std::string voice_dir_value = get_first_form_field(
                    req.form, {"voice_dir"});

                if (req.form.has_field("params"))
                {
                    try {
                        auto j = json::parse(req.form.get_field("params"));
                        if (!j.is_object()) {
                            json err = { {"error", "Params JSON must be an object"} };
                            res.set_content(err.dump(), "application/json");
                            res.status = 400;
                            return;
                        }

                        if (j.contains("max_new_tokens")) {
                            int32_t val = j["max_new_tokens"].get<int32_t>();
                            pipelineParams.gen.max_new_tokens = std::max(0, val);
                        }

                        if (j.contains("temperature")) {
                            float val = j["temperature"].get<float>();
                            pipelineParams.gen.temperature = std::max(0.0f, val);
                        }

                        if (j.contains("top_p")) {
                            float val = j["top_p"].get<float>();
                            pipelineParams.gen.top_p = std::max(0.0f, val);
                        }

                        if (j.contains("top_k")) {
                            int32_t val = j["top_k"].get<int32_t>();
                            pipelineParams.gen.top_k = std::max(0, val);
                        }

                        if (j.contains("min_tokens_before_end")) {
                            int32_t val = j["min_tokens_before_end"].get<int32_t>();
                            pipelineParams.gen.min_tokens_before_end = std::max(0, val);
                        }

                        if (j.contains("n_threads")) {
                            int32_t val = j["n_threads"].get<int32_t>();
                            pipelineParams.gen.n_threads = std::max(1, val);
                        }

                        if (j.contains("codec_follow_backend")) {
                            pipelineParams.codec_auto_backend = false;
                            pipelineParams.codec_follow_backend = j["codec_follow_backend"].get<bool>();
                        }

                        if (j.contains("codec_auto_backend")) {
                            pipelineParams.codec_auto_backend = j["codec_auto_backend"].get<bool>();
                        }

                        if (j.contains("voice")) {
                            voice_value = j["voice"].get<std::string>();
                        }

                        if (j.contains("voice_id")) {
                            voice_value = j["voice_id"].get<std::string>();
                        }

                        if (j.contains("voice_dir")) {
                            voice_dir_value = j["voice_dir"].get<std::string>();
                        }

                        if (j.contains("stream_decode_stride_frames")) {
                            int32_t val = j["stream_decode_stride_frames"].get<int32_t>();
                            pipelineParams.stream_decode_stride_frames = std::max(0, val);
                            explicit_stream_stride = true;
                        }

                        if (j.contains("stream_decode_stride")) {
                            int32_t val = j["stream_decode_stride"].get<int32_t>();
                            pipelineParams.stream_decode_stride_frames = std::max(0, val);
                            explicit_stream_stride = true;
                        }

                        if (j.contains("stream_holdback_frames")) {
                            int32_t val = j["stream_holdback_frames"].get<int32_t>();
                            pipelineParams.stream_holdback_frames = std::max(0, val);
                            explicit_stream_holdback = true;
                        }

                        if (j.contains("codec_decode_context_frames")) {
                            int32_t val = j["codec_decode_context_frames"].get<int32_t>();
                            pipelineParams.codec_decode_context_frames = std::max(0, val);
                        }

                        if (j.contains("codec_context_frames")) {
                            int32_t val = j["codec_context_frames"].get<int32_t>();
                            pipelineParams.codec_decode_context_frames = std::max(0, val);
                        }

                        if (j.contains("stream_start_buffer_ms")) {
                            int32_t val = j["stream_start_buffer_ms"].get<int32_t>();
                            stream_start_buffer_ms = std::max(0, val);
                            explicit_start_buffer = true;
                        }

                        if (j.contains("segment_sentences")) {
                            segment_sentences = j["segment_sentences"].get<bool>();
                        }

                        if (j.contains("sentence_pause_ms")) {
                            int32_t val = j["sentence_pause_ms"].get<int32_t>();
                            sentence_pause_ms = std::max(0, val);
                        }

                        if (j.contains("segment_max_chars")) {
                            int32_t val = j["segment_max_chars"].get<int32_t>();
                            segment_max_chars = std::max(0, val);
                        }

                        if (j.contains("low_latency")) {
                            low_latency_mode = j["low_latency"].get<bool>();
                        }

                        if (j.contains("output_format")) {
                            stream_audio_format =
                                parse_stream_audio_format(j["output_format"].get<std::string>());
                        }

                        if (j.contains("verbose")) {
                            bool val = j["verbose"].get<bool>();
                            pipelineParams.gen.verbose = val;
                        }

                        if (j.contains("stream")) {
                            stream_response = j["stream"].get<bool>();
                        }

                        if (j.contains("chunked")) {
                            chunked_response = j["chunked"].get<bool>();
                        }

                        if (j.contains("realtime")) {
                            chunked_response = j["realtime"].get<bool>();
                        }
                    }
                    catch (const json::parse_error& e) {
                        json err = { {"error", "JSON parse error"} };
                        res.set_content(err.dump(), "application/json");
                        res.status = 400;
                        return;
                    }
                    catch (const json::exception& e) {
                        json err = { {"error", std::string("Invalid params JSON: ") + e.what()} };
                        res.set_content(err.dump(), "application/json");
                        res.status = 400;
                        return;
                    }
                }

                apply_voice_selection(pipelineParams, voice_value, voice_dir_value);

                if (low_latency_mode) {
                    if (!explicit_stream_stride && pipelineParams.stream_decode_stride_frames <= 0) {
                        pipelineParams.stream_decode_stride_frames = 1;
                    }
                    if (!explicit_stream_holdback && pipelineParams.stream_holdback_frames < 0) {
                        pipelineParams.stream_holdback_frames = 0;
                    }
                    if (!explicit_start_buffer) {
                        stream_start_buffer_ms = 0;
                    }
                } else if (!explicit_start_buffer &&
                           chunked_response &&
                           stream_audio_format == StreamAudioFormat::PcmS16LE) {
                    stream_start_buffer_ms = 3000;
                }

                const void* ref_audio_buffer = nullptr;
                size_t ref_audio_size = 0;

                httplib::FormData ref_file;
                if (get_first_form_file(req.form, {"reference", "reference_audio", "prompt_audio", "ref_audio"}, ref_file)) {
                    if (!ref_file.content.empty()) {
                        ref_audio_buffer = ref_file.content.data();
                        ref_audio_size = ref_file.content.size();
                    }
                }

                if (ref_audio_buffer && ref_audio_size > 0 && pipelineParams.prompt_text.empty()) {
                    res.status = 400;
                    res.set_content("Reference audio requires reference_text (aliases: ref_text, prompt_text).", "text/plain");
                    return;
                }

                std::vector<std::string> segmented_texts;
                if (segment_sentences) {
                    segmented_texts = split_text_for_segmented_synthesis(
                        pipelineParams.text, static_cast<size_t>(segment_max_chars));
                    if (segmented_texts.empty()) {
                        res.status = 400;
                        res.set_content("Text did not produce any non-empty segments.", "text/plain");
                        return;
                    }
                }

                bool expected_idle = false;
                if (!server_busy->compare_exchange_strong(expected_idle, true)) {
                    res.status = 503;
                    res.set_content("Server busy processing another synthesis request.", "text/plain");
                    return;
                }

                auto release_busy = [&]() {
                    server_busy->store(false);
                };

                if (!stream_response && !segment_sentences) {
                    void * wav_buffer = nullptr;
                    size_t wav_size = 0;
                    const bool ok = pipeline->synthesize_to_memory(
                        pipelineParams,
                        ref_audio_buffer,
                        ref_audio_size,
                        &wav_buffer,
                        &wav_size);

                    if (!ok) {
                        release_busy();
                        res.status = 500;
                        res.set_content("Synthesis failed.", "text/plain");
                        return;
                    }

                    res.set_header("Content-Disposition", "attachment; filename=\"generated_audio.wav\"");
                    res.set_content(std::string(reinterpret_cast<const char *>(wav_buffer), wav_size),
                                    "audio/wav");
                    s2::audio_free_memory_wav(&wav_buffer, &wav_size, nullptr);
                    release_busy();
                    return;
                }

                s2::AudioData ref_audio;
                if (ref_audio_buffer && ref_audio_size > 0) {
                    if (!s2::load_audio_from_memory(ref_audio_buffer, ref_audio_size,
                                                    ref_audio, pipeline->output_sample_rate())) {
                        S2_LOG_WARN_STREAM("Failed to load reference audio; continuing without it." << std::endl);
                    }
                }

                if (!chunked_response) {
                    BufferedAudioSink sink(stream_audio_format);
                    const bool ok = segment_sentences
                        ? synthesize_segmented_to_sink(*pipeline, pipelineParams, ref_audio,
                                                       segmented_texts, sentence_pause_ms, sink)
                        : pipeline->synthesize_streaming_raw(pipelineParams, ref_audio, sink);
                    release_busy();

                    if (!ok || !sink.ok()) {
                        res.status = 500;
                        res.set_content(
                            sink.error_message().empty() ? "Streaming synthesis failed." : sink.error_message(),
                            "text/plain");
                        return;
                    }

                    const auto & audio_bytes = sink.bytes();
                    res.set_header(
                        "Content-Disposition",
                        std::string("attachment; filename=\"") +
                        stream_audio_filename(stream_audio_format) + "\"");
                    if (stream_audio_format == StreamAudioFormat::PcmS16LE) {
                        res.set_header("X-Audio-Sample-Rate", std::to_string(pipeline->output_sample_rate()));
                        res.set_header("X-Audio-Channels", "1");
                        res.set_header("X-Audio-Encoding", "pcm_s16le");
                    }
                    res.set_content(
                        std::string(reinterpret_cast<const char *>(audio_bytes.data()), audio_bytes.size()),
                        stream_audio_content_type(stream_audio_format, pipeline->output_sample_rate()));
                    return;
                }

                auto ctx = std::make_shared<StreamContext>();
                if (stream_start_buffer_ms > 0) {
                    const uint64_t bytes_per_second =
                        static_cast<uint64_t>(pipeline->output_sample_rate()) * sizeof(int16_t);
                    ctx->startup_buffer_bytes = static_cast<size_t>(
                        (bytes_per_second * static_cast<uint64_t>(stream_start_buffer_ms)) / 1000u);
                }
                auto sink = std::make_shared<HttpStreamSink>(ctx, stream_audio_format);

                res.set_header(
                    "Content-Disposition",
                    std::string("attachment; filename=\"") +
                    stream_audio_filename(stream_audio_format) + "\"");
                if (stream_audio_format == StreamAudioFormat::PcmS16LE) {
                    res.set_header("X-Audio-Sample-Rate", std::to_string(pipeline->output_sample_rate()));
                    res.set_header("X-Audio-Channels", "1");
                    res.set_header("X-Audio-Encoding", "pcm_s16le");
                }

                res.set_chunked_content_provider(
                    stream_audio_content_type(stream_audio_format, pipeline->output_sample_rate()),
                    [ctx](size_t , httplib::DataSink &sink) -> bool {
                        std::unique_lock<std::mutex> lock(ctx->mtx);
                        while (ctx->chunks.empty() && !ctx->done && !ctx->error) {
                            if (!sink.is_writable()) {
                                ctx->cancelled.store(true);
                                return false;
                            }
                            ctx->cv.wait_for(lock, std::chrono::milliseconds(100));
                        }

                        while (!ctx->started &&
                               ctx->startup_buffer_bytes > 0 &&
                               ctx->queued_bytes < ctx->startup_buffer_bytes &&
                               !ctx->done && !ctx->error) {
                            if (!sink.is_writable()) {
                                ctx->cancelled.store(true);
                                return false;
                            }
                            ctx->cv.wait_for(lock, std::chrono::milliseconds(100));
                        }

                        if (!ctx->started) {
                            ctx->started = true;
                        }

                        if (ctx->error && ctx->chunks.empty()) {
                            return false;
                        }

                        while (!ctx->chunks.empty()) {
                            auto &chunk = ctx->chunks.front();
                            lock.unlock();
                            if (!sink.write(reinterpret_cast<const char*>(chunk.data()),
                                           chunk.size())) {
                                ctx->cancelled.store(true);
                                return false;
                            }
                            lock.lock();
                            if (ctx->queued_bytes >= chunk.size()) {
                                ctx->queued_bytes -= chunk.size();
                            } else {
                                ctx->queued_bytes = 0;
                            }
                            ctx->chunks.pop_front();
                        }

                        if (ctx->done) {
                            sink.done();
                            return true;
                        }
                        return true;
                    },
                    [ctx](bool success) {
                        if (!success) {
                            ctx->cancelled.store(true);
                            std::lock_guard<std::mutex> lock(ctx->mtx);
                            ctx->cv.notify_all();
                        }
                    }
                );

                std::thread synth_thread([pipeline, pipelineParams, ref_audio, sink, server_busy,
                                          segment_sentences, segmented_texts, sentence_pause_ms]() mutable {
                    const bool ok = segment_sentences
                        ? synthesize_segmented_to_sink(*pipeline, pipelineParams, ref_audio,
                                                       segmented_texts, sentence_pause_ms, *sink)
                        : pipeline->synthesize_streaming_raw(pipelineParams, ref_audio, *sink);
                    if (!ok) {
                        std::cerr << "Streaming synthesis failed." << std::endl;

                    }
                    server_busy->store(false);
                });
                {
                    std::lock_guard<std::mutex> lock(active_threads_mtx);
                    active_threads.push_back(std::move(synth_thread));
                }
            });

        S2_LOG_INFO_STREAM("Server starting on http://" << params.host << ":" << params.port << "..." << std::endl);

        if (!svr.listen(params.host.c_str(), params.port)) {
            std::cerr << "Server not initialized." << std::endl;
            return 0;
        }

        return 1;
    }

}
