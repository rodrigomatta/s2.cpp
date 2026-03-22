#include "../include/s2_pipeline.h"
#include <iostream>

namespace s2 {

Pipeline::Pipeline() {}
Pipeline::~Pipeline() {}

bool Pipeline::init(const PipelineParams & params) {
    std::cout << "--- Pipeline Init ---" << std::endl;
    if (!tokenizer_.load(params.tokenizer_path)) {
        std::cerr << "Pipeline error: could not load tokenizer from " << params.tokenizer_path << std::endl;
        return false;
    }

    if (!model_.load(params.model_path, params.gpu_device, params.backend_type)) {
        std::cerr << "Pipeline error: could not load model from " << params.model_path << std::endl;
        return false;
    }

    // Codec runs only twice per synthesis (encode ref audio + decode output),
    // not in the hot generation loop — always keep on CPU to save VRAM.
    if (!codec_.load(params.model_path, -1, -1)) {
        std::cerr << "Pipeline error: could not load codec from " << params.model_path << std::endl;
        return false;
    }

    // Sync tokenizer config from model hparams so that semantic token IDs,
    // codebook count, and vocab size are consistent between generation and
    // prompt-building regardless of what the tokenizer.json says.
    {
        const ModelHParams & hp = model_.hparams();
        TokenizerConfig & tc    = tokenizer_.config();
        if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
        if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
        if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
        if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
        if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
    }

    initialized_ = true;
    return true;
}


std::vector<std::string> Pipeline::split_text_into_chunks(const std::string & text, int32_t max_new_tokens, bool split_sentences) {
    std::vector<std::string> chunks;
    if (text.empty()) return chunks;

    // Split into segments at sentence-ending punctuation, preserving newlines
    std::vector<std::string> segments;
    size_t start = 0;
    while (start < text.length()) {
        start = text.find_first_not_of(" \t", start);
        if (start == std::string::npos) break;

        size_t end = std::string::npos;
        size_t pos = start;
        while (pos < text.length()) {
            char c = text[pos];
            if (c == '.' || c == '!' || c == '?') {
                if (pos + 1 == text.length() || std::isspace(static_cast<unsigned char>(text[pos + 1]))) {
                    end = pos + 1;
                    break;
                }
            }
            pos++;
        }

        if (end == std::string::npos) {
            segments.push_back(text.substr(start));
            break;
        } else {
            std::string segment = text.substr(start, end - start);
            // Include trailing newlines after punctuation
            size_t after_punct = end;
            while (after_punct < text.length() && (text[after_punct] == '\n' || text[after_punct] == '\r')) {
                after_punct++;
            }
            if (after_punct > end) {
                segment += text.substr(end, after_punct - end);
            }
            segments.push_back(segment);
            start = after_punct;
        }
    }

    // Group segments into chunks - split at every sentence
    std::string current_chunk;
    int32_t current_tokens = 0;

    for (size_t i = 0; i < segments.size(); ++i) {
        const std::string & seg = segments[i];

        if (!current_chunk.empty()) {
            chunks.push_back(current_chunk);
            current_chunk = seg;
        } else {
            current_chunk = seg;
        }
    }

    if (!current_chunk.empty()) {
        chunks.push_back(current_chunk);
    }

    return chunks;
}

bool Pipeline::synthesize(const PipelineParams & params) {
    if (!initialized_) {
        std::cerr << "Pipeline not initialized." << std::endl;
        return false;
    }

    std::cout << "--- Pipeline Synthesize ---" << std::endl;
    
    std::vector<std::string> chunks = split_text_into_chunks(params.text, params.gen.max_new_tokens, params.split_sentences);
    if (chunks.empty()) {
        std::cerr << "Pipeline warning: no text to synthesize." << std::endl;
        return true; 
    }

    std::cout << "Text split into " << chunks.size() << " chunks." << std::endl;

    const int32_t num_codebooks = model_.hparams().num_codebooks;

    // 1. Audio Prompt Loading
    std::vector<int32_t> ref_codes;
    int32_t T_prompt = 0;
    if (!params.prompt_audio_path.empty()) {
        std::cout << "Loading reference audio: " << params.prompt_audio_path << std::endl;
        AudioData ref_audio;
        if (load_audio(params.prompt_audio_path, ref_audio, codec_.sample_rate())) {
            if (!codec_.encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
                               params.gen.n_threads, ref_codes, T_prompt)) {
                std::cerr << "Pipeline warning: encode failed, running without reference audio." << std::endl;
                ref_codes.clear();
                T_prompt = 0;
            }
        } else {
            std::cerr << "Pipeline warning: load_audio failed, running without reference audio." << std::endl;
        }
    }

    std::vector<float> final_audio_out;

    for (size_t i = 0; i < chunks.size(); ++i) {
        // Show current chunk text
        std::string chunk_preview = chunks[i];
        // Replace newlines with visible symbol for display
        std::string cleaned;
        cleaned.reserve(chunk_preview.length());
        for (size_t j = 0; j < chunk_preview.length(); ++j) {
            char c = chunk_preview[j];
            if (c == '\n') {
                cleaned += " \\n ";
            } else if (c == '\r') {
                // Skip carriage return
            } else {
                cleaned += c;
            }
        }
        chunk_preview = cleaned;
        if (chunk_preview.length() > 120) {
            chunk_preview = chunk_preview.substr(0, 120) + "...";
        }
        std::cout << "\nChunk " << (i + 1) << "/" << chunks.size() << ": " << chunk_preview << std::endl;

        if (chunks.size() > 1) {
            float progress = (float)(i + 1) / (float)chunks.size();
            int bar_width = 30;
            int pos = (int)(bar_width * progress);

            std::cout << "[" ;
            for (int b = 0; b < bar_width; ++b) {
                if (b < pos) std::cout << "=";
                else if (b == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << "% | Chunk " << (i + 1) << "/" << chunks.size() << std::flush;
        }

        // 2. Build Prompt Tensor
        PromptTensor prompt = build_prompt(
            tokenizer_, chunks[i], params.prompt_text,
            ref_codes.empty() ? nullptr : ref_codes.data(),
            num_codebooks, T_prompt);

        // 3. Setup KV Cache
        // Reset model state/cache for each chunk
        model_.reset();

        int32_t max_seq_len = prompt.cols + params.gen.max_new_tokens;
        if (!model_.init_kv_cache(max_seq_len)) {
            std::cerr << "Pipeline error: init_kv_cache failed for chunk " << (i + 1) << std::endl;
            return false;
        }

        // 4. Generate
        GenerateResult res = generate(model_, tokenizer_.config(), prompt, params.gen);
        if (res.n_frames == 0) {
            std::cerr << "Pipeline error: generation produced no frames for chunk " << (i + 1) << std::endl;
            continue; 
        }

        // 5. Decode
        std::vector<float> chunk_audio;
        if (!codec_.decode(res.codes.data(), res.n_frames, params.gen.n_threads, chunk_audio)) {
            std::cerr << "Pipeline error: decode failed for chunk " << (i + 1) << std::endl;
            return false;
        }

        // Append to final audio
        final_audio_out.insert(final_audio_out.end(), chunk_audio.begin(), chunk_audio.end());

        // Periodic checkpoint save every 20 chunks to prevent data loss
        if ((i + 1) % 20 == 0 || (i + 1) == chunks.size()) {
            std::string checkpoint_path = params.output_path + ".tmp";
            save_audio(checkpoint_path, final_audio_out, codec_.sample_rate());
        }
    }

    if (final_audio_out.empty()) {
        std::cerr << "Pipeline error: no audio generated." << std::endl;
        return false;
    }

    // 6. Save
    if (!save_audio(params.output_path, final_audio_out, codec_.sample_rate())) {
        std::cerr << "Pipeline error: could not save to " << params.output_path << std::endl;
        std::cerr << "This often happens if the file is open in another program (VLC, Media Player, etc.)." << std::endl;
        
        // Fallback: try saving with a backup name
        std::string fallback_path = "backup_" + params.output_path;
        std::cerr << "Attempting fallback save to: " << fallback_path << std::endl;
        
        if (save_audio(fallback_path, final_audio_out, codec_.sample_rate())) {
            std::cout << "Successfully saved to fallback: " << fallback_path << std::endl;
            return true;
        } else {
            std::cerr << "Fallback save also failed. Please check disk space and file permissions." << std::endl;
            return false;
        }
    }
    // Cleanup checkpoint if final save succeeded
    std::remove((params.output_path + ".tmp").c_str());

    std::cout << std::endl;
    std::cout << "Saved long-form audio to: " << params.output_path << std::endl;
    return true;
}

} // namespace s2
