#include "../include/s2_sampler.h"
#include <cmath>
#include <algorithm>
#include <random>

namespace s2 {

// Helper: Softmax with optional temperature
static void apply_softmax(std::vector<float> & probs, float temp = 1.0f) {
    if (probs.empty()) return;
    float max_val = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (float & p : probs) {
        if (temp > 0.0f) {
            p = std::exp((p - max_val) / temp);
        } else {
            p = (p == max_val) ? 1.0f : 0.0f;
        }
        sum += p;
    }
    if (sum > 0) {
        for (float & p : probs) p /= sum;
    }
}

// Sample a single token from logits
int32_t sample_token(const float * logits, int32_t vocab_size, const SamplerParams & params,
                     std::mt19937 & gen, int32_t always_include_id) {
    if (vocab_size <= 0) return 0;

    std::vector<std::pair<float, int32_t>> items;
    items.reserve(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) {
        items.push_back({logits[i], i});
    }

    // Sort descending by logit
    std::sort(items.begin(), items.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });

    // Top-K
    int32_t k = params.top_k > 0 ? std::min(params.top_k, vocab_size) : vocab_size;
    items.resize(k);

    // Force-include always_include_id after top-k if it was excluded.
    // This ensures EOS can always be sampled regardless of GPU precision differences.
    if (always_include_id >= 0 && always_include_id < vocab_size &&
        logits[always_include_id] > -1e30f)
    {
        bool found = false;
        for (const auto & it : items) {
            if (it.second == always_include_id) { found = true; break; }
        }
        if (!found) {
            items.push_back({logits[always_include_id], always_include_id});
        }
    }

    // Re-sort so temperature/softmax and top-p see items in descending logit order
    std::sort(items.begin(), items.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });

    int32_t n = (int32_t)items.size();

    // Apply temperature and softmax to get probabilities
    std::vector<float> probs(n);
    for (int32_t i = 0; i < n; ++i) probs[i] = items[i].first;
    apply_softmax(probs, params.temperature);

    // Top-P — items/probs are already sorted descending.
    // Track always_include_id position so we can force it in even if top-p cuts it.
    int32_t always_pos = -1;
    if (always_include_id >= 0) {
        for (int32_t i = 0; i < n; ++i) {
            if (items[i].second == always_include_id) { always_pos = i; break; }
        }
    }

    float cumsum = 0.0f;
    int32_t p_idx = 0;
    while (p_idx < n) {
        cumsum += probs[p_idx];
        p_idx++;
        if (cumsum >= params.top_p) break;
    }
    if (p_idx == 0) p_idx = 1;

    // Force always_include_id past top-p cutoff if needed
    if (always_pos >= p_idx) p_idx = always_pos + 1;

    items.resize(p_idx);
    probs.resize(p_idx);

    // Re-normalize after top-p truncation
    float sum_p = 0.0f;
    for (int32_t i = 0; i < p_idx; ++i) sum_p += probs[i];
    if (sum_p > 0) {
        for (int32_t i = 0; i < p_idx; ++i) probs[i] /= sum_p;
    }

    // Sample using std::discrete_distribution
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());

    int32_t sampled_idx = dist(gen);
    return items[sampled_idx].second;
}


RASSampler::RASSampler(int32_t window_size, float high_temp, float high_top_p)
    : window_size_(window_size), high_temp_(high_temp), high_top_p_(high_top_p) 
{
}

int32_t RASSampler::sample(const float * logits, int32_t vocab_size,
               const SamplerParams & params,
               std::mt19937 & gen,
               int32_t sem_begin, int32_t sem_end) {
    
    int32_t token = sample_token(logits, vocab_size, params, gen);
    
    // Check if token is within the repeating window
    if (!window_.empty() && token >= sem_begin && token < sem_end) {
        if (std::find(window_.begin(), window_.end(), token) != window_.end()) {
            // It is a repetition. Resample with higher temperature/top_p.
            SamplerParams high_params = params;
            high_params.temperature = high_temp_;
            high_params.top_p = high_top_p_;
            token = sample_token(logits, vocab_size, high_params, gen);
        }
    }
    
    if (token >= sem_begin && token < sem_end) {
        window_.push_back(token);
        if ((int32_t)window_.size() > window_size_) {
            window_.erase(window_.begin());
        }
    } else {
        window_.clear();
    }
    
    return token;
}

void RASSampler::reset() {
    window_.clear();
}

} // namespace s2
