#include "../include/s2_sampler.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>

namespace s2 {

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

int32_t sample_token(const float * logits, int32_t vocab_size, const SamplerParams & params) {
    if (vocab_size <= 0) return 0;

    const int32_t k     = params.top_k > 0 ? std::min(params.top_k, vocab_size) : vocab_size;
    const float   top_p = std::clamp(params.top_p, 0.0f, 1.0f);
    const float   NEG_INF = -std::numeric_limits<float>::infinity();

    // Single pass: gather finite candidates and the global max. Masked logits
    // (set to -inf by the caller, e.g. the semantic mask covering ~95% of the
    // vocab) contribute exp(-inf)=0 to the softmax and can never be selected,
    // so skipping them is exact and avoids building/sorting the full vocab.
    std::vector<std::pair<float, int32_t>> items;
    items.reserve(static_cast<size_t>(std::min(vocab_size, 8192)));
    float max_val = NEG_INF;
    for (int32_t i = 0; i < vocab_size; ++i) {
        const float v = logits[i];
        if (v == NEG_INF) continue;
        items.push_back({v, i});
        if (v > max_val) max_val = v;
    }
    if (items.empty()) return 0;

    // Full partition function over the finite candidates. This matches the
    // original code's denominator (the -inf terms it summed were exactly 0),
    // so the top_p cumulative mass below is computed identically.
    float Z = 0.0f;
    for (const auto & it : items) Z += std::exp(it.first - max_val);
    if (!(Z > 0.0f)) Z = 1.0f;

    // Order by logit descending, breaking ties by index for determinism.
    auto cmp = [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    };
    const int32_t kk = std::min<int32_t>(k, static_cast<int32_t>(items.size()));
    if (kk < static_cast<int32_t>(items.size())) {
        // top_k is small (default 30); only the top kk elements are ever kept,
        // so a partial sort (O(n log kk)) replaces a full sort (O(n log n)).
        std::partial_sort(items.begin(), items.begin() + kk, items.end(), cmp);
        items.resize(kk);
    } else {
        std::sort(items.begin(), items.end(), cmp);
    }

    // Keep the prefix up to top_k and the top_p cumulative-probability cutoff.
    std::vector<std::pair<float, int32_t>> filtered;
    filtered.reserve(items.size());
    float cumsum = 0.0f;
    for (int32_t i = 0; i < (int32_t)items.size(); ++i) {
        cumsum += std::exp(items[i].first - max_val) / Z;
        if (i > 0 && cumsum > top_p) {
            break; // cumsum is monotonic, so every later item also exceeds top_p
        }
        filtered.push_back(items[i]);
    }

    if (filtered.empty()) {
        filtered.push_back(items.front());
    }

    if (params.temperature <= 0.0f) {
        return filtered[0].second;
    }

    std::vector<float> probs(filtered.size());
    for (size_t i = 0; i < filtered.size(); ++i) {
        probs[i] = filtered[i].first;
    }
    apply_softmax(probs, params.temperature);

    float sum_p = 0.0f;
    for (float p : probs) sum_p += p;
    if (sum_p <= 0.0f) {
        return filtered[0].second;
    }
    for (float & p : probs) p /= sum_p;

    thread_local static std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());

    const int32_t sampled_idx = dist(gen);
    return filtered[sampled_idx].second;
}

}
