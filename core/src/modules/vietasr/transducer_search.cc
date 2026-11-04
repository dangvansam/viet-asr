#include "modules/vietasr/transducer_search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace vietasr {

namespace {

struct Candidate final {
    double score;
    int hyp;
    int token;
};

double LogAdd(double a, double b) {
    if (a < b) std::swap(a, b);
    if (b == -std::numeric_limits<double>::infinity()) return a;
    return a + std::log1p(std::exp(b - a));
}

}

TransducerBeamSearch::TransducerBeamSearch(const TransducerSearchConfig& config)
    : config_(config) {
    if (config_.max_active_paths < 1) config_.max_active_paths = 1;
    if (config_.context_size < 1) config_.context_size = 1;
}

void TransducerBeamSearch::Begin() {
    hyps_.clear();
    decoder_cache_.clear();
    TransducerHypothesis start;
    start.ys.assign(static_cast<std::size_t>(config_.context_size - 1), -1);
    start.ys.push_back(config_.blank_id);
    start.log_prob = 0.0;
    hyps_.push_back(std::move(start));
}

std::vector<std::int64_t> TransducerBeamSearch::ContextOf(const TransducerHypothesis& hyp) const {
    return std::vector<std::int64_t>(hyp.ys.end() - config_.context_size, hyp.ys.end());
}

std::vector<std::int64_t> TransducerBeamSearch::PendingContexts() const {
    std::vector<std::int64_t> flat;
    std::vector<std::vector<std::int64_t>> seen;
    for (const auto& hyp : hyps_) {
        std::vector<std::int64_t> ctx = ContextOf(hyp);
        if (decoder_cache_.count(ctx) != 0) continue;
        if (std::find(seen.begin(), seen.end(), ctx) != seen.end()) continue;
        seen.push_back(ctx);
        flat.insert(flat.end(), ctx.begin(), ctx.end());
    }
    return flat;
}

void TransducerBeamSearch::AcceptDecoderOutputs(const std::vector<std::int64_t>& contexts,
                                                const std::vector<float>& outputs) {
    std::size_t batch = contexts.size() / static_cast<std::size_t>(config_.context_size);
    if (batch == 0) return;
    std::size_t dim = outputs.size() / batch;
    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<std::int64_t> key(contexts.begin() + b * config_.context_size,
                                      contexts.begin() + (b + 1) * config_.context_size);
        decoder_cache_[key].assign(outputs.begin() + b * dim, outputs.begin() + (b + 1) * dim);
    }
}

int TransducerBeamSearch::BuildJoinerInputs(const float* encoder_frame, int encoder_dim,
                                            std::vector<float>* encoder_rows,
                                            std::vector<float>* decoder_rows) const {
    encoder_rows->clear();
    decoder_rows->clear();
    for (const auto& hyp : hyps_) {
        encoder_rows->insert(encoder_rows->end(), encoder_frame, encoder_frame + encoder_dim);
        auto it = decoder_cache_.find(ContextOf(hyp));
        if (it == decoder_cache_.end()) return 0;
        decoder_rows->insert(decoder_rows->end(), it->second.begin(), it->second.end());
    }
    return static_cast<int>(hyps_.size());
}

void TransducerBeamSearch::AcceptLogits(const std::vector<float>& logits, int batch) {
    if (batch <= 0 || batch != static_cast<int>(hyps_.size())) return;
    int vocab = static_cast<int>(logits.size()) / batch;
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(batch) * static_cast<std::size_t>(vocab));
    for (int h = 0; h < batch; ++h) {
        const float* row = logits.data() + static_cast<std::size_t>(h) * vocab;
        float max_logit = *std::max_element(row, row + vocab);
        double sum = 0.0;
        for (int v = 0; v < vocab; ++v) sum += std::exp(static_cast<double>(row[v] - max_logit));
        double log_norm = max_logit + std::log(sum);
        for (int v = 0; v < vocab; ++v) {
            candidates.push_back({hyps_[h].log_prob + row[v] - log_norm, h, v});
        }
    }
    std::size_t k = std::min<std::size_t>(static_cast<std::size_t>(config_.max_active_paths),
                                          candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                      [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    std::vector<TransducerHypothesis> next;
    for (std::size_t i = 0; i < k; ++i) {
        const Candidate& c = candidates[i];
        TransducerHypothesis hyp = hyps_[c.hyp];
        if (c.token != config_.blank_id && c.token != config_.unk_id) {
            hyp.ys.push_back(c.token);
        }
        hyp.log_prob = c.score;
        auto same = std::find_if(next.begin(), next.end(),
            [&](const TransducerHypothesis& o) { return o.ys == hyp.ys; });
        if (same == next.end()) {
            next.push_back(std::move(hyp));
        } else {
            same->log_prob = LogAdd(same->log_prob, hyp.log_prob);
        }
    }
    hyps_ = std::move(next);
}

std::vector<int> TransducerBeamSearch::BestTokens() const {
    std::vector<int> tokens;
    if (hyps_.empty()) return tokens;
    auto score = [&](const TransducerHypothesis& h) {
        return h.log_prob / static_cast<double>(h.ys.size());
    };
    const TransducerHypothesis* best = &hyps_.front();
    for (const auto& hyp : hyps_) {
        if (score(hyp) > score(*best)) best = &hyp;
    }
    for (std::size_t i = static_cast<std::size_t>(config_.context_size); i < best->ys.size(); ++i) {
        tokens.push_back(static_cast<int>(best->ys[i]));
    }
    return tokens;
}

}
