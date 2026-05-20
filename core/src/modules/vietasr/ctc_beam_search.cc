#include "modules/vietasr/ctc_beam_search.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vietasr {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

}

CtcBeamSearch::CtcBeamSearch(const CtcBeamSearchConfig& config)
    : config_(config) {
    Reset();
}

void CtcBeamSearch::Reset() {
    beams_.clear();
    beams_[Prefix{}] = {0.0, kNegInf};
}

double CtcBeamSearch::LogSumExp(double a, double b) {
    if (a == kNegInf) return b;
    if (b == kNegInf) return a;
    double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

double CtcBeamSearch::LogSoftmaxDenominator(const float* logits, int vocab_size) {
    float m = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > m) m = logits[i];
    }
    double s = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        s += std::exp(static_cast<double>(logits[i]) - m);
    }
    return std::log(s) + m;
}

void CtcBeamSearch::Step(const float* logits, int vocab_size) {
    int k = std::min(2 * config_.beam_size, vocab_size);

    std::vector<int> indices(vocab_size);
    for (int i = 0; i < vocab_size; ++i) indices[i] = i;
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [logits](int a, int b) { return logits[a] > logits[b]; });

    double denom = LogSoftmaxDenominator(logits, vocab_size);
    std::vector<double> log_probs(k);
    std::vector<int> top_ids(k);
    for (int i = 0; i < k; ++i) {
        top_ids[i] = indices[i];
        log_probs[i] = static_cast<double>(logits[indices[i]]) - denom;
    }

    std::map<Prefix, Score> next_beams;

    for (const auto& [prefix, score] : beams_) {
        const double pb = score.pb;
        const double pn = score.pn;
        for (int i = 0; i < k; ++i) {
            int tid = top_ids[i];
            double lp = log_probs[i];

            if (tid == config_.blank_id) {
                Score& slot = next_beams.try_emplace(prefix, Score{kNegInf, kNegInf}).first->second;
                slot.pb = LogSumExp(slot.pb, LogSumExp(pb, pn) + lp);
            } else {
                int last = prefix.empty() ? -1 : prefix.back();
                if (tid == last) {
                    Prefix extended = prefix;
                    extended.push_back(tid);
                    Score& s_ext = next_beams.try_emplace(extended, Score{kNegInf, kNegInf}).first->second;
                    s_ext.pn = LogSumExp(s_ext.pn, pb + lp);
                    Score& s_keep = next_beams.try_emplace(prefix, Score{kNegInf, kNegInf}).first->second;
                    s_keep.pn = LogSumExp(s_keep.pn, pn + lp);
                } else {
                    Prefix extended = prefix;
                    extended.push_back(tid);
                    Score& s_ext = next_beams.try_emplace(extended, Score{kNegInf, kNegInf}).first->second;
                    s_ext.pn = LogSumExp(s_ext.pn, LogSumExp(pb, pn) + lp);
                }
            }
        }
    }

    std::vector<std::pair<Prefix, Score>> scored(next_beams.begin(), next_beams.end());
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) {
                  double sa = LogSumExp(a.second.pb, a.second.pn);
                  double sb = LogSumExp(b.second.pb, b.second.pn);
                  return sa > sb;
              });
    if (static_cast<int>(scored.size()) > config_.beam_size) {
        scored.resize(config_.beam_size);
    }

    beams_.clear();
    for (auto& entry : scored) {
        beams_[entry.first] = entry.second;
    }
}

std::vector<int> CtcBeamSearch::Hypothesis() const {
    const Prefix* best = nullptr;
    double best_score = kNegInf;
    for (const auto& [prefix, score] : beams_) {
        double s = LogSumExp(score.pb, score.pn);
        if (s > best_score) {
            best_score = s;
            best = &prefix;
        }
    }
    return best ? *best : Prefix{};
}

}
