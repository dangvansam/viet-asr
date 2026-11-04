#ifndef VIETASR_MODULES_VIETASR_TRANSDUCER_SEARCH_H
#define VIETASR_MODULES_VIETASR_TRANSDUCER_SEARCH_H

#include <cstdint>
#include <map>
#include <vector>

#include "vietasr/types.h"

namespace vietasr {

struct TransducerSearchConfig final {
    int max_active_paths{4};
    int context_size{2};
    int blank_id{0};
    int unk_id{2};
};

struct TransducerHypothesis final {
    std::vector<std::int64_t> ys;
    double log_prob{0.0};
};

class TransducerBeamSearch final {
public:
    explicit TransducerBeamSearch(const TransducerSearchConfig& config);

    void Begin();

    std::vector<std::int64_t> PendingContexts() const;
    void AcceptDecoderOutputs(const std::vector<std::int64_t>& contexts,
                              const std::vector<float>& outputs);

    int BuildJoinerInputs(const float* encoder_frame, int encoder_dim,
                          std::vector<float>* encoder_rows,
                          std::vector<float>* decoder_rows) const;
    void AcceptLogits(const std::vector<float>& logits, int batch);

    std::vector<int> BestTokens() const;
    int num_hypotheses() const { return static_cast<int>(hyps_.size()); }

private:
    std::vector<std::int64_t> ContextOf(const TransducerHypothesis& hyp) const;

    TransducerSearchConfig config_;
    std::vector<TransducerHypothesis> hyps_;
    std::map<std::vector<std::int64_t>, std::vector<float>> decoder_cache_;
};

}

#endif
