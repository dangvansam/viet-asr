#ifndef VIETASR_MODULES_VIETASR_CTC_BEAM_SEARCH_H
#define VIETASR_MODULES_VIETASR_CTC_BEAM_SEARCH_H

#include <map>
#include <vector>

namespace vietasr {

struct CtcBeamSearchConfig final {
    int beam_size{10};
    int blank_id{0};
};

class CtcBeamSearch final {
public:
    explicit CtcBeamSearch(const CtcBeamSearchConfig& config = {});
    void Reset();
    void Step(const float* logits, int vocab_size);
    std::vector<int> Hypothesis() const;

private:
    using Prefix = std::vector<int>;
    struct Score {
        double pb;
        double pn;
    };

    CtcBeamSearchConfig config_;
    std::map<Prefix, Score> beams_;

    static double LogSumExp(double a, double b);
    static double LogSoftmaxDenominator(const float* logits, int vocab_size);
};

}

#endif
