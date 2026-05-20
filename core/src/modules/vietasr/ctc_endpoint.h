#ifndef VIETASR_MODULES_VIETASR_CTC_ENDPOINT_H
#define VIETASR_MODULES_VIETASR_CTC_ENDPOINT_H

namespace vietasr {

struct CtcEndpointConfig final {
    int rule1_ms{5000};
    int rule2_ms{1000};
    int rule3_ms{20000};
    float blank_threshold{0.8f};
};

class CtcEndpoint final {
public:
    explicit CtcEndpoint(const CtcEndpointConfig& config = {});
    void Reset();
    bool Check(int silence_ms, int utterance_ms, bool has_decoded);

private:
    CtcEndpointConfig config_;
};

}

#endif
