#include "modules/vietasr/ctc_endpoint.h"

namespace vietasr {

CtcEndpoint::CtcEndpoint(const CtcEndpointConfig& config) : config_(config) {}

void CtcEndpoint::Reset() {}

bool CtcEndpoint::Check(int silence_ms, int utterance_ms, bool has_decoded) {
    if (utterance_ms >= config_.rule3_ms) return true;
    if (has_decoded && silence_ms >= config_.rule2_ms) return true;
    if (silence_ms >= config_.rule1_ms) return true;
    return false;
}

}
