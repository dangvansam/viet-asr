#include "modules/vad/energy_backend.h"

#include <cmath>

namespace vietasr {
namespace vad {

Status EnergyBackend::Init(const VadConfig& cfg) {
    cfg_ = cfg;
    return Status::Ok();
}

float EnergyBackend::Process(const float* chunk, std::size_t n) {
    if (chunk == nullptr || n == 0) return 0.0f;

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double s = static_cast<double>(chunk[i]);
        sum_sq += s * s;
    }
    const double mean_sq = sum_sq / static_cast<double>(n);
    if (mean_sq <= 0.0) return 0.0f;

    // Map mean-square energy to a probability on a log scale. The ramp is
    // centered so that prob == 0.5 at mean_sq == 1e-6 — the voiced/silence
    // boundary used by the module's original energy heuristic.
    double prob = (std::log10(mean_sq) + 8.0) / 4.0;
    if (prob < 0.0) prob = 0.0;
    if (prob > 1.0) prob = 1.0;
    return static_cast<float>(prob);
}

}  // namespace vad
}  // namespace vietasr
