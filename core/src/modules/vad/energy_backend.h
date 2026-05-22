#ifndef VIETASR_MODULES_VAD_ENERGY_BACKEND_H
#define VIETASR_MODULES_VAD_ENERGY_BACKEND_H

#include "modules/vad/vad_backend.h"

namespace vietasr {
namespace vad {

// A dependency-free RMS-energy detector. It carries no model, so it works on
// every build (including ones without ONNX Runtime) and is the fallback when
// the configured backend (tenvad by default) cannot initialize.
class EnergyBackend final : public VadBackend {
public:
    const char* name() const override { return "energy"; }
    Status Init(const VadConfig& cfg) override;
    std::size_t chunk_samples() const override { return 512; }  // 32 ms @ 16 kHz
    float Process(const float* chunk, std::size_t n) override;
    void Reset() override {}

private:
    VadConfig cfg_;
};

}  // namespace vad
}  // namespace vietasr

#endif  // VIETASR_MODULES_VAD_ENERGY_BACKEND_H
