#ifndef VIETASR_MODULES_VAD_SILERO_BACKEND_H
#define VIETASR_MODULES_VAD_SILERO_BACKEND_H

#include <memory>
#include <vector>

#include "modules/vad/vad_backend.h"
#include "vietasr/engine.h"

namespace vietasr {
namespace vad {

// VAD backend running Silero VAD v5 (silero_vad.onnx, MIT-licensed). It owns
// its own ONNX engine — the pipeline's shared engine already holds the ASR
// model — and carries the model's recurrent state across chunks.
class SileroBackend final : public VadBackend {
public:
    const char* name() const override { return "silero"; }
    Status Init(const VadConfig& cfg) override;
    std::size_t chunk_samples() const override { return 512; }  // 32 ms @ 16 kHz
    float Process(const float* chunk, std::size_t n) override;
    void Reset() override;

private:
    std::unique_ptr<Engine> engine_;
    std::vector<float> state_;    // Silero recurrent state, shape [2, 1, 128]
    std::vector<float> context_;  // last 64 samples of the previous window
    bool ready_ = false;
};

}  // namespace vad
}  // namespace vietasr

#endif  // VIETASR_MODULES_VAD_SILERO_BACKEND_H
