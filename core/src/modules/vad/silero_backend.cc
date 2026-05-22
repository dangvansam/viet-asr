#include "modules/vad/silero_backend.h"

#include <algorithm>

#include "modules/vad/embedded_vad.h"
#include "vietasr.h"  // VIETASR_EP_CPU
#include "vietasr/logger.h"

namespace vietasr {
namespace vad {

namespace {
// Silero VAD v5 fixed dimensions at 16 kHz.
constexpr std::size_t kChunk = 512;          // new samples consumed per call
constexpr std::size_t kContext = 64;         // samples carried from the prior
                                             // window and prepended to input
constexpr std::size_t kWindow = kContext + kChunk;  // model input length, 576
constexpr std::size_t kStateSize = 2 * 1 * 128;     // recurrent state tensor
}  // namespace

Status SileroBackend::Init(const VadConfig& cfg) {
    (void)cfg;

    const EmbeddedBlob model = EmbeddedSileroModel();
    if (model.empty()) {
        return Status::Error(-6,
            "silero: model not embedded (build with VIETASR_EMBED_SILERO=ON)");
    }

    // A dedicated engine on CPU — the model is tiny and the pipeline's shared
    // engine is occupied by the ASR model.
    EpPreference ep;
    ep.ep_enum = VIETASR_EP_CPU;
    engine_ = EngineFactory::CreateAuto(ep);
    if (!engine_) {
        return Status::Error(-6, "silero: ONNX engine unavailable");
    }

    Status loaded = engine_->LoadModelFromBuffer(model.data, model.size);
    if (!loaded.ok()) {
        return Status::Error(-6, "silero: " + loaded.message());
    }

    state_.assign(kStateSize, 0.0f);
    context_.assign(kContext, 0.0f);
    ready_ = true;
    VIETASR_LOG_INFO("vad")
        << "silero backend ready (" << model.size << " bytes)";
    return Status::Ok();
}

float SileroBackend::Process(const float* chunk, std::size_t n) {
    if (!ready_ || !engine_ || chunk == nullptr || n != kChunk) {
        return 0.0f;
    }

    // Silero v5 expects a 576-sample input: 64 samples of context from the
    // previous window prepended to the 512 new samples.
    std::vector<float> window;
    window.reserve(kWindow);
    window.insert(window.end(), context_.begin(), context_.end());
    window.insert(window.end(), chunk, chunk + n);

    std::vector<Tensor> inputs(3);
    // input: float32 [1, 576]
    inputs[0].shape = {1, static_cast<std::int64_t>(kWindow)};
    inputs[0].data = window;
    // state: float32 [2, 1, 128]
    inputs[1].shape = {2, 1, 128};
    inputs[1].data = state_;
    // sr: int64 scalar — the engine converts the float value to int64.
    inputs[2].shape = {};
    inputs[2].data = {16000.0f};

    std::vector<Tensor> outputs;
    Status s = engine_->Run(inputs, &outputs);
    if (!s.ok() || outputs.size() < 2 || outputs[0].data.empty()) {
        return 0.0f;
    }

    // outputs: [0] = speech probability, [1] = next recurrent state.
    if (outputs[1].data.size() == state_.size()) {
        state_ = outputs[1].data;
    }
    // Carry the last 64 samples of this window into the next call.
    context_.assign(window.end() - kContext, window.end());

    float prob = outputs[0].data[0];
    if (prob < 0.0f) prob = 0.0f;
    if (prob > 1.0f) prob = 1.0f;
    return prob;
}

void SileroBackend::Reset() {
    std::fill(state_.begin(), state_.end(), 0.0f);
    std::fill(context_.begin(), context_.end(), 0.0f);
}

}  // namespace vad
}  // namespace vietasr
