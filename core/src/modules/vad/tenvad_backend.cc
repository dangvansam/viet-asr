#include "modules/vad/tenvad_backend.h"

#include <cmath>

#include "vietasr/logger.h"

extern "C" {
#include "ten_vad.h"
}

namespace vietasr {
namespace vad {

void TenVadBackend::Destroy() {
    if (handle_) {
        ten_vad_destroy(&handle_);  // sets handle_ to NULL
        handle_ = nullptr;
    }
}

TenVadBackend::~TenVadBackend() { Destroy(); }

Status TenVadBackend::Init(const VadConfig& cfg) {
    // TEN-VAD supports two analysis hops at 16 kHz: 256 (16 ms) or 160 (10 ms).
    hop_size_ = (cfg.hop_size == 160) ? 160u : 256u;
    threshold_ = cfg.threshold;
    scratch_.assign(hop_size_, 0);

    if (ten_vad_create(&handle_, hop_size_, threshold_) != 0
            || handle_ == nullptr) {
        handle_ = nullptr;
        return Status::Error(-6, "tenvad: ten_vad_create failed");
    }
    VIETASR_LOG_INFO("vad")
        << "tenvad backend ready (version " << ten_vad_get_version()
        << ", hop=" << hop_size_ << ")";
    return Status::Ok();
}

float TenVadBackend::Process(const float* chunk, std::size_t n) {
    if (!handle_ || chunk == nullptr || n != hop_size_) return 0.0f;

    // TEN-VAD consumes int16 PCM; vietasr audio is normalized float [-1, 1].
    for (std::size_t i = 0; i < n; ++i) {
        float v = chunk[i] * 32768.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        scratch_[i] = static_cast<std::int16_t>(std::lrintf(v));
    }

    float prob = 0.0f;
    int flag = 0;
    if (ten_vad_process(handle_, scratch_.data(), n, &prob, &flag) != 0) {
        return 0.0f;
    }
    return prob;
}

void TenVadBackend::Reset() {
    // TEN-VAD exposes no reset; recreate the handle for clean per-stream state.
    Destroy();
    if (ten_vad_create(&handle_, hop_size_, threshold_) != 0) {
        handle_ = nullptr;
        VIETASR_LOG_WARN("vad") << "tenvad: handle re-create failed on reset";
    }
}

}  // namespace vad
}  // namespace vietasr
