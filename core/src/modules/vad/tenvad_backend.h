#ifndef VIETASR_MODULES_VAD_TENVAD_BACKEND_H
#define VIETASR_MODULES_VAD_TENVAD_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "modules/vad/vad_backend.h"

namespace vietasr {
namespace vad {

// VAD backend wrapping TEN-VAD (https://github.com/TEN-framework/ten-vad),
// compiled from the vendored source under core/third_party/ten-vad/. The
// neural model is embedded into libvietasr; see embedded_vad.h.
//
// Powered by ten-vad. TEN-VAD is licensed under Apache-2.0 with additional
// conditions — see core/third_party/ten-vad/LICENSE.
class TenVadBackend final : public VadBackend {
public:
    TenVadBackend() = default;
    ~TenVadBackend() override;

    TenVadBackend(const TenVadBackend&) = delete;
    TenVadBackend& operator=(const TenVadBackend&) = delete;

    const char* name() const override { return "tenvad"; }
    Status Init(const VadConfig& cfg) override;
    std::size_t chunk_samples() const override { return hop_size_; }
    float Process(const float* chunk, std::size_t n) override;
    void Reset() override;

private:
    void Destroy();

    void* handle_ = nullptr;  // ten_vad_handle_t (opaque)
    std::size_t hop_size_ = 256;
    float threshold_ = 0.5f;
    std::vector<std::int16_t> scratch_;
};

}  // namespace vad
}  // namespace vietasr

#endif  // VIETASR_MODULES_VAD_TENVAD_BACKEND_H
