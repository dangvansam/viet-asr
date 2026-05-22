#ifndef VIETASR_MODULES_VAD_VAD_BACKEND_H
#define VIETASR_MODULES_VAD_VAD_BACKEND_H

#include <cstddef>
#include <memory>
#include <string>

#include "vietasr/types.h"

namespace vietasr {
namespace vad {

// Resolved configuration for the vad module, parsed from the module's JSON
// config. Every field has a default, so an empty config ("{}") yields the
// tenvad backend with sensible thresholds. If tenvad is not compiled in (or
// fails to initialize), MakeVadBackend falls back to the energy backend.
struct VadConfig final {
    // Backend selector: "energy" | "silero" | "tenvad".
    std::string backend = "tenvad";

    // Speech-probability cutoff in [0, 1]. A chunk counts as voiced when its
    // backend probability is >= threshold.
    float threshold = 0.5f;

    // TEN-VAD analysis hop in samples (16 kHz): 256 (16 ms) or 160 (10 ms).
    int hop_size = 256;

    // Silence tolerated inside a voiced region before it is closed (ms).
    int hangover_ms = 200;

    // Voiced regions shorter than this are dropped (ms).
    int min_speech_ms = 100;

    // Two regions closer than this gap are merged (ms).
    int min_silence_ms = 100;

    // Padding added to each side of every emitted region (ms).
    int speech_pad_ms = 30;

    // Audio sample rate the module runs at (the pipeline feeds 16 kHz).
    float sample_rate = 16000.0f;
};

// A voice-activity detector for a single streaming session. Backends are
// stateful and NOT thread-safe — VadModule::Clone() builds a fresh backend per
// Session (see vad.cc).
class VadBackend {
public:
    virtual ~VadBackend() = default;

    // Short backend name for logging ("energy" / "silero" / "tenvad").
    virtual const char* name() const = 0;

    // One-time setup. Returns a non-ok Status on hard failure; MakeVadBackend
    // falls back to the energy backend when a requested backend fails here.
    virtual Status Init(const VadConfig& cfg) = 0;

    // Number of 16 kHz samples the backend consumes per Process() call.
    virtual std::size_t chunk_samples() const = 0;

    // Process exactly chunk_samples() samples of normalized float PCM
    // ([-1, 1], 16 kHz). Returns the speech probability in [0, 1].
    virtual float Process(const float* chunk, std::size_t n) = 0;

    // Drop all per-stream state (called between sessions / on Reset()).
    virtual void Reset() = 0;
};

// Build the backend named by cfg.backend, fully initialized. On any failure
// (unknown name, missing model, ONNX unavailable) it logs a warning and
// returns an initialized energy backend instead — so the vad module always
// produces a working detector.
std::unique_ptr<VadBackend> MakeVadBackend(const VadConfig& cfg);

}  // namespace vad
}  // namespace vietasr

#endif  // VIETASR_MODULES_VAD_VAD_BACKEND_H
