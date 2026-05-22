// vad — Voice Activity Detection module.
//
// A thin dispatcher over pluggable VAD backends. The backend is chosen by the
// "backend" field of the module's JSON config:
//
//   {}                          -> tenvad  (default; energy fallback if absent)
//   {"backend": "energy"}       -> RMS-energy detector, no model
//   {"backend": "silero"}       -> Silero VAD v5 (silero_vad.onnx)
//   {"backend": "tenvad"}       -> TEN-VAD (compiled from vendored source)
//
// All backends emit the same "voiced_regions" result field. The module owns
// config parsing, fixed-size chunk buffering, and the shared segmenter that
// turns per-chunk probabilities into merged regions.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "modules/vad/energy_backend.h"
#include "modules/vad/vad_backend.h"
#include "modules/vad/vad_segmenter.h"
#include "utils/json.h"
#include "vietasr/logger.h"
#include "vietasr/module.h"

#if defined(VIETASR_VAD_SILERO)
#include "modules/vad/silero_backend.h"
#endif
#if defined(VIETASR_VAD_TENVAD)
#include "modules/vad/tenvad_backend.h"
#endif

namespace vietasr {
namespace vad {

namespace {

// Recommended per-backend tuning. The neural backends score speech probability
// very differently from the RMS-energy detector, so each carries its own
// defaults. ApplyBackendDefaults runs before any JSON field is read, so an
// empty config ("{}") yields these values while any explicit field still wins.
void ApplyBackendDefaults(VadConfig& cfg) {
    if (cfg.backend == "silero") {
        cfg.threshold     = 0.43f;  // activation threshold
        cfg.min_speech_ms = 100;    // min speech duration   0.10 s
        cfg.hangover_ms   = 250;    // min silence duration  0.25 s
        cfg.speech_pad_ms = 500;    // prefix padding        0.50 s
    } else if (cfg.backend == "tenvad") {
        cfg.threshold     = 0.41f;  // activation threshold
        cfg.min_speech_ms = 100;    // min speech duration   0.10 s
        cfg.hangover_ms   = 250;    // min silence duration  0.25 s
        cfg.speech_pad_ms = 500;    // prefix padding        0.50 s
        cfg.hop_size      = 256;    // 16 ms analysis hop @ 16 kHz
    }
    // "energy" keeps the VadConfig struct defaults.
}

// Parse the module's JSON config into a fully-defaulted VadConfig. Per-backend
// defaults are applied first (see ApplyBackendDefaults); any field present in
// the JSON then overrides them — so SDK callers stay free to tune every
// parameter while an empty config still yields sensible per-backend values.
// backend_override forces the backend regardless of the JSON (used when
// MakeVadBackend falls back) without discarding the caller's field overrides.
VadConfig ParseVadConfig(const std::string& json, float sample_rate,
                         const std::string& backend_override = "") {
    VadConfig cfg;
    cfg.sample_rate = sample_rate > 0.0f ? sample_rate : 16000.0f;

    std::string str;
    if (JsonGetString(json, "backend", &str)) cfg.backend = str;
    if (!backend_override.empty()) cfg.backend = backend_override;

    ApplyBackendDefaults(cfg);

    double num = 0.0;
    if (JsonGetNumber(json, "threshold", &num))
        cfg.threshold = static_cast<float>(num);
    if (JsonGetNumber(json, "hop_size", &num))
        cfg.hop_size = static_cast<int>(num);
    if (JsonGetNumber(json, "hangover_ms", &num))
        cfg.hangover_ms = static_cast<int>(num);
    if (JsonGetNumber(json, "min_speech_ms", &num))
        cfg.min_speech_ms = static_cast<int>(num);
    if (JsonGetNumber(json, "min_silence_ms", &num))
        cfg.min_silence_ms = static_cast<int>(num);
    if (JsonGetNumber(json, "speech_pad_ms", &num))
        cfg.speech_pad_ms = static_cast<int>(num);

    // Clamp to sane ranges.
    if (cfg.threshold < 0.0f) cfg.threshold = 0.0f;
    if (cfg.threshold > 1.0f) cfg.threshold = 1.0f;
    if (cfg.hop_size != 160 && cfg.hop_size != 256) cfg.hop_size = 256;
    if (cfg.hangover_ms < 0) cfg.hangover_ms = 0;
    if (cfg.min_speech_ms < 0) cfg.min_speech_ms = 0;
    if (cfg.min_silence_ms < 0) cfg.min_silence_ms = 0;
    if (cfg.speech_pad_ms < 0) cfg.speech_pad_ms = 0;
    return cfg;
}

}  // namespace

std::unique_ptr<VadBackend> MakeVadBackend(const VadConfig& cfg) {
    std::unique_ptr<VadBackend> backend;

    if (cfg.backend == "tenvad") {
#if defined(VIETASR_VAD_TENVAD)
        backend = std::unique_ptr<VadBackend>(new TenVadBackend());
#else
        VIETASR_LOG_WARN("vad")
            << "tenvad backend not compiled in; using energy";
#endif
    } else if (cfg.backend == "silero") {
#if defined(VIETASR_VAD_SILERO)
        backend = std::unique_ptr<VadBackend>(new SileroBackend());
#else
        VIETASR_LOG_WARN("vad")
            << "silero backend not compiled in; using energy";
#endif
    } else if (cfg.backend != "energy") {
        VIETASR_LOG_WARN("vad")
            << "unknown vad backend '" << cfg.backend << "'; using energy";
    }

    if (backend) {
        Status s = backend->Init(cfg);
        if (s.ok()) return backend;
        VIETASR_LOG_WARN("vad")
            << "vad backend '" << backend->name() << "' failed to init ("
            << s.message() << "); falling back to energy";
    }

    auto energy = std::unique_ptr<VadBackend>(new EnergyBackend());
    energy->Init(cfg);
    return energy;
}

}  // namespace vad

// --------------------------------------------------------------------------

class VadModule final : public Module {
public:
    const char* name() const override { return "vad"; }
    const char* version() const override { return "2"; }

    Status Init(const ModuleConfig& config,
                ModelManager* models,
                Engine* engine) override {
        (void)models;
        (void)engine;
        config_json_ = config.json;
        cfg_ = vad::ParseVadConfig(config.json, config.sample_rate);
        BuildBackend();
        VIETASR_LOG_INFO("vad")
            << "module ready (backend=" << cfg_.backend
            << ", threshold=" << cfg_.threshold << ")";
        return Status::Ok();
    }

    std::unique_ptr<Module> Clone() const override {
        auto copy = std::unique_ptr<VadModule>(new VadModule());
        copy->cfg_ = cfg_;
        copy->config_json_ = config_json_;
        copy->BuildBackend();  // fresh, independent backend per Session
        return copy;
    }

    void Reset() override {
        if (backend_) backend_->Reset();
        if (segmenter_) segmenter_->Reset();
        accum_.Reset();
        chunks_done_ = 0;
        samples_seen_ = 0;
    }

    void OnFrame(const AudioFrame& frame, ResultBuilder* out) override {
        if (!backend_ || !segmenter_ || frame.pcm == nullptr) return;
        samples_seen_ += static_cast<std::int64_t>(frame.samples);
        const std::int64_t chunks_before = chunks_done_;
        accum_.Push(frame.pcm, frame.samples,
                    [this](const float* c, std::size_t n) {
                        ProcessChunk(c, n);
                    });
        // Streaming: refresh the result snapshot whenever new chunks were
        // processed, so Session::Partial() / Result() reflect VAD progress
        // live instead of only at the end of the stream.
        if (out != nullptr && chunks_done_ != chunks_before) {
            EmitSnapshot(out);
        }
    }

    void OnFinalize(ResultBuilder* out) override {
        if (!backend_ || !segmenter_ || out == nullptr) return;
        // Flush the trailing partial chunk (zero-padded to a full window).
        accum_.Flush([this](const float* c, std::size_t n) {
            ProcessChunk(c, n);
        });
        EmitSnapshot(out);
    }

private:
    void BuildBackend() {
        backend_ = vad::MakeVadBackend(cfg_);
        // MakeVadBackend may fall back to a different backend when the
        // requested one is unavailable. Re-parse against the backend actually
        // in use so the segmenter gets that backend's defaults — the caller's
        // explicit JSON field overrides are preserved by the re-parse.
        if (backend_->name() != cfg_.backend) {
            cfg_ = vad::ParseVadConfig(config_json_, cfg_.sample_rate,
                                       backend_->name());
        }
        accum_.SetChunk(backend_->chunk_samples());
        segmenter_ = std::unique_ptr<vad::VadSegmenter>(
            new vad::VadSegmenter(cfg_));
        chunks_done_ = 0;
        samples_seen_ = 0;
    }

    void ProcessChunk(const float* chunk, std::size_t n) {
        const float prob = backend_->Process(chunk, n);
        const double rate = static_cast<double>(cfg_.sample_rate);
        const double start =
            static_cast<double>(chunks_done_ * static_cast<std::int64_t>(n))
            / rate;
        const double dur = static_cast<double>(n) / rate;
        segmenter_->Accept(prob, start, dur);
        ++chunks_done_;
    }

    // Write the current voiced_regions + speech classification into the
    // result. Safe to call repeatedly during streaming: VadSegmenter::Finish()
    // does not mutate segmenter state, so each snapshot is consistent and the
    // final result simply reflects all audio seen so far.
    void EmitSnapshot(ResultBuilder* out) {
        const double end_s = cfg_.sample_rate > 0.0f
            ? static_cast<double>(samples_seen_) / cfg_.sample_rate
            : 0.0;
        const std::vector<vad::VoicedRegion> regions =
            segmenter_->Finish(end_s);
        EmitRegions(regions, out);
        EmitClassification(regions, end_s, out);
    }

    void EmitRegions(const std::vector<vad::VoicedRegion>& regions,
                     ResultBuilder* out) {
        JsonWriter writer;
        writer.BeginArray();
        for (const auto& r : regions) {
            writer.BeginObject();
            writer.Key("start").Number(r.start);
            writer.Key("end").Number(r.end);
            writer.EndObject();
        }
        writer.EndArray();
        out->SetField("voiced_regions", writer.Str());
    }

    // Emit the speech-vs-no-speech classification summary. `detected` is the
    // direct answer to "does this audio contain speech": true when the VAD
    // found at least one voiced region (already filtered by min_speech_ms).
    void EmitClassification(const std::vector<vad::VoicedRegion>& regions,
                            double total_s, ResultBuilder* out) {
        double voiced_s = 0.0;
        for (const auto& r : regions) voiced_s += r.end - r.start;
        if (total_s > 0.0 && voiced_s > total_s) voiced_s = total_s;

        double ratio = total_s > 0.0 ? voiced_s / total_s : 0.0;
        if (ratio < 0.0) ratio = 0.0;
        if (ratio > 1.0) ratio = 1.0;

        JsonWriter writer;
        writer.BeginObject();
        writer.Key("detected").Boolean(!regions.empty());
        writer.Key("voiced_ratio").Number(ratio);
        writer.Key("voiced_duration_s").Number(voiced_s);
        writer.Key("total_duration_s").Number(total_s);
        writer.EndObject();
        out->SetField("speech", writer.Str());
    }

    vad::VadConfig cfg_;
    std::string config_json_;  // raw module config, kept for fallback re-parse
    std::unique_ptr<vad::VadBackend> backend_;
    std::unique_ptr<vad::VadSegmenter> segmenter_;
    vad::ChunkAccumulator accum_;
    std::int64_t chunks_done_ = 0;
    std::int64_t samples_seen_ = 0;
};

VIETASR_REGISTER_MODULE("vad", VadModule)

}  // namespace vietasr
