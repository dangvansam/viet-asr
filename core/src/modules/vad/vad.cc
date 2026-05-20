#include "vietasr/module.h"

#include "vietasr/logger.h"

namespace vietasr {

class VadModule final : public Module {
public:
    const char* name() const override { return "vad"; }
    const char* version() const override { return "1"; }

    Status Init(const ModuleConfig& config,
                ModelManager* models,
                Engine* engine) override {
        (void)config;
        (void)models;
        engine_ = engine;
        return Status::Ok();
    }

    std::unique_ptr<Module> Clone() const override {
        auto copy = std::unique_ptr<VadModule>(new VadModule());
        copy->engine_ = engine_;
        copy->voiced_samples_ = 0;
        copy->total_samples_ = 0;
        return copy;
    }

    void Reset() override {
        voiced_samples_ = 0;
        total_samples_ = 0;
    }

    void OnFrame(const AudioFrame& frame, ResultBuilder* out) override {
        (void)out;
        total_samples_ += static_cast<std::int64_t>(frame.samples);
        float energy = 0.0f;
        for (std::size_t i = 0; i < frame.samples; ++i) {
            energy += frame.pcm[i] * frame.pcm[i];
        }
        if (frame.samples > 0 && energy / frame.samples > 1e-6f) {
            voiced_samples_ += static_cast<std::int64_t>(frame.samples);
        }
    }

    void OnFinalize(ResultBuilder* out) override {
        (void)out;
        VIETASR_LOG_DEBUG("vad")
            << "voiced=" << voiced_samples_ << "/" << total_samples_;
    }

private:
    Engine* engine_{nullptr};
    std::int64_t voiced_samples_{0};
    std::int64_t total_samples_{0};
};

VIETASR_REGISTER_MODULE("vad", VadModule)

}
