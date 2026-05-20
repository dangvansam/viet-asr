#include "vietasr/module.h"

namespace vietasr {

namespace {

class DiarizationModule final : public Module {
public:
    const char* name() const override { return "diarization"; }
    const char* version() const override { return "0"; }

    Status Init(const ModuleConfig& config,
                ModelManager* models,
                Engine* engine) override {
        (void)config; (void)models; (void)engine;
        return Status::Ok();
    }

    std::unique_ptr<Module> Clone() const override {
        return std::unique_ptr<Module>(new DiarizationModule());
    }

    void OnFinalize(ResultBuilder* out) override {
        out->SetField("speakers", "[]");
    }
};

VIETASR_REGISTER_MODULE("diarization", DiarizationModule)

}

}
