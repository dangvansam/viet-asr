#include "vietasr/module.h"

namespace vietasr {

namespace {

class NoiseModule final : public Module {
public:
    const char* name() const override { return "noise"; }
    const char* version() const override { return "0"; }

    Status Init(const ModuleConfig& config,
                ModelManager* models,
                Engine* engine) override {
        (void)config; (void)models; (void)engine;
        return Status::Ok();
    }

    std::unique_ptr<Module> Clone() const override {
        return std::unique_ptr<Module>(new NoiseModule());
    }

    void OnFinalize(ResultBuilder* out) override {
        out->SetField("noise", "{\"db\":0.0,\"type\":\"unknown\"}");
    }
};

VIETASR_REGISTER_MODULE("noise", NoiseModule)

}

}
