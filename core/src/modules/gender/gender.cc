#include "vietasr/module.h"

namespace vietasr {

namespace {

class GenderModule final : public Module {
public:
    const char* name() const override { return "gender"; }
    const char* version() const override { return "0"; }

    Status Init(const ModuleConfig& config,
                ModelManager* models,
                Engine* engine) override {
        (void)config; (void)models; (void)engine;
        return Status::Ok();
    }

    std::unique_ptr<Module> Clone() const override {
        return std::unique_ptr<Module>(new GenderModule());
    }

    void OnFinalize(ResultBuilder* out) override {
        out->SetField("gender", "{\"value\":\"U\",\"score\":0.0}");
    }
};

VIETASR_REGISTER_MODULE("gender", GenderModule)

}

}
