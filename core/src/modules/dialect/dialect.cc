#include "vietasr/module.h"

namespace vietasr {

namespace {

class DialectModule final : public Module {
public:
    const char* name() const override { return "dialect"; }
    const char* version() const override { return "0"; }

    Status Init(const ModuleConfig& config,
                ModelManager* models,
                Engine* engine) override {
        (void)config; (void)models; (void)engine;
        return Status::Ok();
    }

    std::unique_ptr<Module> Clone() const override {
        return std::unique_ptr<Module>(new DialectModule());
    }

    void OnFinalize(ResultBuilder* out) override {
        out->SetField("dialect", "{\"value\":\"unknown\",\"score\":0.0}");
    }
};

VIETASR_REGISTER_MODULE("dialect", DialectModule)

}

}
