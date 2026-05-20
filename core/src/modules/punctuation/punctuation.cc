#include "vietasr/module.h"

namespace vietasr {

namespace {

class PunctuationModule final : public Module {
public:
    const char* name() const override { return "punctuation"; }
    const char* version() const override { return "0"; }

    Status Init(const ModuleConfig& config,
                ModelManager* models,
                Engine* engine) override {
        (void)config; (void)models; (void)engine;
        return Status::Ok();
    }

    std::unique_ptr<Module> Clone() const override {
        return std::unique_ptr<Module>(new PunctuationModule());
    }

    void OnFinalize(ResultBuilder* out) override {
        out->SetField("punctuation", "{\"applied\":false}");
    }
};

VIETASR_REGISTER_MODULE("punctuation", PunctuationModule)

}

}
