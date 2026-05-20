#include "vietasr/module.h"

namespace vietasr {

namespace {

class EmotionModule final : public Module {
public:
    const char* name() const override { return "emotion"; }
    const char* version() const override { return "0"; }

    Status Init(const ModuleConfig& config,
                ModelManager* models,
                Engine* engine) override {
        (void)config; (void)models; (void)engine;
        return Status::Ok();
    }

    std::unique_ptr<Module> Clone() const override {
        return std::unique_ptr<Module>(new EmotionModule());
    }

    void OnFinalize(ResultBuilder* out) override {
        out->SetField("emotion", "{\"value\":\"neutral\",\"score\":0.0}");
    }
};

VIETASR_REGISTER_MODULE("emotion", EmotionModule)

}

}
