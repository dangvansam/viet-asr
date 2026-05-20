#include "vietasr/engine.h"

#include "vietasr/logger.h"
#include "vietasr.h"

namespace vietasr {

class NoopEngine final : public Engine {
public:
    const char* backend_name() const override { return "noop"; }

    Status LoadModel(const std::string& model_path) override {
        (void)model_path;
        return Status::Ok();
    }

    Status Run(const std::vector<Tensor>& inputs,
               std::vector<Tensor>* outputs) override {
        (void)inputs;
        if (outputs) outputs->clear();
        return Status::Ok();
    }
};

#ifdef VIETASR_BACKEND_ONNX_ENABLED
std::unique_ptr<Engine> CreateOnnxEngine();
#else
static std::unique_ptr<Engine> CreateOnnxEngine() { return nullptr; }
#endif

std::unique_ptr<Engine> EngineFactory::Create(int backend_enum) {
    switch (backend_enum) {
        case VIETASR_BACKEND_ONNX: {
            auto engine = CreateOnnxEngine();
            if (engine) return engine;
            VIETASR_LOG_WARN("engine") << "ONNX backend not available; using noop";
            return std::make_unique<NoopEngine>();
        }
        case VIETASR_BACKEND_COREML:
            VIETASR_LOG_WARN("engine") << "CoreML backend not yet implemented; using noop";
            return std::make_unique<NoopEngine>();
        case VIETASR_BACKEND_AUTO:
        default:
            return CreateAuto();
    }
}

std::unique_ptr<Engine> EngineFactory::CreateAuto() {
    auto engine = CreateOnnxEngine();
    if (engine) return engine;
    return std::make_unique<NoopEngine>();
}

}
