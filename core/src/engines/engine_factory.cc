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
std::unique_ptr<Engine> CreateOnnxEngine(const EpPreference& ep);
#else
static std::unique_ptr<Engine> CreateOnnxEngine(const EpPreference&) { return nullptr; }
std::vector<std::string> OnnxAvailableProviders() { return {}; }
#endif

const char* EpEnumName(int ep_enum) {
    switch (ep_enum) {
        case VIETASR_EP_AUTO:     return "auto";
        case VIETASR_EP_CPU:      return "cpu";
        case VIETASR_EP_CUDA:     return "cuda";
        case VIETASR_EP_TENSORRT: return "tensorrt";
        case VIETASR_EP_COREML:   return "coreml";
        case VIETASR_EP_NNAPI:    return "nnapi";
        case VIETASR_EP_XNNPACK:  return "xnnpack";
        case VIETASR_EP_DIRECTML: return "directml";
        case VIETASR_EP_QNN:      return "qnn";
        default:                  return "unknown";
    }
}

std::unique_ptr<Engine> EngineFactory::Create(int backend_enum, EpPreference ep) {
    switch (backend_enum) {
        case VIETASR_BACKEND_ONNX: {
            auto engine = CreateOnnxEngine(ep);
            if (engine) return engine;
            VIETASR_LOG_WARN("engine") << "ONNX backend not available; using noop";
            return std::make_unique<NoopEngine>();
        }
        case VIETASR_BACKEND_COREML: {
            // Deprecated: CoreML is an ONNX Runtime execution provider, not a
            // separate backend. Remap to the ONNX backend with the CoreML EP.
            VIETASR_LOG_WARN("engine")
                << "VIETASR_BACKEND_COREML is deprecated; use "
                   "vietasr_pipeline_set_execution_provider(VIETASR_EP_COREML)";
            EpPreference coreml_ep{VIETASR_EP_COREML, ep.device_id};
            auto engine = CreateOnnxEngine(coreml_ep);
            if (engine) return engine;
            VIETASR_LOG_WARN("engine") << "ONNX backend not available; using noop";
            return std::make_unique<NoopEngine>();
        }
        case VIETASR_BACKEND_AUTO:
        default:
            return CreateAuto(ep);
    }
}

std::unique_ptr<Engine> EngineFactory::CreateAuto(EpPreference ep) {
    auto engine = CreateOnnxEngine(ep);
    if (engine) return engine;
    return std::make_unique<NoopEngine>();
}

}
