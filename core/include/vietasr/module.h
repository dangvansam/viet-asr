#ifndef VIETASR_MODULE_H
#define VIETASR_MODULE_H

#include <functional>
#include <memory>
#include <string>

#include "vietasr/engine.h"
#include "vietasr/model_manager.h"
#include "vietasr/result.h"
#include "vietasr/types.h"

namespace vietasr {

struct ModuleConfig final {
    std::string json;
    float sample_rate = 16000.0f;
};

class Module {
public:
    virtual ~Module() = default;

    virtual const char* name() const = 0;
    virtual const char* version() const { return "1.0.0"; }

    virtual Status Init(const ModuleConfig& config,
                        ModelManager* models,
                        Engine* engine) = 0;

    virtual std::unique_ptr<Module> Clone() const {
        return nullptr;
    }

    virtual void Reset() {}

    virtual void OnFrame(const AudioFrame& frame, ResultBuilder* out) {
        (void)frame; (void)out;
    }
    virtual void OnFeature(const FeatureFrame& frame, ResultBuilder* out) {
        (void)frame; (void)out;
    }
    virtual void OnLogits(const LogitsFrame& frame, ResultBuilder* out) {
        (void)frame; (void)out;
    }
    virtual void OnSegment(const Segment& segment, ResultBuilder* out) {
        (void)segment; (void)out;
    }
    virtual void OnText(const TextSegment& segment, ResultBuilder* out) {
        (void)segment; (void)out;
    }
    virtual void OnFinalize(ResultBuilder* out) { (void)out; }

    virtual bool produces_features() const { return false; }
    virtual bool produces_logits() const { return false; }
    virtual bool produces_text() const { return false; }
    virtual bool consumes_features() const { return false; }
    virtual bool consumes_logits() const { return false; }
    virtual bool consumes_text() const { return false; }
};

using ModuleFactory = std::function<std::unique_ptr<Module>()>;

class ModuleRegistry final {
public:
    static ModuleRegistry& Instance();

    void Register(const std::string& name, ModuleFactory factory);
    std::unique_ptr<Module> Create(const std::string& name) const;
    std::vector<std::string> List() const;

private:
    ModuleRegistry();
    ~ModuleRegistry();
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#define VIETASR_REGISTER_MODULE(name, type)                                   \
    namespace {                                                               \
    struct AutoRegister_##type {                                              \
        AutoRegister_##type() {                                               \
            ::vietasr::ModuleRegistry::Instance().Register(                   \
                name, [] { return std::unique_ptr<::vietasr::Module>(        \
                    new type()); });                                          \
        }                                                                     \
    };                                                                        \
    static AutoRegister_##type s_register_##type;                             \
    }

}

#endif
