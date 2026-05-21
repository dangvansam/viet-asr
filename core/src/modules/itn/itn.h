#ifndef VIETASR_MODULES_ITN_ITN_H
#define VIETASR_MODULES_ITN_ITN_H

#include <memory>

#include "vietasr/module.h"

namespace vietasr {

namespace itn {
class ItnProcessor;
}  // namespace itn

// Inverse Text Normalization module. Rewrites the spoken-form ASR transcript
// into written form (numbers, dates, currency, time, ...) using a WFST grammar
// ported from FunASR. Runs in OnFinalize, after the ASR module has set text.
class ItnModule final : public Module {
public:
    ItnModule();
    ~ItnModule() override;

    const char* name() const override { return "itn"; }
    const char* version() const override { return "1"; }

    Status Init(const ModuleConfig& config,
                ModelManager* models,
                Engine* engine) override;

    std::unique_ptr<Module> Clone() const override;

    void OnFinalize(ResultBuilder* out) override;

    bool consumes_text() const override { return true; }
    bool produces_text() const override { return true; }

private:
    // Read-only once Init() succeeds — shared across Clone()d instances.
    std::shared_ptr<itn::ItnProcessor> processor_;
};

}  // namespace vietasr

#endif  // VIETASR_MODULES_ITN_ITN_H
