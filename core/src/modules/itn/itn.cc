#include "modules/itn/itn.h"

#include <utility>

#include "modules/itn/embedded_itn.h"
#include "modules/itn/processor.h"
#include "utils/json.h"
#include "vietasr/logger.h"

namespace vietasr {

ItnModule::ItnModule() = default;
ItnModule::~ItnModule() = default;

Status ItnModule::Init(const ModuleConfig& config,
                       ModelManager* models,
                       Engine* engine) {
    (void)config;
    (void)models;
    (void)engine;

    const itn::ItnBlob tagger = itn::EmbeddedItnTagger();
    const itn::ItnBlob verbalizer = itn::EmbeddedItnVerbalizer();
    if (tagger.empty() || verbalizer.empty()) {
        VIETASR_LOG_WARN("itn")
            << "no embedded ITN grammar; text will pass through unchanged";
        return Status::Ok();
    }

    auto processor = std::make_shared<itn::ItnProcessor>();
    if (!processor->Load(tagger.data, tagger.size,
                         verbalizer.data, verbalizer.size)) {
        VIETASR_LOG_WARN("itn")
            << "failed to load ITN grammar; text will pass through unchanged";
        return Status::Ok();
    }

    processor_ = std::move(processor);
    VIETASR_LOG_INFO("itn") << "ITN grammar loaded (vi)";
    return Status::Ok();
}

std::unique_ptr<Module> ItnModule::Clone() const {
    auto copy = std::unique_ptr<ItnModule>(new ItnModule());
    // The grammar FSTs are immutable after loading, so every cloned session
    // can safely share the same processor instance.
    copy->processor_ = processor_;
    return copy;
}

void ItnModule::OnFinalize(ResultBuilder* out) {
    if (!processor_) {
        out->SetField("itn", R"({"applied":false,"lang":"vi"})");
        return;
    }

    const std::string text = out->CurrentText();
    if (text.empty()) {
        out->SetField("itn", R"({"applied":false,"lang":"vi"})");
        return;
    }

    const std::string normalized = processor_->Normalize(text);
    const bool applied = !normalized.empty() && normalized != text;
    if (applied) {
        // Overwrite the canonical transcript with the written form.
        out->SetText("itn", normalized);
    }

    JsonWriter writer;
    writer.BeginObject();
    writer.Key("applied").Boolean(applied);
    writer.Key("lang").String("vi");
    if (applied) {
        writer.Key("text").String(normalized);
    }
    writer.EndObject();
    out->SetField("itn", writer.Str());
}

VIETASR_REGISTER_MODULE("itn", ItnModule)

}  // namespace vietasr
