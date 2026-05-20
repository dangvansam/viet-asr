#include "vietasr/result.h"

#include <map>
#include <mutex>

#include "utils/json.h"

namespace vietasr {

class ResultBuilder::Impl {
public:
    mutable std::mutex mutex;
    std::map<std::string, std::string> fields;
    std::vector<TextSegment> segments;
    std::vector<std::pair<std::string, double>> speakers;
    std::string partial;
    bool is_final{false};

    std::string RenderUnlocked() const {
        JsonWriter writer;
        writer.BeginObject();

        if (auto it = fields.find("text"); it != fields.end()) {
            writer.Key("text").RawValue(it->second);
        }
        if (!partial.empty()) {
            writer.Key("partial").String(partial);
        }
        writer.Key("is_final").Boolean(is_final);

        if (!segments.empty()) {
            writer.Key("segments").BeginArray();
            for (const auto& seg : segments) {
                writer.BeginObject();
                writer.Key("start").Number(seg.start_s);
                writer.Key("end").Number(seg.end_s);
                writer.Key("text").String(seg.text);
                if (seg.confidence > 0.0f) {
                    writer.Key("confidence").Number(seg.confidence);
                }
                if (!seg.speaker_id.empty()) {
                    writer.Key("speaker").String(seg.speaker_id);
                }
                writer.EndObject();
            }
            writer.EndArray();
        }

        if (!speakers.empty()) {
            writer.Key("speakers").BeginArray();
            for (const auto& [id, total] : speakers) {
                writer.BeginObject();
                writer.Key("id").String(id);
                writer.Key("total_time_s").Number(total);
                writer.EndObject();
            }
            writer.EndArray();
        }

        for (const auto& [key, raw] : fields) {
            if (key == "text") continue;
            writer.Key(key).RawValue(raw);
        }

        writer.EndObject();
        return writer.Str();
    }
};

ResultBuilder::ResultBuilder() : impl_(std::make_unique<Impl>()) {}
ResultBuilder::~ResultBuilder() = default;

void ResultBuilder::Reset() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->fields.clear();
    impl_->segments.clear();
    impl_->speakers.clear();
    impl_->partial.clear();
    impl_->is_final = false;
}

void ResultBuilder::SetText(const std::string& module, std::string text) {
    (void)module;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->fields["text"] = JsonEscape(text);
}

void ResultBuilder::SetPartial(std::string text) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->partial = std::move(text);
}

void ResultBuilder::SetField(const std::string& dotted_path, std::string json_value) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->fields[dotted_path] = std::move(json_value);
}

void ResultBuilder::AppendSegment(TextSegment segment) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->segments.push_back(std::move(segment));
}

void ResultBuilder::AppendSpeaker(std::string speaker_id, double total_time_s) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->speakers.emplace_back(std::move(speaker_id), total_time_s);
}

void ResultBuilder::MarkFinal(bool is_final) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->is_final = is_final;
}

std::string ResultBuilder::SnapshotJson() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->RenderUnlocked();
}

std::string ResultBuilder::FinalJson() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->is_final = true;
    return impl_->RenderUnlocked();
}

}
