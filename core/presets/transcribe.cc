#include "vietasr/pipeline.h"

namespace vietasr {

static Status BuildTranscribePreset(Pipeline* pipeline) {
    auto status = pipeline->AddModule("vad", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("vietasr", "{}");
    if (!status.ok()) return status;
    return Status::Ok();
}

VIETASR_REGISTER_PRESET("transcribe", BuildTranscribePreset)

static Status BuildVadPreset(Pipeline* pipeline) {
    return pipeline->AddModule("vad", R"({"backend":"tenvad"})");
}

VIETASR_REGISTER_PRESET("vad", BuildVadPreset)

static Status BuildStreamPreset(Pipeline* pipeline) {
    return pipeline->AddModule("vietasr",
        R"({"emit_voiced_regions":true,"emit_segments":true})");
}

VIETASR_REGISTER_PRESET("stream", BuildStreamPreset)

static Status BuildBatchPreset(Pipeline* pipeline) {
    auto status = pipeline->AddModule("vad", R"({"backend":"tenvad"})");
    if (!status.ok()) return status;
    return pipeline->AddModule("vietasr", R"({"emit_segments":true})");
}

VIETASR_REGISTER_PRESET("batch", BuildBatchPreset)

static Status BuildTranscribeRichPreset(Pipeline* pipeline) {
    auto status = pipeline->AddModule("vad", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("vietasr", R"({"emit_segments":true})");
    if (!status.ok()) return status;
    status = pipeline->AddModule("punctuation", "{}");
    if (!status.ok()) return status;
    return Status::Ok();
}

VIETASR_REGISTER_PRESET("transcribe-rich", BuildTranscribeRichPreset)

static Status BuildMeetingPreset(Pipeline* pipeline) {
    auto status = pipeline->AddModule("vad", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("diarization", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("vietasr", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("punctuation", "{}");
    if (!status.ok()) return status;
    return Status::Ok();
}

VIETASR_REGISTER_PRESET("meeting", BuildMeetingPreset)

static Status BuildAnalyticsPreset(Pipeline* pipeline) {
    auto status = pipeline->AddModule("vad", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("vietasr", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("gender", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("emotion", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("dialect", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("noise", "{}");
    if (!status.ok()) return status;
    return Status::Ok();
}

VIETASR_REGISTER_PRESET("analytics", BuildAnalyticsPreset)

}
