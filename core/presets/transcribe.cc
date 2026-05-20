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

static Status BuildTranscribeRichPreset(Pipeline* pipeline) {
    auto status = pipeline->AddModule("vad", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("vietasr", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("punctuation", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("itn", "{}");
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
