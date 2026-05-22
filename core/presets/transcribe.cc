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

// VAD-only pipeline: classifies whether audio contains speech, with no ASR.
// The result carries a "speech" summary ({"detected": ...}) plus the detailed
// "voiced_regions". Uses the TEN-VAD backend — a real speech/non-speech
// classifier (it falls back to the energy detector if ONNX Runtime is absent).
static Status BuildVadPreset(Pipeline* pipeline) {
    return pipeline->AddModule("vad", R"({"backend":"tenvad"})");
}

VIETASR_REGISTER_PRESET("vad", BuildVadPreset)

// Streaming-oriented pipeline: ASR only, no separate VAD module. The CTC
// endpointer segments on silence; the vietasr module emits voiced_regions +
// speech itself (from CTC-blank), exposes the per-sentence segments[] array
// with start/end times, and ITN-normalizes each sentence as it ends — while
// the encoder cache keeps streaming uninterrupted. `text` is the running
// aggregate of the whole stream.
static Status BuildStreamPreset(Pipeline* pipeline) {
    return pipeline->AddModule("vietasr",
        R"({"emit_voiced_regions":true,"emit_segments":true,"itn":true})");
}

VIETASR_REGISTER_PRESET("stream", BuildStreamPreset)

// Batch-oriented pipeline: a real (TEN-VAD) VAD plus ASR. The vad module emits
// voiced_regions + speech; the vietasr module emits the per-sentence segments[]
// array (text + start/end) and ITN-normalizes each sentence — so a batch result
// carries voiced_regions, speech, segments, and a fully-normalized transcript.
static Status BuildBatchPreset(Pipeline* pipeline) {
    auto status = pipeline->AddModule("vad", R"({"backend":"tenvad"})");
    if (!status.ok()) return status;
    return pipeline->AddModule("vietasr",
        R"({"emit_segments":true,"itn":true})");
}

VIETASR_REGISTER_PRESET("batch", BuildBatchPreset)

static Status BuildTranscribeRichPreset(Pipeline* pipeline) {
    auto status = pipeline->AddModule("vad", "{}");
    if (!status.ok()) return status;
    status = pipeline->AddModule("vietasr", "{}");
    if (!status.ok()) return status;
    // itn before punctuation: ITN expects lower-cased text without punctuation,
    // so it must rewrite the raw ASR transcript before punctuation restores it.
    status = pipeline->AddModule("itn", "{}");
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
