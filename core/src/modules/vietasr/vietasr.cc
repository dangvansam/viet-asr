#include "modules/vietasr/vietasr.h"

#include <algorithm>
#include <chrono>
#include <utility>

#include "utils/json.h"
#include "vietasr/engine.h"
#include "vietasr/logger.h"

#if VIETASR_HAS_ITN
#include "modules/itn/itn_normalizer.h"
#endif

namespace vietasr {

namespace {

constexpr int kSampleRate = 16000;
constexpr std::size_t kMinDecodeSamples = 1600;
constexpr std::size_t kStreamingFrameLimit = 2 * kSampleRate;

int ReadInt(const std::string& json, const char* key, int fallback) {
    double value = 0.0;
    return JsonGetNumber(json, key, &value) ? static_cast<int>(value) : fallback;
}

bool ReadBool(const std::string& json, const char* key, bool fallback) {
    bool value = false;
    return JsonGetBool(json, key, &value) ? value : fallback;
}

}

VietAsrConfig VietAsrConfig::Parse(const std::string& json) {
    VietAsrConfig cfg;
    std::string method;
    if (JsonGetString(json, "decoding_method", &method) && method == "greedy_search") {
        cfg.max_active_paths = 1;
    }
    cfg.max_active_paths = std::max(1, ReadInt(json, "max_active_paths", cfg.max_active_paths));
    cfg.num_threads = std::max(1, ReadInt(json, "num_threads", cfg.num_threads));
    cfg.partial_interval_ms = std::max(0, ReadInt(json, "partial_interval_ms", cfg.partial_interval_ms));
    cfg.emit_segments = ReadBool(json, "emit_segments", cfg.emit_segments);
    cfg.emit_voiced_regions = ReadBool(json, "emit_voiced_regions", cfg.emit_voiced_regions);
    cfg.itn = ReadBool(json, "itn", cfg.itn);
    JsonGetString(json, "model_dir", &cfg.model_dir);
    JsonGetString(json, "vad_backend", &cfg.segmenter.backend);
    double threshold = 0.0;
    if (JsonGetNumber(json, "vad_threshold", &threshold)) {
        cfg.segmenter.threshold = static_cast<float>(threshold);
    }
    cfg.segmenter.min_silence_ms = ReadInt(json, "min_silence_ms", cfg.segmenter.min_silence_ms);
    cfg.segmenter.min_speech_ms = ReadInt(json, "min_speech_ms", cfg.segmenter.min_speech_ms);
    cfg.segmenter.speech_pad_ms = ReadInt(json, "speech_pad_ms", cfg.segmenter.speech_pad_ms);
    cfg.segmenter.max_utterance_ms = std::max(2000, ReadInt(json, "max_utterance_ms", cfg.segmenter.max_utterance_ms));
    cfg.segmenter.sample_rate = kSampleRate;
    return cfg;
}

std::string SegmentJoiner::Join(const std::vector<AsrSegment>& segments, const std::string& partial) {
    std::string out;
    for (const auto& seg : segments) {
        if (seg.text.empty()) continue;
        if (!out.empty()) out.push_back(' ');
        out += seg.text;
    }
    if (!partial.empty()) {
        if (!out.empty()) out.push_back(' ');
        out += partial;
    }
    return out;
}

VietAsrRecognizer::VietAsrRecognizer(std::shared_ptr<const TransducerModel> model,
                                     std::shared_ptr<const Units> units,
                                     int max_active_paths)
    : model_(model),
      units_(std::move(units)),
      decoder_(model, TransducerSearchConfig{max_active_paths, model->context_size(), model->blank_id(), 2}) {}

std::string VietAsrRecognizer::Transcribe(const std::vector<float>& pcm) const {
    if (pcm.size() < kMinDecodeSamples) return std::string();
    std::vector<float> features = fbank_.Compute(pcm.data(), pcm.size());
    int frames = static_cast<int>(features.size() / static_cast<std::size_t>(fbank_.num_bins()));
    EncoderOutput encoded;
    Status status = model_->Encode(features, frames, &encoded);
    if (!status.ok()) {
        VIETASR_LOG_WARN("vietasr") << "encode failed: " << status.message();
        return std::string();
    }
    std::vector<int> ids;
    status = decoder_.Decode(encoded, &ids);
    if (!status.ok()) {
        VIETASR_LOG_WARN("vietasr") << "decode failed: " << status.message();
        return std::string();
    }
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (int id : ids) tokens.push_back(units_->At(id));
    return post_processor_.Detokenize(tokens);
}

VietAsrModule::VietAsrModule() = default;
VietAsrModule::~VietAsrModule() = default;

Status VietAsrModule::Init(const ModuleConfig& config, ModelManager* models, Engine* engine) {
    config_ = VietAsrConfig::Parse(config.json);
#if !VIETASR_HAS_ITN
    config_.itn = false;
#endif
    segmenter_ = std::make_unique<UtteranceSegmenter>(config_.segmenter);

    std::string cache_dir = models ? models->cache_dir() : std::string();
    TransducerAssets assets = TransducerAssetLoader::Resolve(config_.model_dir, cache_dir);
    if (!assets.complete()) {
        VIETASR_LOG_WARN("vietasr") << "no transducer model found; vietasr will return empty results";
        return Status::Ok();
    }
    auto units = std::make_shared<Units>();
    Status status = units->LoadFromText(assets.tokens);
    if (!status.ok()) return status;
    if (engine == nullptr) return Status::Error(-6, "vietasr: no engine");
    engine->SetIntraOpThreads(config_.num_threads);
    auto model = std::make_shared<TransducerModel>();
    status = model->Load(engine, assets);
    if (!status.ok()) {
        VIETASR_LOG_WARN("vietasr") << "model load failed: " << status.message();
        return Status::Ok();
    }
    model_ = model;
    units_ = units;
    recognizer_ = std::make_unique<VietAsrRecognizer>(model_, units_, config_.max_active_paths);
    VIETASR_LOG_INFO("vietasr") << "transducer loaded origin=" << assets.origin
        << " tokens=" << units_->size() << " beam=" << config_.max_active_paths
        << " provider=" << engine->active_provider();
    return Status::Ok();
}

std::unique_ptr<Module> VietAsrModule::Clone() const {
    auto copy = std::make_unique<VietAsrModule>();
    copy->config_ = config_;
    copy->model_ = model_;
    copy->units_ = units_;
    if (model_) {
        copy->recognizer_ = std::make_unique<VietAsrRecognizer>(model_, units_, config_.max_active_paths);
    }
    copy->segmenter_ = std::make_unique<UtteranceSegmenter>(config_.segmenter);
    return copy;
}

void VietAsrModule::Reset() {
    if (segmenter_) segmenter_->Reset();
    segments_.clear();
    partial_.clear();
    partial_decoded_samples_ = 0;
}

double VietAsrModule::SamplesToSeconds(std::int64_t samples) const {
    return static_cast<double>(samples) / kSampleRate;
}

void VietAsrModule::CommitText(std::string text, double start_s, double end_s) {
    if (text.empty()) return;
#if VIETASR_HAS_ITN
    if (config_.itn) text = itn::NormalizeText(text);
#endif
    segments_.push_back(AsrSegment{std::move(text), start_s, end_s});
}

void VietAsrModule::DecodeCompleted() {
    for (auto& utt : segmenter_->PopCompleted()) {
        auto t0 = std::chrono::steady_clock::now();
        std::string text = recognizer_->Transcribe(utt.pcm);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        VIETASR_LOG_DEBUG("vietasr") << "utterance audio_s=" << SamplesToSeconds(static_cast<std::int64_t>(utt.pcm.size()))
            << " decode_ms=" << ms;
        CommitText(std::move(text), SamplesToSeconds(utt.speech_start_sample),
                   SamplesToSeconds(utt.speech_end_sample));
        partial_.clear();
        partial_decoded_samples_ = 0;
    }
}

void VietAsrModule::UpdatePartial() {
    if (!segmenter_->in_speech()) {
        partial_.clear();
        partial_decoded_samples_ = 0;
        return;
    }
    const std::vector<float>& active = segmenter_->active_pcm();
    std::size_t interval = static_cast<std::size_t>(config_.partial_interval_ms) * kSampleRate / 1000;
    interval = std::max(interval, partial_decoded_samples_ / 4);
    if (active.size() < partial_decoded_samples_ + interval) return;
    partial_ = recognizer_->Transcribe(active);
    partial_decoded_samples_ = active.size();
}

void VietAsrModule::OnFrame(const AudioFrame& frame, ResultBuilder* out) {
    if (!recognizer_ || !segmenter_) return;
    segmenter_->Push(frame.pcm, frame.samples);
    DecodeCompleted();
    if (frame.samples <= kStreamingFrameLimit) UpdatePartial();
    if (out) {
        std::string text = SegmentJoiner::Join(segments_, partial_);
        if (!text.empty()) {
            out->SetPartial(text);
            out->SetText("vietasr", text);
        }
        if (config_.emit_voiced_regions) EmitVoicedRegions(out);
        if (config_.emit_segments) EmitSegments(out);
    }
}

void VietAsrModule::OnFinalize(ResultBuilder* out) {
    if (!recognizer_ || !segmenter_) {
        out->SetText("vietasr", "[vietasr: model not loaded]");
        return;
    }
    bool fallback = segmenter_->fallback_available();
    std::vector<float> retained;
    if (fallback) retained = segmenter_->retained_stream();
    segmenter_->Finish();
    DecodeCompleted();
    if (segments_.empty() && fallback) {
        CommitText(recognizer_->Transcribe(retained), 0.0,
                   SamplesToSeconds(static_cast<std::int64_t>(retained.size())));
    }
    partial_.clear();
    partial_decoded_samples_ = 0;
    Emit(out);
}

void VietAsrModule::Emit(ResultBuilder* out) const {
    out->SetText("vietasr", SegmentJoiner::Join(segments_, std::string()));
    if (config_.emit_voiced_regions) EmitVoicedRegions(out);
    if (config_.emit_segments) EmitSegments(out);
}

void VietAsrModule::EmitSegments(ResultBuilder* out) const {
    JsonWriter writer;
    writer.BeginArray();
    for (const auto& seg : segments_) {
        writer.BeginObject();
        writer.Key("text").String(seg.text);
        writer.Key("start").Number(seg.start_s);
        writer.Key("end").Number(seg.end_s);
        writer.EndObject();
    }
    writer.EndArray();
    out->SetField("segments", writer.Str());
}

void VietAsrModule::EmitVoicedRegions(ResultBuilder* out) const {
    std::vector<std::pair<double, double>> regions;
    for (const auto& seg : segments_) regions.emplace_back(seg.start_s, seg.end_s);
    if (segmenter_->in_speech()) {
        regions.emplace_back(SamplesToSeconds(segmenter_->active_speech_start_sample()),
                             SamplesToSeconds(segmenter_->samples_seen()));
    }
    double total_s = SamplesToSeconds(segmenter_->samples_seen());
    JsonWriter regions_json;
    regions_json.BeginArray();
    double voiced_s = 0.0;
    for (const auto& r : regions) {
        regions_json.BeginObject();
        regions_json.Key("start").Number(r.first);
        regions_json.Key("end").Number(r.second);
        regions_json.EndObject();
        if (r.second > r.first) voiced_s += r.second - r.first;
    }
    regions_json.EndArray();
    out->SetField("voiced_regions", regions_json.Str());
    double ratio = total_s > 0.0 ? std::min(1.0, voiced_s / total_s) : 0.0;
    JsonWriter speech;
    speech.BeginObject();
    speech.Key("detected").Boolean(!regions.empty());
    speech.Key("voiced_ratio").Number(ratio);
    speech.Key("voiced_duration_s").Number(voiced_s);
    speech.Key("total_duration_s").Number(total_s);
    speech.EndObject();
    out->SetField("speech", speech.Str());
}

VIETASR_REGISTER_MODULE("vietasr", VietAsrModule)

}
