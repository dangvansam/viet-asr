#include "modules/vietasr/vietasr.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "embedded_model.h"
#include "utils/json.h"
#include "vietasr/engine.h"
#include "vietasr/logger.h"

#if VIETASR_HAS_ITN
#include "modules/itn/itn_normalizer.h"  // per-sentence ITN (fst-free facade)
#endif

namespace vietasr {

namespace {

constexpr int kInputFrames = 43;
constexpr int kFeatureDim = 80;
constexpr int kNumLayers = 12;
constexpr int kAttHeads = 4;
constexpr int kAttCacheLen = 40;
constexpr int kAttCacheDim = 224;
constexpr int kCnnCacheW = 448;
constexpr int kCnnCacheH = 30;
constexpr int kOutputFrames = 10;
constexpr int kVocabSize = 4972;
constexpr int kBlankId = 0;
constexpr int kFrameMs = 40;
constexpr float kBlankThreshold = 0.8f;

// The encoder's positional-encoding table is a fixed [5040, 448] initializer.
// Each chunk indexes a 50-wide span [encoder_offset, encoder_offset+49] into
// it, so a stream is bounded to ~5040 output frames (~3.6 min) before the
// index overflows. RunOneChunk() re-anchors the streaming caches before that
// limit, keeping arbitrarily long / live streams working.
constexpr int kEncoderPosTableLen = 5040;
constexpr int kEncoderPosWindow = 50;
constexpr int kMaxEncoderOffset =
    kEncoderPosTableLen - kEncoderPosWindow - kOutputFrames;  // 4980

std::size_t AttCacheSize() {
    return 1ull * kNumLayers * kAttHeads * kAttCacheLen * kAttCacheDim;
}

std::size_t CnnCacheSize() {
    return 1ull * kNumLayers * kCnnCacheW * kCnnCacheH;
}

// Maps a silence gap to the joiner text placed before the next utterance.
std::string PauseSeparator(int pause_ms, bool pause_punctuation,
                           int sentence_pause_ms, int clause_pause_ms) {
    if (!pause_punctuation) return "\n";
    if (pause_ms >= sentence_pause_ms) return ". ";
    if (pause_ms >= clause_pause_ms) return ", ";
    return " ";
}

}

std::string ApplyPausePunctuation(const std::vector<AsrSegment>& segments,
                                  const std::string& partial,
                                  int partial_lead_pause_ms,
                                  bool pause_punctuation,
                                  int sentence_pause_ms,
                                  int clause_pause_ms) {
    std::string out;
    for (const auto& seg : segments) {
        if (seg.text.empty()) continue;
        if (!out.empty()) {
            out += PauseSeparator(seg.lead_pause_ms, pause_punctuation,
                                  sentence_pause_ms, clause_pause_ms);
        }
        out += seg.text;
    }
    if (!partial.empty()) {
        if (!out.empty()) {
            out += PauseSeparator(partial_lead_pause_ms, pause_punctuation,
                                  sentence_pause_ms, clause_pause_ms);
        }
        out += partial;
    }
    return out;
}

VietAsrModule::VietAsrModule() = default;
VietAsrModule::~VietAsrModule() = default;

std::unique_ptr<Module> VietAsrModule::Clone() const {
    auto copy = std::unique_ptr<VietAsrModule>(new VietAsrModule());
    copy->engine_ = engine_;
    copy->model_loaded_ = model_loaded_;
    copy->units_ = units_;
    copy->features_ = std::make_unique<FeaturePipeline>();
    copy->beam_search_ = std::make_unique<CtcBeamSearch>(CtcBeamSearchConfig{10, kBlankId});
    copy->endpoint_ = std::make_unique<CtcEndpoint>(
        CtcEndpointConfig{5000, 1000, 20000, kBlankThreshold});
    copy->att_cache_.assign(AttCacheSize(), 0.0f);
    copy->cnn_cache_.assign(CnnCacheSize(), 0.0f);
    copy->encoder_offset_ = 0;
    copy->samples_seen_ = 0;
    copy->silence_frames_ = 0;
    copy->total_frames_ = 0;
    copy->has_decoded_ = false;
    copy->completed_segments_.clear();
    copy->completed_join_.clear();
    copy->pause_run_frames_ = 0;
    copy->lead_pause_ms_ = 0;
    copy->pause_punctuation_ = pause_punctuation_;
    copy->sentence_pause_ms_ = sentence_pause_ms_;
    copy->clause_pause_ms_ = clause_pause_ms_;
    copy->emit_voiced_regions_ = emit_voiced_regions_;
    copy->emit_segments_ = emit_segments_;
    copy->itn_ = itn_;
    copy->output_frame_clock_ = 0;
    copy->utt_start_frame_ = 0;
    copy->utt_last_speech_frame_ = 0;
    return copy;
}

Status VietAsrModule::Init(const ModuleConfig& config,
                         ModelManager* models,
                         Engine* engine) {
    (void)models;
    engine_ = engine;

    // Pause-driven punctuation config (consumed by ApplyPausePunctuation).
    bool cfg_flag = true;
    double cfg_num = 0.0;
    if (JsonGetBool(config.json, "pause_punctuation", &cfg_flag))
        pause_punctuation_ = cfg_flag;
    if (JsonGetNumber(config.json, "sentence_pause_ms", &cfg_num))
        sentence_pause_ms_ = static_cast<int>(cfg_num);
    if (JsonGetNumber(config.json, "clause_pause_ms", &cfg_num))
        clause_pause_ms_ = static_cast<int>(cfg_num);
    if (sentence_pause_ms_ < 0) sentence_pause_ms_ = 0;
    if (clause_pause_ms_ < 0) clause_pause_ms_ = 0;
    // Opt-in: emit voiced_regions + speech from the CTC-blank endpointer, so a
    // streaming vietasr-only pipeline needs no separate VAD model.
    if (JsonGetBool(config.json, "emit_voiced_regions", &cfg_flag))
        emit_voiced_regions_ = cfg_flag;
    // Opt-in: emit the per-sentence segments[] array (text + start/end).
    if (JsonGetBool(config.json, "emit_segments", &cfg_flag))
        emit_segments_ = cfg_flag;
    // Opt-in: ITN-normalize each sentence as it is committed (needs the itn
    // module — quietly ignored if it was not built into this library).
    if (JsonGetBool(config.json, "itn", &cfg_flag))
        itn_ = cfg_flag;
#if !VIETASR_HAS_ITN
    if (itn_) {
        VIETASR_LOG_WARN("vietasr")
            << "itn=true but the itn module is not built; ignoring";
        itn_ = false;
    }
#endif

    features_ = std::make_unique<FeaturePipeline>();
    beam_search_ = std::make_unique<CtcBeamSearch>(CtcBeamSearchConfig{10, kBlankId});
    endpoint_ = std::make_unique<CtcEndpoint>(
        CtcEndpointConfig{5000, 1000, 20000, kBlankThreshold});

    samples_seen_ = 0;
    att_cache_.assign(AttCacheSize(), 0.0f);
    cnn_cache_.assign(CnnCacheSize(), 0.0f);
    encoder_offset_ = 0;
    completed_segments_.clear();
    completed_join_.clear();
    silence_frames_ = 0;
    total_frames_ = 0;
    has_decoded_ = false;
    pause_run_frames_ = 0;
    lead_pause_ms_ = 0;
    output_frame_clock_ = 0;
    utt_start_frame_ = 0;
    utt_last_speech_frame_ = 0;

    EmbeddedBlob vocab = EmbeddedAsrVocab();
    if (vocab.empty()) {
        VIETASR_LOG_WARN("vietasr")
            << "no embedded vocab; vietasr will return empty results";
        return Status::Ok();
    }
    auto units_status = units_.LoadFromText(
        std::string(reinterpret_cast<const char*>(vocab.data), vocab.size));
    if (!units_status.ok()) {
        VIETASR_LOG_WARN("vietasr") << "vocab: " << units_status.message();
    } else {
        VIETASR_LOG_INFO("vietasr") << "vocab loaded: " << units_.size() << " tokens";
    }

    EmbeddedBlob model = EmbeddedAsrModel();
    if (model.empty()) {
        VIETASR_LOG_WARN("vietasr")
            << "no embedded model; vietasr will return empty results";
        return Status::Ok();
    }

    if (engine_) {
        auto load_status = engine_->LoadModelFromBuffer(model.data, model.size);
        if (!load_status.ok()) {
            VIETASR_LOG_WARN("vietasr")
                << "engine LoadModelFromBuffer failed: " << load_status.message();
        } else {
            VIETASR_LOG_INFO("vietasr")
                << "engine loaded (" << engine_->backend_name()
                << "): embedded model, " << model.size << " bytes";
            model_loaded_ = true;
        }
    }

    return Status::Ok();
}

void VietAsrModule::Reset() {
    samples_seen_ = 0;
    encoder_offset_ = 0;
    if (features_) features_->Reset();
    if (beam_search_) beam_search_->Reset();
    if (endpoint_) endpoint_->Reset();
    std::fill(att_cache_.begin(), att_cache_.end(), 0.0f);
    std::fill(cnn_cache_.begin(), cnn_cache_.end(), 0.0f);
    completed_segments_.clear();
    completed_join_.clear();
    silence_frames_ = 0;
    total_frames_ = 0;
    has_decoded_ = false;
    pause_run_frames_ = 0;
    lead_pause_ms_ = 0;
    output_frame_clock_ = 0;
    utt_start_frame_ = 0;
    utt_last_speech_frame_ = 0;
}

void VietAsrModule::OnFrame(const AudioFrame& frame, ResultBuilder* out) {
    if (!model_loaded_ || !features_) {
        samples_seen_ += static_cast<std::int64_t>(frame.samples);
        return;
    }
    features_->AcceptWaveform(frame.pcm, frame.samples);
    samples_seen_ += static_cast<std::int64_t>(frame.samples);
    while (features_->num_frames_ready() >= static_cast<std::size_t>(kInputFrames)) {
        RunOneChunk();
    }
    if (out) {
        std::string current = BuildTranscript();
        if (!current.empty()) {
            out->SetPartial(current);
            out->SetText("vietasr", current);
        }
        if (emit_voiced_regions_) EmitVoicedRegions(out);
        if (emit_segments_) EmitSegments(out);
    }
}

void VietAsrModule::OnSegment(const Segment& segment, ResultBuilder* out) {
    (void)segment;
    (void)out;
}

void VietAsrModule::OnFinalize(ResultBuilder* out) {
    if (!model_loaded_) {
        out->SetText("vietasr", "[vietasr: model not loaded]");
        return;
    }
    Flush();
    // Commit the trailing in-progress utterance so the final segments[] /
    // transcript cover the whole stream end to end.
    CommitCurrentBeamAsSegment();
    out->SetText("vietasr", BuildTranscript());
    if (emit_voiced_regions_) EmitVoicedRegions(out);
    if (emit_segments_) EmitSegments(out);
}

// The per-sentence segments[] array: each committed chunk with its
// (ITN-normalized) text and start/end time — the streaming "result per chunk".
void VietAsrModule::EmitSegments(ResultBuilder* out) const {
    JsonWriter writer;
    writer.BeginArray();
    for (const auto& seg : completed_segments_) {
        writer.BeginObject();
        writer.Key("text").String(seg.text);
        writer.Key("start").Number(seg.start_s);
        writer.Key("end").Number(seg.end_s);
        writer.EndObject();
    }
    writer.EndArray();
    out->SetField("segments", writer.Str());
}

// Voiced regions + speech derived from the CTC-blank endpointer: each utterance
// (a speech run between endpoints) is one region. Lets a streaming vietasr-only
// pipeline report speech activity without a separate VAD model.
void VietAsrModule::EmitVoicedRegions(ResultBuilder* out) const {
    std::vector<std::pair<double, double>> regions;
    regions.reserve(completed_segments_.size() + 1);
    for (const auto& seg : completed_segments_) {
        regions.emplace_back(seg.start_s, seg.end_s);
    }
    // The current, not-yet-committed utterance.
    if (has_decoded_ && utt_last_speech_frame_ >= utt_start_frame_) {
        regions.emplace_back(
            static_cast<double>(utt_start_frame_) * kFrameMs / 1000.0,
            static_cast<double>(utt_last_speech_frame_) * kFrameMs / 1000.0);
    }
    const double total_s =
        static_cast<double>(output_frame_clock_) * kFrameMs / 1000.0;

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

    double ratio = total_s > 0.0 ? voiced_s / total_s : 0.0;
    if (ratio > 1.0) ratio = 1.0;
    JsonWriter speech;
    speech.BeginObject();
    speech.Key("detected").Boolean(!regions.empty());
    speech.Key("voiced_ratio").Number(ratio);
    speech.Key("voiced_duration_s").Number(voiced_s);
    speech.Key("total_duration_s").Number(total_s);
    speech.EndObject();
    out->SetField("speech", speech.Str());
}

void VietAsrModule::RunOneChunk() {
    if (!engine_ || !features_ || !beam_search_) return;

    // Re-anchor the encoder before its positional-encoding index would overrun
    // the fixed [5040, 448] table. Zeroing the caches alongside the offset is
    // required: the offset is the absolute position of the cached frames, so
    // the two must reset together. The decoder/beam continues uninterrupted.
    if (encoder_offset_ >= kMaxEncoderOffset) {
        std::fill(att_cache_.begin(), att_cache_.end(), 0.0f);
        std::fill(cnn_cache_.begin(), cnn_cache_.end(), 0.0f);
        encoder_offset_ = 0;
        VIETASR_LOG_DEBUG("vietasr")
            << "encoder re-anchored (positional-encoding limit reached)";
    }

    std::vector<float> frames = features_->PopFrames(kInputFrames);
    if (static_cast<int>(frames.size()) != kInputFrames * kFeatureDim) return;

    std::vector<Tensor> inputs(4);
    inputs[0].shape = {1, kInputFrames, kFeatureDim};
    inputs[0].data = std::move(frames);

    inputs[1].shape = {1, 1};
    inputs[1].data = { static_cast<float>(encoder_offset_) };

    inputs[2].shape = {1, kNumLayers, kAttHeads, kAttCacheLen, kAttCacheDim};
    inputs[2].data = att_cache_;

    inputs[3].shape = {1, kNumLayers, kCnnCacheW, kCnnCacheH};
    inputs[3].data = cnn_cache_;

    std::vector<Tensor> outputs;
    auto status = engine_->Run(inputs, &outputs);
    if (!status.ok()) {
        VIETASR_LOG_WARN("vietasr") << "engine Run failed: " << status.message();
        return;
    }
    if (outputs.size() < 3) {
        VIETASR_LOG_WARN("vietasr") << "engine returned " << outputs.size() << " outputs";
        return;
    }

    const Tensor& logits = outputs[0];
    int vocab_dim = logits.shape.empty() ? 0
        : static_cast<int>(logits.shape.back());
    int n_frames = logits.shape.size() >= 3
        ? static_cast<int>(logits.shape[logits.shape.size() - 2]) : 0;
    if (vocab_dim <= 0 || n_frames <= 0) {
        VIETASR_LOG_WARN("vietasr") << "unexpected logits shape";
        return;
    }
    int dim = std::min(vocab_dim, kVocabSize);

    for (int f = 0; f < n_frames; ++f) {
        const float* row = logits.data.data() + static_cast<std::size_t>(f) * vocab_dim;
        beam_search_->Step(row, dim);
        ProcessFrameEndpoint(row, vocab_dim);
    }

    att_cache_ = outputs[1].data;
    cnn_cache_ = outputs[2].data;
    encoder_offset_ += kOutputFrames;
}

void VietAsrModule::ProcessFrameEndpoint(const float* logits, int vocab_dim) {
    if (!endpoint_) return;
    total_frames_ += 1;
    output_frame_clock_ += 1;

    float m = logits[0];
    for (int i = 1; i < vocab_dim; ++i) {
        if (logits[i] > m) m = logits[i];
    }
    double denom_sum = 0.0;
    for (int i = 0; i < vocab_dim; ++i) {
        denom_sum += std::exp(static_cast<double>(logits[i]) - m);
    }
    double log_denom = std::log(denom_sum) + m;
    double blank_logp = static_cast<double>(logits[kBlankId]) - log_denom;
    double blank_prob = std::exp(blank_logp);
    if (blank_prob > static_cast<double>(kBlankThreshold)) {
        silence_frames_ += 1;
        pause_run_frames_ += 1;
    } else {
        // First speech frame of a new utterance: the silence run that just
        // ended is this utterance's lead-in pause. pause_run_frames_ spans the
        // whole gap — unlike silence_frames_ it is not reset by the endpoint
        // commit that happened mid-pause.
        if (!has_decoded_) {
            lead_pause_ms_ = pause_run_frames_ * kFrameMs;
            utt_start_frame_ = output_frame_clock_;
        }
        utt_last_speech_frame_ = output_frame_clock_;
        silence_frames_ = 0;
        pause_run_frames_ = 0;
        has_decoded_ = true;
    }

    int silence_ms = silence_frames_ * kFrameMs;
    int utterance_ms = total_frames_ * kFrameMs;
    if (endpoint_->Check(silence_ms, utterance_ms, has_decoded_)) {
        CommitCurrentBeamAsSegment();
    }
}

void VietAsrModule::CommitCurrentBeamAsSegment() {
    if (!beam_search_) return;
    std::vector<int> ids = beam_search_->Hypothesis();
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (int id : ids) {
        if (id == kBlankId) continue;
        tokens.push_back(units_.At(id));
    }
    std::string text = post_processor_.Detokenize(tokens);
    if (!text.empty()) {
#if VIETASR_HAS_ITN
        // Sentence end reached — ITN-normalize this sentence now. This touches
        // only the committed text; the encoder/beam caches are untouched, so
        // the rest of the stream continues uninterrupted.
        if (itn_) text = itn::NormalizeText(text);
#endif
        const double start_s =
            static_cast<double>(utt_start_frame_) * kFrameMs / 1000.0;
        const double end_s =
            static_cast<double>(utt_last_speech_frame_) * kFrameMs / 1000.0;
        completed_segments_.push_back(
            AsrSegment{std::move(text), lead_pause_ms_, start_s, end_s});
        // Refresh the cached join of all completed sentences (used by
        // BuildTranscript so the per-frame cost stays O(partial), not O(n)).
        completed_join_ = ApplyPausePunctuation(
            completed_segments_, "", 0, pause_punctuation_,
            sentence_pause_ms_, clause_pause_ms_);
    }
    beam_search_->Reset();
    silence_frames_ = 0;
    total_frames_ = 0;
    has_decoded_ = false;
}

void VietAsrModule::Flush() {
    if (!features_) return;
    while (features_->num_frames_ready() >= static_cast<std::size_t>(kInputFrames)) {
        RunOneChunk();
    }
    if (features_->num_frames_ready() > 0) {
        std::size_t avail = features_->num_frames_ready();
        std::vector<float> partial = features_->PopFrames(avail);
        partial.resize(static_cast<std::size_t>(kInputFrames) * kFeatureDim, 0.0f);

        std::vector<Tensor> inputs(4);
        inputs[0].shape = {1, kInputFrames, kFeatureDim};
        inputs[0].data = std::move(partial);
        inputs[1].shape = {1, 1};
        inputs[1].data = { static_cast<float>(encoder_offset_) };
        inputs[2].shape = {1, kNumLayers, kAttHeads, kAttCacheLen, kAttCacheDim};
        inputs[2].data = att_cache_;
        inputs[3].shape = {1, kNumLayers, kCnnCacheW, kCnnCacheH};
        inputs[3].data = cnn_cache_;

        std::vector<Tensor> outputs;
        if (engine_->Run(inputs, &outputs).ok() && outputs.size() >= 1) {
            const Tensor& logits = outputs[0];
            int vocab_dim = logits.shape.empty() ? 0
                : static_cast<int>(logits.shape.back());
            int n_frames = logits.shape.size() >= 3
                ? static_cast<int>(logits.shape[logits.shape.size() - 2]) : 0;
            int dim = std::min(vocab_dim, kVocabSize);
            int valid = static_cast<int>(
                std::ceil(static_cast<double>(avail) / 4.0));
            valid = std::min(valid, n_frames);
            for (int f = 0; f < valid; ++f) {
                const float* row = logits.data.data() + static_cast<std::size_t>(f) * vocab_dim;
                beam_search_->Step(row, dim);
            }
        }
    }
}

std::string VietAsrModule::BuildTranscript() const {
    std::string partial;
    if (beam_search_) {
        std::vector<int> ids = beam_search_->Hypothesis();
        std::vector<std::string> tokens;
        tokens.reserve(ids.size());
        for (int id : ids) {
            if (id == kBlankId) continue;
            tokens.push_back(units_.At(id));
        }
        partial = post_processor_.Detokenize(tokens);
    }
    // completed_join_ caches the join of all committed sentences (refreshed on
    // each commit); only the in-progress partial is appended here.
    if (partial.empty()) return completed_join_;
    if (completed_join_.empty()) return partial;
    return completed_join_
         + PauseSeparator(lead_pause_ms_, pause_punctuation_,
                          sentence_pause_ms_, clause_pause_ms_)
         + partial;
}

VIETASR_REGISTER_MODULE("vietasr", VietAsrModule)

}
