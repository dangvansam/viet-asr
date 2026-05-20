#include "modules/vietasr/vietasr.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "embedded_model.h"
#include "vietasr/engine.h"
#include "vietasr/logger.h"

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

std::size_t AttCacheSize() {
    return 1ull * kNumLayers * kAttHeads * kAttCacheLen * kAttCacheDim;
}

std::size_t CnnCacheSize() {
    return 1ull * kNumLayers * kCnnCacheW * kCnnCacheH;
}

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
    return copy;
}

Status VietAsrModule::Init(const ModuleConfig& config,
                         ModelManager* models,
                         Engine* engine) {
    (void)config;
    (void)models;
    engine_ = engine;

    features_ = std::make_unique<FeaturePipeline>();
    beam_search_ = std::make_unique<CtcBeamSearch>(CtcBeamSearchConfig{10, kBlankId});
    endpoint_ = std::make_unique<CtcEndpoint>(
        CtcEndpointConfig{5000, 1000, 20000, kBlankThreshold});

    samples_seen_ = 0;
    att_cache_.assign(AttCacheSize(), 0.0f);
    cnn_cache_.assign(CnnCacheSize(), 0.0f);
    encoder_offset_ = 0;
    completed_segments_.clear();
    silence_frames_ = 0;
    total_frames_ = 0;
    has_decoded_ = false;

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
    silence_frames_ = 0;
    total_frames_ = 0;
    has_decoded_ = false;
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
    out->SetText("vietasr", BuildTranscript());
}

void VietAsrModule::RunOneChunk() {
    if (!engine_ || !features_ || !beam_search_) return;

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
    } else {
        silence_frames_ = 0;
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
        completed_segments_.push_back(std::move(text));
        if (completed_segments_.size() > 4) {
            completed_segments_.erase(completed_segments_.begin());
        }
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
    std::string out;
    for (const auto& seg : completed_segments_) {
        if (!out.empty()) out.push_back('\n');
        out += seg;
    }
    if (!beam_search_) return out;
    std::vector<int> ids = beam_search_->Hypothesis();
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (int id : ids) {
        if (id == kBlankId) continue;
        tokens.push_back(units_.At(id));
    }
    std::string partial = post_processor_.Detokenize(tokens);
    if (!partial.empty()) {
        if (!out.empty()) out.push_back('\n');
        out += partial;
    }
    return out;
}

VIETASR_REGISTER_MODULE("vietasr", VietAsrModule)

}
