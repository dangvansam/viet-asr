#include "modules/vietasr/utterance_segmenter.h"

#include <algorithm>

namespace vietasr {

UtteranceSegmenter::UtteranceSegmenter(const UtteranceSegmenterConfig& config)
    : config_(config) {
    vad::VadConfig vad_config;
    vad_config.backend = config_.backend;
    vad_config.threshold = config_.threshold;
    vad_config.sample_rate = static_cast<float>(config_.sample_rate);
    backend_ = vad::MakeVadBackend(vad_config);
    accumulator_.SetChunk(backend_ ? backend_->chunk_samples() : 512);
}

std::int64_t UtteranceSegmenter::MsToSamples(int ms) const {
    return static_cast<std::int64_t>(ms) * config_.sample_rate / 1000;
}

std::int64_t UtteranceSegmenter::RequiredSilence() const {
    if (!config_.adaptive_silence) return MsToSamples(config_.min_silence_ms);
    std::int64_t length = static_cast<std::int64_t>(active_.size());
    if (length >= MsToSamples(config_.adaptive_long_ms)) return MsToSamples(config_.adaptive_long_silence_ms);
    if (length >= MsToSamples(config_.adaptive_short_ms)) return MsToSamples(config_.adaptive_mid_silence_ms);
    return MsToSamples(config_.min_silence_ms);
}

void UtteranceSegmenter::Reset() {
    if (backend_) backend_->Reset();
    accumulator_.Reset();
    preroll_.clear();
    active_.clear();
    chunk_marks_.clear();
    retained_.clear();
    completed_.clear();
    in_speech_ = false;
    ever_detected_ = false;
    retain_enabled_ = true;
    samples_seen_ = 0;
    active_start_ = 0;
    speech_start_ = 0;
    last_voiced_end_ = 0;
    silence_run_ = 0;
    voiced_samples_ = 0;
}

void UtteranceSegmenter::Push(const float* pcm, std::size_t n) {
    if (retain_enabled_ && !ever_detected_) {
        std::size_t limit = static_cast<std::size_t>(MsToSamples(config_.max_utterance_ms));
        if (retained_.size() + n <= limit) {
            retained_.insert(retained_.end(), pcm, pcm + n);
        } else {
            retain_enabled_ = false;
            retained_.clear();
            retained_.shrink_to_fit();
        }
    }
    accumulator_.Push(pcm, n, [this](const float* chunk, std::size_t size) {
        ProcessChunk(chunk, size, size);
    });
}

void UtteranceSegmenter::Finish() {
    std::size_t chunk = accumulator_.chunk();
    std::size_t real = accumulator_.Flush([this](const float* data, std::size_t size) {
        ProcessChunk(data, size, size);
    });
    if (real > 0 && real < chunk) {
        std::int64_t pad = static_cast<std::int64_t>(chunk - real);
        samples_seen_ -= pad;
        if (in_speech_) {
            std::size_t keep = active_.size() > static_cast<std::size_t>(pad)
                ? active_.size() - static_cast<std::size_t>(pad) : 0;
            active_.resize(keep);
        }
        last_voiced_end_ = std::min(last_voiced_end_, samples_seen_);
    }
    if (in_speech_) CloseUtterance(samples_seen_);
}

void UtteranceSegmenter::ProcessChunk(const float* chunk, std::size_t n, std::size_t real) {
    (void)real;
    std::int64_t chunk_start = samples_seen_;
    samples_seen_ += static_cast<std::int64_t>(n);
    float prob = backend_ ? backend_->Process(chunk, n) : 0.0f;
    bool voiced = prob >= config_.threshold;

    if (!in_speech_) {
        if (!voiced) {
            preroll_.insert(preroll_.end(), chunk, chunk + n);
            std::size_t pad = static_cast<std::size_t>(MsToSamples(config_.speech_pad_ms));
            while (preroll_.size() > pad) preroll_.pop_front();
            return;
        }
        OpenUtterance(chunk_start);
    }

    active_.insert(active_.end(), chunk, chunk + n);
    chunk_marks_.emplace_back(chunk_start, prob);
    if (voiced) {
        last_voiced_end_ = samples_seen_;
        silence_run_ = 0;
        voiced_samples_ += static_cast<std::int64_t>(n);
    } else {
        silence_run_ += static_cast<std::int64_t>(n);
        if (silence_run_ >= RequiredSilence()) {
            CloseUtterance(last_voiced_end_ + MsToSamples(config_.speech_pad_ms));
            return;
        }
    }
    if (static_cast<std::int64_t>(active_.size()) >= MsToSamples(config_.max_utterance_ms)) {
        SplitLongUtterance();
    }
}

void UtteranceSegmenter::OpenUtterance(std::int64_t chunk_start) {
    active_.assign(preroll_.begin(), preroll_.end());
    active_start_ = chunk_start - static_cast<std::int64_t>(preroll_.size());
    speech_start_ = chunk_start;
    preroll_.clear();
    chunk_marks_.clear();
    in_speech_ = true;
    silence_run_ = 0;
    voiced_samples_ = 0;
}

void UtteranceSegmenter::CloseUtterance(std::int64_t end_sample) {
    std::int64_t active_end = active_start_ + static_cast<std::int64_t>(active_.size());
    std::int64_t end = std::min(end_sample, active_end);
    std::size_t keep = static_cast<std::size_t>(std::max<std::int64_t>(0, end - active_start_));
    if (voiced_samples_ >= MsToSamples(config_.min_speech_ms)) {
        Utterance utt;
        utt.pcm.assign(active_.begin(), active_.begin() + keep);
        utt.start_sample = active_start_;
        utt.speech_start_sample = speech_start_;
        utt.speech_end_sample = std::min(last_voiced_end_, end);
        completed_.push_back(std::move(utt));
        ever_detected_ = true;
        retained_.clear();
        retained_.shrink_to_fit();
    }
    preroll_.assign(active_.begin() + keep, active_.end());
    std::size_t pad = static_cast<std::size_t>(MsToSamples(config_.speech_pad_ms));
    while (preroll_.size() > pad) preroll_.pop_front();
    active_.clear();
    chunk_marks_.clear();
    in_speech_ = false;
    silence_run_ = 0;
    voiced_samples_ = 0;
}

void UtteranceSegmenter::SplitLongUtterance() {
    std::int64_t window_start = samples_seen_ - MsToSamples(config_.split_search_ms);
    std::int64_t cut = samples_seen_;
    float best = 2.0f;
    std::size_t half = accumulator_.chunk() / 2;
    for (const auto& mark : chunk_marks_) {
        if (mark.first < window_start) continue;
        if (mark.second < best) {
            best = mark.second;
            cut = mark.first + static_cast<std::int64_t>(half);
        }
    }
    std::size_t keep = static_cast<std::size_t>(std::max<std::int64_t>(0, cut - active_start_));
    keep = std::min(keep, active_.size());

    Utterance utt;
    utt.pcm.assign(active_.begin(), active_.begin() + keep);
    utt.start_sample = active_start_;
    utt.speech_start_sample = speech_start_;
    utt.speech_end_sample = active_start_ + static_cast<std::int64_t>(keep);
    completed_.push_back(std::move(utt));
    ever_detected_ = true;
    retained_.clear();
    retained_.shrink_to_fit();

    std::vector<float> tail(active_.begin() + keep, active_.end());
    active_ = std::move(tail);
    active_start_ += static_cast<std::int64_t>(keep);
    speech_start_ = active_start_;
    std::vector<std::pair<std::int64_t, float>> marks;
    voiced_samples_ = 0;
    for (const auto& mark : chunk_marks_) {
        if (mark.first < active_start_) continue;
        marks.push_back(mark);
        if (mark.second >= config_.threshold) {
            voiced_samples_ += static_cast<std::int64_t>(accumulator_.chunk());
        }
    }
    chunk_marks_ = std::move(marks);
}

std::vector<Utterance> UtteranceSegmenter::PopCompleted() {
    std::vector<Utterance> out = std::move(completed_);
    completed_.clear();
    return out;
}

}
