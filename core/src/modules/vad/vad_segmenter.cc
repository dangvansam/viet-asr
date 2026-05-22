#include "modules/vad/vad_segmenter.h"

#include <algorithm>

namespace vietasr {
namespace vad {

VadSegmenter::VadSegmenter(const VadConfig& cfg) : cfg_(cfg) {}

void VadSegmenter::Reset() {
    in_speech_ = false;
    region_start_ = 0.0;
    last_voiced_end_ = 0.0;
    silence_run_ = 0.0;
    regions_.clear();
}

void VadSegmenter::Accept(float prob, double chunk_start_s, double chunk_dur_s) {
    const bool voiced = prob >= cfg_.threshold;
    if (voiced) {
        if (!in_speech_) {
            in_speech_ = true;
            region_start_ = chunk_start_s;
        }
        last_voiced_end_ = chunk_start_s + chunk_dur_s;
        silence_run_ = 0.0;
        return;
    }
    if (in_speech_) {
        silence_run_ += chunk_dur_s;
        if (silence_run_ * 1000.0 >= cfg_.hangover_ms) {
            regions_.push_back({region_start_, last_voiced_end_});
            in_speech_ = false;
        }
    }
}

std::vector<VoicedRegion> VadSegmenter::Finish(double end_of_stream_s) const {
    std::vector<VoicedRegion> out = regions_;
    if (in_speech_) {
        out.push_back({region_start_,
                       std::min(last_voiced_end_, end_of_stream_s)});
    }
    if (out.empty()) return out;

    // Pad each region, clamped to the stream bounds.
    const double pad = cfg_.speech_pad_ms / 1000.0;
    for (auto& r : out) {
        r.start = std::max(0.0, r.start - pad);
        r.end = std::min(end_of_stream_s, r.end + pad);
    }

    // Merge regions whose gap is below min_silence_ms (padding may have made
    // neighbors overlap, too).
    const double min_gap = cfg_.min_silence_ms / 1000.0;
    std::vector<VoicedRegion> merged;
    merged.push_back(out.front());
    for (std::size_t i = 1; i < out.size(); ++i) {
        VoicedRegion& cur = merged.back();
        if (out[i].start - cur.end < min_gap) {
            cur.end = std::max(cur.end, out[i].end);
        } else {
            merged.push_back(out[i]);
        }
    }

    // Drop regions shorter than min_speech_ms.
    const double min_speech = cfg_.min_speech_ms / 1000.0;
    std::vector<VoicedRegion> kept;
    kept.reserve(merged.size());
    for (const auto& r : merged) {
        if (r.end - r.start >= min_speech) kept.push_back(r);
    }
    return kept;
}

}  // namespace vad
}  // namespace vietasr
