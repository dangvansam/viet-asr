#ifndef VIETASR_MODULES_VIETASR_UTTERANCE_SEGMENTER_H
#define VIETASR_MODULES_VIETASR_UTTERANCE_SEGMENTER_H

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "modules/vad/vad_backend.h"
#include "modules/vad/vad_segmenter.h"

namespace vietasr {

struct UtteranceSegmenterConfig final {
    std::string backend{"silero"};
    float threshold{0.5f};
    int min_silence_ms{1200};
    int min_speech_ms{200};
    int speech_pad_ms{1000};
    int max_utterance_ms{30000};
    int split_search_ms{3000};
    bool adaptive_silence{true};
    int adaptive_short_ms{4000};
    int adaptive_long_ms{10000};
    int adaptive_mid_silence_ms{500};
    int adaptive_long_silence_ms{300};
    int sample_rate{16000};
};

struct Utterance final {
    std::vector<float> pcm;
    std::int64_t start_sample{0};
    std::int64_t speech_start_sample{0};
    std::int64_t speech_end_sample{0};
};

class UtteranceSegmenter final {
public:
    explicit UtteranceSegmenter(const UtteranceSegmenterConfig& config);

    void Reset();
    void Push(const float* pcm, std::size_t n);
    void Finish();

    std::vector<Utterance> PopCompleted();

    bool in_speech() const { return in_speech_; }
    const std::vector<float>& active_pcm() const { return active_; }
    std::int64_t active_start_sample() const { return active_start_; }
    std::int64_t active_speech_start_sample() const { return speech_start_; }
    std::int64_t samples_seen() const { return samples_seen_; }
    bool fallback_available() const { return !ever_detected_ && retain_enabled_ && !retained_.empty(); }
    const std::vector<float>& retained_stream() const { return retained_; }

private:
    void ProcessChunk(const float* chunk, std::size_t n, std::size_t real);
    void OpenUtterance(std::int64_t chunk_start);
    void CloseUtterance(std::int64_t end_sample);
    void SplitLongUtterance();
    std::int64_t MsToSamples(int ms) const;
    std::int64_t RequiredSilence() const;

    UtteranceSegmenterConfig config_;
    std::unique_ptr<vad::VadBackend> backend_;
    vad::ChunkAccumulator accumulator_;

    std::deque<float> preroll_;
    std::vector<float> active_;
    std::vector<std::pair<std::int64_t, float>> chunk_marks_;
    std::vector<float> retained_;
    std::vector<Utterance> completed_;

    bool in_speech_{false};
    bool ever_detected_{false};
    bool retain_enabled_{true};
    std::int64_t samples_seen_{0};
    std::int64_t active_start_{0};
    std::int64_t speech_start_{0};
    std::int64_t last_voiced_end_{0};
    std::int64_t silence_run_{0};
    std::int64_t voiced_samples_{0};
};

}

#endif
