#ifndef VIETASR_MODULES_VAD_VAD_SEGMENTER_H
#define VIETASR_MODULES_VAD_VAD_SEGMENTER_H

#include <cstddef>
#include <vector>

#include "modules/vad/vad_backend.h"

namespace vietasr {
namespace vad {

// A contiguous span of detected speech, in seconds from the stream start.
struct VoicedRegion final {
    double start;
    double end;
};

// Buffers a variable-length float PCM stream and hands it out in fixed-size
// chunks. Audio frames from the pipeline are arbitrary length (one huge frame
// in batch mode, small frames when streaming); VAD backends need fixed windows.
class ChunkAccumulator final {
public:
    explicit ChunkAccumulator(std::size_t chunk = 512) : chunk_(chunk) {}

    void SetChunk(std::size_t chunk) {
        chunk_ = chunk ? chunk : 1;
        buf_.clear();
    }
    void Reset() { buf_.clear(); }
    std::size_t chunk() const { return chunk_; }

    // Append n samples; invoke fn(ptr, chunk()) for each complete window.
    template <class Fn>
    void Push(const float* pcm, std::size_t n, Fn&& fn) {
        buf_.insert(buf_.end(), pcm, pcm + n);
        std::size_t off = 0;
        while (buf_.size() - off >= chunk_) {
            fn(buf_.data() + off, chunk_);
            off += chunk_;
        }
        if (off) buf_.erase(buf_.begin(), buf_.begin() + off);
    }

    // Zero-pad and emit the trailing partial window, if any. Returns the
    // number of real (non-padding) samples in that final window, or 0 when
    // the buffer was already empty.
    template <class Fn>
    std::size_t Flush(Fn&& fn) {
        if (buf_.empty()) return 0;
        std::size_t real = buf_.size();
        buf_.resize(chunk_, 0.0f);
        fn(buf_.data(), chunk_);
        buf_.clear();
        return real;
    }

private:
    std::size_t chunk_;
    std::vector<float> buf_;
};

// Turns a stream of per-chunk speech probabilities into merged voiced regions.
// Implements the standard hysteresis state machine: a chunk above threshold
// opens a region; a silence run longer than hangover_ms closes it. Finish()
// applies padding, merges near regions, and drops too-short ones.
class VadSegmenter final {
public:
    explicit VadSegmenter(const VadConfig& cfg);

    void Reset();

    // Feed one chunk's probability. chunk_start_s is the chunk's start time
    // (seconds from stream start); chunk_dur_s is its duration.
    void Accept(float prob, double chunk_start_s, double chunk_dur_s);

    // Close any open region at end_of_stream_s and return the final, merged,
    // filtered region list. Does not mutate state (safe to call once).
    std::vector<VoicedRegion> Finish(double end_of_stream_s) const;

private:
    VadConfig cfg_;
    bool in_speech_ = false;
    double region_start_ = 0.0;
    double last_voiced_end_ = 0.0;
    double silence_run_ = 0.0;
    std::vector<VoicedRegion> regions_;
};

}  // namespace vad
}  // namespace vietasr

#endif  // VIETASR_MODULES_VAD_VAD_SEGMENTER_H
