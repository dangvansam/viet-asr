#ifndef VIETASR_TYPES_H
#define VIETASR_TYPES_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vietasr {

class Status final {
public:
    Status() : code_(0) {}
    Status(int code, std::string message)
        : code_(code), message_(std::move(message)) {}

    static Status Ok() { return Status(); }
    static Status Error(int code, std::string message) {
        return Status(code, std::move(message));
    }

    bool ok() const { return code_ == 0; }
    int code() const { return code_; }
    const std::string& message() const { return message_; }

private:
    int code_;
    std::string message_;
};

struct AudioFrame final {
    const float* pcm;
    std::size_t samples;
    float sample_rate;
    std::int64_t offset_samples;
};

struct FeatureFrame final {
    const float* data;
    std::size_t rows;
    std::size_t cols;
    std::int64_t offset_frames;
};

struct LogitsFrame final {
    const float* data;
    std::size_t rows;
    std::size_t cols;
    std::int64_t offset_frames;
};

struct Segment final {
    double start_s;
    double end_s;
    std::int64_t start_sample;
    std::int64_t end_sample;
    bool is_voiced;
    bool is_endpoint;
};

struct TextSegment final {
    std::string text;
    double start_s;
    double end_s;
    float confidence;
    std::string speaker_id;
};

struct AudioClip final {
    std::vector<float> pcm;
    float sample_rate;
};

}

#endif
