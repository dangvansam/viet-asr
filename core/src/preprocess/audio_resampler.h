#ifndef VIETASR_PREPROCESS_AUDIO_RESAMPLER_H
#define VIETASR_PREPROCESS_AUDIO_RESAMPLER_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vietasr {

class AudioResampler final {
public:
    AudioResampler(int input_rate, int output_rate,
                   int kernel_half_width = 16);

    void Reset();

    void AcceptWaveform(const float* pcm, std::size_t n);
    void Flush();
    std::vector<float> PopAll();

    int input_rate() const { return input_rate_; }
    int output_rate() const { return output_rate_; }
    bool is_passthrough() const { return input_rate_ == output_rate_; }

private:
    int input_rate_;
    int output_rate_;
    int kernel_half_width_;
    double ratio_;

    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    double next_input_index_{0.0};
    std::int64_t input_consumed_{0};

    static double SincWindow(double x, double half_width);
};

}

#endif
