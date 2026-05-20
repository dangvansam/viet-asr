#include "preprocess/audio_resampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vietasr {

namespace {

constexpr double kPi = 3.14159265358979323846;

double Sinc(double x) {
    if (std::fabs(x) < 1e-9) return 1.0;
    double px = kPi * x;
    return std::sin(px) / px;
}

}

AudioResampler::AudioResampler(int input_rate, int output_rate,
                               int kernel_half_width)
    : input_rate_(input_rate),
      output_rate_(output_rate),
      kernel_half_width_(kernel_half_width),
      ratio_(static_cast<double>(input_rate) / static_cast<double>(output_rate)) {}

void AudioResampler::Reset() {
    input_buffer_.clear();
    output_buffer_.clear();
    next_input_index_ = 0.0;
    input_consumed_ = 0;
}

double AudioResampler::SincWindow(double x, double half_width) {
    if (std::fabs(x) >= half_width) return 0.0;
    double cutoff = 1.0;
    double hann = 0.5 + 0.5 * std::cos(kPi * x / half_width);
    return Sinc(cutoff * x) * hann;
}

void AudioResampler::AcceptWaveform(const float* pcm, std::size_t n) {
    if (is_passthrough()) {
        output_buffer_.insert(output_buffer_.end(), pcm, pcm + n);
        return;
    }
    input_buffer_.insert(input_buffer_.end(), pcm, pcm + n);

    double effective_half = static_cast<double>(kernel_half_width_);
    double safety_margin = effective_half + 2.0;

    double available_end = static_cast<double>(input_consumed_)
        + static_cast<double>(input_buffer_.size());
    double last_safe = available_end - safety_margin;

    while (next_input_index_ < last_safe) {
        double abs_idx = next_input_index_;
        double base = std::floor(abs_idx);
        double frac = abs_idx - base;

        double acc = 0.0;
        double weight_sum = 0.0;
        for (int k = -kernel_half_width_ + 1; k <= kernel_half_width_; ++k) {
            double sample_pos = base + static_cast<double>(k);
            std::int64_t rel = static_cast<std::int64_t>(sample_pos)
                - input_consumed_;
            if (rel < 0 || rel >= static_cast<std::int64_t>(input_buffer_.size())) {
                continue;
            }
            double w = SincWindow(static_cast<double>(k) - frac, effective_half);
            acc += w * input_buffer_[rel];
            weight_sum += w;
        }
        if (weight_sum > 1e-9) acc /= weight_sum;
        output_buffer_.push_back(static_cast<float>(acc));
        next_input_index_ += ratio_;
    }

    double trim_floor = next_input_index_ - safety_margin;
    std::int64_t trim_to = static_cast<std::int64_t>(std::floor(trim_floor));
    std::int64_t can_trim = trim_to - input_consumed_;
    if (can_trim > 0 && can_trim <= static_cast<std::int64_t>(input_buffer_.size())) {
        input_buffer_.erase(input_buffer_.begin(),
                            input_buffer_.begin() + can_trim);
        input_consumed_ += can_trim;
    }
}

void AudioResampler::Flush() {
    if (is_passthrough()) return;

    double available_end = static_cast<double>(input_consumed_)
        + static_cast<double>(input_buffer_.size());

    while (next_input_index_ < available_end) {
        double abs_idx = next_input_index_;
        double base = std::floor(abs_idx);
        double frac = abs_idx - base;

        double acc = 0.0;
        double weight_sum = 0.0;
        for (int k = -kernel_half_width_ + 1; k <= kernel_half_width_; ++k) {
            double sample_pos = base + static_cast<double>(k);
            std::int64_t rel = static_cast<std::int64_t>(sample_pos)
                - input_consumed_;
            if (rel < 0 || rel >= static_cast<std::int64_t>(input_buffer_.size())) {
                continue;
            }
            double w = SincWindow(static_cast<double>(k) - frac,
                                  static_cast<double>(kernel_half_width_));
            acc += w * input_buffer_[rel];
            weight_sum += w;
        }
        if (weight_sum > 1e-9) acc /= weight_sum;
        output_buffer_.push_back(static_cast<float>(acc));
        next_input_index_ += ratio_;
    }
}

std::vector<float> AudioResampler::PopAll() {
    std::vector<float> out;
    out.swap(output_buffer_);
    return out;
}

}

