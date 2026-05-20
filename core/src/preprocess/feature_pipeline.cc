#include "preprocess/feature_pipeline.h"

#include <algorithm>
#include <cmath>
#include <complex>

namespace vietasr {

namespace {

constexpr float kPi = 3.14159265358979323846f;

bool IsPowerOfTwo(std::size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

void Radix2Fft(std::vector<std::complex<float>>* buf) {
    std::size_t n = buf->size();
    if (n <= 1) return;

    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; (j & bit); bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap((*buf)[i], (*buf)[j]);
    }

    for (std::size_t len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * kPi / static_cast<float>(len);
        std::complex<float> wlen(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            std::size_t half = len >> 1;
            for (std::size_t k = 0; k < half; ++k) {
                std::complex<float> u = (*buf)[i + k];
                std::complex<float> v = (*buf)[i + k + half] * w;
                (*buf)[i + k] = u + v;
                (*buf)[i + k + half] = u - v;
                w *= wlen;
            }
        }
    }
}

void FftReal(const float* input, std::size_t n,
             std::vector<std::complex<float>>* out) {
    std::vector<std::complex<float>> work(n);
    if (IsPowerOfTwo(n)) {
        for (std::size_t i = 0; i < n; ++i) {
            work[i] = std::complex<float>(input[i], 0.0f);
        }
        Radix2Fft(&work);
    } else {
        for (std::size_t k = 0; k < n / 2 + 1; ++k) {
            float re = 0.0f, im = 0.0f;
            for (std::size_t i = 0; i < n; ++i) {
                float angle = -2.0f * kPi * static_cast<float>(k * i) / static_cast<float>(n);
                re += input[i] * std::cos(angle);
                im += input[i] * std::sin(angle);
            }
            work.resize(n / 2 + 1);
            work[k] = std::complex<float>(re, im);
        }
        *out = std::move(work);
        return;
    }
    out->resize(n / 2 + 1);
    for (std::size_t k = 0; k < out->size(); ++k) {
        (*out)[k] = work[k];
    }
}

}

FeaturePipeline::FeaturePipeline(const FbankOptions& opts) : opts_(opts) {
    frame_len_samples_ = opts_.sample_rate * opts_.frame_length_ms / 1000;
    frame_shift_samples_ = opts_.sample_rate * opts_.frame_shift_ms / 1000;
    n_fft_ = 512;
    while (n_fft_ < frame_len_samples_) n_fft_ *= 2;
    n_freq_bins_ = n_fft_ / 2 + 1;
    BuildWindow();
    BuildMelFilterbank();
}

void FeaturePipeline::Reset() {
    buffer_.clear();
    features_.clear();
}

void FeaturePipeline::BuildWindow() {
    window_.resize(frame_len_samples_);
    for (int i = 0; i < frame_len_samples_; ++i) {
        float w = 0.5f - 0.5f * std::cos(2.0f * kPi * i / (frame_len_samples_ - 1));
        window_[i] = std::pow(w, 0.85f);
    }
}

float FeaturePipeline::HzToMel(float hz) {
    return 1127.0f * std::log(1.0f + hz / 700.0f);
}

float FeaturePipeline::MelToHz(float mel) {
    return 700.0f * (std::exp(mel / 1127.0f) - 1.0f);
}

void FeaturePipeline::BuildMelFilterbank() {
    float high_hz = (opts_.high_freq < 0.0f)
        ? static_cast<float>(opts_.sample_rate) / 2.0f + opts_.high_freq
        : opts_.high_freq;
    float mel_low = HzToMel(opts_.low_freq);
    float mel_high = HzToMel(high_hz);
    int M = opts_.num_bins;

    std::vector<float> mels(M + 2);
    for (int i = 0; i < M + 2; ++i) {
        mels[i] = mel_low + (mel_high - mel_low) * i / (M + 1);
    }

    std::vector<int> bins(M + 2);
    for (int i = 0; i < M + 2; ++i) {
        float hz = MelToHz(mels[i]);
        bins[i] = static_cast<int>(std::floor((n_fft_ + 1) * hz / opts_.sample_rate));
        if (bins[i] < 0) bins[i] = 0;
        if (bins[i] >= n_freq_bins_) bins[i] = n_freq_bins_ - 1;
    }

    mel_filterbank_.assign(static_cast<std::size_t>(M) * n_freq_bins_, 0.0f);
    for (int m = 1; m <= M; ++m) {
        int l = bins[m - 1];
        int c = bins[m];
        int r = bins[m + 1];
        for (int k = l; k < c; ++k) {
            float denom = static_cast<float>(c - l);
            if (denom <= 0.0f) continue;
            mel_filterbank_[(m - 1) * n_freq_bins_ + k] = (k - l) / denom;
        }
        for (int k = c; k < r; ++k) {
            float denom = static_cast<float>(r - c);
            if (denom <= 0.0f) continue;
            mel_filterbank_[(m - 1) * n_freq_bins_ + k] = (r - k) / denom;
        }
    }
}

void FeaturePipeline::AcceptWaveform(const float* pcm, std::size_t n) {
    std::size_t start = buffer_.size();
    buffer_.resize(start + n);
    for (std::size_t i = 0; i < n; ++i) {
        float sample = pcm[i];
        if (std::fabs(sample) <= 1.0f) sample *= 32768.0f;
        buffer_[start + i] = sample;
    }
    ComputeFrames();
}

void FeaturePipeline::ComputeFrames() {
    if (static_cast<int>(buffer_.size()) < frame_len_samples_) return;
    std::size_t n_frames = 1 + (buffer_.size() - frame_len_samples_) / frame_shift_samples_;
    if (n_frames == 0) return;

    std::vector<float> emphasised(buffer_.size());
    emphasised[0] = buffer_[0];
    for (std::size_t i = 1; i < buffer_.size(); ++i) {
        emphasised[i] = buffer_[i] - opts_.pre_emphasis * buffer_[i - 1];
    }

    std::vector<float> frame(n_fft_, 0.0f);
    std::vector<std::complex<float>> spectrum;
    std::size_t M = static_cast<std::size_t>(opts_.num_bins);

    for (std::size_t f = 0; f < n_frames; ++f) {
        std::size_t offset = f * frame_shift_samples_;
        for (int i = 0; i < frame_len_samples_; ++i) {
            frame[i] = emphasised[offset + i] * window_[i];
        }
        for (int i = frame_len_samples_; i < n_fft_; ++i) {
            frame[i] = 0.0f;
        }
        FftReal(frame.data(), static_cast<std::size_t>(n_fft_), &spectrum);

        std::vector<float> power(n_freq_bins_);
        for (int k = 0; k < n_freq_bins_; ++k) {
            float re = spectrum[k].real();
            float im = spectrum[k].imag();
            power[k] = re * re + im * im;
        }

        std::vector<float> mel(M, 0.0f);
        for (std::size_t m = 0; m < M; ++m) {
            float acc = 0.0f;
            const float* row = &mel_filterbank_[m * n_freq_bins_];
            for (int k = 0; k < n_freq_bins_; ++k) {
                acc += power[k] * row[k];
            }
            mel[m] = std::log(std::max(acc, 1e-10f));
        }

        features_.insert(features_.end(), mel.begin(), mel.end());
    }

    std::size_t consumed = n_frames * static_cast<std::size_t>(frame_shift_samples_);
    buffer_.erase(buffer_.begin(), buffer_.begin() + consumed);
}

std::size_t FeaturePipeline::num_frames_ready() const {
    return features_.size() / static_cast<std::size_t>(opts_.num_bins);
}

std::vector<float> FeaturePipeline::PopFrames(std::size_t n) {
    std::size_t avail = num_frames_ready();
    if (n > avail) n = avail;
    std::size_t row = static_cast<std::size_t>(opts_.num_bins);
    std::size_t bytes = n * row;
    std::vector<float> out(features_.begin(), features_.begin() + bytes);
    features_.erase(features_.begin(), features_.begin() + bytes);
    return out;
}

}
