#include "preprocess/kaldi_fbank.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <complex>

namespace vietasr {

namespace {

constexpr double kPi = 3.14159265358979323846;


}

KaldiFbank::KaldiFbank(const KaldiFbankOptions& opts) : opts_(opts) {
    padded_size_ = 1;
    while (padded_size_ < opts_.frame_length_samples) padded_size_ <<= 1;
    twiddle_.resize(static_cast<std::size_t>(padded_size_ / 2));
    for (int k = 0; k < padded_size_ / 2; ++k) {
        double a = -2.0 * kPi * k / padded_size_;
        twiddle_[k] = std::complex<double>(std::cos(a), std::sin(a));
    }
    BuildWindow();
    BuildMelBanks();
}

void KaldiFbank::PowerSpectrum(const std::vector<float>& input,
                               std::vector<double>* power) const {
    std::size_t n = static_cast<std::size_t>(padded_size_);
    std::vector<std::complex<double>> buf(n);
    for (std::size_t i = 0; i < n; ++i) buf[i] = std::complex<double>(input[i], 0.0);
    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(buf[i], buf[j]);
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        std::size_t half = len >> 1;
        std::size_t step = n / len;
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t k = 0; k < half; ++k) {
                std::complex<double> v = buf[i + k + half] * twiddle_[k * step];
                std::complex<double> u = buf[i + k];
                buf[i + k] = u + v;
                buf[i + k + half] = u - v;
            }
        }
    }
    power->resize(n / 2 + 1);
    for (std::size_t k = 0; k <= n / 2; ++k) (*power)[k] = std::norm(buf[k]);
}

double KaldiFbank::MelScale(double hz) {
    return 1127.0 * std::log(1.0 + hz / 700.0);
}

void KaldiFbank::BuildWindow() {
    int n = opts_.frame_length_samples;
    window_.resize(static_cast<std::size_t>(n));
    double a = 2.0 * kPi / (n - 1);
    for (int i = 0; i < n; ++i) {
        window_[i] = static_cast<float>(std::pow(0.5 - 0.5 * std::cos(a * i), 0.85));
    }
}

void KaldiFbank::BuildMelBanks() {
    int num_fft_bins = padded_size_ / 2;
    double nyquist = 0.5 * opts_.sample_rate;
    double high = opts_.high_freq > 0.0f ? opts_.high_freq : nyquist + opts_.high_freq;
    double bin_width = static_cast<double>(opts_.sample_rate) / padded_size_;
    double mel_low = MelScale(opts_.low_freq);
    double mel_high = MelScale(high);
    double delta = (mel_high - mel_low) / (opts_.num_bins + 1);

    bank_offset_.assign(static_cast<std::size_t>(opts_.num_bins), 0);
    bank_weights_.assign(static_cast<std::size_t>(opts_.num_bins), {});
    for (int b = 0; b < opts_.num_bins; ++b) {
        double left = mel_low + b * delta;
        double center = mel_low + (b + 1) * delta;
        double right = mel_low + (b + 2) * delta;
        std::vector<float> full(static_cast<std::size_t>(num_fft_bins), 0.0f);
        int first = -1;
        int last = -1;
        for (int i = 0; i < num_fft_bins; ++i) {
            double mel = MelScale(bin_width * i);
            if (mel > left && mel < right) {
                double w = mel <= center ? (mel - left) / (center - left)
                                         : (right - mel) / (right - center);
                full[i] = static_cast<float>(w);
                if (first < 0) first = i;
                last = i;
            }
        }
        if (first < 0) continue;
        bank_offset_[b] = first;
        bank_weights_[b].assign(full.begin() + first, full.begin() + last + 1);
    }
}

std::size_t KaldiFbank::NumFrames(std::size_t n) const {
    std::size_t shift = static_cast<std::size_t>(opts_.frame_shift_samples);
    if (n == 0) return 0;
    return (n + shift / 2) / shift;
}

void KaldiFbank::ExtractFrame(const float* pcm, std::size_t n, std::size_t index,
                              std::vector<float>* frame) const {
    long long shift = opts_.frame_shift_samples;
    long long length = opts_.frame_length_samples;
    long long start = static_cast<long long>(index) * shift + shift / 2 - length / 2;
    long long size = static_cast<long long>(n);
    frame->assign(static_cast<std::size_t>(padded_size_), 0.0f);
    for (long long i = 0; i < length; ++i) {
        long long s = start + i;
        while (s < 0 || s >= size) {
            if (s < 0) s = -s - 1;
            else s = 2 * size - 1 - s;
        }
        (*frame)[static_cast<std::size_t>(i)] = pcm[s];
    }
}

void KaldiFbank::ComputeFrame(std::vector<float>* frame, float* out) const {
    int length = opts_.frame_length_samples;
    std::vector<float>& w = *frame;
    if (opts_.remove_dc_offset) {
        double sum = 0.0;
        for (int i = 0; i < length; ++i) sum += w[i];
        float mean = static_cast<float>(sum / length);
        for (int i = 0; i < length; ++i) w[i] -= mean;
    }
    if (opts_.preemph_coeff != 0.0f) {
        for (int i = length - 1; i > 0; --i) w[i] -= opts_.preemph_coeff * w[i - 1];
        w[0] -= opts_.preemph_coeff * w[0];
    }
    for (int i = 0; i < length; ++i) w[i] *= window_[i];

    std::vector<double> power;
    PowerSpectrum(w, &power);

    for (int b = 0; b < opts_.num_bins; ++b) {
        const std::vector<float>& weights = bank_weights_[b];
        int offset = bank_offset_[b];
        double energy = 0.0;
        for (std::size_t k = 0; k < weights.size(); ++k) {
            energy += weights[k] * power[offset + k];
        }
        out[b] = static_cast<float>(std::log(std::max(energy, static_cast<double>(FLT_EPSILON))));
    }
}

std::vector<float> KaldiFbank::Compute(const float* pcm, std::size_t n) const {
    std::size_t frames = NumFrames(n);
    std::vector<float> feats(frames * static_cast<std::size_t>(opts_.num_bins));
    std::vector<float> frame;
    for (std::size_t f = 0; f < frames; ++f) {
        ExtractFrame(pcm, n, f, &frame);
        ComputeFrame(&frame, feats.data() + f * static_cast<std::size_t>(opts_.num_bins));
    }
    return feats;
}

}
