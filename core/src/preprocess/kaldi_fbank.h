#ifndef VIETASR_PREPROCESS_KALDI_FBANK_H
#define VIETASR_PREPROCESS_KALDI_FBANK_H

#include <complex>
#include <cstddef>
#include <vector>

namespace vietasr {

struct KaldiFbankOptions final {
    int sample_rate{16000};
    int num_bins{80};
    int frame_length_samples{400};
    int frame_shift_samples{160};
    float preemph_coeff{0.97f};
    float low_freq{20.0f};
    float high_freq{-400.0f};
    bool remove_dc_offset{true};
};

class KaldiFbank final {
public:
    explicit KaldiFbank(const KaldiFbankOptions& opts = {});

    std::vector<float> Compute(const float* pcm, std::size_t n) const;

    std::size_t NumFrames(std::size_t n) const;
    int num_bins() const { return opts_.num_bins; }

private:
    void BuildWindow();
    void BuildMelBanks();
    void ExtractFrame(const float* pcm, std::size_t n, std::size_t index,
                      std::vector<float>* frame) const;
    void ComputeFrame(std::vector<float>* frame, float* out) const;
    void PowerSpectrum(const std::vector<float>& input, std::vector<double>* power) const;

    static double MelScale(double hz);

    KaldiFbankOptions opts_;
    int padded_size_{512};
    std::vector<float> window_;
    std::vector<std::complex<double>> twiddle_;
    std::vector<int> bank_offset_;
    std::vector<std::vector<float>> bank_weights_;
};

}

#endif
