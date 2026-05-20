#ifndef VIETASR_PREPROCESS_FEATURE_PIPELINE_H
#define VIETASR_PREPROCESS_FEATURE_PIPELINE_H

#include <vector>

namespace vietasr {

struct FbankOptions final {
    int sample_rate{16000};
    int num_bins{80};
    int frame_length_ms{25};
    int frame_shift_ms{10};
    float pre_emphasis{0.97f};
    float low_freq{20.0f};
    float high_freq{-400.0f};
};

class FeaturePipeline final {
public:
    explicit FeaturePipeline(const FbankOptions& opts = {});
    void Reset();
    void AcceptWaveform(const float* pcm, std::size_t n);
    std::size_t num_frames_ready() const;
    std::vector<float> PopFrames(std::size_t n);

    int num_bins() const { return opts_.num_bins; }

private:
    FbankOptions opts_;
    int frame_len_samples_{};
    int frame_shift_samples_{};
    int n_fft_{};
    int n_freq_bins_{};
    std::vector<float> window_;
    std::vector<float> mel_filterbank_;
    std::vector<float> buffer_;
    std::vector<float> features_;

    void BuildWindow();
    void BuildMelFilterbank();
    void ComputeFrames();

    static float HzToMel(float hz);
    static float MelToHz(float mel);
};

}

#endif
