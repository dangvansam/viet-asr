#ifndef VIETASR_MODULES_VIETASR_H
#define VIETASR_MODULES_VIETASR_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "modules/vietasr/post_processor.h"
#include "modules/vietasr/transducer_model.h"
#include "modules/vietasr/transducer_decoder.h"
#include "modules/vietasr/units.h"
#include "modules/vietasr/utterance_segmenter.h"
#include "preprocess/kaldi_fbank.h"
#include "vietasr/module.h"

namespace vietasr {

struct WordTiming final {
    std::string word;
    double start_s;
    double end_s;
};

struct Recognition final {
    std::string text;
    std::vector<WordTiming> words;
};

struct AsrSegment final {
    std::string text;
    double start_s;
    double end_s;
    std::vector<WordTiming> words;
};

struct VietAsrConfig final {
    int max_active_paths{4};
    int num_threads{1};
    int partial_interval_ms{640};
    bool emit_segments{false};
    bool emit_voiced_regions{false};
    bool itn{false};
    bool emit_words{false};
    std::string model_dir;
    UtteranceSegmenterConfig segmenter;

    static VietAsrConfig Parse(const std::string& json);
};

class SignalLevel final {
public:
    static double Rms(const std::vector<float>& pcm);
};

class LongAudioSplitter final {
public:
    static std::vector<std::pair<std::size_t, std::size_t>> Split(const std::vector<float>& pcm,
                                                                  std::size_t max_samples,
                                                                  std::size_t search_samples);
};

class WordGrouper final {
public:
    static std::vector<WordTiming> Group(const std::vector<std::string>& tokens,
                                         const std::vector<int>& frames,
                                         double frame_seconds,
                                         double offset_s,
                                         double window_end_s,
                                         const PostProcessor& post);
};

class SegmentJoiner final {
public:
    static std::string Join(const std::vector<AsrSegment>& segments, const std::string& partial);
};

class VietAsrRecognizer final {
public:
    VietAsrRecognizer(std::shared_ptr<const TransducerModel> model,
                      std::shared_ptr<const Units> units,
                      int max_active_paths);

    Recognition Transcribe(const std::vector<float>& pcm, double offset_s) const;

private:
    Recognition TranscribeWindow(const std::vector<float>& pcm, double offset_s) const;

    std::shared_ptr<const TransducerModel> model_;
    std::shared_ptr<const Units> units_;
    KaldiFbank fbank_;
    TransducerDecoder decoder_;
    PostProcessor post_processor_;
};

class VietAsrModule final : public Module {
public:
    VietAsrModule();
    ~VietAsrModule() override;

    const char* name() const override { return "vietasr"; }
    const char* version() const override { return "2"; }

    Status Init(const ModuleConfig& config, ModelManager* models, Engine* engine) override;
    std::unique_ptr<Module> Clone() const override;

    void Reset() override;
    void OnFrame(const AudioFrame& frame, ResultBuilder* out) override;
    void OnFinalize(ResultBuilder* out) override;

    bool produces_text() const override { return true; }

private:
    void DecodeCompleted();
    void CommitText(Recognition recognition, double start_s, double end_s);
    void UpdatePartial();
    void Emit(ResultBuilder* out) const;
    void EmitSegments(ResultBuilder* out) const;
    void EmitVoicedRegions(ResultBuilder* out) const;
    double SamplesToSeconds(std::int64_t samples) const;

    VietAsrConfig config_;
    std::shared_ptr<const TransducerModel> model_;
    std::shared_ptr<const Units> units_;
    std::unique_ptr<VietAsrRecognizer> recognizer_;
    std::unique_ptr<UtteranceSegmenter> segmenter_;

    std::vector<AsrSegment> segments_;
    std::string partial_;
    std::size_t partial_decoded_samples_{0};
};

}

#endif
