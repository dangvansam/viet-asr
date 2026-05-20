#ifndef VIETASR_MODULES_VIETASR_H
#define VIETASR_MODULES_VIETASR_H

#include <memory>
#include <string>
#include <vector>

#include "modules/vietasr/ctc_beam_search.h"
#include "modules/vietasr/ctc_endpoint.h"
#include "modules/vietasr/post_processor.h"
#include "modules/vietasr/units.h"
#include "preprocess/feature_pipeline.h"
#include "vietasr/module.h"

namespace vietasr {

class VietAsrModule final : public Module {
public:
    VietAsrModule();
    ~VietAsrModule() override;

    const char* name() const override { return "vietasr"; }
    const char* version() const override { return "1"; }

    Status Init(const ModuleConfig& config,
                ModelManager* models,
                Engine* engine) override;

    std::unique_ptr<Module> Clone() const override;

    void Reset() override;
    void OnFrame(const AudioFrame& frame, ResultBuilder* out) override;
    void OnSegment(const Segment& segment, ResultBuilder* out) override;
    void OnFinalize(ResultBuilder* out) override;

    bool produces_logits() const override { return true; }
    bool produces_text() const override { return true; }

private:
    void RunOneChunk();
    void Flush();
    void CommitCurrentBeamAsSegment();
    std::string BuildTranscript() const;
    void ProcessFrameEndpoint(const float* logits, int vocab_dim);

    Engine* engine_{nullptr};
    bool model_loaded_{false};

    std::unique_ptr<FeaturePipeline> features_;
    std::unique_ptr<CtcBeamSearch> beam_search_;
    std::unique_ptr<CtcEndpoint> endpoint_;
    PostProcessor post_processor_;
    Units units_;

    std::vector<float> att_cache_;
    std::vector<float> cnn_cache_;
    std::int64_t encoder_offset_{0};

    std::int64_t samples_seen_{0};
    std::vector<std::string> completed_segments_;
    int silence_frames_{0};
    int total_frames_{0};
    bool has_decoded_{false};
};

}

#endif
