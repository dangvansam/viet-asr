#ifndef VIETASR_MODULES_VIETASR_H
#define VIETASR_MODULES_VIETASR_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "modules/vietasr/ctc_beam_search.h"
#include "modules/vietasr/ctc_endpoint.h"
#include "modules/vietasr/post_processor.h"
#include "modules/vietasr/units.h"
#include "preprocess/feature_pipeline.h"
#include "vietasr/module.h"

namespace vietasr {

// One ASR utterance ("chunk"), committed at a CTC endpoint = a sentence end.
// lead_pause_ms is the silence gap before it (drives `.` / `,` punctuation);
// start_s / end_s are its time span; text is ITN-normalized when itn is on.
struct AsrSegment final {
    std::string text;
    int lead_pause_ms;
    double start_s;
    double end_s;
};

// Joins ASR segments plus the trailing in-progress partial into one transcript.
// When pause_punctuation is true the silence gap before each segment becomes
// ". " (>= sentence_pause_ms), ", " (>= clause_pause_ms) or " "; otherwise
// segments are newline-joined (legacy behavior). Free function so it is
// unit-testable without driving audio through the model.
std::string ApplyPausePunctuation(const std::vector<AsrSegment>& segments,
                                  const std::string& partial,
                                  int partial_lead_pause_ms,
                                  bool pause_punctuation,
                                  int sentence_pause_ms,
                                  int clause_pause_ms);

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
    // Emits voiced_regions + speech derived from the CTC-blank endpointer, so a
    // streaming vietasr-only pipeline needs no separate VAD model.
    void EmitVoicedRegions(ResultBuilder* out) const;
    // Emits the segments[] array — one entry per committed sentence/chunk with
    // its (ITN-normalized) text and start/end time.
    void EmitSegments(ResultBuilder* out) const;

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
    // Every committed sentence, uncapped — the full transcript history. text
    // joins all of them, segments[] exposes each one.
    std::vector<AsrSegment> completed_segments_;
    std::string completed_join_;  // cached join of completed_segments_ text
    int silence_frames_{0};
    int total_frames_{0};
    bool has_decoded_{false};

    // Pause-driven punctuation. pause_run_frames_ counts contiguous silence
    // frames and — unlike silence_frames_ — is NOT reset by an endpoint commit,
    // so it measures the full gap between two utterances.
    int pause_run_frames_{0};
    int lead_pause_ms_{0};            // silence before the current utterance
    bool pause_punctuation_{true};    // config: insert `.` / `,` from pauses
    int sentence_pause_ms_{1500};     // config: gap >= this -> `.`
    int clause_pause_ms_{500};        // config: gap >= this -> `,`

    // Voiced-region detection from the CTC-blank endpointer (opt-in). When
    // enabled, a streaming vietasr-only pipeline emits voiced_regions + speech
    // itself — no separate VAD model in parallel. Each utterance (speech run
    // between endpoints) is one voiced region.
    bool emit_voiced_regions_{false};       // config: emit voiced_regions/speech
    bool emit_segments_{false};             // config: emit the segments[] array
    bool itn_{false};                       // config: ITN-normalize each sentence
    std::int64_t output_frame_clock_{0};    // total encoder output frames seen
    std::int64_t utt_start_frame_{0};       // first speech frame of current utt
    std::int64_t utt_last_speech_frame_{0}; // last speech frame of current utt
};

}

#endif
