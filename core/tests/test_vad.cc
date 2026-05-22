#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "modules/vad/vad_segmenter.h"
#include "utils/json.h"
#include "vietasr.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kChunkDur = 512.0 / 16000.0;  // 32 ms

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// Build a single-vad pipeline configured by `cfg`, run `pcm` through it in
// batch mode, and return the result JSON.
std::string RunVad(const std::string& cfg, const std::vector<short>& pcm) {
    VietasrPipeline* pipeline = vietasr_pipeline_new();
    EXPECT_NE(pipeline, nullptr);
    EXPECT_EQ(vietasr_pipeline_add_module(pipeline, "vad", cfg.c_str()),
              VIETASR_OK);
    EXPECT_EQ(vietasr_pipeline_build(pipeline), VIETASR_OK);
    const char* result = vietasr_transcribe_buffer(
        pipeline, pcm.data(), static_cast<int>(pcm.size()), 16000.0f);
    std::string json = result ? result : "";
    vietasr_pipeline_free(pipeline);
    return json;
}

std::vector<short> Silence(int n) { return std::vector<short>(n, 0); }

// A loud 300 Hz tone — high energy, so the energy backend marks it voiced.
std::vector<short> Tone(int n) {
    std::vector<short> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = static_cast<short>(
            8000.0 * std::sin(2.0 * kPi * 300.0 * i / 16000.0));
    }
    return v;
}

// silence | tone | silence, each `seg` samples long.
std::vector<short> SilenceToneSilence(int seg) {
    std::vector<short> v;
    auto s1 = Silence(seg), t = Tone(seg), s2 = Silence(seg);
    v.insert(v.end(), s1.begin(), s1.end());
    v.insert(v.end(), t.begin(), t.end());
    v.insert(v.end(), s2.begin(), s2.end());
    return v;
}

}  // namespace

// --------------------------------------------------------------------------
// JSON config parser
// --------------------------------------------------------------------------

TEST(VadJson, EmptyObjectYieldsNoKeys) {
    std::string s;
    double d;
    bool b;
    EXPECT_FALSE(vietasr::JsonGetString("{}", "backend", &s));
    EXPECT_FALSE(vietasr::JsonGetNumber("{}", "threshold", &d));
    EXPECT_FALSE(vietasr::JsonGetBool("{}", "flag", &b));
}

TEST(VadJson, ReadsScalars) {
    const std::string json =
        R"({"backend":"silero","threshold":0.7,"hop_size":160,"on":true})";
    std::string backend;
    ASSERT_TRUE(vietasr::JsonGetString(json, "backend", &backend));
    EXPECT_EQ(backend, "silero");

    double threshold = 0.0;
    ASSERT_TRUE(vietasr::JsonGetNumber(json, "threshold", &threshold));
    EXPECT_NEAR(threshold, 0.7, 1e-9);

    double hop = 0.0;
    ASSERT_TRUE(vietasr::JsonGetNumber(json, "hop_size", &hop));
    EXPECT_EQ(static_cast<int>(hop), 160);

    bool on = false;
    ASSERT_TRUE(vietasr::JsonGetBool(json, "on", &on));
    EXPECT_TRUE(on);
}

TEST(VadJson, SkipsNestedObjectsAndMissingKeys) {
    const std::string json = R"({"nested":{"x":1},"backend":"tenvad"})";
    std::string backend;
    ASSERT_TRUE(vietasr::JsonGetString(json, "backend", &backend));
    EXPECT_EQ(backend, "tenvad");

    std::string missing;
    EXPECT_FALSE(vietasr::JsonGetString(json, "absent", &missing));
}

// --------------------------------------------------------------------------
// VadSegmenter
// --------------------------------------------------------------------------

namespace {

// Feed `n` chunks at `prob`, advancing the clock; returns next chunk index.
int Feed(vietasr::vad::VadSegmenter* seg, int from, int n, float prob) {
    for (int i = 0; i < n; ++i) {
        seg->Accept(prob, (from + i) * kChunkDur, kChunkDur);
    }
    return from + n;
}

}  // namespace

TEST(VadSegmenter, EmitsOneRegionForContiguousSpeech) {
    vietasr::vad::VadConfig cfg;  // defaults
    vietasr::vad::VadSegmenter seg(cfg);
    int idx = 0;
    idx = Feed(&seg, idx, 10, 0.0f);   // 320 ms silence
    idx = Feed(&seg, idx, 30, 1.0f);   // ~960 ms speech
    idx = Feed(&seg, idx, 20, 0.0f);   // 640 ms silence

    auto regions = seg.Finish(idx * kChunkDur);
    ASSERT_EQ(regions.size(), 1u);
    EXPECT_LT(regions[0].start, regions[0].end);
    EXPECT_LT(regions[0].start, 0.5);
    EXPECT_GT(regions[0].end, 1.0);
}

TEST(VadSegmenter, DropsRegionShorterThanMinSpeech) {
    vietasr::vad::VadConfig cfg;
    cfg.min_speech_ms = 200;          // 200 ms floor
    cfg.speech_pad_ms = 0;
    vietasr::vad::VadSegmenter seg(cfg);
    int idx = 0;
    idx = Feed(&seg, idx, 5, 0.0f);
    idx = Feed(&seg, idx, 2, 1.0f);   // 64 ms speech — below the floor
    idx = Feed(&seg, idx, 20, 0.0f);

    auto regions = seg.Finish(idx * kChunkDur);
    EXPECT_TRUE(regions.empty());
}

TEST(VadSegmenter, MergesRegionsWithinMinSilence) {
    vietasr::vad::VadConfig cfg;
    cfg.hangover_ms = 100;            // each burst closes after 100 ms silence
    cfg.min_silence_ms = 10000;       // ...but gaps under 10 s are merged
    cfg.speech_pad_ms = 0;
    vietasr::vad::VadSegmenter seg(cfg);
    int idx = 0;
    idx = Feed(&seg, idx, 20, 1.0f);  // burst 1
    idx = Feed(&seg, idx, 10, 0.0f);  // 320 ms gap (closes burst 1)
    idx = Feed(&seg, idx, 20, 1.0f);  // burst 2

    auto regions = seg.Finish(idx * kChunkDur);
    EXPECT_EQ(regions.size(), 1u);
}

TEST(VadSegmenter, KeepsSeparateRegionsWhenGapExceedsMinSilence) {
    vietasr::vad::VadConfig cfg;
    cfg.hangover_ms = 100;
    cfg.min_silence_ms = 50;
    cfg.speech_pad_ms = 0;
    vietasr::vad::VadSegmenter seg(cfg);
    int idx = 0;
    idx = Feed(&seg, idx, 20, 1.0f);
    idx = Feed(&seg, idx, 15, 0.0f);  // 480 ms gap — well above 50 ms
    idx = Feed(&seg, idx, 20, 1.0f);

    auto regions = seg.Finish(idx * kChunkDur);
    EXPECT_EQ(regions.size(), 2u);
}

// --------------------------------------------------------------------------
// Pipeline integration — energy backend
// --------------------------------------------------------------------------

TEST(VadModule, DefaultConfigEmitsVoicedRegionsField) {
    std::string json = RunVad("{}", Silence(32000));
    EXPECT_TRUE(Contains(json, "voiced_regions"));
    EXPECT_TRUE(Contains(json, "is_final"));
}

TEST(VadModule, EnergyBackendSilenceHasNoRegions) {
    std::string json = RunVad(R"({"backend":"energy"})", Silence(32000));
    EXPECT_TRUE(Contains(json, "\"voiced_regions\":[]"));
}

TEST(VadModule, SilenceClassifiedAsNoSpeech) {
    std::string json = RunVad(R"({"backend":"energy"})", Silence(32000));
    EXPECT_TRUE(Contains(json, "\"detected\":false"));
}

TEST(VadModule, LoudAudioClassifiedAsSpeech) {
    std::string json = RunVad(R"({"backend":"energy"})",
                              SilenceToneSilence(16000));
    EXPECT_TRUE(Contains(json, "\"speech\""));
    EXPECT_TRUE(Contains(json, "\"detected\":true"));
    EXPECT_TRUE(Contains(json, "\"voiced_ratio\""));
}

TEST(VadModule, EnergyBackendDetectsLoudSpeechRegion) {
    std::string json = RunVad(R"({"backend":"energy"})",
                              SilenceToneSilence(16000));
    EXPECT_TRUE(Contains(json, "voiced_regions"));
    EXPECT_TRUE(Contains(json, "\"start\""));
    EXPECT_TRUE(Contains(json, "\"end\""));
}

TEST(VadModule, UnknownBackendFallsBackToEnergy) {
    std::string json = RunVad(R"({"backend":"does-not-exist"})",
                              SilenceToneSilence(16000));
    // Still produces a valid, non-empty detection rather than failing.
    EXPECT_TRUE(Contains(json, "\"voiced_regions\":["));
    EXPECT_TRUE(Contains(json, "\"start\""));
}

// --------------------------------------------------------------------------
// Pipeline integration — neural backends
//
// Synthetic audio is not real speech, so these assert structural validity
// (the pipeline runs and emits a well-formed voiced_regions array) rather
// than exact boundaries.
// --------------------------------------------------------------------------

TEST(VadModule, TenVadBackendRunsAndEmitsField) {
    std::string json = RunVad(R"({"backend":"tenvad"})",
                              SilenceToneSilence(16000));
    EXPECT_TRUE(Contains(json, "\"voiced_regions\":["));
}

TEST(VadModule, SileroBackendRunsAndEmitsField) {
    std::string json = RunVad(R"({"backend":"silero"})",
                              SilenceToneSilence(16000));
    EXPECT_TRUE(Contains(json, "\"voiced_regions\":["));
}

// --------------------------------------------------------------------------
// VAD-only preset — classify audio as speech / no-speech without ASR.
// --------------------------------------------------------------------------

TEST(VadPreset, ListedAmongPresets) {
    std::string presets = vietasr_list_presets();
    EXPECT_TRUE(Contains(presets, "vad"));
}

TEST(VadPreset, EmitsSpeechClassificationAndNoTranscript) {
    VietasrPipeline* pipeline = vietasr_pipeline_preset("vad");
    ASSERT_NE(pipeline, nullptr);
    auto pcm = SilenceToneSilence(16000);
    const char* result = vietasr_transcribe_buffer(
        pipeline, pcm.data(), static_cast<int>(pcm.size()), 16000.0f);
    ASSERT_NE(result, nullptr);
    std::string json = result;
    // Carries the speech classification + regions, but no ASR transcript.
    EXPECT_TRUE(Contains(json, "\"speech\""));
    EXPECT_TRUE(Contains(json, "\"detected\""));
    EXPECT_TRUE(Contains(json, "\"voiced_regions\":["));
    EXPECT_FALSE(Contains(json, "\"text\""));
    vietasr_pipeline_free(pipeline);
}

TEST(VadPreset, SilenceThroughPresetIsNoSpeech) {
    VietasrPipeline* pipeline = vietasr_pipeline_preset("vad");
    ASSERT_NE(pipeline, nullptr);
    auto pcm = Silence(32000);
    const char* result = vietasr_transcribe_buffer(
        pipeline, pcm.data(), static_cast<int>(pcm.size()), 16000.0f);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(Contains(std::string(result), "\"detected\":false"));
    vietasr_pipeline_free(pipeline);
}

// --------------------------------------------------------------------------
// Streaming mode — VAD results update live via partial()/result().
// --------------------------------------------------------------------------

namespace {

VietasrPipeline* BuildVadPipeline(const char* cfg) {
    VietasrPipeline* pipeline = vietasr_pipeline_new();
    EXPECT_NE(pipeline, nullptr);
    EXPECT_EQ(vietasr_pipeline_add_module(pipeline, "vad", cfg), VIETASR_OK);
    EXPECT_EQ(vietasr_pipeline_build(pipeline), VIETASR_OK);
    return pipeline;
}

}  // namespace

TEST(VadStreaming, PartialResultReflectsSpeechMidStream) {
    VietasrPipeline* pipeline = BuildVadPipeline(R"({"backend":"energy"})");
    VietasrSession* session = vietasr_session_new(pipeline, 16000.0f);
    ASSERT_NE(session, nullptr);

    // Stream 1.6 s of loud audio in 100 ms chunks.
    auto chunk = Tone(1600);
    for (int i = 0; i < 16; ++i) {
        vietasr_accept_waveform_s16(session, chunk.data(),
                                    static_cast<int>(chunk.size()));
    }

    // The partial result — fetched before finalizing — already carries the
    // speech classification, not just the final result.
    std::string partial = vietasr_partial_result(session);
    EXPECT_TRUE(Contains(partial, "\"speech\""));
    EXPECT_TRUE(Contains(partial, "\"voiced_regions\""));
    EXPECT_TRUE(Contains(partial, "\"detected\":true"));

    std::string final = vietasr_final_result(session);
    EXPECT_TRUE(Contains(final, "\"detected\":true"));

    vietasr_session_free(session);
    vietasr_pipeline_free(pipeline);
}

TEST(VadStreaming, SilenceStreamClassifiedAsNoSpeech) {
    VietasrPipeline* pipeline = BuildVadPipeline(R"({"backend":"energy"})");
    VietasrSession* session = vietasr_session_new(pipeline, 16000.0f);
    ASSERT_NE(session, nullptr);

    auto chunk = Silence(1600);
    for (int i = 0; i < 10; ++i) {
        vietasr_accept_waveform_s16(session, chunk.data(),
                                    static_cast<int>(chunk.size()));
    }
    std::string final = vietasr_final_result(session);
    EXPECT_TRUE(Contains(final, "\"detected\":false"));

    vietasr_session_free(session);
    vietasr_pipeline_free(pipeline);
}

TEST(VadStreaming, NeuralBackendRunsInStreamingMode) {
    VietasrPipeline* pipeline = BuildVadPipeline(R"({"backend":"silero"})");
    VietasrSession* session = vietasr_session_new(pipeline, 16000.0f);
    ASSERT_NE(session, nullptr);

    auto chunk = SilenceToneSilence(4000);
    vietasr_accept_waveform_s16(session, chunk.data(),
                                static_cast<int>(chunk.size()));
    std::string final = vietasr_final_result(session);
    EXPECT_TRUE(Contains(final, "\"speech\""));
    EXPECT_TRUE(Contains(final, "\"voiced_regions\":["));

    vietasr_session_free(session);
    vietasr_pipeline_free(pipeline);
}

// --------------------------------------------------------------------------
// Configurable parameters — JSON config flows through to behavior.
// --------------------------------------------------------------------------

TEST(VadConfig, MinSpeechMsIsConfigurable) {
    auto audio = SilenceToneSilence(16000);  // 1 s of voiced audio
    // The default 100 ms floor keeps a 1 s region.
    EXPECT_TRUE(Contains(RunVad(R"({"backend":"energy"})", audio),
                         "\"detected\":true"));
    // Raising min_speech_ms to 10 s drops it — proving the param took effect.
    EXPECT_TRUE(Contains(
        RunVad(R"({"backend":"energy","min_speech_ms":10000})", audio),
        "\"detected\":false"));
}

TEST(VadConfig, AllKnobsAcceptedWithoutError) {
    // Every documented parameter, set at once, must parse and run cleanly.
    const char* cfg =
        R"({"backend":"energy","threshold":0.4,"hop_size":160,)"
        R"("hangover_ms":150,"min_speech_ms":80,"min_silence_ms":120,)"
        R"("speech_pad_ms":50})";
    std::string json = RunVad(cfg, SilenceToneSilence(16000));
    EXPECT_TRUE(Contains(json, "\"voiced_regions\":["));
    EXPECT_TRUE(Contains(json, "\"speech\""));
}

// --------------------------------------------------------------------------
// Clone isolation — two sessions of one pipeline must not share VAD state.
// --------------------------------------------------------------------------

TEST(VadModule, SessionsAreIndependent) {
    VietasrPipeline* pipeline = vietasr_pipeline_new();
    ASSERT_NE(pipeline, nullptr);
    ASSERT_EQ(vietasr_pipeline_add_module(pipeline, "vad",
                                          R"({"backend":"energy"})"),
              VIETASR_OK);
    ASSERT_EQ(vietasr_pipeline_build(pipeline), VIETASR_OK);

    VietasrSession* quiet = vietasr_session_new(pipeline, 16000.0f);
    VietasrSession* loud = vietasr_session_new(pipeline, 16000.0f);
    ASSERT_NE(quiet, nullptr);
    ASSERT_NE(loud, nullptr);

    auto silence = Silence(32000);
    auto tone = SilenceToneSilence(16000);
    vietasr_accept_waveform_s16(quiet, silence.data(),
                                static_cast<int>(silence.size()));
    vietasr_accept_waveform_s16(loud, tone.data(),
                                static_cast<int>(tone.size()));

    std::string quiet_json = vietasr_final_result(quiet);
    std::string loud_json = vietasr_final_result(loud);

    EXPECT_TRUE(Contains(quiet_json, "\"voiced_regions\":[]"));
    EXPECT_TRUE(Contains(loud_json, "\"start\""));

    vietasr_session_free(quiet);
    vietasr_session_free(loud);
    vietasr_pipeline_free(pipeline);
}
