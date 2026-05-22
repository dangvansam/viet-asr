#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include <vietasr.h>  // public C API — angle brackets to skip the sibling vietasr.h

#include "modules/vietasr/ctc_endpoint.h"
#include "modules/vietasr/post_processor.h"
#include "modules/vietasr/vietasr.h"

namespace vietasr {

TEST(CtcEndpoint, Rule3_MaxUtterance) {
    CtcEndpoint detector({5000, 1000, 20000, 0.8f});
    EXPECT_FALSE(detector.Check(0, 19999, true));
    EXPECT_TRUE(detector.Check(0, 20000, true));
}

TEST(CtcEndpoint, Rule2_TrailingSilenceAfterDecode) {
    CtcEndpoint detector({5000, 1000, 20000, 0.8f});
    EXPECT_FALSE(detector.Check(500, 1500, true));
    EXPECT_TRUE(detector.Check(1000, 1500, true));
}

TEST(CtcEndpoint, Rule1_LongInitialSilence) {
    CtcEndpoint detector({5000, 1000, 20000, 0.8f});
    EXPECT_FALSE(detector.Check(4999, 5000, false));
    EXPECT_TRUE(detector.Check(5000, 5000, false));
}

TEST(PostProcessor, BpeUnderscorePrefixBecomesSpace) {
    PostProcessor processor;
    std::vector<std::string> tokens = {
        "\xe2\x96\x81" "xin",
        "\xe2\x96\x81" "ch" "\xc3\xa0" "o",
        "\xe2\x96\x81" "b" "\xe1\xba\xa1" "n"
    };
    EXPECT_EQ(processor.Detokenize(tokens), "xin ch" "\xc3\xa0" "o b" "\xe1\xba\xa1" "n");
}

TEST(PostProcessor, StripsUnkAndContextTokens) {
    PostProcessor processor;
    std::vector<std::string> tokens = {
        "\xe2\x96\x81" "hello", "<unk>", "\xe2\x96\x81" "world", "<context>", "</context>"
    };
    EXPECT_EQ(processor.Detokenize(tokens), "hello world");
}

// --------------------------------------------------------------------------
// Pause-driven punctuation: long pause -> ". ", short pause -> ", ".
// --------------------------------------------------------------------------

TEST(PausePunctuation, LongPauseBecomesPeriod) {
    std::vector<AsrSegment> segs = {{"xin chao", 0}, {"ban khoe khong", 1800}};
    EXPECT_EQ(ApplyPausePunctuation(segs, "", 0, true, 1500, 500),
              "xin chao. ban khoe khong");
}

TEST(PausePunctuation, ShortPauseBecomesComma) {
    std::vector<AsrSegment> segs = {{"hom nay", 0}, {"troi dep", 800}};
    EXPECT_EQ(ApplyPausePunctuation(segs, "", 0, true, 1500, 500),
              "hom nay, troi dep");
}

TEST(PausePunctuation, TinyPauseStaysSpace) {
    std::vector<AsrSegment> segs = {{"mot", 0}, {"hai", 100}};
    EXPECT_EQ(ApplyPausePunctuation(segs, "", 0, true, 1500, 500), "mot hai");
}

TEST(PausePunctuation, MixedPauses) {
    std::vector<AsrSegment> segs = {
        {"cau mot", 0}, {"ve ngan", 700}, {"cau hai", 2000}};
    EXPECT_EQ(ApplyPausePunctuation(segs, "", 0, true, 1500, 500),
              "cau mot, ve ngan. cau hai");
}

TEST(PausePunctuation, AppendsTrailingPartial) {
    std::vector<AsrSegment> segs = {{"xong roi", 0}};
    EXPECT_EQ(ApplyPausePunctuation(segs, "dang noi", 1800, true, 1500, 500),
              "xong roi. dang noi");
}

TEST(PausePunctuation, DisabledJoinsWithNewline) {
    std::vector<AsrSegment> segs = {{"a", 0}, {"b", 3000}};
    EXPECT_EQ(ApplyPausePunctuation(segs, "", 0, false, 1500, 500), "a\nb");
}

TEST(PausePunctuation, ThresholdsAreConfigurable) {
    std::vector<AsrSegment> segs = {{"x", 0}, {"y", 1000}};
    // A 1 s gap is a comma by default...
    EXPECT_EQ(ApplyPausePunctuation(segs, "", 0, true, 1500, 500), "x, y");
    // ...but a sentence break once sentence_pause_ms is lowered to 1 s.
    EXPECT_EQ(ApplyPausePunctuation(segs, "", 0, true, 1000, 300), "x. y");
}

TEST(PausePunctuation, EmptyInputIsEmpty) {
    EXPECT_EQ(ApplyPausePunctuation({}, "", 0, true, 1500, 500), "");
}

// --------------------------------------------------------------------------
// Long-stream regression: the encoder's positional-encoding table is finite
// (~3.6 min); a longer stream must re-anchor instead of crashing with an ONNX
// Gather out-of-bounds. Heavy (~500 model inferences) — opt in by setting
// VIETASR_SLOW_TESTS=1.
// --------------------------------------------------------------------------

TEST(VietAsrStreaming, LongStreamReanchorsEncoderInsteadOfCrashing) {
    if (std::getenv("VIETASR_SLOW_TESTS") == nullptr) {
        GTEST_SKIP() << "set VIETASR_SLOW_TESTS=1 to run (~20 s)";
    }
    VietasrPipeline* pipeline = vietasr_pipeline_preset("transcribe");
    ASSERT_NE(pipeline, nullptr);
    VietasrSession* session = vietasr_session_new(pipeline, 16000.0f);
    ASSERT_NE(session, nullptr);

    // 240 s of audio — past the ~216 s positional-encoding window, so the
    // encoder must re-anchor at least once.
    std::vector<short> one_second(16000, 0);
    for (int sec = 0; sec < 240; ++sec) {
        vietasr_accept_waveform_s16(session, one_second.data(),
                                    static_cast<int>(one_second.size()));
    }
    // Completes without the ONNX Gather overflow.
    EXPECT_NE(vietasr_final_result(session), nullptr);

    vietasr_session_free(session);
    vietasr_pipeline_free(pipeline);
}

}
