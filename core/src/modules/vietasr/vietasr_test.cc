#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <vietasr.h>

#include "modules/vietasr/post_processor.h"
#include "modules/vietasr/transducer_search.h"
#include "modules/vietasr/utterance_segmenter.h"
#include "modules/vietasr/vietasr.h"
#include "preprocess/kaldi_fbank.h"

namespace vietasr {

TEST(PostProcessor, BpeUnderscorePrefixBecomesSpace) {
    PostProcessor processor;
    std::vector<std::string> tokens = {"\xe2\x96\x81" "Xin", "\xe2\x96\x81" "ch\xc3\xa0o", "."};
    EXPECT_EQ(processor.Detokenize(tokens), "Xin ch\xc3\xa0o.");
}

TEST(PostProcessor, ByteFallbackTokensDecodeToUtf8) {
    PostProcessor processor;
    std::vector<std::string> tokens = {"\xe2\x96\x81" "a", "<0xE1>", "<0xBA>", "<0xA1>"};
    EXPECT_EQ(processor.Detokenize(tokens), "a\xe1\xba\xa1");
}

TEST(PostProcessor, DropsSpecialTokens) {
    PostProcessor processor;
    std::vector<std::string> tokens = {"<blk>", "\xe2\x96\x81" "hello", "<unk>", "<sos/eos>"};
    EXPECT_EQ(processor.Detokenize(tokens), "hello");
}

TEST(SegmentJoiner, JoinsSegmentsAndPartialWithSpaces) {
    std::vector<AsrSegment> segs = {{"Xin ch\xc3\xa0o.", 0.0, 1.0}, {"T\xc3\xb4i l\xc3\xa0 An.", 1.5, 2.5}};
    EXPECT_EQ(SegmentJoiner::Join(segs, "\xc4\x90" "ang"), "Xin ch\xc3\xa0o. T\xc3\xb4i l\xc3\xa0 An. \xc4\x90" "ang");
    EXPECT_EQ(SegmentJoiner::Join({}, ""), "");
}

TEST(KaldiFbank, FrameCountMatchesSnipEdgesFalse) {
    KaldiFbank fbank;
    EXPECT_EQ(fbank.NumFrames(16000), 100u);
    EXPECT_EQ(fbank.NumFrames(16079), 100u);
    EXPECT_EQ(fbank.NumFrames(16080), 101u);
    std::vector<float> pcm(16000);
    for (std::size_t i = 0; i < pcm.size(); ++i) pcm[i] = 0.1f * std::sin(0.05f * static_cast<float>(i));
    std::vector<float> feats = fbank.Compute(pcm.data(), pcm.size());
    ASSERT_EQ(feats.size(), 100u * 80u);
    for (float v : feats) EXPECT_TRUE(std::isfinite(v));
}

TEST(TransducerBeamSearch, PicksArgmaxAndSkipsBlank) {
    TransducerBeamSearch search(TransducerSearchConfig{1, 2, 0, 2});
    search.Begin();
    std::vector<std::int64_t> pending = search.PendingContexts();
    ASSERT_EQ(pending, (std::vector<std::int64_t>{-1, 0}));
    search.AcceptDecoderOutputs(pending, std::vector<float>(4, 0.0f));
    std::vector<float> enc(4, 0.0f);
    std::vector<float> enc_rows;
    std::vector<float> dec_rows;
    ASSERT_EQ(search.BuildJoinerInputs(enc.data(), 4, &enc_rows, &dec_rows), 1);
    search.AcceptLogits({0.0f, 0.0f, 0.0f, 0.0f, 9.0f}, 1);
    ASSERT_EQ(search.BuildJoinerInputs(enc.data(), 4, &enc_rows, &dec_rows), 0);
    pending = search.PendingContexts();
    ASSERT_EQ(pending, (std::vector<std::int64_t>{0, 4}));
    search.AcceptDecoderOutputs(pending, std::vector<float>(4, 0.0f));
    ASSERT_EQ(search.BuildJoinerInputs(enc.data(), 4, &enc_rows, &dec_rows), 1);
    search.AcceptLogits({9.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 1);
    EXPECT_EQ(search.BestTokens(), (std::vector<int>{4}));
}

TEST(TransducerBeamSearch, UnknownTokenIsNeverEmitted) {
    TransducerBeamSearch search(TransducerSearchConfig{2, 2, 0, 2});
    search.Begin();
    search.AcceptDecoderOutputs(search.PendingContexts(), std::vector<float>(2, 0.0f));
    std::vector<float> enc(2, 0.0f);
    std::vector<float> enc_rows;
    std::vector<float> dec_rows;
    ASSERT_EQ(search.BuildJoinerInputs(enc.data(), 2, &enc_rows, &dec_rows), 1);
    search.AcceptLogits({0.0f, 0.0f, 9.0f, 0.0f}, 1);
    EXPECT_TRUE(search.BestTokens().empty());
}

TEST(UtteranceSegmenter, SilenceProducesNoUtterance) {
    UtteranceSegmenterConfig cfg;
    cfg.backend = "energy";
    UtteranceSegmenter segmenter(cfg);
    std::vector<float> silence(16000 * 3, 0.0f);
    segmenter.Push(silence.data(), silence.size());
    segmenter.Finish();
    EXPECT_TRUE(segmenter.PopCompleted().empty());
    EXPECT_FALSE(segmenter.in_speech());
}

TEST(UtteranceSegmenter, LongSpeechIsSplitBelowMaxLength) {
    UtteranceSegmenterConfig cfg;
    cfg.backend = "energy";
    cfg.max_utterance_ms = 4000;
    UtteranceSegmenter segmenter(cfg);
    std::vector<float> tone(16000 * 10);
    for (std::size_t i = 0; i < tone.size(); ++i) tone[i] = 0.5f * std::sin(0.2f * static_cast<float>(i));
    segmenter.Push(tone.data(), tone.size());
    segmenter.Finish();
    std::vector<Utterance> utts = segmenter.PopCompleted();
    ASSERT_GE(utts.size(), 2u);
    std::size_t total = 0;
    for (const auto& u : utts) {
        EXPECT_LE(u.pcm.size(), 4000u * 16u);
        total += u.pcm.size();
    }
    EXPECT_GE(total, tone.size() - 16000u);
}

TEST(VietAsrStreaming, LongSilentStreamCompletes) {
    if (std::getenv("VIETASR_SLOW_TESTS") == nullptr) {
        GTEST_SKIP() << "set VIETASR_SLOW_TESTS=1 to run";
    }
    VietasrPipeline* pipeline = vietasr_pipeline_preset("transcribe");
    ASSERT_NE(pipeline, nullptr);
    VietasrSession* session = vietasr_session_new(pipeline, 16000.0f);
    ASSERT_NE(session, nullptr);
    std::vector<short> one_second(16000, 0);
    for (int sec = 0; sec < 240; ++sec) {
        vietasr_accept_waveform_s16(session, one_second.data(), static_cast<int>(one_second.size()));
    }
    EXPECT_NE(vietasr_final_result(session), nullptr);
    vietasr_session_free(session);
    vietasr_pipeline_free(pipeline);
}

}
