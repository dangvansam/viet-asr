// Golden tests for the Vietnamese ITN module. The expected outputs were
// cross-checked against FunASR's reference Python InverseNormalizer(lang="vi")
// using the same grammar that scripts/gen_itn_fst.py compiles.

#include <gtest/gtest.h>

#include <string>

#include "modules/itn/embedded_itn.h"
#include "modules/itn/itn.h"
#include "modules/itn/processor.h"
#include "vietasr/result.h"

namespace {

// One processor shared across the ItnProcessor cases (loading is not free).
vietasr::itn::ItnProcessor& Processor() {
    static vietasr::itn::ItnProcessor proc;
    static const bool loaded = [] {
        const auto tagger = vietasr::itn::EmbeddedItnTagger();
        const auto verbalizer = vietasr::itn::EmbeddedItnVerbalizer();
        return proc.Load(tagger.data, tagger.size,
                         verbalizer.data, verbalizer.size);
    }();
    EXPECT_TRUE(loaded);
    return proc;
}

}  // namespace

TEST(ItnEmbeddedGrammar, IsBakedIntoLibrary) {
    EXPECT_FALSE(vietasr::itn::EmbeddedItnTagger().empty());
    EXPECT_FALSE(vietasr::itn::EmbeddedItnVerbalizer().empty());
}

TEST(ItnProcessor, LoadsGrammar) {
    EXPECT_TRUE(Processor().ready());
}

TEST(ItnProcessor, ConvertsCurrency) {
    EXPECT_EQ(Processor().Normalize("tôi có hai mươi nghìn đồng"),
              "tôi có 20000₫");
}

TEST(ItnProcessor, ConvertsTime) {
    EXPECT_EQ(Processor().Normalize("lúc tám giờ ba mươi phút"), "lúc 8:30");
}

TEST(ItnProcessor, ConvertsDate) {
    EXPECT_EQ(
        Processor().Normalize(
            "ngày hai mươi tháng năm năm hai nghìn không trăm hai mươi sáu"),
        "ngày 20 tháng 5 năm 2026");
}

TEST(ItnProcessor, ConvertsPercent) {
    EXPECT_EQ(Processor().Normalize("năm phần trăm"), "5 %");
}

TEST(ItnProcessor, PassesPlainTextThrough) {
    EXPECT_EQ(Processor().Normalize("xin chào bạn"), "xin chào bạn");
}

TEST(ItnProcessor, HandlesEmptyInput) {
    EXPECT_EQ(Processor().Normalize(""), "");
}

TEST(ItnModule, RewritesCanonicalTranscript) {
    vietasr::ItnModule module;
    ASSERT_TRUE(module.Init(vietasr::ModuleConfig{}, nullptr, nullptr).ok());

    vietasr::ResultBuilder builder;
    builder.SetText("vietasr", "tôi có hai mươi nghìn đồng");
    module.OnFinalize(&builder);

    EXPECT_EQ(builder.CurrentText(), "tôi có 20000₫");

    const std::string json = builder.SnapshotJson();
    EXPECT_NE(json.find("\"applied\":true"), std::string::npos);
}

TEST(ItnModule, LeavesPlainTextUnchanged) {
    vietasr::ItnModule module;
    ASSERT_TRUE(module.Init(vietasr::ModuleConfig{}, nullptr, nullptr).ok());

    vietasr::ResultBuilder builder;
    builder.SetText("vietasr", "xin chào bạn");
    module.OnFinalize(&builder);

    EXPECT_EQ(builder.CurrentText(), "xin chào bạn");
    const std::string json = builder.SnapshotJson();
    EXPECT_NE(json.find("\"applied\":false"), std::string::npos);
}

TEST(ItnModule, ClonedInstanceSharesGrammar) {
    vietasr::ItnModule module;
    ASSERT_TRUE(module.Init(vietasr::ModuleConfig{}, nullptr, nullptr).ok());

    auto clone = module.Clone();
    ASSERT_NE(clone, nullptr);

    vietasr::ResultBuilder builder;
    builder.SetText("vietasr", "lúc tám giờ ba mươi phút");
    clone->OnFinalize(&builder);

    EXPECT_EQ(builder.CurrentText(), "lúc 8:30");
}
