#include <gtest/gtest.h>

#include "modules/vietasr/ctc_endpoint.h"
#include "modules/vietasr/post_processor.h"

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

}
