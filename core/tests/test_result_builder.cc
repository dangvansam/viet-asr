#include <gtest/gtest.h>

#include <string>

#include "vietasr/result.h"

namespace vietasr {

TEST(ResultBuilder, EmptySnapshotIsValidJson) {
    ResultBuilder builder;
    std::string json = builder.SnapshotJson();
    EXPECT_NE(json.find("\"is_final\":false"), std::string::npos);
}

TEST(ResultBuilder, SetTextAppearsInOutput) {
    ResultBuilder builder;
    builder.SetText("vietasr", "xin chào");
    std::string json = builder.SnapshotJson();
    EXPECT_NE(json.find("\"text\":\"xin chào\""), std::string::npos);
}

TEST(ResultBuilder, FinalJsonMarksFinal) {
    ResultBuilder builder;
    builder.SetText("vietasr", "hello");
    std::string json = builder.FinalJson();
    EXPECT_NE(json.find("\"is_final\":true"), std::string::npos);
}

TEST(ResultBuilder, ExtraFieldsRenderAsRawJson) {
    ResultBuilder builder;
    builder.SetField("gender", "{\"value\":\"M\",\"score\":0.9}");
    std::string json = builder.SnapshotJson();
    EXPECT_NE(json.find("\"gender\":{\"value\":\"M\",\"score\":0.9}"), std::string::npos);
}

TEST(ResultBuilder, ResetClearsState) {
    ResultBuilder builder;
    builder.SetText("vietasr", "hello");
    builder.Reset();
    std::string json = builder.SnapshotJson();
    EXPECT_EQ(json.find("hello"), std::string::npos);
}

}
