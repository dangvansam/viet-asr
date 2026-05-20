#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "vietasr.h"

namespace {

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}

TEST(Pipeline, ListModulesIncludesVietasr) {
    std::string modules = vietasr_list_modules();
    EXPECT_TRUE(Contains(modules, "vietasr"));
    EXPECT_TRUE(Contains(modules, "vad"));
}

TEST(Pipeline, ListPresetsIncludesTranscribe) {
    std::string presets = vietasr_list_presets();
    EXPECT_TRUE(Contains(presets, "transcribe"));
    EXPECT_TRUE(Contains(presets, "meeting"));
}

TEST(Pipeline, BuildEmptyPipelineSucceeds) {
    VietasrPipeline* pipeline = vietasr_pipeline_new();
    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(vietasr_pipeline_build(pipeline), VIETASR_OK);
    vietasr_pipeline_free(pipeline);
}

TEST(Pipeline, AddUnknownModuleFails) {
    VietasrPipeline* pipeline = vietasr_pipeline_new();
    auto status = vietasr_pipeline_add_module(pipeline,
                                              "nonexistent-module",
                                              "{}");
    EXPECT_NE(status, VIETASR_OK);
    vietasr_pipeline_free(pipeline);
}

TEST(Pipeline, TranscribePresetReturnsJson) {
    VietasrPipeline* pipeline = vietasr_pipeline_preset("transcribe");
    ASSERT_NE(pipeline, nullptr);

    short pcm[16000] = {0};
    for (int i = 0; i < 16000; ++i) {
        pcm[i] = static_cast<short>((i % 2000) - 1000);
    }
    const char* result = vietasr_transcribe_buffer(pipeline, pcm, 16000, 16000.0f);
    ASSERT_NE(result, nullptr);
    std::string json = result;
    EXPECT_TRUE(Contains(json, "is_final"));
    vietasr_pipeline_free(pipeline);
}

TEST(Pipeline, AnalyticsPresetSurfacesAllModuleFields) {
    VietasrPipeline* pipeline = vietasr_pipeline_preset("analytics");
    ASSERT_NE(pipeline, nullptr);

    short pcm[16000] = {0};
    const char* result = vietasr_transcribe_buffer(pipeline, pcm, 16000, 16000.0f);
    ASSERT_NE(result, nullptr);
    std::string json = result;
    EXPECT_TRUE(Contains(json, "gender"));
    EXPECT_TRUE(Contains(json, "emotion"));
    EXPECT_TRUE(Contains(json, "dialect"));
    EXPECT_TRUE(Contains(json, "noise"));
    vietasr_pipeline_free(pipeline);
}

TEST(Session, StreamingFinalJson) {
    VietasrPipeline* pipeline = vietasr_pipeline_preset("transcribe");
    ASSERT_NE(pipeline, nullptr);
    VietasrSession* session = vietasr_session_new(pipeline, 16000.0f);
    ASSERT_NE(session, nullptr);

    short pcm[1600] = {0};
    for (int i = 0; i < 10; ++i) {
        vietasr_accept_waveform_s16(session, pcm, 1600);
    }
    std::string final = vietasr_final_result(session);
    EXPECT_TRUE(Contains(final, "is_final\":true"));

    vietasr_session_free(session);
    vietasr_pipeline_free(pipeline);
}
