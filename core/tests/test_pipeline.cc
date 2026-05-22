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

TEST(Pipeline, StreamPresetEmitsVoicedRegionsFromAsr) {
    // The `stream` preset is vietasr-only; the ASR itself emits voiced_regions
    // + speech (derived from CTC-blank) — no separate VAD module.
    VietasrPipeline* pipeline = vietasr_pipeline_preset("stream");
    ASSERT_NE(pipeline, nullptr);

    short pcm[16000] = {0};
    for (int i = 0; i < 16000; ++i) {
        pcm[i] = static_cast<short>((i % 2000) - 1000);
    }
    const char* result = vietasr_transcribe_buffer(pipeline, pcm, 16000, 16000.0f);
    ASSERT_NE(result, nullptr);
    std::string json = result;
    EXPECT_TRUE(Contains(json, "voiced_regions"));
    EXPECT_TRUE(Contains(json, "speech"));
    vietasr_pipeline_free(pipeline);
}

TEST(Pipeline, BatchPresetSurfacesVadAndSegments) {
    // The `batch` preset = Silero VAD + ASR; the result carries voiced_regions
    // + speech (from the vad module) and a per-sentence segments[] array.
    VietasrPipeline* pipeline = vietasr_pipeline_preset("batch");
    ASSERT_NE(pipeline, nullptr);

    short pcm[16000] = {0};
    for (int i = 0; i < 16000; ++i) {
        pcm[i] = static_cast<short>((i % 2000) - 1000);
    }
    const char* result = vietasr_transcribe_buffer(pipeline, pcm, 16000, 16000.0f);
    ASSERT_NE(result, nullptr);
    std::string json = result;
    EXPECT_TRUE(Contains(json, "voiced_regions"));
    EXPECT_TRUE(Contains(json, "speech"));
    EXPECT_TRUE(Contains(json, "segments"));
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

TEST(ExecutionProvider, AvailableProvidersIncludeCpu) {
    // ONNX Runtime always ships the CPU provider, whatever the build variant.
    std::string providers = vietasr_available_execution_providers();
    EXPECT_TRUE(Contains(providers, "CPUExecutionProvider"));
}

TEST(ExecutionProvider, AutoSelectionResolvesToCpuOnPlainBuild) {
    VietasrPipeline* pipeline = vietasr_pipeline_preset("transcribe");
    ASSERT_NE(pipeline, nullptr);
    VietasrSession* session = vietasr_session_new(pipeline, 16000.0f);
    ASSERT_NE(session, nullptr);
    // A CPU-only ONNX Runtime build has no accelerator to auto-detect.
    EXPECT_STREQ(vietasr_active_execution_provider(session),
                 "CPUExecutionProvider");
    vietasr_session_free(session);
    vietasr_pipeline_free(pipeline);
}

TEST(ExecutionProvider, UnavailableProviderFallsBackAndStillTranscribes) {
    VietasrPipeline* pipeline = vietasr_pipeline_new();
    ASSERT_NE(pipeline, nullptr);
    ASSERT_EQ(vietasr_pipeline_add_module(pipeline, "vietasr", "{}"), VIETASR_OK);
    // CUDA is not compiled into the default (CPU) ONNX Runtime build; the
    // request must fall back to CPU rather than fail.
    EXPECT_EQ(vietasr_pipeline_set_execution_provider(pipeline,
                                                      VIETASR_EP_CUDA, 0),
              VIETASR_OK);
    ASSERT_EQ(vietasr_pipeline_build(pipeline), VIETASR_OK);

    VietasrSession* session = vietasr_session_new(pipeline, 16000.0f);
    ASSERT_NE(session, nullptr);
    EXPECT_STREQ(vietasr_active_execution_provider(session),
                 "CPUExecutionProvider");

    short pcm[16000] = {0};
    for (int i = 0; i < 16000; ++i) {
        pcm[i] = static_cast<short>((i % 2000) - 1000);
    }
    vietasr_accept_waveform_s16(session, pcm, 16000);
    std::string final = vietasr_final_result(session);
    EXPECT_TRUE(Contains(final, "is_final"));

    vietasr_session_free(session);
    vietasr_pipeline_free(pipeline);
}

TEST(ExecutionProvider, SetExecutionProviderRejectsNullPipeline) {
    EXPECT_EQ(vietasr_pipeline_set_execution_provider(nullptr,
                                                      VIETASR_EP_AUTO, 0),
              VIETASR_ERR_INVALID_ARG);
}
