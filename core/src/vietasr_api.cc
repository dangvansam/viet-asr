#include "vietasr.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include "utils/json.h"
#include "vietasr/logger.h"
#include "vietasr/pipeline.h"

namespace {

thread_local std::string tls_result;
thread_local std::string tls_error;

const char* StoreResult(std::string value) {
    tls_result = std::move(value);
    return tls_result.c_str();
}

void SetError(const std::string& message) {
    tls_error = message;
    if (!message.empty()) {
        VIETASR_LOG_ERROR("api") << message;
    }
}

::vietasr::Pipeline* ToCxx(VietasrPipeline* handle) {
    return reinterpret_cast<::vietasr::Pipeline*>(handle);
}

::vietasr::Session* ToCxx(VietasrSession* handle) {
    return reinterpret_cast<::vietasr::Session*>(handle);
}

}

extern "C" {

VietasrPipeline* vietasr_pipeline_preset(const char* name) {
    if (!name) {
        SetError("preset name is null");
        return nullptr;
    }
    auto pipeline = ::vietasr::Pipeline::FromPreset(name);
    if (!pipeline) {
        SetError(std::string("preset not found or build failed: ") + name);
        return nullptr;
    }
    return reinterpret_cast<VietasrPipeline*>(pipeline.release());
}

VietasrPipeline* vietasr_pipeline_new(void) {
    return reinterpret_cast<VietasrPipeline*>(new ::vietasr::Pipeline());
}

VietasrStatus vietasr_pipeline_add_module(VietasrPipeline* pipeline,
                                          const char* module_name,
                                          const char* json_config) {
    if (!pipeline || !module_name) return VIETASR_ERR_INVALID_ARG;
    auto status = ToCxx(pipeline)->AddModule(module_name, json_config ? json_config : "{}");
    if (!status.ok()) {
        SetError(status.message());
        return static_cast<VietasrStatus>(status.code());
    }
    return VIETASR_OK;
}

VietasrStatus vietasr_pipeline_set_backend(VietasrPipeline* pipeline,
                                           VietasrBackend backend) {
    if (!pipeline) return VIETASR_ERR_INVALID_ARG;
    ToCxx(pipeline)->SetBackend(static_cast<int>(backend));
    return VIETASR_OK;
}

VietasrStatus vietasr_pipeline_set_model_dir(VietasrPipeline* pipeline,
                                             const char* model_dir) {
    if (!pipeline || !model_dir) return VIETASR_ERR_INVALID_ARG;
    ToCxx(pipeline)->SetModelDir(model_dir);
    return VIETASR_OK;
}

VietasrStatus vietasr_pipeline_build(VietasrPipeline* pipeline) {
    if (!pipeline) return VIETASR_ERR_INVALID_ARG;
    auto status = ToCxx(pipeline)->Build();
    if (!status.ok()) {
        SetError(status.message());
        return static_cast<VietasrStatus>(status.code());
    }
    return VIETASR_OK;
}

void vietasr_pipeline_free(VietasrPipeline* pipeline) {
    delete ToCxx(pipeline);
}

const char* vietasr_list_modules(void) {
    ::vietasr::JsonWriter writer;
    writer.BeginArray();
    for (const auto& name : ::vietasr::ModuleRegistry::Instance().List()) {
        writer.String(name);
    }
    writer.EndArray();
    return StoreResult(writer.Str());
}

const char* vietasr_list_presets(void) {
    ::vietasr::JsonWriter writer;
    writer.BeginArray();
    for (const auto& name : ::vietasr::PresetRegistry::Instance().List()) {
        writer.String(name);
    }
    writer.EndArray();
    return StoreResult(writer.Str());
}

VietasrSession* vietasr_session_new(VietasrPipeline* pipeline, float sample_rate) {
    if (!pipeline) return nullptr;
    auto session = ToCxx(pipeline)->NewSession(sample_rate);
    if (!session) {
        SetError("session creation failed");
        return nullptr;
    }
    return reinterpret_cast<VietasrSession*>(session.release());
}

void vietasr_session_free(VietasrSession* session) {
    delete ToCxx(session);
}

void vietasr_session_reset(VietasrSession* session) {
    if (session) ToCxx(session)->Reset();
}

VietasrFrameStatus vietasr_accept_waveform_s16(VietasrSession* session,
                                               const short* pcm, int len) {
    if (!session || !pcm) return VIETASR_FRAME_PARTIAL;
    return static_cast<VietasrFrameStatus>(
        ToCxx(session)->AcceptWaveformS16(pcm, len));
}

VietasrFrameStatus vietasr_accept_waveform_f32(VietasrSession* session,
                                               const float* pcm, int len) {
    if (!session || !pcm) return VIETASR_FRAME_PARTIAL;
    return static_cast<VietasrFrameStatus>(
        ToCxx(session)->AcceptWaveformF32(pcm, len));
}

VietasrFrameStatus vietasr_accept_waveform_bytes(VietasrSession* session,
                                                 const char* pcm, int len) {
    if (!session || !pcm) return VIETASR_FRAME_PARTIAL;
    int samples = len / 2;
    return vietasr_accept_waveform_s16(
        session, reinterpret_cast<const short*>(pcm), samples);
}

const char* vietasr_partial_result(VietasrSession* session) {
    if (!session) return "{}";
    return StoreResult(ToCxx(session)->Partial());
}

const char* vietasr_result(VietasrSession* session) {
    if (!session) return "{}";
    return StoreResult(ToCxx(session)->Result());
}

const char* vietasr_final_result(VietasrSession* session) {
    if (!session) return "{}";
    return StoreResult(ToCxx(session)->Final());
}

const char* vietasr_transcribe_file(VietasrPipeline* pipeline, const char* wav_path) {
    if (!pipeline || !wav_path) return nullptr;
    return StoreResult(ToCxx(pipeline)->TranscribeFile(wav_path));
}

const char* vietasr_transcribe_buffer(VietasrPipeline* pipeline,
                                      const short* pcm, int len, float sample_rate) {
    if (!pipeline || !pcm) return nullptr;
    return StoreResult(ToCxx(pipeline)->TranscribeBuffer(pcm, len, sample_rate));
}

const char* vietasr_result_field(const char* json, const char* dotted_path) {
    (void)json; (void)dotted_path;
    SetError("result_field traversal not yet implemented; parse JSON in caller");
    return nullptr;
}

VietasrStatus vietasr_ensure_models(VietasrPipeline* pipeline) {
    if (!pipeline) return VIETASR_ERR_INVALID_ARG;
    auto status = ToCxx(pipeline)->EnsureModels();
    return status.ok() ? VIETASR_OK
                       : static_cast<VietasrStatus>(status.code());
}

const char* vietasr_default_cache_dir(void) {
    return StoreResult(::vietasr::ModelManager::DefaultCacheDir());
}

void vietasr_set_log_level(VietasrLogLevel level) {
    ::vietasr::Logger::Instance().SetLevel(static_cast<::vietasr::LogLevel>(level));
}

const char* vietasr_version(void) {
    return "0.1.0";
}

const char* vietasr_last_error(void) {
    return tls_error.c_str();
}

}
