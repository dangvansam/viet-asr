#ifndef VIETASR_H
#define VIETASR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef VIETASR_BUILDING_DLL
    #define VIETASR_API __declspec(dllexport)
  #else
    #define VIETASR_API __declspec(dllimport)
  #endif
#else
  #define VIETASR_API __attribute__((visibility("default")))
#endif

#define VIETASR_VERSION_MAJOR 0
#define VIETASR_VERSION_MINOR 1
#define VIETASR_VERSION_PATCH 0

typedef struct VietasrPipeline VietasrPipeline;
typedef struct VietasrSession  VietasrSession;

typedef enum {
    VIETASR_OK              =  0,
    VIETASR_ERR_INVALID_ARG = -1,
    VIETASR_ERR_NOT_FOUND   = -2,
    VIETASR_ERR_IO          = -3,
    VIETASR_ERR_NETWORK     = -4,
    VIETASR_ERR_CHECKSUM    = -5,
    VIETASR_ERR_BACKEND     = -6,
    VIETASR_ERR_RUNTIME     = -7,
    VIETASR_ERR_OOM         = -8
} VietasrStatus;

typedef enum {
    VIETASR_BACKEND_AUTO   = 0,
    VIETASR_BACKEND_ONNX   = 1,
    VIETASR_BACKEND_COREML = 2
} VietasrBackend;

typedef enum {
    VIETASR_LOG_TRACE = 0,
    VIETASR_LOG_DEBUG = 1,
    VIETASR_LOG_INFO  = 2,
    VIETASR_LOG_WARN  = 3,
    VIETASR_LOG_ERROR = 4,
    VIETASR_LOG_OFF   = 5
} VietasrLogLevel;

typedef enum {
    VIETASR_FRAME_PARTIAL = 0,
    VIETASR_FRAME_FINAL   = 1
} VietasrFrameStatus;

VIETASR_API VietasrPipeline* vietasr_pipeline_preset(const char* name);

VIETASR_API VietasrPipeline* vietasr_pipeline_new(void);
VIETASR_API VietasrStatus    vietasr_pipeline_add_module(VietasrPipeline* pipeline,
                                                         const char* module_name,
                                                         const char* json_config);
VIETASR_API VietasrStatus    vietasr_pipeline_set_backend(VietasrPipeline* pipeline,
                                                          VietasrBackend backend);
VIETASR_API VietasrStatus    vietasr_pipeline_set_model_dir(VietasrPipeline* pipeline,
                                                            const char* model_dir);
VIETASR_API VietasrStatus    vietasr_pipeline_build(VietasrPipeline* pipeline);
VIETASR_API void             vietasr_pipeline_free(VietasrPipeline* pipeline);

VIETASR_API const char*      vietasr_list_modules(void);
VIETASR_API const char*      vietasr_list_presets(void);

VIETASR_API VietasrSession*  vietasr_session_new(VietasrPipeline* pipeline, float sample_rate);
VIETASR_API void             vietasr_session_free(VietasrSession* session);
VIETASR_API void             vietasr_session_reset(VietasrSession* session);

VIETASR_API VietasrFrameStatus vietasr_accept_waveform_s16(VietasrSession* session,
                                                           const short* pcm, int len);
VIETASR_API VietasrFrameStatus vietasr_accept_waveform_f32(VietasrSession* session,
                                                           const float* pcm, int len);
VIETASR_API VietasrFrameStatus vietasr_accept_waveform_bytes(VietasrSession* session,
                                                             const char* pcm, int len);

VIETASR_API const char* vietasr_partial_result(VietasrSession* session);
VIETASR_API const char* vietasr_result(VietasrSession* session);
VIETASR_API const char* vietasr_final_result(VietasrSession* session);

VIETASR_API const char* vietasr_transcribe_file(VietasrPipeline* pipeline,
                                                const char* wav_path);
VIETASR_API const char* vietasr_transcribe_buffer(VietasrPipeline* pipeline,
                                                  const short* pcm, int len,
                                                  float sample_rate);

VIETASR_API const char* vietasr_result_field(const char* json, const char* dotted_path);

VIETASR_API VietasrStatus vietasr_ensure_models(VietasrPipeline* pipeline);
VIETASR_API const char*   vietasr_default_cache_dir(void);

VIETASR_API void        vietasr_set_log_level(VietasrLogLevel level);
VIETASR_API const char* vietasr_version(void);
VIETASR_API const char* vietasr_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
