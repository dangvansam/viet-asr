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
#define VIETASR_VERSION_MINOR 2
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
    /* Deprecated: CoreML is an ONNX Runtime execution provider, not a separate
     * backend. Kept for ABI compatibility; it is remapped internally to the
     * ONNX backend with VIETASR_EP_COREML. Use
     * vietasr_pipeline_set_execution_provider() instead. */
    VIETASR_BACKEND_COREML = 2
} VietasrBackend;

/* ONNX Runtime execution provider. VIETASR_EP_AUTO probes the best provider
 * available on the device and falls back to CPU; an explicit value forces a
 * provider and still falls back to CPU if it is unavailable. Whether a given
 * provider is usable depends on which ONNX Runtime binary the SDK was built
 * against (see vietasr_available_execution_providers). */
typedef enum {
    VIETASR_EP_AUTO     = 0,
    VIETASR_EP_CPU      = 1,
    VIETASR_EP_CUDA     = 2,
    VIETASR_EP_TENSORRT = 3,
    VIETASR_EP_COREML   = 4,
    VIETASR_EP_NNAPI    = 5,
    VIETASR_EP_XNNPACK  = 6,
    VIETASR_EP_DIRECTML = 7,
    VIETASR_EP_QNN      = 8
} VietasrExecutionProvider;

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
/* Select the ONNX Runtime execution provider for this pipeline. device_id picks
 * the accelerator index on multi-device hosts (ignored for CPU/AUTO). Takes
 * effect on the next vietasr_pipeline_build(). */
VIETASR_API VietasrStatus    vietasr_pipeline_set_execution_provider(
                                 VietasrPipeline* pipeline,
                                 VietasrExecutionProvider ep,
                                 int device_id);
VIETASR_API VietasrStatus    vietasr_pipeline_set_model_dir(VietasrPipeline* pipeline,
                                                            const char* model_dir);
VIETASR_API VietasrStatus    vietasr_pipeline_build(VietasrPipeline* pipeline);
VIETASR_API void             vietasr_pipeline_free(VietasrPipeline* pipeline);

VIETASR_API const char*      vietasr_list_modules(void);
VIETASR_API const char*      vietasr_list_presets(void);

/* JSON array of execution providers compiled into the linked ONNX Runtime, by
 * canonical name (e.g. ["CUDAExecutionProvider","CPUExecutionProvider"]).
 * Returns "[]" when the ONNX backend is not compiled in. */
VIETASR_API const char*      vietasr_available_execution_providers(void);

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

/* Canonical name of the execution provider this session actually runs on
 * (e.g. "CoreMLExecutionProvider", "CPUExecutionProvider"). Reflects the EP
 * fallback outcome — may differ from the requested provider. */
VIETASR_API const char* vietasr_active_execution_provider(VietasrSession* session);

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
