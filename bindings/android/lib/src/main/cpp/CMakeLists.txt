cmake_minimum_required(VERSION 3.22)
project(vietasr_android LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(VIETASR_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../core" CACHE PATH "Path to vietasr core")
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../ort-android" CACHE PATH "Path to extracted onnxruntime-android AAR")

if(NOT DEFINED ANDROID_ABI)
    message(FATAL_ERROR "ANDROID_ABI must be set (e.g., arm64-v8a)")
endif()

set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI}")
# Default to the headers/ dir inside an extracted onnxruntime-android AAR;
# overridable with -DORT_INCLUDE_DIR= for other layouts.
if(NOT ORT_INCLUDE_DIR)
    set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/headers")
    if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
        set(ORT_INCLUDE_DIR "${VIETASR_CORE_DIR}/../../onnxruntime-headers")
    endif()
endif()

message(STATUS "ABI=${ANDROID_ABI}")
message(STATUS "ORT_LIB_DIR=${ORT_LIB_DIR}")
message(STATUS "ORT_INCLUDE_DIR=${ORT_INCLUDE_DIR}")

set(VIETASR_CORE_SOURCES
    "${VIETASR_CORE_DIR}/src/vietasr_api.cc"
    "${VIETASR_CORE_DIR}/src/pipeline.cc"
    "${VIETASR_CORE_DIR}/src/result_builder.cc"
    "${VIETASR_CORE_DIR}/src/module_registry.cc"
    "${VIETASR_CORE_DIR}/src/preset_registry.cc"
    "${VIETASR_CORE_DIR}/src/engines/engine_factory.cc"
    "${VIETASR_CORE_DIR}/src/engines/engine_onnx.cc"
    "${VIETASR_CORE_DIR}/src/preprocess/feature_pipeline.cc"
    "${VIETASR_CORE_DIR}/src/preprocess/audio_buffer.cc"
    "${VIETASR_CORE_DIR}/src/preprocess/audio_resampler.cc"
    "${VIETASR_CORE_DIR}/src/preprocess/audio_io.cc"
    "${VIETASR_CORE_DIR}/src/utils/model_manager.cc"
    "${VIETASR_CORE_DIR}/src/utils/http_client.cc"
    "${VIETASR_CORE_DIR}/src/utils/logger.cc"
    "${VIETASR_CORE_DIR}/src/utils/json.cc"
    "${VIETASR_CORE_DIR}/src/utils/fs.cc"
    "${VIETASR_CORE_DIR}/src/utils/md5.cc"
    "${VIETASR_CORE_DIR}/src/modules/vietasr/vietasr.cc"
    "${VIETASR_CORE_DIR}/src/modules/vietasr/ctc_beam_search.cc"
    "${VIETASR_CORE_DIR}/src/modules/vietasr/ctc_endpoint.cc"
    "${VIETASR_CORE_DIR}/src/modules/vietasr/post_processor.cc"
    "${VIETASR_CORE_DIR}/src/modules/vietasr/units.cc"
    "${VIETASR_CORE_DIR}/src/modules/vad/vad.cc"
    "${VIETASR_CORE_DIR}/src/modules/punctuation/punctuation.cc"
    "${VIETASR_CORE_DIR}/src/modules/itn/itn.cc"
    "${VIETASR_CORE_DIR}/src/modules/diarization/diarization.cc"
    "${VIETASR_CORE_DIR}/src/modules/gender/gender.cc"
    "${VIETASR_CORE_DIR}/src/modules/emotion/emotion.cc"
    "${VIETASR_CORE_DIR}/src/modules/dialect/dialect.cc"
    "${VIETASR_CORE_DIR}/src/modules/noise/noise.cc"
    "${VIETASR_CORE_DIR}/presets/transcribe.cc"
)

add_library(vietasr SHARED ${VIETASR_CORE_SOURCES} vietasr_jni.cc)
target_include_directories(vietasr PRIVATE
    "${VIETASR_CORE_DIR}/include"
    "${VIETASR_CORE_DIR}/src"
    "${ORT_INCLUDE_DIR}"
)
target_compile_definitions(vietasr PRIVATE
    VIETASR_BACKEND_ONNX_ENABLED=1
    VIETASR_HAS_ONNXRUNTIME=1
)
target_compile_options(vietasr PRIVATE
    -Wall -Wextra -fvisibility=hidden -O2
)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_LIB_DIR}/libonnxruntime.so"
)

target_link_libraries(vietasr
    PRIVATE
        onnxruntime
        log
        android
)
