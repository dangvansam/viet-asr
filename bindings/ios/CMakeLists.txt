cmake_minimum_required(VERSION 3.20)
project(vietasr_ios LANGUAGES C CXX)

# Builds libvietasr.a (static) for one iOS slice. Configured once per slice by
# build-xcframework.sh with the iOS toolchain variables:
#   -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=<iphoneos|iphonesimulator>
#   -DCMAKE_OSX_ARCHITECTURES=<arch list> -DORT_INCLUDE_DIR=<onnxruntime-c headers>
# This mirrors bindings/android/lib/src/main/cpp/CMakeLists.txt: it compiles the
# core sources directly rather than going through core/CMakeLists.txt, whose
# FetchContent step targets desktop prebuilts only.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(VIETASR_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../core" CACHE PATH "vietasr core")

# Headers/ directory from the extracted onnxruntime-c iOS pod archive. The C API
# headers are slice-independent, so one path serves device and simulator builds.
set(ORT_INCLUDE_DIR "" CACHE PATH "onnxruntime-c iOS C API headers")
if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
    message(FATAL_ERROR
        "set -DORT_INCLUDE_DIR to the onnxruntime-c Headers directory")
endif()

set(VIETASR_CORE_SOURCES
    "${VIETASR_CORE_DIR}/src/vietasr_api.cc"
    "${VIETASR_CORE_DIR}/src/pipeline.cc"
    "${VIETASR_CORE_DIR}/src/result_builder.cc"
    "${VIETASR_CORE_DIR}/src/module_registry.cc"
    "${VIETASR_CORE_DIR}/src/preset_registry.cc"
    # Stub EmbeddedAsrVocab()/EmbeddedAsrModel() — iOS downloads the model at
    # runtime (as Android does); without it vietasr.cc fails to link.
    "${VIETASR_CORE_DIR}/src/embedded_model.cc"
    "${VIETASR_CORE_DIR}/src/engines/engine_factory.cc"
    "${VIETASR_CORE_DIR}/src/engines/engine_onnx.cc"
    "${VIETASR_CORE_DIR}/src/preprocess/kaldi_fbank.cc"
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
    "${VIETASR_CORE_DIR}/src/modules/vietasr/transducer_model.cc"
    "${VIETASR_CORE_DIR}/src/modules/vietasr/transducer_search.cc"
    "${VIETASR_CORE_DIR}/src/modules/vietasr/transducer_decoder.cc"
    "${VIETASR_CORE_DIR}/src/modules/vietasr/utterance_segmenter.cc"
    "${VIETASR_CORE_DIR}/src/modules/vietasr/post_processor.cc"
    "${VIETASR_CORE_DIR}/src/modules/vietasr/units.cc"
    "${VIETASR_CORE_DIR}/src/modules/vad/vad.cc"
    "${VIETASR_CORE_DIR}/src/modules/vad/vad_segmenter.cc"
    "${VIETASR_CORE_DIR}/src/modules/vad/energy_backend.cc"
    "${VIETASR_CORE_DIR}/src/modules/punctuation/punctuation.cc"
    "${VIETASR_CORE_DIR}/src/modules/itn/itn.cc"
    "${VIETASR_CORE_DIR}/src/modules/diarization/diarization.cc"
    "${VIETASR_CORE_DIR}/src/modules/gender/gender.cc"
    "${VIETASR_CORE_DIR}/src/modules/emotion/emotion.cc"
    "${VIETASR_CORE_DIR}/src/modules/dialect/dialect.cc"
    "${VIETASR_CORE_DIR}/src/modules/noise/noise.cc"
    "${VIETASR_CORE_DIR}/presets/transcribe.cc"
)

add_library(vietasr STATIC ${VIETASR_CORE_SOURCES})
target_include_directories(vietasr PRIVATE
    "${VIETASR_CORE_DIR}/include"
    "${VIETASR_CORE_DIR}/src"
    "${ORT_INCLUDE_DIR}"
)
target_compile_definitions(vietasr PRIVATE
    VIETASR_BACKEND_ONNX_ENABLED=1
    VIETASR_HAS_ONNXRUNTIME=1
)
# The onnxruntime-c iOS package bundles the CoreML execution provider.
if(EXISTS "${ORT_INCLUDE_DIR}/coreml_provider_factory.h")
    target_compile_definitions(vietasr PRIVATE VIETASR_EP_COREML_AVAILABLE=1)
    message(STATUS "vietasr: CoreML execution provider available")
endif()
target_compile_options(vietasr PRIVATE
    -Wall -Wextra -fvisibility=hidden -O2
)
