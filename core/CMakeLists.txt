cmake_minimum_required(VERSION 3.20)
project(vietasr
    VERSION 0.1.0
    DESCRIPTION "Universal Vietnamese Speech AI SDK"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VIETASR_BUILD_SHARED "Build shared library (libvietasr.so/.dylib/.dll)" ON)
option(VIETASR_BUILD_STATIC "Build static library (libvietasr.a)" OFF)
option(VIETASR_BUILD_TESTS  "Build unit tests" ON)
option(VIETASR_BUILD_EXAMPLES "Build example CLIs" OFF)

option(VIETASR_BACKEND_ONNX   "Enable ONNX Runtime backend (default: everywhere)" ON)
option(VIETASR_BACKEND_COREML "Enable CoreML backend (Apple platforms)" OFF)

message(STATUS "vietasr backends: onnx=${VIETASR_BACKEND_ONNX} coreml=${VIETASR_BACKEND_COREML}")

option(VIETASR_WITH_CURL "Use libcurl for HTTP model downloads" ON)

if(VIETASR_WITH_CURL)
    find_package(CURL QUIET)
    if(NOT CURL_FOUND)
        message(STATUS "libcurl not found; HTTP downloads will be stubbed")
        set(VIETASR_WITH_CURL OFF)
    endif()
endif()

if(VIETASR_BACKEND_ONNX)
    if(NOT DEFINED ONNXRUNTIME_VERSION)
        set(ONNXRUNTIME_VERSION "1.20.1")
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
        set(_ort_dirname "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz")
        set(_ort_dirname "onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
        set(_ort_dirname "onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}")
    elseif(WIN32)
        set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
        set(_ort_dirname "onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
    else()
        message(WARNING "no onnxruntime prebuilt for this platform; engine_onnx will be a stub")
        set(_ort_url "")
    endif()

    if(_ort_url)
        include(FetchContent)
        FetchContent_Declare(onnxruntime URL "${_ort_url}")
        FetchContent_MakeAvailable(onnxruntime)
        set(ONNXRUNTIME_ROOT         "${onnxruntime_SOURCE_DIR}" CACHE INTERNAL "")
        set(ONNXRUNTIME_INCLUDE_DIR  "${ONNXRUNTIME_ROOT}/include" CACHE INTERNAL "")
        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime
            PATHS "${ONNXRUNTIME_ROOT}/lib"
            NO_DEFAULT_PATH
        )
        if(ONNXRUNTIME_LIB)
            message(STATUS "onnxruntime: ${ONNXRUNTIME_LIB}")
            set(VIETASR_HAS_ONNXRUNTIME 1 CACHE INTERNAL "")
            set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib" CACHE INTERNAL "")
        else()
            message(WARNING "onnxruntime lib not found under ${ONNXRUNTIME_ROOT}")
        endif()
    endif()
endif()

option(VIETASR_MODULE_VIETASR      "Build vietasr ASR module" ON)
option(VIETASR_MODULE_VAD          "Build vad module" ON)
option(VIETASR_MODULE_PUNCTUATION  "Build punctuation module (Phase 11 stub)" ON)
option(VIETASR_MODULE_ITN          "Build itn module (Phase 12 stub)" ON)
option(VIETASR_MODULE_DIARIZATION  "Build diarization module (Phase 13 stub)" ON)
option(VIETASR_MODULE_GENDER       "Build gender module (Phase 14 stub)" ON)
option(VIETASR_MODULE_EMOTION      "Build emotion module (Phase 15 stub)" ON)
option(VIETASR_MODULE_DIALECT      "Build dialect module (Phase 16 stub)" ON)
option(VIETASR_MODULE_NOISE        "Build noise module (Phase 17 stub)" ON)

option(VIETASR_EMBED_MODEL "Embed the vietasr model into libvietasr at build time" ON)

include(GNUInstallDirs)

set(VIETASR_PUBLIC_HEADERS
    include/vietasr.h
    include/vietasr/engine.h
    include/vietasr/logger.h
    include/vietasr/model_manager.h
    include/vietasr/module.h
    include/vietasr/pipeline.h
    include/vietasr/result.h
    include/vietasr/types.h
)

set(VIETASR_CORE_SOURCES
    src/vietasr_api.cc
    src/pipeline.cc
    src/result_builder.cc
    src/module_registry.cc
    src/preset_registry.cc

    src/engines/engine_factory.cc

    src/preprocess/feature_pipeline.cc
    src/preprocess/audio_buffer.cc
    src/preprocess/audio_resampler.cc
    src/preprocess/audio_io.cc

    src/utils/model_manager.cc
    src/utils/http_client.cc
    src/utils/logger.cc
    src/utils/json.cc
    src/utils/fs.cc
    src/utils/md5.cc
)

if(VIETASR_BACKEND_ONNX)
    list(APPEND VIETASR_CORE_SOURCES src/engines/engine_onnx.cc)
endif()

set(VIETASR_BACKEND_DEFINES)
if(VIETASR_BACKEND_ONNX)
    list(APPEND VIETASR_BACKEND_DEFINES VIETASR_BACKEND_ONNX_ENABLED=1)
endif()
if(VIETASR_BACKEND_COREML)
    list(APPEND VIETASR_BACKEND_DEFINES VIETASR_BACKEND_COREML_ENABLED=1)
endif()

if(VIETASR_BACKEND_COREML)
    list(APPEND VIETASR_CORE_SOURCES src/engines/engine_coreml.mm)
endif()

if(VIETASR_MODULE_VIETASR)
    list(APPEND VIETASR_CORE_SOURCES
        src/modules/vietasr/vietasr.cc
        src/modules/vietasr/ctc_beam_search.cc
        src/modules/vietasr/ctc_endpoint.cc
        src/modules/vietasr/post_processor.cc
        src/modules/vietasr/units.cc
    )
endif()

if(VIETASR_MODULE_VAD)
    list(APPEND VIETASR_CORE_SOURCES src/modules/vad/vad.cc)
endif()

if(VIETASR_MODULE_PUNCTUATION)
    list(APPEND VIETASR_CORE_SOURCES src/modules/punctuation/punctuation.cc)
endif()
if(VIETASR_MODULE_ITN)
    list(APPEND VIETASR_CORE_SOURCES src/modules/itn/itn.cc)
endif()
if(VIETASR_MODULE_DIARIZATION)
    list(APPEND VIETASR_CORE_SOURCES src/modules/diarization/diarization.cc)
endif()
if(VIETASR_MODULE_GENDER)
    list(APPEND VIETASR_CORE_SOURCES src/modules/gender/gender.cc)
endif()
if(VIETASR_MODULE_EMOTION)
    list(APPEND VIETASR_CORE_SOURCES src/modules/emotion/emotion.cc)
endif()
if(VIETASR_MODULE_DIALECT)
    list(APPEND VIETASR_CORE_SOURCES src/modules/dialect/dialect.cc)
endif()
if(VIETASR_MODULE_NOISE)
    list(APPEND VIETASR_CORE_SOURCES src/modules/noise/noise.cc)
endif()

list(APPEND VIETASR_CORE_SOURCES
    presets/transcribe.cc
)

list(APPEND VIETASR_CORE_SOURCES src/embedded_model.cc)

if(VIETASR_EMBED_MODEL)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedModel.cmake)
    vietasr_embed_model(VIETASR_EMBED_SOURCES)
    list(APPEND VIETASR_CORE_SOURCES ${VIETASR_EMBED_SOURCES})
    message(STATUS "vietasr: VIETASR_EMBED_MODEL=ON (model baked into libvietasr)")
else()
    message(STATUS "vietasr: VIETASR_EMBED_MODEL=OFF (vietasr module will have no model)")
endif()

add_library(vietasr_objects OBJECT ${VIETASR_CORE_SOURCES})
target_include_directories(vietasr_objects
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(vietasr_objects PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(VIETASR_BACKEND_DEFINES)
    target_compile_definitions(vietasr_objects PRIVATE ${VIETASR_BACKEND_DEFINES})
endif()

if(VIETASR_EMBED_MODEL)
    target_compile_definitions(vietasr_objects PRIVATE VIETASR_HAS_EMBEDDED_MODEL=1)
endif()

if(VIETASR_WITH_CURL)
    target_compile_definitions(vietasr_objects PRIVATE VIETASR_WITH_CURL=1)
    target_link_libraries(vietasr_objects PUBLIC CURL::libcurl)
endif()

if(VIETASR_HAS_ONNXRUNTIME)
    target_compile_definitions(vietasr_objects PRIVATE VIETASR_HAS_ONNXRUNTIME=1)
    target_include_directories(vietasr_objects PUBLIC "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(vietasr_objects PUBLIC "${ONNXRUNTIME_LIB}")
endif()

if(VIETASR_BUILD_SHARED)
    add_library(vietasr SHARED $<TARGET_OBJECTS:vietasr_objects>)
    target_compile_definitions(vietasr PRIVATE VIETASR_BUILDING_DLL=1)
else()
    add_library(vietasr STATIC $<TARGET_OBJECTS:vietasr_objects>)
endif()

target_include_directories(vietasr
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(VIETASR_WITH_CURL)
    target_link_libraries(vietasr PUBLIC CURL::libcurl)
endif()
if(VIETASR_HAS_ONNXRUNTIME)
    target_link_libraries(vietasr PUBLIC "${ONNXRUNTIME_LIB}")
endif()

set_target_properties(vietasr PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${VIETASR_PUBLIC_HEADERS}"
)

if(VIETASR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(VIETASR_BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../examples/transcribe-wav
                     ${CMAKE_BINARY_DIR}/_examples_cli)
endif()

install(TARGETS vietasr
    EXPORT vietasrTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/vietasr
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
