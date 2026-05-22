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
# Consumers that add_subdirectory() this core (e.g. the Python wheel) install
# libvietasr themselves and turn this OFF to avoid duplicate install rules.
option(VIETASR_INSTALL "Emit install() rules for libvietasr" ON)

option(VIETASR_BACKEND_ONNX   "Enable ONNX Runtime backend (default: everywhere)" ON)
option(VIETASR_BACKEND_COREML "Enable CoreML backend (Apple platforms)" OFF)

message(STATUS "vietasr backends: onnx=${VIETASR_BACKEND_ONNX} coreml=${VIETASR_BACKEND_COREML}")

option(VIETASR_WITH_CURL "Use libcurl for HTTP model downloads" ON)

# iOS has no system libcurl, and the embedded model needs no HTTP downloads.
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(VIETASR_WITH_CURL OFF)
endif()

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

    if(DEFINED ONNXRUNTIME_FRAMEWORK_DIR)
        # Apple builds (the iOS/macOS xcframework) supply a prebuilt
        # onnxruntime.framework slice from the onnxruntime-c CocoaPods archive
        # — onnxruntime ships no .tgz for iOS. libvietasr links it directly so
        # the dylib records @rpath/onnxruntime.framework/onnxruntime; the app
        # embeds both frameworks side by side. See bindings/apple/.
        set(_ort_url "")
        get_filename_component(ONNXRUNTIME_FRAMEWORK_DIR
            "${ONNXRUNTIME_FRAMEWORK_DIR}" ABSOLUTE)
        if(EXISTS "${ONNXRUNTIME_FRAMEWORK_DIR}/onnxruntime")
            set(VIETASR_HAS_ONNXRUNTIME 1 CACHE INTERNAL "")
            set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_FRAMEWORK_DIR}/Headers" CACHE INTERNAL "")
            set(ONNXRUNTIME_LIB "${ONNXRUNTIME_FRAMEWORK_DIR}/onnxruntime" CACHE INTERNAL "")
            set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_FRAMEWORK_DIR}" CACHE INTERNAL "")
            # The onnxruntime pod framework leaves its CoreML / CoreFoundation
            # dependencies for the consumer to satisfy.
            set(ONNXRUNTIME_APPLE_SYSTEM_LIBS
                "-framework CoreML" "-framework Foundation"
                "-framework CoreFoundation" CACHE INTERNAL "")
            message(STATUS "onnxruntime (Apple framework): ${ONNXRUNTIME_FRAMEWORK_DIR}")
        else()
            message(WARNING "ONNXRUNTIME_FRAMEWORK_DIR has no onnxruntime binary; engine_onnx will be a stub")
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
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
    # ONNXRUNTIME_LIB may be unset if a platform has no prebuilt onnxruntime.
    if(ONNXRUNTIME_LIB)
        target_link_libraries(vietasr_objects PUBLIC "${ONNXRUNTIME_LIB}")
    endif()
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
if(VIETASR_HAS_ONNXRUNTIME AND ONNXRUNTIME_LIB)
    target_link_libraries(vietasr PUBLIC "${ONNXRUNTIME_LIB}")
endif()
if(ONNXRUNTIME_APPLE_SYSTEM_LIBS)
    target_link_libraries(vietasr PUBLIC ${ONNXRUNTIME_APPLE_SYSTEM_LIBS})
endif()

set_target_properties(vietasr PROPERTIES
    PUBLIC_HEADER "${VIETASR_PUBLIC_HEADERS}"
)

# Apple: emit libvietasr as a .framework bundle (correctly structured per
# platform — flat for iOS, versioned for macOS). The Swift binding turns these
# into VietASR.xcframework. Off by default so desktop/Linux still build a plain
# libvietasr.{dylib,so}. See bindings/apple/build-xcframework.sh.
option(VIETASR_APPLE_FRAMEWORK "Build libvietasr as a .framework bundle (Apple)" OFF)
if(VIETASR_APPLE_FRAMEWORK)
    if(NOT VIETASR_BUILD_SHARED)
        message(FATAL_ERROR "VIETASR_APPLE_FRAMEWORK requires VIETASR_BUILD_SHARED=ON")
    endif()
    # Bundle name is CVietASR — the Swift wrapper target is VietASR and the two
    # module names must differ. Only the C ABI header ships in the framework.
    set_target_properties(vietasr PROPERTIES
        FRAMEWORK TRUE
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER io.github.dangvansam.cvietasr
        MACOSX_FRAMEWORK_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${PROJECT_VERSION}
        OUTPUT_NAME CVietASR
        PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/vietasr.h"
    )
endif()

if(VIETASR_INSTALL)
    # SONAME versioning matters for a system-installed library. An embedded
    # build (e.g. the Python wheel) sets VIETASR_INSTALL=OFF and wants a single
    # unversioned libvietasr.so instead of versioned SONAME symlinks.
    set_target_properties(vietasr PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

if(VIETASR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(VIETASR_BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../examples/transcribe-wav
                     ${CMAKE_BINARY_DIR}/_examples_cli)
endif()

if(VIETASR_INSTALL)
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
endif()
