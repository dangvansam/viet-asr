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

option(VIETASR_BACKEND_ONNX "Enable ONNX Runtime backend (default: everywhere)" ON)

# Which ONNX Runtime build to link against. "cpu" is the default and the only
# variant that ships as the standard release artifacts; "cuda" and "directml"
# produce opt-in GPU builds (see Phase 3 of the EP coverage plan).
set(VIETASR_ONNX_VARIANT "cpu" CACHE STRING
    "ONNX Runtime variant: cpu | cuda (NVIDIA, Linux/Win x64) | directml (Win x64)")
set_property(CACHE VIETASR_ONNX_VARIANT PROPERTY STRINGS cpu cuda directml)

message(STATUS "vietasr backend: onnx=${VIETASR_BACKEND_ONNX} variant=${VIETASR_ONNX_VARIANT}")

option(VIETASR_WITH_CURL "Use libcurl for HTTP model downloads (Linux, macOS)" ON)
option(VIETASR_REQUIRE_HTTP "Fail the configure step when no HTTP backend is available" OFF)
set(VIETASR_WITH_WINHTTP OFF)

if(WIN32)
    set(VIETASR_WITH_CURL OFF)
    set(VIETASR_WITH_WINHTTP ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR ANDROID OR EMSCRIPTEN)
    set(VIETASR_WITH_CURL OFF)
endif()

if(VIETASR_WITH_CURL)
    find_package(CURL QUIET)
    if(NOT CURL_FOUND)
        message(STATUS "libcurl not found; HTTP downloads will be stubbed")
        set(VIETASR_WITH_CURL OFF)
    endif()
endif()

if(VIETASR_REQUIRE_HTTP AND NOT VIETASR_WITH_CURL AND NOT VIETASR_WITH_WINHTTP)
    message(FATAL_ERROR "VIETASR_REQUIRE_HTTP=ON but no HTTP backend was found (install libcurl development files)")
endif()

if(VIETASR_BACKEND_ONNX)
    if(NOT DEFINED ONNXRUNTIME_VERSION)
        set(ONNXRUNTIME_VERSION "1.20.1")
    endif()

    string(TOLOWER "${VIETASR_ONNX_VARIANT}" _ort_variant)
    set(_ort_base "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}")
    set(_ort_is_x64 OFF)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(_ort_is_x64 ON)
    endif()

    # Exactly one of these is set: _ort_url is a GitHub release archive that
    # FetchContent extracts directly; _ort_nupkg is a NuGet package (a zip with
    # no extension CMake recognises) fetched + extracted by hand further down.
    set(_ort_url "")
    set(_ort_nupkg "")

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
    elseif(_ort_variant STREQUAL "cuda")
        # CUDA + TensorRT EPs ship only in the dedicated -gpu packages, and only
        # for Linux x64 and Windows x64.
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _ort_is_x64)
            set(_ort_url "${_ort_base}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz")
        elseif(WIN32 AND _ort_is_x64)
            set(_ort_url "${_ort_base}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip")
        else()
            message(FATAL_ERROR
                "VIETASR_ONNX_VARIANT=cuda is only available for Linux x64 and Windows x64")
        endif()
    elseif(_ort_variant STREQUAL "directml")
        # DirectML is NOT published as a GitHub release archive — it ships only
        # in the Microsoft.ML.OnnxRuntime.DirectML NuGet package. Windows x64
        # only. The DirectML.dll redistributable is a separate NuGet dependency,
        # resolved in the manual-extract branch below.
        if(WIN32 AND _ort_is_x64)
            set(_ort_nupkg "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/${ONNXRUNTIME_VERSION}")
        else()
            message(FATAL_ERROR
                "VIETASR_ONNX_VARIANT=directml is only available for Windows x64")
        endif()
    else()
        # cpu (default): the standard per-platform prebuilt.
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _ort_is_x64)
            set(_ort_url "${_ort_base}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(_ort_url "${_ort_base}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            set(_ort_url "${_ort_base}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
        elseif(WIN32 AND _ort_is_x64)
            set(_ort_url "${_ort_base}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
        elseif(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
            set(_ort_url "${_ort_base}/onnxruntime-win-arm64-${ONNXRUNTIME_VERSION}.zip")
        else()
            message(WARNING "no onnxruntime prebuilt for this platform; engine_onnx will be a stub")
        endif()
    endif()

    if(_ort_url)
        include(FetchContent)
        FetchContent_Declare(onnxruntime URL "${_ort_url}")
        FetchContent_MakeAvailable(onnxruntime)
        set(ONNXRUNTIME_ROOT        "${onnxruntime_SOURCE_DIR}" CACHE INTERNAL "")
        set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include" CACHE INTERNAL "")
        set(_ort_lib_dir            "${ONNXRUNTIME_ROOT}/lib")
    elseif(_ort_nupkg)
        # A .nupkg has no extension FetchContent recognises, so download then
        # extract by hand. NuGet lays native libs out under runtimes/<rid>/native
        # and headers under build/native/include. The Microsoft.AI.DirectML
        # dependency (which provides DirectML.dll) is resolved by reading the
        # .nuspec rather than hard-coding a version.
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchDirectML.cmake)
        vietasr_fetch_directml("${_ort_nupkg}" "${CMAKE_BINARY_DIR}")
        set(ONNXRUNTIME_ROOT        "${VIETASR_DIRECTML_ROOT}" CACHE INTERNAL "")
        set(ONNXRUNTIME_INCLUDE_DIR "${VIETASR_DIRECTML_ROOT}/build/native/include" CACHE INTERNAL "")
        set(_ort_lib_dir            "${VIETASR_DIRECTML_NATIVE_DIR}")
    endif()

    if(DEFINED _ort_lib_dir)
        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime
            PATHS "${_ort_lib_dir}"
            NO_DEFAULT_PATH
        )
        if(ONNXRUNTIME_LIB)
            message(STATUS "onnxruntime (${_ort_variant}): ${ONNXRUNTIME_LIB}")
            set(VIETASR_HAS_ONNXRUNTIME 1 CACHE INTERNAL "")
            set(ONNXRUNTIME_LIB_DIR "${_ort_lib_dir}" CACHE INTERNAL "")
        else()
            message(WARNING "onnxruntime lib not found under ${_ort_lib_dir}")
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

option(VIETASR_EMBED_MODEL "Download the vietasr model from Hugging Face at build time and embed it into libvietasr" OFF)
option(VIETASR_EMBED_ITN   "Embed the Vietnamese ITN grammar into libvietasr" ON)

# vad-module backends. The neural backends additionally require ONNX Runtime;
# they auto-disable (with the energy backend remaining) when it is unavailable.
option(VIETASR_VAD_TENVAD   "Build the TEN-VAD backend of the vad module" ON)
option(VIETASR_VAD_SILERO   "Build the Silero VAD backend of the vad module" ON)
option(VIETASR_EMBED_TENVAD "Embed ten-vad.onnx into libvietasr"   ON)
option(VIETASR_EMBED_SILERO "Embed silero_vad.onnx into libvietasr" ON)

# ---------------------------------------------------------------------------
# OpenFst — required by the itn module's WFST inverse text normalizer. We use
# a CMake-friendly fork (the one sherpa-onnx ships, including to WebAssembly),
# so the dependency stays light and portable across all bindings. Only the
# core `fst` library is built; bin/script/extension targets are disabled.
# ---------------------------------------------------------------------------
if(VIETASR_MODULE_ITN)
    set(HAVE_BIN       OFF CACHE BOOL "" FORCE)
    set(HAVE_SCRIPT    OFF CACHE BOOL "" FORCE)
    set(HAVE_COMPACT   OFF CACHE BOOL "" FORCE)
    set(HAVE_COMPRESS  OFF CACHE BOOL "" FORCE)
    set(HAVE_CONST     OFF CACHE BOOL "" FORCE)
    set(HAVE_FAR       OFF CACHE BOOL "" FORCE)
    set(HAVE_GRM       OFF CACHE BOOL "" FORCE)
    set(HAVE_PDT       OFF CACHE BOOL "" FORCE)
    set(HAVE_MPDT      OFF CACHE BOOL "" FORCE)
    set(HAVE_LINEAR    OFF CACHE BOOL "" FORCE)
    set(HAVE_LOOKAHEAD OFF CACHE BOOL "" FORCE)
    set(HAVE_NGRAM     OFF CACHE BOOL "" FORCE)
    set(HAVE_SPECIAL   OFF CACHE BOOL "" FORCE)
    set(HAVE_FSTS      OFF CACHE BOOL "" FORCE)
    set(OPENFST_USE_ABSL OFF CACHE BOOL "" FORCE)

    include(FetchContent)
    FetchContent_Declare(openfst
        URL      "https://github.com/csukuangfj/openfst/archive/refs/tags/v1.8.5-2026-04-11.tar.gz"
        URL_HASH SHA256=57fbc4b950ae81b1a0e1e298af15652da968a6723a592b7874e9b4027a80a5b4
    )
    # Build a static `fst` so it links cleanly into libvietasr — PIC is already
    # on globally. BUILD_SHARED_LIBS is restored afterwards for other deps.
    set(_vietasr_saved_shared "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(openfst)
    set(BUILD_SHARED_LIBS "${_vietasr_saved_shared}")
    set(VIETASR_OPENFST_INCLUDE_DIR "${openfst_SOURCE_DIR}/src/include"
        CACHE INTERNAL "")
    message(STATUS "vietasr: openfst (itn) -> ${VIETASR_OPENFST_INCLUDE_DIR}")
endif()

include(GNUInstallDirs)

set(VIETASR_PUBLIC_HEADERS
    include/vietasr.h
    include/vietasr/engine.h
    include/vietasr/ep_config.h
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

    src/preprocess/kaldi_fbank.cc
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
# Detect which execution providers ship inside the fetched ONNX Runtime tree and
# expose them as compile defines. engine_onnx.cc #ifdef-guards every EP path on
# these, so one source file compiles against any ORT package (CPU, GPU, mobile).
if(VIETASR_HAS_ONNXRUNTIME)
    if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/coreml_provider_factory.h")
        list(APPEND VIETASR_BACKEND_DEFINES VIETASR_EP_COREML_AVAILABLE=1)
    endif()
    if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/nnapi_provider_factory.h")
        list(APPEND VIETASR_BACKEND_DEFINES VIETASR_EP_NNAPI_AVAILABLE=1)
    endif()
    if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/dml_provider_factory.h")
        list(APPEND VIETASR_BACKEND_DEFINES VIETASR_EP_DIRECTML_AVAILABLE=1)
    endif()
    file(GLOB _ort_cuda_libs "${ONNXRUNTIME_LIB_DIR}/*onnxruntime_providers_cuda*")
    if(_ort_cuda_libs)
        list(APPEND VIETASR_BACKEND_DEFINES VIETASR_EP_CUDA_AVAILABLE=1)
    endif()
    file(GLOB _ort_trt_libs "${ONNXRUNTIME_LIB_DIR}/*onnxruntime_providers_tensorrt*")
    if(_ort_trt_libs)
        list(APPEND VIETASR_BACKEND_DEFINES VIETASR_EP_TENSORRT_AVAILABLE=1)
    endif()
    message(STATUS "vietasr execution-provider defines: ${VIETASR_BACKEND_DEFINES}")
endif()

if(VIETASR_MODULE_VIETASR)
    list(APPEND VIETASR_CORE_SOURCES
        src/modules/vietasr/vietasr.cc
        src/modules/vietasr/transducer_model.cc
        src/modules/vietasr/model_fetcher.cc
        src/modules/vietasr/transducer_search.cc
        src/modules/vietasr/transducer_decoder.cc
        src/modules/vietasr/utterance_segmenter.cc
        src/modules/vietasr/post_processor.cc
        src/modules/vietasr/units.cc
    )
    if(NOT VIETASR_MODULE_VAD)
        message(FATAL_ERROR "VIETASR_MODULE_VIETASR requires VIETASR_MODULE_VAD (utterance segmentation)")
    endif()
endif()

if(VIETASR_MODULE_VAD)
    list(APPEND VIETASR_CORE_SOURCES
        src/modules/vad/vad.cc
        src/modules/vad/vad_segmenter.cc
        src/modules/vad/energy_backend.cc
    )

    # The silero and tenvad backends both need ONNX Runtime; disable them (the
    # energy backend always remains) when it is not available in this build.
    set(_vad_tenvad OFF)
    set(_vad_silero OFF)
    if(VIETASR_VAD_TENVAD AND VIETASR_HAS_ONNXRUNTIME)
        set(_vad_tenvad ON)
    elseif(VIETASR_VAD_TENVAD)
        message(STATUS "vietasr: VIETASR_VAD_TENVAD set but ONNX Runtime is "
                       "unavailable; tenvad backend disabled")
    endif()
    if(VIETASR_VAD_SILERO AND VIETASR_HAS_ONNXRUNTIME)
        set(_vad_silero ON)
    elseif(VIETASR_VAD_SILERO)
        message(STATUS "vietasr: VIETASR_VAD_SILERO set but ONNX Runtime is "
                       "unavailable; silero backend disabled")
    endif()

    set(_vad_embed_tenvad "")
    set(_vad_embed_silero "")

    if(_vad_silero)
        list(APPEND VIETASR_CORE_SOURCES src/modules/vad/silero_backend.cc)
        if(VIETASR_EMBED_SILERO)
            set(_vad_embed_silero
                "${CMAKE_CURRENT_SOURCE_DIR}/models/vad/silero_vad.onnx")
        endif()
    endif()

    if(_vad_tenvad)
        set(_tenvad_dir "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ten-vad")
        # Vendored TEN-VAD sources, compiled straight into libvietasr.
        set(VIETASR_TENVAD_SOURCES
            "${_tenvad_dir}/src/ten_vad.cc"
            "${_tenvad_dir}/src/aed.cc"
            "${_tenvad_dir}/src/biquad.cc"
            "${_tenvad_dir}/src/fscvrt.cc"
            "${_tenvad_dir}/src/pitch_est.cc"
            "${_tenvad_dir}/src/stft.cc"
            "${_tenvad_dir}/src/fftw.c"
        )
        list(APPEND VIETASR_CORE_SOURCES
            ${VIETASR_TENVAD_SOURCES}
            src/modules/vad/tenvad_backend.cc
        )
        # Third-party code does not pass vietasr's -Wall -Wextra -Wpedantic.
        if(MSVC)
            set_source_files_properties(${VIETASR_TENVAD_SOURCES}
                PROPERTIES COMPILE_OPTIONS "/w")
        else()
            set_source_files_properties(${VIETASR_TENVAD_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-w")
        endif()
        if(VIETASR_EMBED_TENVAD)
            set(_vad_embed_tenvad "${_tenvad_dir}/src/onnx_model/ten-vad.onnx")
        endif()
    endif()

    # embedded_vad.cc exposes the embedded model blobs; build it whenever a
    # neural backend is present (it returns empty blobs when nothing embedded).
    if(_vad_tenvad OR _vad_silero)
        list(APPEND VIETASR_CORE_SOURCES src/modules/vad/embedded_vad.cc)
    endif()

    if(NOT _vad_embed_tenvad STREQUAL "" OR NOT _vad_embed_silero STREQUAL "")
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedVadModel.cmake)
        vietasr_embed_vad(VIETASR_VAD_EMBED_SOURCES
            "${_vad_embed_tenvad}" "${_vad_embed_silero}")
        list(APPEND VIETASR_CORE_SOURCES ${VIETASR_VAD_EMBED_SOURCES})
    endif()
endif()

if(VIETASR_MODULE_PUNCTUATION)
    list(APPEND VIETASR_CORE_SOURCES src/modules/punctuation/punctuation.cc)
endif()
if(VIETASR_MODULE_ITN)
    list(APPEND VIETASR_CORE_SOURCES
        src/modules/itn/itn.cc
        src/modules/itn/processor.cc
        src/modules/itn/itn_normalizer.cc
        src/modules/itn/embedded_itn.cc
    )
    if(VIETASR_EMBED_ITN)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedItn.cmake)
        vietasr_embed_itn(VIETASR_ITN_EMBED_SOURCES)
        list(APPEND VIETASR_CORE_SOURCES ${VIETASR_ITN_EMBED_SOURCES})
        message(STATUS "vietasr: VIETASR_EMBED_ITN=ON (ITN grammar baked into libvietasr)")
    endif()
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

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ModelSource.cmake)
vietasr_model_source_header(VIETASR_MODEL_SOURCE_INCLUDE_DIR)

if(VIETASR_EMBED_MODEL)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedModel.cmake)
    vietasr_embed_model(VIETASR_EMBED_SOURCES)
    list(APPEND VIETASR_CORE_SOURCES ${VIETASR_EMBED_SOURCES})
    message(STATUS "vietasr: VIETASR_EMBED_MODEL=ON (model baked into libvietasr)")
else()
    message(STATUS "vietasr: VIETASR_EMBED_MODEL=OFF (model downloaded from Hugging Face on first use)")
endif()

add_library(vietasr_objects OBJECT ${VIETASR_CORE_SOURCES})
target_include_directories(vietasr_objects
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE
        ${VIETASR_MODEL_SOURCE_INCLUDE_DIR}
)
if(WIN32)
    target_compile_definitions(vietasr_objects PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
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

if(VIETASR_MODULE_ITN)
    # SYSTEM so openfst's own header warnings don't fail/flood vietasr's -Wall.
    target_include_directories(vietasr_objects SYSTEM PRIVATE
        "${VIETASR_OPENFST_INCLUDE_DIR}")
    target_link_libraries(vietasr_objects PUBLIC fst)
    # VIETASR_HAS_ITN lets the vietasr module call the ITN normalizer (for
    # per-sentence ITN while streaming) only when the itn module is built.
    target_compile_definitions(vietasr_objects PRIVATE VIETASR_HAS_ITN=1)
    if(VIETASR_EMBED_ITN)
        target_compile_definitions(vietasr_objects PRIVATE VIETASR_HAS_EMBEDDED_ITN=1)
    endif()
endif()

if(VIETASR_MODULE_VAD AND _vad_tenvad)
    # TEN-VAD's own headers + the ONNX Runtime C API it uses.
    target_include_directories(vietasr_objects PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ten-vad/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ten-vad/src")
    target_compile_definitions(vietasr_objects PRIVATE VIETASR_VAD_TENVAD=1)
    if(VIETASR_EMBED_TENVAD)
        target_compile_definitions(vietasr_objects PRIVATE
            VIETASR_HAS_EMBEDDED_TENVAD=1)
    endif()
endif()

if(VIETASR_MODULE_VAD AND _vad_silero)
    target_compile_definitions(vietasr_objects PRIVATE VIETASR_VAD_SILERO=1)
    if(VIETASR_EMBED_SILERO)
        target_compile_definitions(vietasr_objects PRIVATE
            VIETASR_HAS_EMBEDDED_SILERO=1)
    endif()
endif()

if(VIETASR_WITH_CURL)
    target_compile_definitions(vietasr_objects PRIVATE VIETASR_WITH_CURL=1)
    target_link_libraries(vietasr_objects PUBLIC CURL::libcurl)
endif()
if(VIETASR_WITH_WINHTTP)
    target_compile_definitions(vietasr_objects PRIVATE VIETASR_WITH_WINHTTP=1)
    target_link_libraries(vietasr_objects PUBLIC winhttp)
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
if(VIETASR_WITH_WINHTTP)
    target_link_libraries(vietasr PUBLIC winhttp)
endif()
if(VIETASR_HAS_ONNXRUNTIME AND ONNXRUNTIME_LIB)
    target_link_libraries(vietasr PUBLIC "${ONNXRUNTIME_LIB}")
endif()
if(VIETASR_MODULE_ITN)
    # `fst` is static; fold it into libvietasr ($<TARGET_OBJECTS> does not
    # carry the object library's link deps).
    target_link_libraries(vietasr PRIVATE fst)
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
