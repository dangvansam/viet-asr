cmake_minimum_required(VERSION 3.20)
project(vietasr-python-binding LANGUAGES C CXX)

set(VIETASR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(VIETASR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VIETASR_BUILD_SHARED ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../core ${CMAKE_BINARY_DIR}/_vietasr_core)

set_target_properties(vietasr PROPERTIES
    BUILD_RPATH    "$ORIGIN"
    INSTALL_RPATH  "$ORIGIN"
    SKIP_BUILD_RPATH FALSE
)

install(TARGETS vietasr
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

if(VIETASR_HAS_ONNXRUNTIME)
    get_filename_component(_ort_lib_dir "${ONNXRUNTIME_LIB}" DIRECTORY)
    file(GLOB _ort_libs
        "${_ort_lib_dir}/libonnxruntime.so*"
        "${_ort_lib_dir}/libonnxruntime.*.dylib"
        "${_ort_lib_dir}/libonnxruntime.dylib"
        "${_ort_lib_dir}/onnxruntime*.dll"
        "${_ort_lib_dir}/onnxruntime_providers*.dll"
        "${_ort_lib_dir}/libonnxruntime_providers*.so*"
    )
    foreach(_lib IN LISTS _ort_libs)
        install(FILES "${_lib}" DESTINATION .)
    endforeach()
    message(STATUS "wheel will bundle onnxruntime libs from: ${_ort_lib_dir}")
endif()
