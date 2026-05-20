cmake_minimum_required(VERSION 3.20)
project(vietasr-python-binding LANGUAGES C CXX)

set(VIETASR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(VIETASR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VIETASR_BUILD_SHARED ON CACHE BOOL "" FORCE)
# The wheel installs libvietasr itself (below). Suppress core's own install
# rules — otherwise they duplicate the 66 MB library into the wheel as
# vietasr/_native/lib/ plus an unused vietasr/_native/include/ tree.
set(VIETASR_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../core ${CMAKE_BINARY_DIR}/_vietasr_core)

# With VIETASR_INSTALL=OFF, core leaves the target unversioned, so this ships a
# single libvietasr.so — cffi loads it by plain name and a wheel zip would
# otherwise materialise versioned SONAME symlinks as full 66 MB copies.
# RPATH must point at the bundled onnxruntime sibling: $ORIGIN on Linux,
# @loader_path on macOS (the dynamic loader does not understand $ORIGIN there).
if(APPLE)
    set(_vietasr_rpath "@loader_path")
else()
    set(_vietasr_rpath "$ORIGIN")
endif()
set_target_properties(vietasr PROPERTIES
    BUILD_RPATH    "${_vietasr_rpath}"
    INSTALL_RPATH  "${_vietasr_rpath}"
    SKIP_BUILD_RPATH FALSE
)

install(TARGETS vietasr
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

if(VIETASR_HAS_ONNXRUNTIME)
    get_filename_component(_ort_lib_dir "${ONNXRUNTIME_LIB}" DIRECTORY)
    # Bundle the onnxruntime runtime library ONCE. The release archive ships a
    # real lib plus SONAME symlinks; installing the glob verbatim copied every
    # symlink into the wheel as a full file. Resolve to the real lib instead.
    if(WIN32)
        file(GLOB _ort_dlls "${_ort_lib_dir}/onnxruntime*.dll")
        foreach(_dll IN LISTS _ort_dlls)
            install(FILES "${_dll}" DESTINATION .)
        endforeach()
    elseif(APPLE)
        file(REAL_PATH "${ONNXRUNTIME_LIB}" _ort_real)
        install(FILES "${_ort_real}" DESTINATION .)
    else()
        file(REAL_PATH "${ONNXRUNTIME_LIB}" _ort_real)
        # onnxruntime 1.x SONAME is libonnxruntime.so.1 — install under it so
        # libvietasr's RUNPATH ($ORIGIN) resolves the dependency directly.
        install(FILES "${_ort_real}" DESTINATION . RENAME libonnxruntime.so.1)
        file(GLOB _ort_providers "${_ort_lib_dir}/libonnxruntime_providers*.so*")
        foreach(_prov IN LISTS _ort_providers)
            if(NOT IS_SYMLINK "${_prov}")
                install(FILES "${_prov}" DESTINATION .)
            endif()
        endforeach()
    endif()
    message(STATUS "wheel bundles onnxruntime from: ${_ort_lib_dir}")
endif()
