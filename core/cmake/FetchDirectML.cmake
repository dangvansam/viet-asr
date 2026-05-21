# FetchDirectML.cmake — acquires a Windows-x64 DirectML build of ONNX Runtime.
#
# DirectML is the one execution-provider variant with no GitHub release archive.
# It is distributed solely through NuGet, and the DirectML.dll redistributable
# lives in a *separate* package (Microsoft.AI.DirectML) declared as a NuGet
# dependency. This module fetches both without hard-coding the DirectML version:
# it reads the dependency straight out of the ONNX Runtime .nuspec.
#
# vietasr_fetch_directml(<ort_directml_nupkg_url> <work_dir>) sets, in the
# caller's scope:
#   VIETASR_DIRECTML_ROOT        — extracted ORT.DirectML package root
#   VIETASR_DIRECTML_NATIVE_DIR  — runtimes/win-x64/native (onnxruntime.dll,
#                                  onnxruntime.lib, and the staged DirectML.dll)
#
# This path runs only for VIETASR_ONNX_VARIANT=directml on Windows x64; the
# exact NuGet layout is verified by CI rather than on developer machines.

function(vietasr_fetch_directml ort_nupkg_url work_dir)
    set(_ort_root "${work_dir}/onnxruntime-directml")
    set(_ort_native "${_ort_root}/runtimes/win-x64/native")

    if(NOT EXISTS "${_ort_native}/onnxruntime.dll")
        set(_ort_dl "${work_dir}/onnxruntime-directml.nupkg.zip")
        message(STATUS "DirectML: downloading ${ort_nupkg_url}")
        file(DOWNLOAD "${ort_nupkg_url}" "${_ort_dl}" STATUS _st)
        list(GET _st 0 _code)
        if(NOT _code EQUAL 0)
            message(FATAL_ERROR "DirectML: ONNX Runtime nupkg download failed: ${_st}")
        endif()
        file(ARCHIVE_EXTRACT INPUT "${_ort_dl}" DESTINATION "${_ort_root}")
    endif()

    # Resolve the Microsoft.AI.DirectML dependency version from the .nuspec so
    # DirectML.dll always matches the ONNX Runtime build.
    file(GLOB _nuspec "${_ort_root}/*.nuspec")
    if(NOT _nuspec)
        message(FATAL_ERROR "DirectML: .nuspec missing from extracted nupkg")
    endif()
    file(READ "${_nuspec}" _nuspec_text)
    string(REGEX MATCH
        "id=\"Microsoft.AI.DirectML\"[ \t]+version=\"([0-9.]+)\""
        _ "${_nuspec_text}")
    if(NOT CMAKE_MATCH_1)
        message(FATAL_ERROR
            "DirectML: could not find Microsoft.AI.DirectML dependency in .nuspec")
    endif()
    set(_dml_version "${CMAKE_MATCH_1}")
    message(STATUS "DirectML: Microsoft.AI.DirectML ${_dml_version}")

    set(_dml_root "${work_dir}/directml-redist")
    if(NOT EXISTS "${_dml_root}/bin/x64-win/DirectML.dll")
        set(_dml_dl "${work_dir}/directml.nupkg.zip")
        file(DOWNLOAD
            "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/${_dml_version}"
            "${_dml_dl}" STATUS _st)
        list(GET _st 0 _code)
        if(NOT _code EQUAL 0)
            message(FATAL_ERROR "DirectML: redistributable download failed: ${_st}")
        endif()
        file(ARCHIVE_EXTRACT INPUT "${_dml_dl}" DESTINATION "${_dml_root}")
    endif()

    # Stage DirectML.dll next to onnxruntime.dll so the loader and the
    # native-packaging step treat them as one bundle.
    file(COPY "${_dml_root}/bin/x64-win/DirectML.dll" DESTINATION "${_ort_native}")

    set(VIETASR_DIRECTML_ROOT       "${_ort_root}"   PARENT_SCOPE)
    set(VIETASR_DIRECTML_NATIVE_DIR "${_ort_native}" PARENT_SCOPE)
endfunction()
