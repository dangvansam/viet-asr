# EmbedVadModel.cmake — bakes the VAD ONNX models (ten-vad.onnx and/or
# silero_vad.onnx) straight into libvietasr, so the tenvad and silero backends
# of the vad module need no separate download or working-directory lookup.
# Mirrors EmbedItn.cmake / EmbedModel.cmake.
#
# Usage:
#   include(cmake/EmbedVadModel.cmake)
#   vietasr_embed_vad(OUT_VAR "<ten-vad.onnx path>" "<silero_vad.onnx path>")
#   # Pass an empty string for a model that should not be embedded.

if(NOT MSVC)
    enable_language(ASM)
endif()

function(vietasr_embed_vad OUT_SOURCES_VAR TENVAD_PATH SILERO_PATH)
    # (symbol | path | rc-id) for each model that should be embedded.
    set(_specs "")
    if(NOT TENVAD_PATH STREQUAL "")
        list(APPEND _specs "vietasr_tenvad_onnx|${TENVAD_PATH}|300")
    endif()
    if(NOT SILERO_PATH STREQUAL "")
        list(APPEND _specs "vietasr_silero_onnx|${SILERO_PATH}|301")
    endif()

    if(_specs STREQUAL "")
        set(${OUT_SOURCES_VAR} "" PARENT_SCOPE)
        return()
    endif()

    set(_deps "")
    set(_asm_body "")
    set(_rc_entries "")
    foreach(_spec ${_specs})
        string(REPLACE "|" ";" _parts "${_spec}")
        list(GET _parts 0 _sym)
        list(GET _parts 1 _path)
        list(GET _parts 2 _rcid)
        if(NOT EXISTS "${_path}")
            message(FATAL_ERROR "VAD embed: model file missing: ${_path}")
        endif()
        list(APPEND _deps "${_path}")
        string(APPEND _asm_body
            "    .balign 16\n"
            "    .global VA_SYM(${_sym})\n"
            "VA_SYM(${_sym}):\n"
            "    .incbin \"${_path}\"\n"
            "VA_SYM(${_sym}_end):\n"
            "    .balign 8\n"
            "    .global VA_SYM(${_sym}_size)\n"
            "VA_SYM(${_sym}_size):\n"
            "    .quad VA_SYM(${_sym}_end) - VA_SYM(${_sym})\n")
        string(APPEND _rc_entries "${_rcid} RCDATA \"${_path}\"\n")
    endforeach()

    # Reconfigure if any embedded model changes.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_deps})
    message(STATUS "vietasr: embedding VAD model(s): ${_deps}")

    set(_gen_dir "${CMAKE_BINARY_DIR}/embedded")
    file(MAKE_DIRECTORY "${_gen_dir}")

    if(MSVC)
        set(VIETASR_VAD_RC_ENTRIES "${_rc_entries}")
        set(_rc "${_gen_dir}/embedded_vad.rc")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/embedded_vad.rc.in"
                       "${_rc}" @ONLY)
        set_source_files_properties("${_rc}" PROPERTIES OBJECT_DEPENDS "${_deps}")
        set(${OUT_SOURCES_VAR} "${_rc}" PARENT_SCOPE)
    else()
        set(VIETASR_VAD_ASM_BODY "${_asm_body}")
        set(_asm "${_gen_dir}/embedded_vad.S")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/embedded_vad.S.in"
                       "${_asm}" @ONLY)
        # .incbin reads the models at assemble time — rebuild if any change.
        set_source_files_properties("${_asm}" PROPERTIES OBJECT_DEPENDS "${_deps}")
        set(${OUT_SOURCES_VAR} "${_asm}" PARENT_SCOPE)
    endif()
endfunction()
