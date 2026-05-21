# EmbedItn.cmake — bakes the Vietnamese ITN grammar FSTs straight into
# libvietasr, so the itn module needs no separate download or config.
#
# The grammars are small (< 1 MB total) and committed under
# src/modules/itn/models/. Regenerate them with scripts/gen_itn_fst.py.
#
# Usage:
#   include(cmake/EmbedItn.cmake)
#   vietasr_embed_itn(VIETASR_ITN_EMBED_SOURCES)
#   # ... add ${VIETASR_ITN_EMBED_SOURCES} to the library target ...

if(NOT MSVC)
    enable_language(ASM)
endif()

function(vietasr_embed_itn OUT_SOURCES_VAR)
    set(_itn_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/itn/models")
    set(_tagger "${_itn_dir}/vi_itn_tagger.fst")
    set(_verbalizer "${_itn_dir}/vi_itn_verbalizer.fst")

    if(NOT EXISTS "${_tagger}" OR NOT EXISTS "${_verbalizer}")
        message(FATAL_ERROR
            "VIETASR_EMBED_ITN is ON but the grammar FSTs are missing:\n"
            "  ${_tagger}\n  ${_verbalizer}\n"
            "Run scripts/gen_itn_fst.py to generate them.")
    endif()

    # Reconfigure if the grammars change.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_tagger}" "${_verbalizer}")
    message(STATUS "vietasr: embedding ITN grammar (vi)")

    set(VIETASR_ITN_TAGGER_PATH "${_tagger}")
    set(VIETASR_ITN_VERBALIZER_PATH "${_verbalizer}")

    set(_gen_dir "${CMAKE_BINARY_DIR}/embedded")
    file(MAKE_DIRECTORY "${_gen_dir}")

    if(MSVC)
        set(_rc "${_gen_dir}/embedded_itn.rc")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/embedded_itn.rc.in"
                       "${_rc}" @ONLY)
        set_source_files_properties("${_rc}" PROPERTIES
            OBJECT_DEPENDS "${_tagger};${_verbalizer}")
        set(${OUT_SOURCES_VAR} "${_rc}" PARENT_SCOPE)
    else()
        set(_asm "${_gen_dir}/embedded_itn.S")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/embedded_itn.S.in"
                       "${_asm}" @ONLY)
        # .incbin reads the FSTs at assemble time — rebuild if either changes.
        set_source_files_properties("${_asm}" PROPERTIES
            OBJECT_DEPENDS "${_tagger};${_verbalizer}")
        set(${OUT_SOURCES_VAR} "${_asm}" PARENT_SCOPE)
    endif()
endfunction()
