# EmbedModel.cmake — bakes the chunked vietasr model + vocab straight into
# libvietasr. The model chunks are .incbin'd back-to-back by the assembler,
# so they land contiguous in .rodata — no concatenation step is needed.
#
# Usage:
#   include(cmake/EmbedModel.cmake)
#   vietasr_embed_model(VIETASR_EMBED_SOURCES)
#   # ... add ${VIETASR_EMBED_SOURCES} to the library target ...

if(NOT MSVC)
    enable_language(ASM)
endif()

function(vietasr_embed_model OUT_SOURCES_VAR)
    set(_chunk_dir "${CMAKE_CURRENT_SOURCE_DIR}/../models/vietasr")
    cmake_path(NORMAL_PATH _chunk_dir)
    set(_manifest "${_chunk_dir}/chunks.json")
    set(_vocab "${_chunk_dir}/vocab.txt")

    if(NOT EXISTS "${_manifest}")
        message(FATAL_ERROR
            "VIETASR_EMBED_MODEL is ON but ${_manifest} is missing. "
            "Run scripts/split_model.py to generate the model chunks.")
    endif()
    if(NOT EXISTS "${_vocab}")
        message(FATAL_ERROR "VIETASR_EMBED_MODEL is ON but ${_vocab} is missing.")
    endif()

    # Resolve the ordered chunk list and verify each chunk against the manifest.
    file(READ "${_manifest}" _chunks_json)
    string(JSON _n_chunks LENGTH "${_chunks_json}" "chunks")
    if(_n_chunks LESS 1)
        message(FATAL_ERROR "chunks.json lists no chunks")
    endif()
    math(EXPR _last "${_n_chunks} - 1")
    set(_chunk_files "")
    foreach(_i RANGE ${_last})
        string(JSON _cname GET "${_chunks_json}" "chunks" ${_i} "name")
        string(JSON _csha  GET "${_chunks_json}" "chunks" ${_i} "sha256")
        set(_cpath "${_chunk_dir}/${_cname}")
        if(NOT EXISTS "${_cpath}")
            message(FATAL_ERROR "model chunk missing: ${_cpath}")
        endif()
        file(SHA256 "${_cpath}" _got)
        if(NOT _got STREQUAL _csha)
            message(FATAL_ERROR "chunk ${_cname} sha256 mismatch (corrupt?)")
        endif()
        list(APPEND _chunk_files "${_cpath}")
    endforeach()

    # Reconfigure if the chunk set changes.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_manifest}")
    message(STATUS "vietasr: embedding ${_n_chunks} model chunk(s) + vocab")

    set(_gen_dir "${CMAKE_BINARY_DIR}/embedded")
    file(MAKE_DIRECTORY "${_gen_dir}")
    set(VIETASR_VOCAB_PATH "${_vocab}")

    if(MSVC)
        set(_entries "")
        set(_rid 100)
        foreach(_f ${_chunk_files})
            string(APPEND _entries "${_rid} RCDATA \"${_f}\"\n")
            math(EXPR _rid "${_rid} + 1")
        endforeach()
        set(VIETASR_MODEL_RC_ENTRIES "${_entries}")
        set(_rc "${_gen_dir}/embedded_model.rc")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/embedded_model.rc.in"
                       "${_rc}" @ONLY)
        set_source_files_properties("${_rc}" PROPERTIES
            OBJECT_DEPENDS "${_chunk_files};${_vocab}")
        set(${OUT_SOURCES_VAR} "${_rc}" PARENT_SCOPE)
    else()
        set(_incbins "")
        foreach(_f ${_chunk_files})
            string(APPEND _incbins "    .incbin \"${_f}\"\n")
        endforeach()
        set(VIETASR_MODEL_INCBINS "${_incbins}")
        set(_asm "${_gen_dir}/embedded_model.S")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/embedded_model.S.in"
                       "${_asm}" @ONLY)
        # .incbin reads the chunks at assemble time — rebuild if any change.
        set_source_files_properties("${_asm}" PROPERTIES
            OBJECT_DEPENDS "${_chunk_files};${_vocab}")
        set(${OUT_SOURCES_VAR} "${_asm}" PARENT_SCOPE)
    endif()
endfunction()
