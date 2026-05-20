include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

function(vietasr_add_test target)
    add_executable(${target} ${ARGN} $<TARGET_OBJECTS:vietasr_objects>)
    target_link_libraries(${target} PRIVATE gtest_main)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    if(VIETASR_WITH_CURL)
        target_link_libraries(${target} PRIVATE CURL::libcurl)
    endif()
    if(VIETASR_HAS_ONNXRUNTIME)
        target_link_libraries(${target} PRIVATE "${ONNXRUNTIME_LIB}")
        target_include_directories(${target} PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    endif()
endfunction()

vietasr_add_test(test_pipeline test_pipeline.cc test_result_builder.cc)

include(GoogleTest)
gtest_discover_tests(test_pipeline)

if(VIETASR_MODULE_VIETASR)
    vietasr_add_test(test_vietasr ../src/modules/vietasr/vietasr_test.cc)
    gtest_discover_tests(test_vietasr)
endif()
