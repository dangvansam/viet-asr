cmake_minimum_required(VERSION 3.20)
project(vietasr-cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT TARGET vietasr)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../core ${CMAKE_BINARY_DIR}/_vietasr_core)
endif()

add_executable(vietasr-cli main.cc)
target_link_libraries(vietasr-cli PRIVATE vietasr)
