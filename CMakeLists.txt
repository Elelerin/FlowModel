cmake_minimum_required(VERSION 3.15)
project(FlowControl 
            VERSION 0.1
            DESCRIPTION "A Basic Flow Mesh Generating Program"
            LANGUAGES CXX)



set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 86)

enable_language(CUDA)

add_executable(FlowControl
    src/main.cu)

find_library(CUDA cuda)

if (CUDA)
    target_link_libraries(FlowControl PRIVATE ${CUDA})
else()
    message([FATAL_ERROR] "CUDA is necessary for this project. If you don't have a NVIDIA Card, it won't work.")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(FlowControl PRIVATE /W4)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(FlowControl PRIVATE -Wall -Wextra -Wpedantic)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # other macOS-specific flags for Clang
endif()


