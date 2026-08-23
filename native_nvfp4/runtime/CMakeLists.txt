cmake_minimum_required(VERSION 3.24)
project(nvfp4_runtime LANGUAGES CXX)

find_package(OpenCL REQUIRED)

add_library(nvfp4_runtime SHARED nvfp4_runtime.cpp nvfp4_cpu.cpp)
target_compile_features(nvfp4_runtime PRIVATE cxx_std_17)
target_compile_definitions(nvfp4_runtime PRIVATE NVFP4_RUNTIME_BUILD)
target_include_directories(nvfp4_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nvfp4_runtime PRIVATE OpenCL::OpenCL)

if(MSVC)
    target_compile_options(nvfp4_runtime PRIVATE /W4)
else()
    target_compile_options(nvfp4_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()
