cmake_minimum_required(VERSION 3.18)

# Project name - start with CXX, add CUDA later if needed
project(RayTracingCUDA LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Collect all source files
file(GLOB_RECURSE CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CU_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE HEADERS
    "${CMAKE_SOURCE_DIR}/src/*.h"
    "${CMAKE_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_SOURCE_DIR}/src/*.cuh"
)

# Check if we have CUDA files
if(CU_SOURCES)
    # Enable CUDA language
    enable_language(CUDA)

    # Set CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # CUDA architectures (adjust based on your GPU)
    # Common options: 75 (RTX 20xx), 86 (RTX 30xx), 89 (RTX 40xx)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

    # Enable CUDA separable compilation
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    # Include CUDA directories
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    message(STATUS "CUDA files detected - enabling CUDA support")
endif()

# Combine all sources
set(ALL_SOURCES ${CPP_SOURCES} ${CU_SOURCES})

# Create executable
add_executable(${PROJECT_NAME} ${ALL_SOURCES} ${HEADERS})

# Set CUDA properties if we have CUDA files
if(CU_SOURCES)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Link CUDA runtime
    target_link_libraries(${PROJECT_NAME} PRIVATE cudart)
endif()

# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Visual Studio specific settings
if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
endif()
