# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)

if((NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
   AND DEFINED ENV{NVTE_CUDA_ARCHS} AND NOT "$ENV{NVTE_CUDA_ARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "$ENV{NVTE_CUDA_ARCHS}")
endif()

project(transformer_engine_benchmarks LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../transformer_engine/common"
  "${CMAKE_CURRENT_BINARY_DIR}/transformer_engine_common")

add_executable(
  benchmark_grouped_fp8_block_quantize
  linear/benchmark_grouped_fp8_block_quantize_native.cu)

target_link_libraries(
  benchmark_grouped_fp8_block_quantize
  PRIVATE
    transformer_engine
    CUDA::cudart
    CUDA::nvrtc
    Threads::Threads)

target_include_directories(
  benchmark_grouped_fp8_block_quantize
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../transformer_engine/common/include")

set_target_properties(
  benchmark_grouped_fp8_block_quantize
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
