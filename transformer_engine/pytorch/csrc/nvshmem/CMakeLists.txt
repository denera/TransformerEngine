# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
cmake_minimum_required (VERSION 3.18)
project(te_nvshmem LANGUAGES CXX CUDA)

# Configure dependencies
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}" CACHE STRING "Location of NVSHMEM installation")
set(NCCL_HOME "$ENV{NCCL_HOME}" CACHE STRING "Location of NCCL installation")
message(STATUS "Building TE_NVSHMEM with NVSHMEM_HOME=${NVSHMEM_HOME} and NCCL_HOME=${NCCL_HOME}")

# Configure te_nvshmem library
add_library(transformer_engine_nvshmem SHARED
            te_nvshmem.cu cublasmp.cu cuda.cu)
target_link_directories(transformer_engine_nvshmem PRIVATE ${NVSHMEM_HOME}/lib)
target_link_libraries(transformer_engine_nvshmem PRIVATE -static-libstdc++ nvshmem_device nvshmem_host CUDA::nvml CUDA::cublas)
target_include_directories(transformer_engine_nvshmem PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           ${NVSHMEM_HOME}/include/
                           "${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(transformer_engine_nvshmem PROPERTIES
    CUDA_STANDARD 17
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Test driver
foreach(test ag_gemm gemm_rs gemm_rs_atomic)
    add_executable(${test} "${test}.cu")
    target_include_directories(${test} PRIVATE ${NCCL_HOME}/include "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_directories(${test} PRIVATE ${NCCL_HOME}/lib)
    target_link_libraries(${test} PRIVATE MPI::MPI_CXX transformer_engine_nvshmem -static-libstdc++ CUDA::nvml CUDA::cublas CUDA::cuda_driver CUDA::cudart_static nccl)
    set_target_properties(${test} PROPERTIES
        CUDA_STANDARD 17
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()

# Install library
install(TARGETS transformer_engine_nvshmem DESTINATION .)
