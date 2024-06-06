# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
cmake_minimum_required (VERSION 3.18)
project(cublasmplite LANGUAGES CXX CUDA)

# Configure dependencies
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}" CACHE STRING "Location of NVSHMEM installation")
set(NCCL_HOME "$ENV{NCCL_HOME}" CACHE STRING "Location of NCCL installation")
message(STATUS "Building TE_NVSHMEM with NVSHMEM_HOME=${NVSHMEM_HOME} and NCCL_HOME=${NCCL_HOME}")

add_subdirectory(libcublasmplite)
add_subdirectory(tests)
