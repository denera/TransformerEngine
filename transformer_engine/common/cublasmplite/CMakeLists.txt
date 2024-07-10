##########################################################################
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
##########################################################################

cmake_minimum_required (VERSION 3.18)
project(cublasmplite LANGUAGES CXX CUDA)

# Configure dependencies
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}" CACHE STRING "Location of NVSHMEM installation")
set(NCCL_HOME "$ENV{NCCL_HOME}" CACHE STRING "Location of NCCL installation")
message(STATUS "Building cublasmplite with NVSHMEM_HOME=${NVSHMEM_HOME}, NCCL_HOME=${NCCL_HOME} and CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

add_subdirectory(libcublasmplite)
add_subdirectory(tests)

get_directory_property(has_parent PARENT_DIRECTORY)
if(has_parent)
    set(CUBLASMPLITE_INCLUDE_DIR "${CUBLASMPLITE_INCLUDE_DIR}" PARENT_SCOPE)
endif()