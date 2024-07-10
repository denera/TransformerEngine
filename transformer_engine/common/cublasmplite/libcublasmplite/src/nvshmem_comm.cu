/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <nvshmem.h>
#include <cuda.h>

#include <cstdio>
#include <cuda_bf16.h>
#include <string>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>

#include "cublasmplite.h"

#include "macros.hpp.inc"

using namespace cublasmplite;

extern const bool TE_NVSHMEM_DEBUG = (std::getenv("NVTE_NVSHMEM_DEBUG") != nullptr && std::string(std::getenv("NVTE_NVSHMEM_DEBUG")) == "1");

// Used in vector.hpp
namespace cublasmplite {

    void* impl_nvshmem_malloc(size_t size) {
        return nvshmem_malloc(size);
    }

    void impl_nvshmem_free(void* ptr) {
        nvshmem_free(ptr);
    }

}

std::unique_ptr<nvshmem_comm_t> nvshmem_comm_t::create(int my_rank, int num_ranks, broadcast_fun_type broadcast) {
    CUBLASMPLITE_ASSERT(nvshmem_comm_t::initialize(my_rank, num_ranks, broadcast) == status_t::SUCCESS);
    return std::unique_ptr<nvshmem_comm_t>(new nvshmem_comm_t());
}

nvshmem_comm_t::~nvshmem_comm_t() {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] NVSHMEM finalizing...\n", my_pe);
    }
    nvshmem_finalize();
}

status_t nvshmem_comm_t::initialize(int my_rank, int num_ranks, broadcast_fun_type broadcast) {
    nvshmemx_init_attr_t attr = {};
    nvshmemx_uniqueid_t id = {};
    if (my_rank == 0) {
       nvshmemx_get_uniqueid(&id);
    }

    broadcast((void*)&id, sizeof(nvshmemx_uniqueid_t), 0, num_ranks);

    if(TE_NVSHMEM_DEBUG) {
        std::stringstream ss;
        ss << std::hex;
        for(size_t b = 0; b < sizeof(nvshmemx_uniqueid_t); b++) {
            ss << (int)((char*)(&id))[b];
        }
        printf("[%d] NVSHMEM initialized with UID PE %d/%d, UID = %s\n", my_rank, my_rank, num_ranks, ss.str().c_str());
    }

    nvshmemx_set_attr_uniqueid_args(my_rank, num_ranks, &id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    CUBLASMPLITE_ASSERT(num_ranks == nvshmem_n_pes());
    CUBLASMPLITE_ASSERT(my_rank == nvshmem_my_pe());

    return status_t::SUCCESS; 
}

nvshmem_comm_t::nvshmem_comm_t() : my_pe(nvshmem_my_pe()), n_pes(nvshmem_n_pes()) {};

void* nvshmem_comm_t::malloc(size_t size) {
    if(size == 0) {
        size = 1;
    }
    void* ptr = nvshmem_malloc(size);
    CUBLASMPLITE_ASSERT(ptr != nullptr);
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] nvshmem_malloc returned %p\n", my_pe, ptr);
    }
    return ptr;
}

void nvshmem_comm_t::free(void* ptr) {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] nvshmem_free %p\n", my_pe, ptr);
    }
    nvshmem_free(ptr);
}

status_t nvshmem_comm_t::barrier_all() {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] barrier_all\n", my_pe);
    }
    nvshmem_barrier_all();
    return status_t::SUCCESS; 
}

status_t nvshmem_comm_t::sync_all_on_stream(cudaStream_t stream) {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] sync_all_on_stream stream %p\n", my_pe, (void*)stream);
    }
    nvshmemx_sync_all_on_stream(stream);
    return status_t::SUCCESS; 
}

status_t nvshmem_comm_t::barrier_all_on_stream(cudaStream_t stream) {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] barrier_all_on_stream stream %p\n", my_pe, (void*)stream);
    }
    nvshmemx_barrier_all_on_stream(stream);
    return status_t::SUCCESS; 
}

int nvshmem_comm_t::this_pe() const {
    return nvshmem_my_pe();
}

int nvshmem_comm_t::num_pes() const {
    return nvshmem_n_pes();
}

__global__ void set_kernel(int *flag, int value) {
    *flag = value;
    asm volatile("fence.sc.gpu;\n");
}

status_t nvshmem_comm_t::set(int* flag, int value, cudaStream_t stream) {
    set_kernel<<<1, 1, 0, stream>>>(flag, value);
    CUBLASMPLITE_CUDA_CHECK(cudaGetLastError());
    return status_t::SUCCESS;
}

// For producer consumer: wait_on_atomic_and_set_kernel(flag, signal=0, value=1)
__global__ void wait_on_atomic_and_set_kernel(int *flag, int signal, int value) {
    while (signal != (atomicCAS(flag, signal, signal))) {
        // spin
    }
    *flag = value;
    // fence, to ensure results are visible to following kernel
    asm volatile("fence.sc.gpu;\n");
}

status_t nvshmem_comm_t::wait_on_atomic_and_set(int* flag, int signal, int value, cudaStream_t stream) {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] wait_on_atomic_and_set flag %p signal %d set %d stream %p\n", my_pe, flag, signal, value, (void*)stream);
    }
    wait_on_atomic_and_set_kernel<<<1, 1, 0, stream>>>(flag, signal, value);
    CUBLASMPLITE_CUDA_CHECK(cudaGetLastError());
    return status_t::SUCCESS;
}

template<typename T> 
nvshmem_vector_t<T> nvshmem_comm_t::make_vector(size_t size) {
    return nvshmem_vector_t<T>(size);
}

template<typename T> 
nvshmem_vector_t<T> nvshmem_comm_t::make_vector(const std::vector<T>& data) {
    return nvshmem_vector_t<T>(data);
}

///////////// Explicit instantiations

template nvshmem_vector_t<nv_bfloat16> nvshmem_comm_t::make_vector<nv_bfloat16>(size_t size);
template nvshmem_vector_t<nv_bfloat16> nvshmem_comm_t::make_vector<nv_bfloat16>(const std::vector<nv_bfloat16>& data);

template nvshmem_vector_t<char> nvshmem_comm_t::make_vector<char>(size_t size);
template nvshmem_vector_t<char> nvshmem_comm_t::make_vector<char>(const std::vector<char>& data);