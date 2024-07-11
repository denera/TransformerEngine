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
#include "nvshmem_comm.hpp"

#include "macros.hpp.inc"

using namespace cublasmplite;

nvshmem_pipelined_p2p_t::signal_kind nvshmem_pipelined_p2p_t::get_signal_kind(int k) {
    switch (k) {
        case 0:
            return nvshmem_pipelined_p2p_t::signal_kind::set;
        case 1:
            return nvshmem_pipelined_p2p_t::signal_kind::add;
        default:
            CUBLASMPLITE_ASSERT(false);
    }
    return nvshmem_pipelined_p2p_t::signal_kind::set;
}

nvshmem_pipelined_p2p_t::wait_kind nvshmem_pipelined_p2p_t::get_wait_kind(int k) {
    switch (k) {
        case 0:
            return nvshmem_pipelined_p2p_t::wait_kind::nvshmem_wait_device;
        case 1:
            return nvshmem_pipelined_p2p_t::wait_kind::nvshmem_wait;
        case 2:
            return nvshmem_pipelined_p2p_t::wait_kind::cu_stream_wait;
        default:
            CUBLASMPLITE_ASSERT(false);
    }
    return nvshmem_pipelined_p2p_t::wait_kind::nvshmem_wait;
}

nvshmem_pipelined_p2p_t::nvshmem_pipelined_p2p_t(int pipeline_depth, signal_kind signal, wait_kind wait) :
    nvshmem_comm_t(), signalk(signal), waitk(wait),
    pipeline_depth(pipeline_depth), 
    signal_flags(n_pes * pipeline_depth, (uint64_t)0),
    sync_flag(n_pes, (uint64_t)0),
    signals_step(n_pes, (uint64_t)0),
    waits_step(n_pes, (uint64_t)0)
    {
        CUdevice dev;
        CUBLASMPLITE_CU_CHECK(cuCtxGetDevice(&dev));
        // Check that cuStreamWaitValue/cuStreamWriteValue are supported
        int memops = 0;
        CUBLASMPLITE_CU_CHECK(cuDeviceGetAttribute(&memops, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, dev));
        CUBLASMPLITE_ASSERT(memops == 1);
        // This is necessary for NVSHMEM to use cuStreamWaitValue 
        {
            int flush = 0;
            int rdma_flush = 0;
            CUBLASMPLITE_CU_CHECK(cuDeviceGetAttribute(&flush, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, dev));
            CUBLASMPLITE_CU_CHECK(cuDeviceGetAttribute(&rdma_flush, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, dev));
            if(!flush) {
                printf("Note: CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES %d CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS %d.\n", flush, rdma_flush);
            }
        }
    }

std::unique_ptr<nvshmem_pipelined_p2p_t> nvshmem_pipelined_p2p_t::create(int my_rank, int num_ranks, broadcast_fun_type broadcast, int pipeline_depth, nvshmem_pipelined_p2p_t::signal_kind signal, nvshmem_pipelined_p2p_t::wait_kind wait) {
    CUBLASMPLITE_ASSERT(nvshmem_comm_t::initialize(my_rank, num_ranks, broadcast) == status_t::SUCCESS);
    return std::unique_ptr<nvshmem_pipelined_p2p_t>(new nvshmem_pipelined_p2p_t(pipeline_depth, signal, wait));
}

size_t nvshmem_pipelined_p2p_t::idx(int step, int pe) {
    CUBLASMPLITE_ASSERT(step >= 0 && step < this->pipeline_depth);
    CUBLASMPLITE_ASSERT(pe >= 0 && pe < this->num_pes());
    return (size_t)step * (size_t)this->n_pes + (size_t)pe;
}

uint64_t* nvshmem_pipelined_p2p_t::next_signal(int dst_pe) {
    CUBLASMPLITE_ASSERT(dst_pe >= 0 && dst_pe < this->n_pes);
    int step = signals_step[dst_pe];
    CUBLASMPLITE_ASSERT(step >= 0 && step < this->pipeline_depth);
    signals_step[dst_pe]++;
    uint64_t* flag = signal_flags.data() + idx(step, my_pe);
    return flag;
}

uint64_t* nvshmem_pipelined_p2p_t::next_wait(int src_pe) {
    CUBLASMPLITE_ASSERT(src_pe >= 0 && src_pe < this->n_pes);
    int step = waits_step[src_pe];
    CUBLASMPLITE_ASSERT(step >= 0 && step < this->pipeline_depth);
    waits_step[src_pe]++;
    uint64_t* flag = signal_flags.data() + idx(step, src_pe);
    return flag;
}

status_t nvshmem_pipelined_p2p_t::send_and_signal(const void* src, void* dst, size_t size, int peer, cudaStream_t stream) {
    // Push-send mode
    uint64_t* flag = this->next_signal(peer);
    uint64_t signal_value = 1;
    // Starting value is always 0, so we can either add 1 or set to 1, it's the same
    int sig_op = 0;
    if(this->signalk == signal_kind::set) {
        sig_op = NVSHMEM_SIGNAL_SET;
    } else if(this->signalk == signal_kind::add) {
        sig_op = NVSHMEM_SIGNAL_ADD;
    } else {
        CUBLASMPLITE_ASSERT(false);
    }
    char* ptr_dst = (char*)dst;
    const char* ptr_src = (const char*)src;
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] putmem_signal %p -> %p (pe %d/%d, flag %p, signal %d, op %d, stream %p)\n", my_pe, ptr_src, ptr_dst, peer, this->n_pes, flag, (int)signal_value, sig_op, (void*)stream);
    }
    nvshmemx_putmem_signal_on_stream(ptr_dst, ptr_src, size, flag, signal_value, sig_op, peer, stream);
    return status_t::SUCCESS;
}

__global__ void wait_until_on_stream_and_reset(uint64_t* wait_flag, uint64_t wait_value, uint64_t signal_reset) {
    nvshmem_uint64_wait_until(wait_flag, NVSHMEM_CMP_EQ, wait_value);
    *wait_flag = signal_reset;
}

status_t nvshmem_pipelined_p2p_t::wait(int peer, cudaStream_t stream) {
    // Push-send mode
    uint64_t* flag = this->next_wait(peer);
    uint64_t wait_value = 1;
    uint64_t signal_reset = 0;
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] wait_until (pe %d/%d, flag %p, signal %d, stream %p, waitk %d)\n", my_pe, peer, this->n_pes, flag, (int)wait_value, (void*)stream, (int)this->waitk);
    }
    // Single kernel to wait and reset
    if(this->waitk == wait_kind::nvshmem_wait_device) {
        wait_until_on_stream_and_reset<<<1, 1, 0, stream>>>(flag, wait_value, signal_reset);
        CUBLASMPLITE_CUDA_CHECK(cudaGetLastError());
    // NVSHMEM on_stream API to wait + cuStreamWriteValue to reset
    } else if(this->waitk == wait_kind::nvshmem_wait) {
        // Wait until value GEQ > 1
        nvshmemx_uint64_wait_until_on_stream(flag, NVSHMEM_CMP_EQ, wait_value, stream);
        // Reset local flag to 0
        CUBLASMPLITE_CU_CHECK(cuStreamWriteValue64((CUstream)stream, (CUdeviceptr)flag, (cuuint64_t)signal_reset, CU_STREAM_WRITE_VALUE_DEFAULT));
    // This may fail because of CUDA VMM bug on < 12.5. Try NVSHMEM_DISABLE_CUDA_VMM=1.
    } else if(this->waitk == wait_kind::cu_stream_wait) {
        // Wait until value GEQ > 1
        CUBLASMPLITE_CU_CHECK(cuStreamWaitValue64((CUstream)stream, (CUdeviceptr)flag, (cuuint64_t)wait_value, CU_STREAM_WAIT_VALUE_GEQ));
        // Reset local flag to 0
        CUBLASMPLITE_CU_CHECK(cuStreamWriteValue64((CUstream)stream, (CUdeviceptr)flag, (cuuint64_t)signal_reset, CU_STREAM_WRITE_VALUE_DEFAULT));
    } else {
        CUBLASMPLITE_ASSERT(false);
    }
    return status_t::SUCCESS;
}

__global__ void ring_sync_on_stream_kernel(uint64_t* signal_flag, int signal_rank, uint64_t* wait_flag) {
    nvshmemx_signal_op(signal_flag, 1, NVSHMEM_SIGNAL_SET, signal_rank);
    nvshmem_uint64_wait_until(wait_flag, NVSHMEM_CMP_EQ, 1);
    *wait_flag = 0;
}

// signals signal_rank and wait on signal from wait_rank
status_t nvshmem_pipelined_p2p_t::ring_sync_on_stream(int signal_rank, int wait_rank, cudaStream_t stream) {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] pipeline_sync_on_stream, signal_rank %d, wait_rank %d, stream %p\n", my_pe, signal_rank, wait_rank, stream);
    }
    CUBLASMPLITE_ASSERT(signal_rank >= 0 && signal_rank < n_pes);
    CUBLASMPLITE_ASSERT(wait_rank >= 0 && wait_rank < n_pes);
    
    uint64_t* signal_flag = this->sync_flag.data() + my_pe;
    uint64_t* wait_flag   = this->sync_flag.data() + wait_rank;
    
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] sync_on_stream with pe %d,%d/%d (signal %p wait %p)\n", my_pe, signal_rank, wait_rank, n_pes, signal_flag, wait_flag);
    }
    ring_sync_on_stream_kernel<<<1, 1, 0, stream>>>(signal_flag, signal_rank, wait_flag);
    CUBLASMPLITE_CUDA_CHECK(cudaGetLastError());
    return status_t::SUCCESS;
}

status_t nvshmem_pipelined_p2p_t::start_pipeline() {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] start_pipeline\n", my_pe);
    }
    std::fill(this->signals_step.begin(), this->signals_step.end(), 0);
    std::fill(this->waits_step.begin(), this->waits_step.end(), 0);
    return status_t::SUCCESS;
}