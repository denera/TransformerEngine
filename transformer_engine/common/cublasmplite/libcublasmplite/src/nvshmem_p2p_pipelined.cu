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
            return nvshmem_pipelined_p2p_t::wait_kind::nvshmem_wait;
        case 1:
            return nvshmem_pipelined_p2p_t::wait_kind::cu_stream_wait;
        default:
            CUBLASMPLITE_ASSERT(false);
    }
    return nvshmem_pipelined_p2p_t::wait_kind::nvshmem_wait;
}

nvshmem_pipelined_p2p_t::nvshmem_pipelined_p2p_t(int pipeline_depth, signal_kind signal, wait_kind wait) :
    nvshmem_comm_t(), signalk(signal), waitk(wait),
    pipeline_depth(pipeline_depth), 
    flags(n_pes * pipeline_depth, (uint64_t)0),
    sync_flag(n_pes, (uint64_t)0),
    signals_step(n_pes, (uint64_t)0),
    waits_step(n_pes, (uint64_t)0), 
    signals(n_pes * pipeline_depth, (uint64_t)0),
    waits(n_pes * pipeline_depth, (uint64_t)0),
    sync_signals(n_pes, (uint64_t)0),
    sync_waits(n_pes, (uint64_t)0)
    {}

std::unique_ptr<nvshmem_pipelined_p2p_t> nvshmem_pipelined_p2p_t::create(int my_rank, int num_ranks, broadcast_fun_type broadcast, int pipeline_depth, nvshmem_pipelined_p2p_t::signal_kind signal, nvshmem_pipelined_p2p_t::wait_kind wait) {
    CUBLASMPLITE_ASSERT(nvshmem_comm_t::initialize(my_rank, num_ranks, broadcast) == status_t::SUCCESS);
    return std::unique_ptr<nvshmem_pipelined_p2p_t>(new nvshmem_pipelined_p2p_t(pipeline_depth, signal, wait));
}

size_t nvshmem_pipelined_p2p_t::idx(int step, int pe) {
    CUBLASMPLITE_ASSERT(step >= 0 && step < this->pipeline_depth);
    CUBLASMPLITE_ASSERT(pe >= 0 && pe < this->num_pes());
    return (size_t)step * (size_t)this->n_pes + (size_t)pe;
}

std::tuple<int, uint64_t, uint64_t*> nvshmem_pipelined_p2p_t::next_signal(int dst_pe) {
    CUBLASMPLITE_ASSERT(dst_pe >= 0 && dst_pe < this->n_pes);
    int step = signals_step[dst_pe];
    CUBLASMPLITE_ASSERT(step >= 0 && step < this->pipeline_depth);
    signals_step[dst_pe]++;
    uint64_t* flag = flags.data() + idx(step, my_pe);
    uint64_t signal_value = 0;
    int sig_op = 0;
    if(this->signalk == signal_kind::set) {
        signal_value = signals[idx(step, dst_pe)] + 1;
        sig_op = NVSHMEM_SIGNAL_SET;
        signals[idx(step, dst_pe)]++;
    } else if (this->signalk == signal_kind::add) {
        signal_value = 1;
        sig_op = NVSHMEM_SIGNAL_ADD;
        signals[idx(step, dst_pe)]++;
    } else {
        CUBLASMPLITE_ASSERT(false);
    }
    return {sig_op, signal_value, flag};
}

std::tuple<uint64_t, uint64_t*> nvshmem_pipelined_p2p_t::next_wait(int src_pe) {
    CUBLASMPLITE_ASSERT(src_pe >= 0 && src_pe < this->n_pes);
    int step = waits_step[src_pe];
    CUBLASMPLITE_ASSERT(step >= 0 && step < this->pipeline_depth);
    waits_step[src_pe]++;
    uint64_t* f = flags.data() + idx(step, src_pe);
    uint64_t w = waits[idx(step, src_pe)] + 1;
    waits[idx(step, src_pe)] = w;
    return {w, f};
}

status_t nvshmem_pipelined_p2p_t::send_and_signal(const void* src, void* dst, size_t size, int peer, cudaStream_t stream) {
    // Push-send mode
    auto [sig_op, signal, flag] = this->next_signal(peer);
    char* ptr_dst = (char*)dst;
    const char* ptr_src = (const char*)src;
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] putmem %p -> %p (pe %d/%d, flag %p, signal %d, op %d, stream %p)\n", my_pe, ptr_src, ptr_dst, peer, this->n_pes, flag, (int)signal, sig_op, (void*)stream);
    }
    nvshmemx_putmem_signal_on_stream(ptr_dst, ptr_src, size, flag, signal, sig_op, peer, stream);
    return status_t::SUCCESS;
}

status_t nvshmem_pipelined_p2p_t::wait(int peer, cudaStream_t stream) {
    // Push-send mode
    auto [signal, flag] = this->next_wait(peer);
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] wait until (pe %d/%d, flag %p, signal %d, stream %p)\n", my_pe, peer, this->n_pes, flag, (int)signal, (void*)stream);
    }
    if(this->waitk == wait_kind::cu_stream_wait) {
        cuStreamWaitValue64((CUstream)stream, (CUdeviceptr)flag, (cuuint64_t)signal, 0);
    } else if(this->waitk == wait_kind::nvshmem_wait) {
        int sig_op = NVSHMEM_CMP_EQ;
        nvshmemx_uint64_wait_until_on_stream(flag, sig_op, signal, stream);
    } else {
        CUBLASMPLITE_ASSERT(false);
    }
    return status_t::SUCCESS;
}

__global__ void pipeline_sync_on_stream_kernel(uint64_t* signal_flag, int signal_rank, uint64_t signal, uint64_t* wait_flag, uint64_t wait) {
    nvshmemx_signal_op(signal_flag, signal, NVSHMEM_SIGNAL_SET, signal_rank);
    nvshmem_uint64_wait_until(wait_flag, NVSHMEM_CMP_EQ, wait);
}

// signals signal_rank and wait on signal from wait_rank
status_t nvshmem_pipelined_p2p_t::pipeline_sync_on_stream(int signal_rank, int wait_rank, cudaStream_t stream) {
    CUBLASMPLITE_ASSERT(signal_rank >= 0 && signal_rank < n_pes);
    CUBLASMPLITE_ASSERT(wait_rank >= 0 && wait_rank < n_pes);
    
    uint64_t* signal_flag = this->sync_flag.data() + my_pe;
    uint64_t  signal      = this->sync_signals[signal_rank] + 1;
    this->sync_signals[signal_rank] += 1;

    uint64_t* wait_flag = this->sync_flag.data() + wait_rank;
    uint64_t  wait      = this->sync_waits[wait_rank] + 1;
    this->sync_waits[wait_rank] += 1;
    
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] sync_on_stream with pe %d,%d/%d (signal %p %d, wait %p %d)\n", my_pe, signal_rank, wait_rank, n_pes, signal_flag, (int)signal, wait_flag, (int)wait);
    }
    pipeline_sync_on_stream_kernel<<<1, 1, 0, stream>>>(signal_flag, signal_rank, signal, wait_flag, wait);
    CUBLASMPLITE_CUDA_CHECK(cudaGetLastError());
    return status_t::SUCCESS;
}

status_t nvshmem_pipelined_p2p_t::start_pipeline() {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] Start pipeline\n", my_pe);
    }
    std::fill(this->signals_step.begin(), this->signals_step.end(), 0);
    std::fill(this->waits_step.begin(), this->waits_step.end(), 0);
    return status_t::SUCCESS;
}