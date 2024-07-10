/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <memory>
#include <cstdio>
#include <cublas_v2.h>
#include <iostream>
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include "macros.hpp.inc"

#include "cublasmplite.h"
#include "cublas_helpers.hpp"
#include "gemm.hpp"

using namespace cublasmplite;

static const size_t num_max_streams = 3;

cublasmp_split_overlap_t::cublasmp_split_overlap_t(std::unique_ptr<nvshmem_pipelined_p2p_t> p2p, size_t m, size_t n, size_t k,
                                                   std::vector<stream_t> compute, stream_t send, stream_t recv,
                                                   event_t start_comms, event_t start_compute, event_t stop_compute, event_t stop_send, event_t stop_recv) : 
    p2p(std::move(p2p)), m(m), n(n), k(k),
    compute(std::move(compute)), send(std::move(send)), recv(std::move(recv)), 
    start_comms(std::move(start_comms)), start_compute(std::move(start_compute)), stop_compute(std::move(stop_compute)), stop_send(std::move(stop_send)), stop_recv(std::move(stop_recv))
    {};

std::unique_ptr<cublasmp_split_overlap_t> cublasmp_split_overlap_t::create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k, nvshmem_pipelined_p2p_t::signal_kind signal, nvshmem_pipelined_p2p_t::wait_kind wait) {

    auto p2p = nvshmem_pipelined_p2p_t::create(my_rank, num_ranks, broadcast, num_ranks, signal, wait);

    CUBLASMPLITE_ASSERT(n % num_ranks == 0);

    size_t num_streams = std::min<size_t>(num_ranks, num_max_streams);

    std::vector<stream_t> compute_streams(num_streams);
    stream_t send_stream, recv_stream;
    event_t start_compute, stop_compute, stop_send, stop_recv, start_comms;

    return std::unique_ptr<cublasmp_split_overlap_t>(new cublasmp_split_overlap_t(std::move(p2p), m, n, k,
        std::move(compute_streams), std::move(send_stream), std::move(recv_stream),
        std::move(start_comms), std::move(start_compute), std::move(stop_compute), std::move(stop_send), std::move(stop_recv)));
}

cublasmp_split_overlap_t::~cublasmp_split_overlap_t() {}

status_t cublasmp_split_overlap_t::wait_all_on(cudaStream_t main) {
    CUBLASMPLITE_CUDA_CHECK(cudaEventRecord(start_compute, main));
    CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(send, start_compute, 0));
    CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(recv, start_compute, 0));
    for (const auto& s: compute) {
      CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(s, start_compute, 0));
    }
    return status_t::SUCCESS;
}

status_t cublasmp_split_overlap_t::wait_on_all(cudaStream_t main) {
    for (const auto& s: compute) {
      CUBLASMPLITE_CUDA_CHECK(cudaEventRecord(stop_compute, s));
      CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(main, stop_compute, 0));
    }
    CUBLASMPLITE_CUDA_CHECK(cudaEventRecord(stop_send, send));
    CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(main, stop_send, 0));
    CUBLASMPLITE_CUDA_CHECK(cudaEventRecord(stop_recv, recv));
    CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(main, stop_recv, 0));
    return status_t::SUCCESS;
}

cudaStream_t cublasmp_split_overlap_t::compute_cyclic(size_t i) {
    return compute[i % compute.size()];
}

///////////////////////////

template<typename TA, typename TB, typename TC>
cublasmp_ag_gemm_t<TA, TB, TC>::cublasmp_ag_gemm_t(std::unique_ptr<cublasmp_split_overlap_t> overlap, gemm_t<TA, TB, TC> gemm) : overlap(std::move(overlap)), gemm(std::move(gemm)) {};

template<typename TA, typename TB, typename TC>
std::unique_ptr<cublasmp_ag_gemm_t<TA, TB, TC>> cublasmp_ag_gemm_t<TA, TB, TC>::create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k, nvshmem_pipelined_p2p_t::signal_kind signal, nvshmem_pipelined_p2p_t::wait_kind wait, int comms_sm) {
    auto overlap = cublasmp_split_overlap_t::create(my_rank, num_ranks, broadcast, m, n, k, signal, wait);
    CUBLASMPLITE_ASSERT(n % (size_t)num_ranks == 0);
    const size_t n_chunk = n / (size_t)num_ranks;
    int num_sms = 0;
    CUBLASMPLITE_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    CUBLASMPLITE_ASSERT(num_sms > comms_sm);
    CUBLASMPLITE_ASSERT(comms_sm >= 0);
    gemm_t<TA, TB, TC> gemm(m, n_chunk, k, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, num_sms - comms_sm, 0, 0, nullptr);
    return std::unique_ptr<cublasmp_ag_gemm_t<TA, TB, TC>>(new cublasmp_ag_gemm_t<TA, TB, TC>(std::move(overlap), std::move(gemm)));
}

// weights == m x k, local, row-major,  each PE has its own
// input   == n x k, global & symmetric, row-major, only the ith (n // nPEs x k) chunk matter but everything must be allocated
// output  == n x m, local, row-major
template<typename TA, typename TB, typename TC>
status_t cublasmp_ag_gemm_t<TA, TB, TC>::execute(const TA* weights, TB* symm_input, TC* output, cudaStream_t main) const {

    const int num_pes = overlap->p2p->num_pes();
    const int my_pe = overlap->p2p->this_pe();
    const size_t m = overlap->m;
    const size_t n = overlap->n;
    const size_t k = overlap->k;
    CUBLASMPLITE_ASSERT(n % num_pes == 0);
    const size_t n_chunk = n / (size_t)num_pes;
    const size_t chunk_size = k * n_chunk;
    const size_t output_chunk_size = m * n_chunk;

    // Comm
    const int next_pe = (num_pes + my_pe + 1) % num_pes;
    const int prev_pe = (num_pes + my_pe - 1) % num_pes;

    overlap->p2p->start_pipeline();

    // Global (only few PEs) stream sync
    // Since we are sending data to next_pe in the loop below, we need to wait on next_pe here
    // Since prev_pe is sending data to us in the loop below, we need to signal to prev_pe they we reached this point
    overlap->p2p->ring_sync_on_stream(prev_pe, next_pe, main);

    // All streams wait on main
    overlap->wait_all_on(main);

    for (int i = 0; i < num_pes; i++) {

        size_t send_offset = (num_pes + my_pe - i    ) % num_pes;
        size_t recv_offset = (num_pes + my_pe - i - 1) % num_pes;

        TB*       send_chunk  = symm_input  + send_offset * chunk_size;
        TB*       recv_chunk  = symm_input  + recv_offset * chunk_size;

        const TB* gemm_input  = symm_input  + send_offset * chunk_size;
        TC*       gemm_output = output      + send_offset * output_chunk_size;

        gemm.execute(weights, gemm_input, gemm_output, overlap->compute_cyclic(i));

        if (i < num_pes - 1) {
            overlap->p2p->send_and_signal(send_chunk, send_chunk, chunk_size * sizeof(TB), next_pe, overlap->send);
            overlap->p2p->wait(prev_pe, overlap->recv);
            CUBLASMPLITE_CUDA_CHECK(cudaEventRecord(overlap->stop_recv, overlap->recv));
            CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(overlap->send, overlap->stop_recv, 0));
            CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(overlap->compute_cyclic(i+1), overlap->stop_recv, 0));
        }

    }

    // Main waits on all streams
    overlap->wait_on_all(main);

    return status_t::SUCCESS;
}

template<typename TA, typename TB, typename TC>
cublasmp_ag_gemm_t<TA, TB, TC>::~cublasmp_ag_gemm_t() {};

/////////

template<typename TA, typename TB, typename TC>
cublasmp_gemm_rs_t<TA, TB, TC>::cublasmp_gemm_rs_t(std::unique_ptr<cublasmp_split_overlap_t> overlap, gemm_t<TA, TB, TC> gemm) : overlap(std::move(overlap)), gemm(std::move(gemm)) {};

template<typename TA, typename TB, typename TC>
std::unique_ptr<cublasmp_gemm_rs_t<TA, TB, TC>> cublasmp_gemm_rs_t<TA, TB, TC>::create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k) {
    auto overlap = cublasmp_split_overlap_t::create(my_rank, num_ranks, broadcast, m, n, k, nvshmem_pipelined_p2p_t::signal_kind::set, nvshmem_pipelined_p2p_t::wait_kind::cu_stream_wait);
    CUBLASMPLITE_ASSERT(n % num_ranks == 0);
    CUBLASMPLITE_ASSERT(k % num_ranks == 0);
    const size_t n_chunk = n / num_ranks;
    const size_t k_chunk = k / num_ranks;
    gemm_t<TA, TB, TC> gemm(m, n_chunk, k_chunk, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, 0, 0, 0, nullptr);
    return std::unique_ptr<cublasmp_gemm_rs_t<TA, TB, TC>>(new cublasmp_gemm_rs_t<TA, TB, TC>(std::move(overlap), std::move(gemm)));
}

template<typename T>
__global__ void reduce_kernel(const T* input, T* output, size_t chunk_size, size_t num_chunks) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= chunk_size) {
        return;
    }
    T out = (T)0.0;
    for(size_t i = 0; i < num_chunks; i++) {
        // printf("Reduction %d %d | %d %d %f\n", (int)chunk_size, (int)num_chunks, (int)tid, (int)i, (float)input[i * chunk_size + tid]);
        out += input[i * chunk_size + tid];
    }
    output[tid] = out;
}

template<typename T>
void reduce(const T* input, T* output, size_t chunk_size, size_t num_chunks, cudaStream_t stream) {
    size_t block_size = 256;
    size_t num_blocks = (chunk_size + block_size - 1) / block_size;
    reduce_kernel<T><<<num_blocks, block_size, 0, stream>>>(input, output, chunk_size, num_chunks);
    CUBLASMPLITE_CUDA_CHECK(cudaGetLastError());
}

template<typename TA, typename TB, typename TC>
status_t cublasmp_gemm_rs_t<TA, TB, TC>::execute(const TA* weights, const TB* input, void* workspace, TC* output, cudaStream_t main) const {
    const int num_pes = overlap->p2p->num_pes();
    const int my_pe = overlap->p2p->this_pe();
    const size_t m = overlap->m;
    const size_t n = overlap->n;
    const size_t k = overlap->k;
    CUBLASMPLITE_ASSERT(n % num_pes == 0);
    CUBLASMPLITE_ASSERT(k % num_pes == 0);
    const size_t n_chunk = n / num_pes;
    const size_t k_chunk = k / num_pes;

    // Sync main streams, to ensure we're not writing to a buffer that's not ready yet 
    // TODO: remove, probably wasteful
    overlap->p2p->start_pipeline();

    // Global stream sync
    overlap->p2p->sync_all_on_stream(main);

    // All streams wait on main
    overlap->wait_all_on(main);

    // Each PE outputs the full m * n matrix but partial (ie missing reductions)
    // This is done chunk by chunk, by chunks of size m * (n // nPEs)
    TC* gemm_outputs = (TC*)workspace;
    size_t chunk_size = m * n_chunk;
    // This is then shuffled accross PEs ; each PEs received nPEs chunks of size m * (n // nPEs) == m * n total
    TC* comms_outputs = gemm_outputs + chunk_size * num_pes; 
    CUBLASMPLITE_ASSERT(2 * chunk_size * num_pes * sizeof(TC) == this->workspace_size());

    for(int i = 0; i < num_pes; i++) {

        // GEMM
        {
            // PE+1, PE+2, ..., PE-1, PE
            const int gemm_dst_pe = (my_pe + i + 1) % num_pes;

            const size_t gemm_input_chunk_size = k_chunk * n_chunk; // GEMM B
            const TB* gemm_input = input + gemm_dst_pe * gemm_input_chunk_size;

            // No need to bounce through gemm_outputs when sending to self
            TC* gemm_output = (gemm_dst_pe != my_pe) ? (gemm_outputs + i * chunk_size) : (comms_outputs + i * chunk_size);

            gemm.execute(weights, gemm_input, gemm_output, overlap->compute_cyclic(i)); 
        }

        if (i > 0) {
            
            // Send the previous iteration's output
            TC* send_chunk = gemm_outputs  + chunk_size * (i - 1);
            TC* recv_chunk = comms_outputs + chunk_size * (i - 1);
            int send_rank = (          my_pe + i) % num_pes;
            int recv_rank = (num_pes + my_pe - i) % num_pes;
            
            // Previous GEMM wait
            CUBLASMPLITE_CUDA_CHECK(cudaEventRecord(overlap->start_comms, overlap->compute_cyclic(i-1)));
            CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent((cudaStream_t) overlap->send, overlap->start_comms, 0));
            CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent((cudaStream_t) overlap->recv, overlap->start_comms, 0));

            overlap->p2p->send_and_signal(send_chunk, recv_chunk, chunk_size * sizeof(TC), send_rank, overlap->send);
            overlap->p2p->wait(recv_rank, overlap->recv);

        }
    }
    
    // Wait
    CUBLASMPLITE_CUDA_CHECK(cudaEventRecord(overlap->stop_recv, overlap->recv));
    CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(main, overlap->stop_recv, 0));

    // Reduce GEMM output chunks
    CUBLASMPLITE_ASSERT(m * n_chunk * num_pes == m * n);
    reduce<TC>(comms_outputs, output, m * n_chunk, num_pes, main);
    
    // Local sync (probably can skip)
    for(const auto& s: overlap->compute) {
      CUBLASMPLITE_CUDA_CHECK(cudaEventRecord(overlap->stop_compute, s));
      CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(main, overlap->stop_compute, 0));
    }
    CUBLASMPLITE_CUDA_CHECK(cudaEventRecord(overlap->stop_send, overlap->send));
    CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(main, overlap->stop_send, 0));

    return status_t::SUCCESS;

}

template<typename TA, typename TB, typename TC>
cublasmp_gemm_rs_t<TA, TB, TC>::~cublasmp_gemm_rs_t() {};

////////////

template<typename TA, typename TB, typename TC>
cublasmp_gemm_rs_atomic_t<TA, TB, TC>::cublasmp_gemm_rs_atomic_t(std::unique_ptr<cublasmp_split_overlap_t> _overlap, device_vector_t<int32_t> _counters, gemm_t<TA, TB, TC> _gemm) : overlap(std::move(_overlap)), counters(std::move(_counters)), gemm(std::move(_gemm)) {};

template<typename TA, typename TB, typename TC>
std::unique_ptr<cublasmp_gemm_rs_atomic_t<TA, TB, TC>> cublasmp_gemm_rs_atomic_t<TA, TB, TC>::create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k) {
    auto overlap = cublasmp_split_overlap_t::create(my_rank, num_ranks, broadcast, m, n, k, nvshmem_pipelined_p2p_t::signal_kind::set, nvshmem_pipelined_p2p_t::wait_kind::cu_stream_wait);
    CUBLASMPLITE_ASSERT(k % num_ranks == 0);
    CUBLASMPLITE_ASSERT(n % num_ranks == 0);
    const size_t k_chunk = k / num_ranks;
    std::vector<int32_t> data(num_ranks, 1);
    device_vector_t<int32_t> counters(data);
    // Each PE computes a giant m x n gemm, but with only k_chunk reduced entries
    // The output is streamed in chunks of n_chunk columns, n_chunk = n // num_ranks
    gemm_t<TA, TB, TC> gemm(m, n, k_chunk, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, 64 /** FIXME: Pick right number **/, 1 /** row **/, num_ranks /** col **/, counters.data());

    return std::unique_ptr<cublasmp_gemm_rs_atomic_t<TA, TB, TC>>(new cublasmp_gemm_rs_atomic_t<TA, TB, TC>(std::move(overlap), std::move(counters), std::move(gemm)));
}

__global__ void print(int* ptr) {
    printf("print %p = %d\n", ptr, *ptr);
}

template<typename TA, typename TB, typename TC>
status_t cublasmp_gemm_rs_atomic_t<TA, TB, TC>::execute(const TA* weights, const TB* input, void* workspace, TC* output, cudaStream_t main) const {
    
    const int num_pes = overlap->p2p->num_pes();
    const int my_pe = overlap->p2p->this_pe();
    const size_t m = overlap->m;
    const size_t n = overlap->n;
    const size_t k = overlap->k;
    CUBLASMPLITE_ASSERT(n % num_pes == 0);
    const size_t n_chunk = n / num_pes;

    overlap->p2p->start_pipeline();

    // Global sync main streams, to ensure we're not writing to a buffer that's not ready yet 
    // TODO: check, could be wasteful
    overlap->p2p->sync_all_on_stream(main);

    // All streams wait on main
    overlap->wait_all_on(main);

    TC* gemm_outputs = (TC*)workspace;
    TC* comms_outputs = gemm_outputs + m * n;
    size_t chunk_size = m * n_chunk;
    CUBLASMPLITE_ASSERT(2 * m * n * sizeof(TC) == this->workspace_size());
    CUBLASMPLITE_ASSERT(counters.size() == (size_t)num_pes);

    gemm.execute(weights, input, gemm_outputs, counters.data(), main);

    // P2P communication chunk
    // FIXME: ordering sucks
    for (int i = 0; i < num_pes; i++) {
        
        // Send the previous iteration's output
        int dst_pe = i;
        int recv_pe = i;
        TC* send_chunk = gemm_outputs  + chunk_size * dst_pe;
        TC* recv_chunk = comms_outputs + chunk_size * my_pe;

        // Wait until counter reaches 0, then set it to 1 - atomically
        overlap->p2p->wait_on_atomic_and_set(counters.data() + dst_pe, 0, 1, overlap->recv);
        overlap->p2p->send_and_signal(send_chunk, recv_chunk, chunk_size * sizeof(TC), dst_pe, overlap->recv);
        overlap->p2p->wait(recv_pe, overlap->recv);

    }

    CUBLASMPLITE_CUDA_CHECK(cudaEventRecord(overlap->stop_recv, overlap->recv));
    CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(main, overlap->stop_recv, 0));

    // Reduce GEMM output chunks
    CUBLASMPLITE_ASSERT(m * n_chunk * num_pes == m * n);
    reduce<TC>(comms_outputs, output, m * n_chunk, num_pes, main);

    return status_t::SUCCESS;
}

////////////

template cublasmp_ag_gemm_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>::cublasmp_ag_gemm_t(std::unique_ptr<cublasmp_split_overlap_t>, gemm_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>);
template cublasmp_gemm_rs_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>::cublasmp_gemm_rs_t(std::unique_ptr<cublasmp_split_overlap_t>, gemm_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>);
template cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>::cublasmp_gemm_rs_atomic_t(std::unique_ptr<cublasmp_split_overlap_t>, device_vector_t<int32_t>, gemm_t<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>);
template cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, __half>::cublasmp_gemm_rs_atomic_t(std::unique_ptr<cublasmp_split_overlap_t>, device_vector_t<int32_t>, gemm_t<__nv_fp8_e4m3, __nv_fp8_e4m3, __half>);

template std::unique_ptr<cublasmp_ag_gemm_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>> cublasmp_ag_gemm_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>::create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k, nvshmem_pipelined_p2p_t::signal_kind, nvshmem_pipelined_p2p_t::wait_kind, int comm_sms);
template std::unique_ptr<cublasmp_gemm_rs_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>> cublasmp_gemm_rs_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>::create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k);
template std::unique_ptr<cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>> cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>::create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k);
template std::unique_ptr<cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, __half>> cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, __half>::create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k);

template status_t cublasmp_ag_gemm_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>::execute(const nv_bfloat16* weights, nv_bfloat16* input, nv_bfloat16* output, cudaStream_t main) const;
template status_t cublasmp_gemm_rs_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>::execute(const nv_bfloat16* weights, const nv_bfloat16* input, void* workspace, nv_bfloat16* output, cudaStream_t main) const ;
template status_t cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>::execute(const __nv_fp8_e4m3* weights, const __nv_fp8_e4m3* input, void* workspace, nv_bfloat16* output, cudaStream_t main) const ;
template status_t cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, __half>::execute(const __nv_fp8_e4m3* weights, const __nv_fp8_e4m3* input, void* workspace, __half* output, cudaStream_t main) const ;

template cublasmp_ag_gemm_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>::~cublasmp_ag_gemm_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>();
template cublasmp_gemm_rs_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>::~cublasmp_gemm_rs_t<nv_bfloat16, nv_bfloat16, nv_bfloat16>();
template cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>::~cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, nv_bfloat16>();
template cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, __half>::~cublasmp_gemm_rs_atomic_t<__nv_fp8_e4m3, __nv_fp8_e4m3, __half>();