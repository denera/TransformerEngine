/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_CUBLASMPLITE_H
#define TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_CUBLASMPLITE_H

#include <vector>
#include <memory>
#include <variant>
#include <cstdint>
#include <functional>
#include <tuple>

#include "vector.hpp"
#include "gemm.hpp"
#include "cuda_helpers.hpp"

namespace cublasmplite {

using broadcast_fun_type = std::function<void(void*, size_t, int, int)>;

enum class status_t {
    SUCCESS,
    ERROR,
};

// ~Stateless class to encapsulate NVSHMEM init/finalize/alloc/free
class nvshmem_comm_t {

protected:
    const int my_pe;
    const int n_pes;
    nvshmem_comm_t();

public:
    static status_t                   initialize(int my_rank, int num_ranks, broadcast_fun_type broadcast);
    std::unique_ptr<nvshmem_comm_t>   create(int my_rank, int num_ranks, broadcast_fun_type broadcast);
    int                               this_pe() const;
    int                               num_pes() const;
    status_t                          barrier_all();
    status_t                          sync_all_on_stream(cudaStream_t stream);
    status_t                          barrier_all_on_stream(cudaStream_t stream);
    void*                             malloc(size_t size);
    void                              free(void* ptr);
    status_t                          wait_on_atomic_and_set(int* flag, int signal, int value, cudaStream_t stream);
    status_t                          set(int* flag, int value, cudaStream_t stream);
    ~nvshmem_comm_t();

    template<typename T> nvshmem_vector_t<T>  make_vector(size_t size);
    template<typename T> nvshmem_vector_t<T>  make_vector(const std::vector<T>& data);
};

/**
 * This is used for pipelined send/wait ops between PEs
 * This handles
 * - On the sender side, sending and signaling
 * - On the receiver side, waiting
 * The pipeling depth must be specified ahead of time
 * 
 * Starting and ending a pipelined requires an explict call to start() and finalize()
 */
class nvshmem_pipelined_p2p_t : public nvshmem_comm_t {
public:
    enum class signal_kind { set = 0, add = 1 };
    enum class wait_kind { nvshmem_wait = 0, cu_stream_wait = 1 };

    static signal_kind get_signal_kind(int k);
    static wait_kind get_wait_kind(int k);

private:
    signal_kind signalk;
    wait_kind waitk;

    // How many max pipelined send/recv can we have in flight at a given time?
    int pipeline_depth;

    // device symmetric, nPEs * pipeline_depth
    // flags used to notify arrival of data and wait on data
    nvshmem_vector_t<uint64_t> signal_flags;

    // device symmetric, nPEs
    // flags used to synchronize pairs of PEs
    nvshmem_vector_t<uint64_t> sync_flag; 
    
    // host, nPEs
    // What step are we at in the pipeline (on the signaling and on the wait side)
    std::vector<uint64_t> signals_step;
    std::vector<uint64_t> waits_step;
    
    nvshmem_pipelined_p2p_t(int pipeline_depth, signal_kind signal, wait_kind wait);
    size_t idx(int step, int pe);
    uint64_t* next_signal(int pe);
    uint64_t* next_wait(int pe);

public:
    static std::unique_ptr<nvshmem_pipelined_p2p_t> create(int my_rank, int num_ranks, broadcast_fun_type broadcast, int pipeline_depth, signal_kind signal, wait_kind wait);
    status_t send_and_signal(const void* src, void* dst, size_t size, int peer, cudaStream_t stream);
    status_t wait(int peer, cudaStream_t stream);
    // Reset internal pipeline depth counter
    status_t start_pipeline();
    // Synchronize pairs of PEs (useful on rings)
    status_t ring_sync_on_stream(int signal_rank, int wait_rank, cudaStream_t stream);
    ~nvshmem_pipelined_p2p_t() {};
};

class nvshmem_reduce_scatter_t : public nvshmem_comm_t {

private:
    nvshmem_vector_t<uint64_t> flags; // symmetric, one flag per PE
    uint64_t counter;
    nvshmem_reduce_scatter_t();

public:
    
    static  std::unique_ptr<nvshmem_reduce_scatter_t> create(int my_rank, int num_ranks, broadcast_fun_type broadcast);
    template<typename T> status_t reduce_scatter(const T* src, size_t src_rows, size_t src_cols, size_t src_ld, T* dst, size_t dst_ld, cudaStream_t stream);
    ~nvshmem_reduce_scatter_t() {};
};

class cublasmp_split_overlap_t {

private:
    cublasmp_split_overlap_t(std::unique_ptr<nvshmem_pipelined_p2p_t> p2p, size_t m, size_t n, size_t k,
                             std::vector<stream_t> compute, stream_t send, stream_t recv,
                             event_t start_comms, event_t start_compute, event_t stop_compute, event_t stop_send, event_t stop_recv);
public: 
    static std::unique_ptr<cublasmp_split_overlap_t> create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k, nvshmem_pipelined_p2p_t::signal_kind signal, nvshmem_pipelined_p2p_t::wait_kind wait);
    ~cublasmp_split_overlap_t();

    // Same on all PEs
    const size_t m;
    const size_t n;
    const size_t k;
    const std::unique_ptr<nvshmem_pipelined_p2p_t> p2p;

    const std::vector<stream_t> compute;
    const stream_t send;
    const stream_t recv;

    const event_t start_comms;
    const event_t start_compute;
    const event_t stop_compute;
    const event_t stop_send;
    const event_t stop_recv;

    // At the beginning - wait all steams on main
    status_t wait_all_on(cudaStream_t main);

    // At the end - main waits on all streams
    status_t  wait_on_all(cudaStream_t main);

    // Returns the ith compute stream, looping back at the end
    cudaStream_t compute_cyclic(size_t i);
};

template<typename TA, typename TB, typename TC>
class cublasmp_ag_gemm_t {

private:

    std::unique_ptr<cublasmp_split_overlap_t> overlap;
    cublasmp_ag_gemm_t(std::unique_ptr<cublasmp_split_overlap_t>  overlap, gemm_t<TA, TB, TC> gemm);
    const gemm_t<TA, TB, TC> gemm;

public:

    static std::unique_ptr<cublasmp_ag_gemm_t<TA, TB, TC>> create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k, nvshmem_pipelined_p2p_t::signal_kind signal, nvshmem_pipelined_p2p_t::wait_kind wait, int comms_sm);
    status_t execute(const TA* A, TB* B, TC* C, cudaStream_t main) const;
    nvshmem_pipelined_p2p_t* p2p() { return overlap->p2p.get(); }
    ~cublasmp_ag_gemm_t();

};


template<typename TA, typename TB, typename TC>
class cublasmp_gemm_rs_t {

private:

    std::unique_ptr<cublasmp_split_overlap_t> overlap;
    cublasmp_gemm_rs_t(std::unique_ptr<cublasmp_split_overlap_t>  overlap, gemm_t<TA, TB, TC> gemm);
    const gemm_t<TA, TB, TC> gemm;

public:

    static  std::unique_ptr<cublasmp_gemm_rs_t<TA, TB, TC>> create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k);
    status_t execute(const TA* A, const TB* B, void* workspace, TC* C, cudaStream_t main) const;
    nvshmem_pipelined_p2p_t* p2p() { return overlap->p2p.get(); }
    size_t workspace_size() const { return 2 * overlap->m * overlap-> n * sizeof(TC); }
    ~cublasmp_gemm_rs_t();

};

template<typename TA, typename TB, typename TC>
class cublasmp_gemm_rs_atomic_t {

private:

    std::unique_ptr<cublasmp_split_overlap_t> overlap;
    cublasmp_gemm_rs_atomic_t(std::unique_ptr<cublasmp_split_overlap_t> overlap, device_vector_t<int32_t> counters, gemm_t<TA, TB, TC> gemm);
    device_vector_t<int32_t> counters;
    const gemm_t<TA, TB, TC> gemm;

public:

    static std::unique_ptr<cublasmp_gemm_rs_atomic_t<TA, TB, TC>> create(int my_rank, int num_ranks, broadcast_fun_type broadcast, size_t m, size_t n, size_t k);
    status_t execute(const TA* A, const TB* B, void* workspace, TC* C, cudaStream_t main) const;
    nvshmem_pipelined_p2p_t* p2p() { return overlap->p2p.get(); }
    size_t workspace_size() const { return 2 * overlap->m * overlap-> n * sizeof(TC); }
    ~cublasmp_gemm_rs_atomic_t() {};

};

}

#endif // TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_CUBLASMPLITE_H