#ifndef __TE_NVSHMEM_GEMM_RS_HPP__
#define __TE_NVSHMEM_GEMM_RS_HPP__


#include <cxxopts.hpp>
#include <cuda_bf16.h>
#include <mpi.h>
#include <nccl.h>
#include <cublas_v2.h>
#include <iostream>
#include <string>

#include "macros.hpp.inc"

#include "helpers.hpp"
#include "mpi_helpers.hpp"
#include "nccl_helpers.hpp"
#include "timings_helpers.hpp"
#include "cublas_helpers.hpp"

#include "cublasmplite.h"

using namespace cublasmplite;

template<typename TA, typename TB, typename TC, typename F>
int test(const size_t m, const size_t n, const size_t k, const size_t cycles, const size_t skip, const bool verbose, MPI_Comm mpi_comm) {

    // MPI
    const mpi_t mpi(mpi_comm);

    if(mpi.my_rank == 0) {
        printf("GEMM+RS:\nnum_ranks %d\nm %zu\nn %zu\nk %zu\ncycles %zu\nskip %zu\n", mpi.num_ranks, m, n, k, cycles, skip);
    }

    // Allocate and fill values
    ASSERT(n % mpi.num_ranks == 0);
    ASSERT(k % mpi.num_ranks == 0);
    size_t n_chunk = n / mpi.num_ranks;
    size_t k_chunk = k / mpi.num_ranks;

    // weights are distributed
    std::vector<TA> my_weights = random<TA>(k_chunk * m, mpi.my_rank, 37);

    // input is distributed
    std::vector<TB> my_input = random<TB>(k_chunk * n, mpi.my_rank, 29);

    device_vector_t<TA> my_weights_d(my_weights);
    device_vector_t<TB> my_input_d(my_input);

    stream_t main_stream;
    
    if(verbose) {
        print("my_weights_d (before)", my_weights_d);
        print("my_input_d (before)", my_input_d);
    }

    // 1. cuBLAS + NVSHMEM overlap

    auto gemm_rs = F::create(mpi.my_rank, mpi.num_ranks, mpi.broadcast(), m, n, k);
    size_t workspace_B = gemm_rs->workspace_size();
    nvshmem_vector_t<char> symm_workspace_d(workspace_B); // FIXME: why this doesn't work ? p2p->make_vector<typename char>(workspace_B);
    device_vector_t<TC> my_output_d(m * n_chunk);

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(mpi.comm));

    auto bench_nvshmem = [&](const stream_t& stream) {
        gemm_rs->execute(my_weights_d.data(), my_input_d.data(), symm_workspace_d.data(), my_output_d.data(), stream);
    };

    float time_nvshmem_ms = run_and_time(bench_nvshmem, cycles, skip, mpi, main_stream);

    // 2. NCCL reference

    const nccl_t nccl(mpi);
    const gemm_t<TA, TB, TC> gemm(m, n, k_chunk, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N);
    
    device_vector_t<TC> intermediate(m * n);
    device_vector_t<TC> my_ref_output_d(m * n_chunk);

    ASSERT(my_weights_d.size() == m * k_chunk);
    ASSERT(my_input_d.size() == n * k_chunk);
    ASSERT(intermediate.size() == m * n);
    ASSERT(n_chunk * mpi.num_ranks == n);
    ASSERT(k_chunk * mpi.num_ranks == k);
    ASSERT(my_ref_output_d.size() == m * n_chunk);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(mpi.comm));

    auto bench = [&](const stream_t& stream) {
        gemm.execute(my_weights_d, my_input_d, intermediate, stream);
        nccl.reduceScatter(intermediate, my_ref_output_d, stream);
    };

    float time_nccl_ms = run_and_time(bench, cycles, skip, mpi, main_stream);

    if(verbose) {
        print("my_output_d (after)", my_output_d);
        print("my_ref_output_d (after)", my_ref_output_d);
    }

    // Timings and checks

    float max_nvshmem_ms = MPI_max(time_nvshmem_ms, mpi.comm);
    float max_nccl_ms = MPI_max(time_nccl_ms, mpi.comm);
    float average_nvshmem_ms = MPI_sum(time_nvshmem_ms, mpi.comm) / mpi.num_ranks;
    float average_nccl_ms = MPI_sum(time_nccl_ms, mpi.comm) / mpi.num_ranks;
    if(mpi.my_rank == 0) {
        printf("Performance:\nNVSHMEM (max) %f ms\nNVSHMEM (average) %f ms\nNCCL (max) %f ms\nNCCL (average) %f ms\n", max_nvshmem_ms, average_nvshmem_ms, max_nccl_ms, average_nccl_ms);
    }

    // Check correctness
    std::vector<TC> ref_output = (std::vector<TC>)(my_ref_output_d);
    std::vector<TC> test_output = (std::vector<TC>)(my_output_d);
    double output_l2_error = error(ref_output, test_output);
    bool passed = check(output_l2_error, 1e-2);

    return status(passed, mpi);

}

/**
 * W^T X = Y
 * 
 * W: (k // nPEs) x  m,          col-major, packed, distribuged along k
 * X: (k // nPEs) x  n,          col-major, packed, distributed along k
 * Y:  m          x  n // nPEs,  col-major, packed, distributed along n
 * 
 * Algorithm:
 * 0. Input W distributed along k, X distributed along k
 * 1. Local GEMM
 *    Y0 = W^T X
 * 2. Reduce scatter along n
 * 3. Output Y is distributed
 */
template<typename TA, typename TB, typename TC, typename F>
int gemm_rs_main(std::string name, int argc, char** argv) {

    cxxopts::Options options("tester", name + "Computes W^T X = Y where W is distributed along k, col-major, k x m, X is distributed along k, col-major, k x n and Y is distributed along m, col-major, k x n");
    options.add_options()
        ("m", "m", cxxopts::value<size_t>())
        ("n", "n", cxxopts::value<size_t>())
        ("k", "k", cxxopts::value<size_t>())
        ("c,cycles", "Number of cycles for timings", cxxopts::value<size_t>()->default_value("10"))
        ("skip", "Number of cycles to skip for timings", cxxopts::value<size_t>()->default_value("5"))
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ;

    auto result = options.parse(argc, argv);
    const size_t m = result["m"].as<size_t>();
    const size_t n = result["n"].as<size_t>();
    const size_t k = result["k"].as<size_t>();
    const size_t cycles = result["cycles"].as<size_t>();
    const size_t skip = result["skip"].as<size_t>();
    const bool verbose = result["verbose"].as<bool>();

    MPI_CHECK(MPI_Init(nullptr, nullptr));

    int error = test<TA, TB, TC, F>(m, n, k, cycles, skip, verbose, MPI_COMM_WORLD);

    MPI_CHECK(MPI_Finalize());

    return error;
}

#endif // __TE_NVSHMEM_GEMM_RS_HPP__
