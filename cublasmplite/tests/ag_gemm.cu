#include "macros.hpp.inc"

#include <cxxopts.hpp>
#include <cuda_bf16.h>
#include <mpi.h>
#include <nccl.h>
#include <cublas_v2.h>
#include <iostream>

#include "helpers.hpp"
#include "mpi_helpers.hpp"
#include "nccl_helpers.hpp"
#include "timings_helpers.hpp"
#include "cublas_helpers.hpp"

#include "cublasmplite.h"

using namespace cublasmplite;

using T = nv_bfloat16;

int test(const size_t m, const size_t n, const size_t k, const nvshmem_pipelined_p2p_t::signal_kind signal, int comm_sms, const size_t cycles, const size_t skip, const bool verbose, const bool csv, MPI_Comm mpi_comm) {

    // MPI
    const mpi_t mpi(mpi_comm, !csv);

    if(mpi.my_rank == 0 && !csv) {
        printf("AG+GEMM: num_ranks %d | m x n x k = %zu x %zu x %zu | cycles %zu | skip %zu\n", mpi.num_ranks, m, n, k, cycles, skip);
    }

    // Allocate and fill values
    ASSERT(n % mpi.num_ranks == 0);
    size_t n_chunk = n / mpi.num_ranks;

    // weights are identical and replicated
    std::vector<T> all_weights = random<T>(m * k, 0, 37);

    // input is distributed
    std::vector<T> my_input = random<T>(n_chunk * k, mpi.my_rank, 29);

    ASSERT(n_chunk * mpi.num_ranks == n);
    device_vector_t<T> my_input_d(my_input);
    device_vector_t<T> all_weights_d(all_weights);
    device_vector_t<T> all_input_d(n * k);
    device_vector_t<T> ref_output_d(n * m);

    stream_t main_stream;
    
    // 1. cuBLAS + NVSHMEM split overlap

    auto gemm_ag = cublasmp_ag_gemm_t<T, T, T>::create(mpi.my_rank, mpi.num_ranks, m, n, k, signal, comm_sms);
    
    nvshmem_vector_t<T> symm_input_d = gemm_ag->p2p()->make_vector<T>(n * k);
    device_vector_t<T> output_d(m * n);
    CUDA_CHECK(cudaMemcpy(symm_input_d.data() + mpi.my_rank * n_chunk * k, my_input.data(), my_input.size() * sizeof(T), cudaMemcpyDefault));

    if(verbose) {
        print("all_weights_d (before)", all_weights_d);
        print("my_input_d (before)", my_input_d);
        print("symm_input_d (before)", symm_input_d);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(mpi.comm));

    auto bench_nvshmem = [&](const stream_t& stream) {
        gemm_ag->execute(all_weights_d.data(), symm_input_d.data(), output_d.data(), stream);
    };

    float time_nvshmem_ms = run_and_time(bench_nvshmem, cycles, skip, mpi, main_stream);

    // 2. NCCL reference
    const nccl_t nccl(mpi);
    const gemm_t<T, T, T> gemm(m, n, k, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(mpi.comm));

    auto bench = [&](const stream_t& stream) {
        nccl.allGather(my_input_d, all_input_d, stream);
        gemm.execute(all_weights_d, all_input_d, ref_output_d, stream);
    };

    float time_nccl_ms = run_and_time(bench, cycles, skip, mpi, main_stream);

    // Timings and checks

    float max_nvshmem_ms = MPI_max(time_nvshmem_ms, mpi.comm);
    float max_nccl_ms = MPI_max(time_nccl_ms, mpi.comm);
    float average_nvshmem_ms = MPI_sum(time_nvshmem_ms, mpi.comm) / mpi.num_ranks;
    float average_nccl_ms = MPI_sum(time_nccl_ms, mpi.comm) / mpi.num_ranks;
    if(mpi.my_rank == 0 && !csv) {
        printf("Performance:\nNVSHMEM (max) %f ms\nNVSHMEM (average) %f ms\nNCCL (max) %f ms\nNCCL (average) %f ms\n", max_nvshmem_ms, average_nvshmem_ms, max_nccl_ms, average_nccl_ms);
    }

    if(verbose) {
        print("symm_input_d (after)", symm_input_d);
        print("all_input_d (after)", all_input_d);
        print("ref_output_d (after)", ref_output_d);
        print("output_d (after)", output_d);
    }

    // Check correctness
    std::vector<T> ref_input = (std::vector<T>)(all_input_d);
    std::vector<T> test_input = (std::vector<T>)(symm_input_d);
    std::vector<T> ref_output = (std::vector<T>)(ref_output_d);
    std::vector<T> test_output = (std::vector<T>)(output_d);
    double input_l2_error = error(ref_input, test_input);
    double output_l2_error = error(ref_output, test_output);
    bool passed = check(input_l2_error, 1e-2) && check(output_l2_error, 1e-2);

    if(mpi.my_rank == 0 && csv) {
        printf("<<<<, num_ranks, m, n, k, signal, comm_sms, cycles, skip, sizeof(T), NVSHMEM time [ms], NVSHMEM perf [GFlop/s], NCCL time [ms], NCCL perf [GFlop/s], L2 relative error\n");
        printf(">>>>, %d, %zu, %zu, %zu, %d, %d, %zu, %zu, %zu, %e, %e, %e, %e, %e\n", 
            mpi.num_ranks, m, n, k, (int)signal, comm_sms, cycles, skip, sizeof(T), 
            average_nvshmem_ms, matmul_perf_Gflops(m, n, k, average_nvshmem_ms), 
            average_nccl_ms, matmul_perf_Gflops(m, n, k, average_nccl_ms),
            output_l2_error);
    }

    return status(passed, mpi, !csv);

}

/**
 * W^T X = Y
 * 
 * W: k x  m,           col-major, packed, replicated
 * X: k x (n // nPEs),  col-major, packed, distributed along n
 * Y: m x  n            col-major, packed, replicated
 * 
 * Algorithm:
 * 0. Input W replicated, X distributed along n
 * 1. All-gather X along n
 *    X2: k x n, col-major, packed, replicated
 * 2. Local GEMM
 *    W^T x X2 = Y
 * 3. Output Y is replicated
 */
int main(int argc, char** argv) {

    cxxopts::Options options("tester", "AG+GEMM test driver. Computes W^T X = Y, where W is replicated, col-major, k x m, X is col-major, k x n, and Y is col-major, m x n. X is distributed along n, Y is replicated.");
    options.add_options()
        ("m", "m", cxxopts::value<size_t>())
        ("n", "n", cxxopts::value<size_t>())
        ("k", "k", cxxopts::value<size_t>())
        ("signal", "signal for NVSHMEM opt (set or add)", cxxopts::value<std::string>()->default_value("add"))
        ("c,cycles", "Number of cycles for timings", cxxopts::value<size_t>()->default_value("10"))
        ("skip", "Number of cycles to skip for timings", cxxopts::value<size_t>()->default_value("5"))
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ("csv", "CSV output", cxxopts::value<bool>()->default_value("false"))
        ("comm_sms", "Number of SMs for comms", cxxopts::value<int>()->default_value("1"))
        ;

    auto result = options.parse(argc, argv);
    const size_t m = result["m"].as<size_t>();
    const size_t n = result["n"].as<size_t>();
    const size_t k = result["k"].as<size_t>();
    const size_t cycles = result["cycles"].as<size_t>();
    const size_t skip = result["skip"].as<size_t>();
    const bool verbose = result["verbose"].as<bool>();
    const bool csv = result["csv"].as<bool>();
    const int comm_sms = result["comm_sms"].as<int>();
    const std::string signal = result["signal"].as<std::string>();

    nvshmem_pipelined_p2p_t::signal_kind s = (signal == "set") ? nvshmem_pipelined_p2p_t::signal_kind::set : nvshmem_pipelined_p2p_t::signal_kind::add;

    MPI_CHECK(MPI_Init(nullptr, nullptr));

    int error = test(m, n, k, s, comm_sms, cycles, skip, verbose, csv, MPI_COMM_WORLD);

    MPI_CHECK(MPI_Finalize());

    return error;
}
