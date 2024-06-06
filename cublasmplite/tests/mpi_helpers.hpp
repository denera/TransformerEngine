#ifndef __TE_NVSHMEM_MPI_HELPERS_HPP__
#define __TE_NVSHMEM_MPI_HELPERS_HPP__

#include <mpi.h>

#include "macros.hpp.inc"

float MPI_max(float value, MPI_Comm comm) {
    float maxi = 0.0;
    MPI_CHECK(MPI_Allreduce(&value, &maxi, 1, MPI_FLOAT, MPI_MAX, comm));
    return maxi;
}

float MPI_sum(float value, MPI_Comm comm) {
    float sum = 0.0;
    MPI_CHECK(MPI_Allreduce(&value, &sum, 1, MPI_FLOAT, MPI_SUM, comm));
    return sum;
}

int MPI_rank(MPI_Comm comm) {
    int rank = 0;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    return rank;
}

int MPI_size(MPI_Comm comm) {
    int size = 0;
    MPI_CHECK(MPI_Comm_size(comm, &size));
    return size;
}

struct mpi_t {
    const int my_rank;
    const int num_ranks;
    const MPI_Comm comm;
    mpi_t(MPI_Comm comm, bool verbose = true) : my_rank(MPI_rank(comm)), num_ranks(MPI_size(comm)), comm(comm) {
        if(verbose) {
            printf("MPI Hello from %d/%d\n", my_rank, num_ranks);
        }
        CUDA_CHECK(cudaSetDevice(my_rank));
    }
};

int status(bool passed, const mpi_t& mpi, bool print = true) {
    if(passed) {
        if(mpi.my_rank == 0 && print) {
            printf("PASSED\n");
        }
    } else {
        printf("FAILED\n");
    }
    return passed ? 0 : 1;
}


#endif // __TE_NVSHMEM_MPI_HELPERS_HPP__