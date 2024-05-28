#ifndef __TE_NVSHMEM_TIMINGS_HELPERS_HPP__
#define __TE_NVSHMEM_TIMINGS_HELPERS_HPP__

#include "macros.hpp.inc"

#include "te_nvshmem.h"
#include "mpi_helpers.hpp"

template<typename F>
float run_and_time(F& f, size_t cycles, size_t skip, const mpi_t& mpi, const stream_t& stream) {

    event_t start;
    event_t stop;
    stream.synchronize();
    MPI_CHECK(MPI_Barrier(mpi.comm));

    for(size_t cycle = 0; cycle < skip + cycles; cycle++) {
        if (cycle == skip) {
            start.record(stream);
        }
        f(stream);
    }

    stop.record(stream);
    stream.synchronize();
    MPI_CHECK(MPI_Barrier(mpi.comm));
    float time_ms = start.elapsed_time_ms(stop) / (float)cycles;
    return time_ms;
    
}

#endif // __TE_NVSHMEM_TIMINGS_HELPERS_HPP__