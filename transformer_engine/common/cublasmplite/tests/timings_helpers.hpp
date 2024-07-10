/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_TEST_TIMINGS_HELPERS_HPP
#define TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_TEST_TIMINGS_HELPERS_HPP

#include "macros.hpp.inc"

#include "cublasmplite.h"

#include "mpi_helpers.hpp"

template<typename F>
float run_and_time(F& f, size_t cycles, size_t skip, const mpi_t& mpi, const cublasmplite::stream_t& stream) {

    cublasmplite::event_t start;
    cublasmplite::event_t stop;
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

#endif // TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_TEST_TIMINGS_HELPERS_HPP