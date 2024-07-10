/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_TEST_NCCL_HELPERS_HPP
#define TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_TEST_NCCL_HELPERS_HPP

#include <mpi.h>
#include <nccl.h>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "mpi_helpers.hpp"
#include "macros.hpp.inc"

template<typename T>
ncclDataType_t nccl_type_map() {
    static_assert(!std::is_same_v<T, __nv_fp8_e4m3>);
    static_assert(!std::is_same_v<T, __nv_fp8_e5m2>);
    if constexpr (std::is_same_v<T, float>) {
        return ncclDataType_t::ncclFloat32;
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return ncclDataType_t::ncclBfloat16;
    } else if constexpr (std::is_same_v<T, __half>) {
        return ncclDataType_t::ncclHalf;
    } else {
        ASSERT(false);
        return ncclDataType_t::ncclFloat32;
    }
}

struct nccl_t {

    ncclComm_t comm;
    const int my_rank;
    const int num_ranks;

    nccl_t(const mpi_t& mpi) : my_rank(mpi.my_rank), num_ranks(mpi.num_ranks) {
        ncclUniqueId id;
        if (mpi.my_rank == 0) ncclGetUniqueId(&id);
        MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi.comm));
        NCCL_CHECK(ncclCommInitRank(&comm, mpi.num_ranks, id, mpi.my_rank));
    }

    ~nccl_t() {
        NCCL_CHECK(ncclCommDestroy(comm));
    }

    template<typename T, typename U>
    void allGather(const T& input, U& output, cudaStream_t stream) const {
        ASSERT(input.size() * num_ranks == output.size());
        static_assert(std::is_same_v<typename T::value_type, typename U::value_type>);
        NCCL_CHECK(ncclAllGather(input.data(), output.data(), input.size(), nccl_type_map<typename T::value_type>(), this->comm, stream));
    }

    template<typename T, typename U>
    void reduceScatter(const T& input, U& output, cudaStream_t stream) const {
        ASSERT(input.size() == output.size() * num_ranks)
        static_assert(std::is_same_v<typename T::value_type, typename U::value_type>);
        NCCL_CHECK(ncclReduceScatter(input.data(), output.data(), output.size(), nccl_type_map<typename T::value_type>(), ncclSum, this->comm, stream));
    }

};



#endif // TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_TEST_NCCL_HELPERS_HPP