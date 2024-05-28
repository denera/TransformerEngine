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
#include "gemm_rs.hpp"

using T = nv_bfloat16;

using namespace cublasmplite;

int main(int argc, char** argv) {
    return gemm_rs_main<T, T, T, cublasmp_gemm_rs_t<T, T, T>>("GEMM+RS(split). ", argc, argv);
}
