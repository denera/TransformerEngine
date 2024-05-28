#include "macros.hpp.inc"

#include <cxxopts.hpp>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mpi.h>
#include <nccl.h>
#include <cublas_v2.h>
#include <iostream>

#include "helpers.hpp"
#include "mpi_helpers.hpp"
#include "nccl_helpers.hpp"
#include "timings_helpers.hpp"
#include "cublas_helpers.hpp"

#include "te_nvshmem.h"
#include "gemm_rs.hpp"

using TA = __nv_fp8_e4m3;
using TB = __nv_fp8_e4m3;
using TC = __nv_bfloat16;

int main(int argc, char** argv) {
    return gemm_rs_main<TA, TB, TC, cublasmp_gemm_rs_atomic_t<TA, TB, TC>>("GEMM+RS(atomic). ", argc, argv);
}
