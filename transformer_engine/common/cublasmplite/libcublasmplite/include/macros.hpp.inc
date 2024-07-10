/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <iostream>

#ifndef CUBLASMPLITE_CUBLAS_CHECK
#define CUBLASMPLITE_CUBLAS_CHECK(err)                                                                      \
do {                                                                                           \
    cublasStatus_t err_ = (err);                                                               \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
        std::printf("cuBLAS error %s:%d, error %d (%s)\n", __FILE__, __LINE__, (int)err_, cublasGetStatusString(err_)); \
        exit(EXIT_FAILURE);                                                                    \
    }                                                                                          \
} while (0)
#endif

#ifndef CUBLASMPLITE_CUDA_CHECK
#define CUBLASMPLITE_CUDA_CHECK(cmd)                                                                         \
  do {                                                                                             \
    cudaError_t retval = cmd;                                                                      \
    if (retval != 0) {                                                                             \
      printf("Failed: CUDART error %s:%d, error %d (%s)\n", __FILE__, __LINE__, (int)retval, cudaGetErrorString(retval)); \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0);
#endif

#ifndef CUBLASMPLITE_CU_CHECK
#define CUBLASMPLITE_CU_CHECK(cmd)                                                                 \
  do {                                                                                             \
    CUresult retval = cmd;                                                                         \
    if (retval != 0) {                                                                             \
      const char* error = nullptr;                                                                 \
      cuGetErrorString(retval, &error);                                                            \
      printf("Failed: CUDA error %s:%d, error %d (%s)\n", __FILE__, __LINE__, (int)retval, error); \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0);
#endif

#ifndef CUBLASMPLITE_ASSERT
#define CUBLASMPLITE_ASSERT(cmd)                                                                                \
  do {                                                                                             \
    bool retval = cmd;                                                                             \
    if (!retval) {                                                                                 \
      printf("Failed: ASSERT failed at %s:%d\n", __FILE__, __LINE__);                              \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0);
#endif

#ifndef CUBLASMPLITE_ASSERT_EQ
#define CUBLASMPLITE_ASSERT_EQ(lhs, rhs)                                                                        \
  do {                                                                                             \
    auto __lhs = lhs;                                                                              \
    auto __rhs = rhs;                                                                              \
    if (__lhs != __rhs) {                                                                          \
      printf("Failed: ASSERT_EQ failed at %s:%d, got:\n", __FILE__, __LINE__);                    \
      std::cout << __lhs << " != " << __rhs << "\n";                                              \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0);
#endif

