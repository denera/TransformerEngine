#ifndef __CUBLAS_HELPERS_HPP__
#define __CUBLAS_HELPERS_HPP__ 

#include <cublas_v2.h>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

template<typename T>
cudaDataType_t cublas_type_map() {
    if constexpr (std::is_same_v<T, float>) {
        return cudaDataType_t::CUDA_R_32F;
    } else if constexpr (std::is_same_v<T, __half>) {
        return cudaDataType_t::CUDA_R_16F;
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return cudaDataType_t::CUDA_R_16BF;
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        return cudaDataType_t::CUDA_R_8F_E4M3;
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
        return cudaDataType_t::CUDA_R_8F_E5M2;
    } else {
        CUBLASMPLITE_ASSERT(false);
        return cudaDataType_t::CUDA_R_32F;
    }
}

template<typename T>
constexpr bool is_fp8() {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>) {
        return true;
    } else {
        return false;
    }
}

#endif // __CUBLAS_HELPERS_HPP__

