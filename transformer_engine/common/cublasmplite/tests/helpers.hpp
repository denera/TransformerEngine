/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_TEST_HELPERS_HPP
#define TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_TEST_HELPERS_HPP

#include <vector>

template<typename T>
double error(std::vector<T> ref, std::vector<T> test) {
    double norm = 0;
    double diff = 0;
    ASSERT(ref.size() == test.size());
    for(size_t i = 0; i < ref.size(); i++) {
        norm += std::abs((double)ref[i]);
        diff += std::abs((double)ref[i] - (double)test[i]);
    }
    return diff / norm;
}

template<typename T>
std::vector<T> random(size_t count, int seed, int extra) {
    std::vector<T> out(count);
    for(size_t i = 0; i < count; i++) {
        out[i] = T(1 + (seed + i) % extra);
    }
    return out;
}

bool check(double value, double tolerance) {
    ASSERT(value >= 0);
    ASSERT(tolerance >= 0);
    if(value <= tolerance) {
        return true;
    } else {
        printf("FAILED, got %e <!= %e\n", value, tolerance);
        return false;
    }
}

double matmul_perf_Gflops(size_t m, size_t n, size_t k, double time_ms) {
    double perf_Gflops = (double)m * (double)n * (double)k * 2.0 / 1e9;
    double time_s = (1e-3 * time_ms);
    return perf_Gflops / time_s;
}

#endif // TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_TEST_HELPERS_HPP