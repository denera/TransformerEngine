/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_CUDA_HELPERS_HPP
#define TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_CUDA_HELPERS_HPP

namespace cublasmplite {

struct stream_t {
private:
    cudaStream_t stream;
    bool alive;
public:
    stream_t();
    ~stream_t();
    stream_t(const stream_t&) = delete;
    stream_t& operator=(const stream_t&) = delete;
    stream_t(stream_t&&);
    stream_t& operator=(stream_t&&);
    operator cudaStream_t() const;
    cudaStream_t handle() const;
    void wait(cudaEvent_t event) const;
    void synchronize() const;
};

struct event_t {
private:
    cudaEvent_t event;
    bool alive;
public:
    event_t();
    ~event_t();
    event_t(const event_t&) = delete;
    event_t& operator=(const event_t&) = delete;
    event_t(event_t&&);
    event_t& operator=(event_t&&);
    operator cudaEvent_t() const;
    cudaEvent_t handle() const;
    void record(cudaStream_t stream) const;
    float elapsed_time_ms(cudaEvent_t stop) const;
};

}

#endif // TRANSFORMER_ENGINE_COMMON_CUBLASMPLITE_CUDA_HELPERS_HPP