
#include <nvshmem.h>

#include <memory>
#include <cstdio>
#include <cublas_v2.h>
#include <iostream>
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include "cublasmplite.h"

#include "macros.hpp.inc"

using namespace cublasmplite;

// stream_t

stream_t::stream_t() {
    CUBLASMPLITE_CUDA_CHECK(cudaStreamCreate(&stream));
    alive = true;
}

stream_t::~stream_t() {
    if(alive) {
        CUBLASMPLITE_CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

void stream_t::synchronize() const {
    CUBLASMPLITE_ASSERT(alive);
    CUBLASMPLITE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

stream_t::stream_t(stream_t&& that) {
    stream = that.stream;
    alive = that.alive;
    that.stream = (cudaStream_t)nullptr;
    that.alive = false;
}

stream_t& stream_t::operator=(stream_t&& that) {
    std::swap(stream, that.stream);
    std::swap(alive, that.alive);
    return *this;
}

stream_t::operator cudaStream_t() const { 
    CUBLASMPLITE_ASSERT(alive);
    return stream;
}

cudaStream_t stream_t::handle() const { 
    CUBLASMPLITE_ASSERT(alive);
    return stream;
}

void stream_t::wait(cudaEvent_t event) const {
    CUBLASMPLITE_ASSERT(alive);
    CUBLASMPLITE_CUDA_CHECK(cudaStreamWaitEvent(stream, event));
}

// event_t

event_t::event_t() {
    CUBLASMPLITE_CUDA_CHECK(cudaEventCreate(&event));
    alive = true;
}

event_t::~event_t() {
    if(alive) {
        CUBLASMPLITE_CUDA_CHECK(cudaEventDestroy(event));
    }
}

event_t::event_t(event_t&& that) {
    event = that.event;
    alive = that.alive;
    that.event = (cudaEvent_t)nullptr;
    that.alive = false;
}

event_t& event_t::operator=(event_t&& that) {
    std::swap(event, that.event);
    std::swap(alive, that.alive);
    return *this;
}

event_t::operator cudaEvent_t() const { 
    CUBLASMPLITE_ASSERT(alive);
    return event;
}

cudaEvent_t event_t::handle() const { 
    CUBLASMPLITE_ASSERT(alive);
    return event;
}

void event_t::record(cudaStream_t stream) const {
    CUBLASMPLITE_ASSERT(alive);
    CUBLASMPLITE_CUDA_CHECK(cudaEventRecord(event, stream));
}

float event_t::elapsed_time_ms(cudaEvent_t stop) const {
    CUBLASMPLITE_ASSERT(alive);
    float time_ms = 0;
    CUBLASMPLITE_CUDA_CHECK(cudaEventElapsedTime(&time_ms, event, stop));
    return time_ms;
}
