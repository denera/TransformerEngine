#include "te_nvshmem.h"

#include <nvshmem.h>

#include <memory>
#include <cstdio>
#include <cublas_v2.h>
#include <iostream>
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include "macros.hpp.inc"

// stream_t

stream_t::stream_t() {
    CUDA_CHECK(cudaStreamCreate(&stream));
    alive = true;
}

stream_t::~stream_t() {
    if(alive) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

void stream_t::synchronize() const {
    ASSERT(alive);
    CUDA_CHECK(cudaStreamSynchronize(stream));
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
    ASSERT(alive);
    return stream;
}

cudaStream_t stream_t::handle() const { 
    ASSERT(alive);
    return stream;
}

void stream_t::wait(cudaEvent_t event) const {
    ASSERT(alive);
    CUDA_CHECK(cudaStreamWaitEvent(stream, event));
}

// event_t

event_t::event_t() {
    CUDA_CHECK(cudaEventCreate(&event));
    alive = true;
}

event_t::~event_t() {
    if(alive) {
        CUDA_CHECK(cudaEventDestroy(event));
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
    ASSERT(alive);
    return event;
}

cudaEvent_t event_t::handle() const { 
    ASSERT(alive);
    return event;
}

void event_t::record(cudaStream_t stream) const {
    ASSERT(alive);
    CUDA_CHECK(cudaEventRecord(event, stream));
}

float event_t::elapsed_time_ms(cudaEvent_t stop) const {
    ASSERT(alive);
    float time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, event, stop));
    return time_ms;
}

// nvshmem_vector_t

template<typename T> 
nvshmem_vector_t<T>::nvshmem_vector_t(size_t size) : device_vector_view_t<T>(nullptr, size) {
    if (this->_size == 0) {
        this->_ptr_d = nullptr;
    } else {
        this->_ptr_d = (T*)nvshmem_malloc(sizeof(T) * size);
        ASSERT(this->_ptr_d != nullptr);
    }
}

template<typename T> 
nvshmem_vector_t<T>::nvshmem_vector_t(const std::vector<T>& data) : device_vector_view_t<T>(nullptr, data.size()) {
    if (data.size() == 0) {
        this->_ptr_d = nullptr;
    } else {
        this->_ptr_d = (T*)nvshmem_malloc(sizeof(T) * data.size());
        ASSERT(this->_ptr_d != nullptr);
        CUDA_CHECK(cudaMemcpy(this->_ptr_d, data.data(), data.size() * sizeof(T), cudaMemcpyDefault));
    }
}

template<typename T> 
nvshmem_vector_t<T>::~nvshmem_vector_t() {
    if(this->_ptr_d != nullptr) {
        nvshmem_free(this->_ptr_d);
    }
}

// device_vector_const_view_t

template<typename T>
device_vector_const_view_t<T>::device_vector_const_view_t(device_vector_const_view_t<T>&& that) {
    this->_size = that._size;
    this->_ptr_d = that._ptr_d;
    that._size = 0;
    that._ptr_d = nullptr;
}

template<typename T>
device_vector_const_view_t<T>& device_vector_const_view_t<T>::operator=(device_vector_const_view_t<T>&& that) {
    std::swap(this->_size, that._size);
    std::swap(this->_ptr_d, that._ptr_d);
    return *this;
}

template<typename T>
device_vector_const_view_t<T>::operator std::vector<T>() {
    std::vector<T> out(_size);
    CUDA_CHECK(cudaMemcpy(out.data(), this->_ptr_d, this->_size * sizeof(T), cudaMemcpyDefault));
    return out;
}

// device_vector_view_t

template<typename T>
device_vector_view_t<T>::device_vector_view_t(device_vector_view_t<T>&& that) {
    this->_size = that._size;
    this->_ptr_d = that._ptr_d;
    that._size = 0;
    that._ptr_d = nullptr;
}

template<typename T>
device_vector_view_t<T>& device_vector_view_t<T>::operator=(device_vector_view_t<T>&& that) {
    std::swap(this->_size, that._size);
    std::swap(this->_ptr_d, that._ptr_d);
    return *this;
}

template<typename T>
device_vector_view_t<T>::operator std::vector<T>() {
    return (std::vector<T>) device_vector_const_view_t(this->_ptr_d, this->_size);
}

// device_vector_t

template<typename T>
device_vector_t<T>::device_vector_t(size_t size) : device_vector_view_t<T>(nullptr, size) {
    if (size == 0) {
        this->_ptr_d = nullptr;
    } else {
        CUDA_CHECK(cudaMalloc(&this->_ptr_d, size * sizeof(T)));
        CUDA_CHECK(cudaMemset(this->_ptr_d, 0, size * sizeof(T)));
    }
}

template<typename T>
device_vector_t<T>::device_vector_t(const std::vector<T>& data) : device_vector_view_t<T>(nullptr, data.size()) {
    if (data.size() == 0) {
        this->_ptr_d = nullptr;
    } else {
        CUDA_CHECK(cudaMalloc(&this->_ptr_d, data.size() * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(this->_ptr_d, data.data(), data.size() * sizeof(T), cudaMemcpyDefault));
    }
}

template<typename T>
device_vector_t<T>::~device_vector_t() {
    if (this->_ptr_d != nullptr) {
        CUDA_CHECK(cudaFree(this->_ptr_d));
    }
}

//////// TODO: fix this madness

template nvshmem_vector_t<nv_bfloat16>::nvshmem_vector_t(size_t size);
template nvshmem_vector_t<nv_bfloat16>::nvshmem_vector_t(const std::vector<nv_bfloat16>& data);
template nvshmem_vector_t<nv_bfloat16>::~nvshmem_vector_t();

template nvshmem_vector_t<uint64_t>::nvshmem_vector_t(size_t size);
template nvshmem_vector_t<uint64_t>::nvshmem_vector_t(const std::vector<uint64_t>& data);
template nvshmem_vector_t<uint64_t>::~nvshmem_vector_t();

template nvshmem_vector_t<char>::nvshmem_vector_t(size_t size);
template nvshmem_vector_t<char>::nvshmem_vector_t(const std::vector<char>& data);
template nvshmem_vector_t<char>::~nvshmem_vector_t();

template device_vector_const_view_t<nv_bfloat16>::device_vector_const_view_t(device_vector_const_view_t<nv_bfloat16>&&);
template device_vector_const_view_t<nv_bfloat16>& device_vector_const_view_t<nv_bfloat16>::operator=(device_vector_const_view_t<nv_bfloat16>&&);
template device_vector_const_view_t<nv_bfloat16>::operator std::vector<nv_bfloat16>();

template device_vector_const_view_t<__nv_fp8_e4m3>::device_vector_const_view_t(device_vector_const_view_t<__nv_fp8_e4m3>&&);
template device_vector_const_view_t<__nv_fp8_e4m3>& device_vector_const_view_t<__nv_fp8_e4m3>::operator=(device_vector_const_view_t<__nv_fp8_e4m3>&&);
template device_vector_const_view_t<__nv_fp8_e4m3>::operator std::vector<__nv_fp8_e4m3>();

template device_vector_const_view_t<__nv_fp8_e5m2>::device_vector_const_view_t(device_vector_const_view_t<__nv_fp8_e5m2>&&);
template device_vector_const_view_t<__nv_fp8_e5m2>& device_vector_const_view_t<__nv_fp8_e5m2>::operator=(device_vector_const_view_t<__nv_fp8_e5m2>&&);
template device_vector_const_view_t<__nv_fp8_e5m2>::operator std::vector<__nv_fp8_e5m2>();

template device_vector_const_view_t<char>::device_vector_const_view_t(device_vector_const_view_t<char>&&);
template device_vector_const_view_t<char>& device_vector_const_view_t<char>::operator=(device_vector_const_view_t<char>&&);
template device_vector_const_view_t<char>::operator std::vector<char>();

template device_vector_const_view_t<uint64_t>::device_vector_const_view_t(device_vector_const_view_t<uint64_t>&&);
template device_vector_const_view_t<uint64_t>& device_vector_const_view_t<uint64_t>::operator=(device_vector_const_view_t<uint64_t>&&);
template device_vector_const_view_t<uint64_t>::operator std::vector<uint64_t>();

template device_vector_view_t<nv_bfloat16>::operator std::vector<nv_bfloat16>();
template device_vector_view_t<nv_bfloat16>::device_vector_view_t(device_vector_view_t<nv_bfloat16>&&);
template device_vector_view_t<nv_bfloat16>& device_vector_view_t<nv_bfloat16>::operator=(device_vector_view_t<nv_bfloat16>&&);

template device_vector_view_t<char>::operator std::vector<char>();
template device_vector_view_t<char>::device_vector_view_t(device_vector_view_t<char>&&);
template device_vector_view_t<char>& device_vector_view_t<char>::operator=(device_vector_view_t<char>&&);

template device_vector_view_t<int32_t>::operator std::vector<int32_t>();
template device_vector_view_t<int32_t>::device_vector_view_t(device_vector_view_t<int32_t>&&);
template device_vector_view_t<int32_t>& device_vector_view_t<int32_t>::operator=(device_vector_view_t<int32_t>&&);

template device_vector_view_t<uint64_t>::operator std::vector<uint64_t>();
template device_vector_view_t<uint64_t>::device_vector_view_t(device_vector_view_t<uint64_t>&&);
template device_vector_view_t<uint64_t>& device_vector_view_t<uint64_t>::operator=(device_vector_view_t<uint64_t>&&);

template device_vector_t<nv_bfloat16>::device_vector_t(size_t size);
template device_vector_t<nv_bfloat16>::device_vector_t(const std::vector<nv_bfloat16>& data);
template device_vector_t<nv_bfloat16>::~device_vector_t();

template device_vector_t<int32_t>::device_vector_t(size_t size);
template device_vector_t<int32_t>::device_vector_t(const std::vector<int32_t>& data);
template device_vector_t<int32_t>::~device_vector_t();

template device_vector_t<char>::device_vector_t(size_t size);
template device_vector_t<char>::device_vector_t(const std::vector<char>& data);
template device_vector_t<char>::~device_vector_t();

template device_vector_t<__nv_fp8_e4m3>::device_vector_t(size_t size);
template device_vector_t<__nv_fp8_e4m3>::device_vector_t(const std::vector<__nv_fp8_e4m3>& data);
template device_vector_t<__nv_fp8_e4m3>::~device_vector_t();

template device_vector_t<__nv_fp8_e5m2>::device_vector_t(size_t size);
template device_vector_t<__nv_fp8_e5m2>::device_vector_t(const std::vector<__nv_fp8_e5m2>& data);
template device_vector_t<__nv_fp8_e5m2>::~device_vector_t();

