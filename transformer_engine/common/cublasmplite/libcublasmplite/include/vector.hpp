#ifndef __CUBLASMPLITE_VECTOR_HPP__
#define __CUBLASMPLITE_VECTOR_HPP__

#include <cstddef>
#include "macros.hpp.inc"

namespace cublasmplite {

template<typename T>
struct device_vector_const_view_t {
protected:
    const T* _ptr_d;
    size_t _size;
public:
    device_vector_const_view_t(const T* ptr, size_t size) : _ptr_d(ptr), _size(size) {}
    device_vector_const_view_t(device_vector_const_view_t<T>&& that) {
        this->_size = that._size;
        this->_ptr_d = that._ptr_d;
        that._size = 0;
        that._ptr_d = nullptr;
    }
    device_vector_const_view_t<T>& operator=(device_vector_const_view_t<T>&& that) {
        std::swap(this->_size, that._size);
        std::swap(this->_ptr_d, that._ptr_d);
        return *this;
    }
    device_vector_const_view_t(const device_vector_const_view_t<T>&) = delete;
    device_vector_const_view_t& operator=(const device_vector_const_view_t<T>&) = delete;
    T* data() const { return _ptr_d; }
    size_t size() const { return _size; }
    explicit operator std::vector<T>() {
        std::vector<T> out(_size);
        CUBLASMPLITE_CUDA_CHECK(cudaMemcpy(out.data(), this->_ptr_d, this->_size * sizeof(T), cudaMemcpyDefault));
        return out;
    }
    ~device_vector_const_view_t() {}
    using value_type = T;
};

template<typename T>
struct device_vector_view_t {
protected:
    T* _ptr_d;
    size_t _size;
public:
    device_vector_view_t(T* ptr, size_t size) : _ptr_d(ptr), _size(size) {}
    device_vector_view_t(const device_vector_view_t<T>&) = delete;
    device_vector_view_t& operator=(const device_vector_view_t<T>&) = delete;
    device_vector_view_t(device_vector_view_t<T>&& that) {
        this->_size = that._size;
        this->_ptr_d = that._ptr_d;
        that._size = 0;
        that._ptr_d = nullptr;
    }
    device_vector_view_t<T>& operator=(device_vector_view_t<T>&& that) {
        std::swap(this->_size, that._size);
        std::swap(this->_ptr_d, that._ptr_d);
        return *this;
    }
    T* data() const { return _ptr_d; }
    size_t size() const { return _size; }
    explicit operator std::vector<T>() {
        return (std::vector<T>) device_vector_const_view_t(this->_ptr_d, this->_size);
    }
    ~device_vector_view_t() {}
    using value_type = T;
};

// Those are used to hide nvshmem_malloc and nvshmem_free from the public interface
// FIXME: this should probably use nvshmem_comm_t instead
void* impl_nvshmem_malloc(size_t);
void impl_nvshmem_free(void*);

template<typename T>
struct nvshmem_vector_t : public device_vector_view_t<T> {
public:
    nvshmem_vector_t(size_t size) : device_vector_view_t<T>(nullptr, size) {
        if (this->_size == 0) {
            this->_ptr_d = nullptr;
        } else {
            this->_ptr_d = (T*)impl_nvshmem_malloc(sizeof(T) * size);
            CUBLASMPLITE_ASSERT(this->_ptr_d != nullptr);
        }
    }
    nvshmem_vector_t(size_t size, const T& value) : device_vector_view_t<T>(nullptr, size) {
        if (this->_size == 0) {
            this->_ptr_d = nullptr;
        } else {
            this->_ptr_d = (T*)impl_nvshmem_malloc(sizeof(T) * size);
            CUBLASMPLITE_ASSERT(this->_ptr_d != nullptr);
            std::vector<T> data(size, value);
            CUBLASMPLITE_CUDA_CHECK(cudaMemcpy(this->_ptr_d, data.data(), data.size() * sizeof(T), cudaMemcpyDefault));
        }
    }
    nvshmem_vector_t(const std::vector<T>& data) : device_vector_view_t<T>(nullptr, data.size()) {
        if (data.size() == 0) {
            this->_ptr_d = nullptr;
        } else {
            this->_ptr_d = (T*)impl_nvshmem_malloc(sizeof(T) * data.size());
            CUBLASMPLITE_ASSERT(this->_ptr_d != nullptr);
            CUBLASMPLITE_CUDA_CHECK(cudaMemcpy(this->_ptr_d, data.data(), data.size() * sizeof(T), cudaMemcpyDefault));
        }
    }
    nvshmem_vector_t(const nvshmem_vector_t&) = delete;
    nvshmem_vector_t& operator=(const nvshmem_vector_t&) = delete;
    nvshmem_vector_t(nvshmem_vector_t<T>&& that) : device_vector_view_t<T>(std::move(that)) {};
    nvshmem_vector_t& operator=(nvshmem_vector_t<T>&& that) {
        device_vector_view_t<T>::operator=(std::move(that));
        return *this;
    }
    ~nvshmem_vector_t() {
        if(this->_ptr_d != nullptr) {
            impl_nvshmem_free(this->_ptr_d);
        }
    }

};

template<typename T>
struct device_vector_t : public device_vector_view_t<T>  {
public:
    device_vector_t(size_t size) : device_vector_view_t<T>(nullptr, size) {
        if (size == 0) {
            this->_ptr_d = nullptr;
        } else {
            CUBLASMPLITE_CUDA_CHECK(cudaMalloc(&this->_ptr_d, size * sizeof(T)));
            CUBLASMPLITE_CUDA_CHECK(cudaMemset(this->_ptr_d, 0, size * sizeof(T)));
        }
    }
    device_vector_t(const std::vector<T>& data) : device_vector_view_t<T>(nullptr, data.size()) {
        if (data.size() == 0) {
            this->_ptr_d = nullptr;
        } else {
            CUBLASMPLITE_CUDA_CHECK(cudaMalloc(&this->_ptr_d, data.size() * sizeof(T)));
            CUBLASMPLITE_CUDA_CHECK(cudaMemcpy(this->_ptr_d, data.data(), data.size() * sizeof(T), cudaMemcpyDefault));
        }
    }
    device_vector_t(const device_vector_t&) = delete;
    device_vector_t& operator=(const device_vector_t&) = delete;
    device_vector_t(device_vector_t<T>&& that) : device_vector_view_t<T>(std::move(that)) {};
    device_vector_t& operator=(device_vector_t<T>&& that) {
        device_vector_view_t<T>::operator=(std::move(that));
        return *this;
    }
    ~device_vector_t() {
        if (this->_ptr_d != nullptr) {
            CUBLASMPLITE_CUDA_CHECK(cudaFree(this->_ptr_d));
        }
    }
};

template<typename T>
void print(const char* name, const T* ptr, size_t count) {
  std::vector<T> data_h = (std::vector<T>)device_vector_const_view_t<T>(ptr, count);
  std::cout << name << " ptr = " << ptr << " count " << count << " |T| " << sizeof(T) << " : ";
  for(auto v: data_h) {
    std::cout << (double)v << " ";
  }
  std::cout << "\n";
}

template<typename T>
void print(const char* name, const T& vec) {
  print<typename T::value_type>(name, vec.data(), vec.size());
}

}

#endif // __CUBLASMPLITE_VECTOR_HPP__