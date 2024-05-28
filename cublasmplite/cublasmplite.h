#ifndef __CUBLASMPLITE_H__
#define __CUBLASMPLITE_H__

#include <vector>
#include <memory>
#include <variant>
#include <cstdint>
#include <functional>
#include <cublas_v2.h>
#include "gemm.hpp"

// CUDA helpers

namespace cublasmplite {

struct event_t;
struct nvshmem_comm_t;

struct stream_t {
private:
    cudaStream_t stream;
    bool alive;
public:
    stream_t();
    ~stream_t();
    stream_t(const stream_t&)            = delete;
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
    event_t(const event_t&)            = delete;
    event_t& operator=(const event_t&) = delete;
    event_t(event_t&&);
    event_t& operator=(event_t&&);
    operator cudaEvent_t() const;
    cudaEvent_t handle() const;
    void record(cudaStream_t stream) const;
    float elapsed_time_ms(cudaEvent_t stop) const;
};

template<typename T>
struct device_vector_const_view_t {
protected:
    const T* _ptr_d;
    size_t _size;
public:
    device_vector_const_view_t(const T* ptr, size_t size) : _ptr_d(ptr), _size(size) {}
    ~device_vector_const_view_t() {}
    device_vector_const_view_t(const device_vector_const_view_t<T>&) = delete;
    device_vector_const_view_t& operator=(const device_vector_const_view_t<T>&) = delete;
    device_vector_const_view_t(device_vector_const_view_t<T>&&);
    device_vector_const_view_t& operator=(device_vector_const_view_t<T>&&);
    T* data() const { return _ptr_d; }
    size_t size() const { return _size; }
    explicit operator std::vector<T>();
    using value_type = T;
};

template<typename T>
struct device_vector_view_t {
protected:
    T* _ptr_d;
    size_t _size;
public:
    device_vector_view_t(T* ptr, size_t size) : _ptr_d(ptr), _size(size) {}
    ~device_vector_view_t() {}
    device_vector_view_t(const device_vector_view_t<T>&) = delete;
    device_vector_view_t& operator=(const device_vector_view_t<T>&) = delete;
    device_vector_view_t(device_vector_view_t<T>&&);
    device_vector_view_t& operator=(device_vector_view_t<T>&&);
    T* data() const { return _ptr_d; }
    size_t size() const { return _size; }
    explicit operator std::vector<T>();
    using value_type = T;
};

template<typename T>
struct nvshmem_vector_t : public device_vector_view_t<T> {
public:
    nvshmem_vector_t(size_t size);
    nvshmem_vector_t(const std::vector<T>& data);
    nvshmem_vector_t(const nvshmem_vector_t&) = delete;
    nvshmem_vector_t& operator=(const nvshmem_vector_t&) = delete;
    nvshmem_vector_t(nvshmem_vector_t<T>&& that) : device_vector_view_t<T>(std::move(that)) {};
    nvshmem_vector_t& operator=(nvshmem_vector_t<T>&& that) {
        device_vector_view_t<T>::operator=(std::move(that));
        return *this;
    }
    ~nvshmem_vector_t();
};


template<typename T>
struct device_vector_t : public device_vector_view_t<T>  {
public:
    device_vector_t(size_t size);
    device_vector_t(const std::vector<T>& data);
    device_vector_t(const device_vector_t&) = delete;
    device_vector_t& operator=(const device_vector_t&) = delete;
    device_vector_t(device_vector_t<T>&& that) : device_vector_view_t<T>(std::move(that)) {};
    device_vector_t& operator=(device_vector_t<T>&& that) {
        device_vector_view_t<T>::operator=(std::move(that));
        return *this;
    }
    ~device_vector_t();
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


// Stateless class to encapsulate NVSHMEM init/finalize/alloc/free
class nvshmem_comm_t {

public:
    enum class error_t {
        SUCCESS,
        ERROR,
    };

protected:
    const int my_pe;
    const int n_pes;
    nvshmem_comm_t(int my_pe, int n_pes);

public:
    static std::unique_ptr<nvshmem_comm_t>                      create(int my_rank, int num_ranks);
    int                                                         this_pe() const;
    int                                                         num_pes() const;
    error_t                                                     barrier_all();
    error_t                                                     sync_all_on_stream(cudaStream_t stream);
    error_t                                                     barrier_all_on_stream(cudaStream_t stream);
    void*                                                       malloc(size_t size);
    void                                                        free(void* ptr);
    error_t                                                     wait_on_atomic_and_set(int* flag, int signal, int value, cudaStream_t stream);
    error_t                                                     set(int* flag, int value, cudaStream_t stream);

    template<typename T> __device__ __forceinline__  static T   nvshmem_g(const T* ptr, int pe);
    ~nvshmem_comm_t();

    template<typename T> nvshmem_vector_t<T>                    make_vector(size_t size);
    template<typename T> nvshmem_vector_t<T>                    make_vector(const std::vector<T>& data);
};

class nvshmem_p2p_t : public nvshmem_comm_t {
    
private:
    nvshmem_vector_t<uint64_t> flags; // symmetric, one per other PE
    std::vector<uint64_t>  counters;
    nvshmem_p2p_t(int my_pe, int n_pes, nvshmem_vector_t<uint64_t> flags);

public:
    static  std::unique_ptr<nvshmem_p2p_t>  create(int my_rank, int num_ranks);
    static std::unique_ptr<nvshmem_p2p_t>   create(int my_rank, int num_ranks, std::function<void(void*, size_t, int, int)> broadcast);
    nvshmem_comm_t::error_t                 send_and_signal(const void* src, void* dst, size_t size, int peer, cudaStream_t stream);
    nvshmem_comm_t::error_t                 wait(int peer, cudaStream_t stream);
    ~nvshmem_p2p_t() {};

};

class nvshmem_reduce_scatter_t : public nvshmem_comm_t {

private:
    nvshmem_vector_t<uint64_t> flags; // symmetric, one flag per PE
    uint64_t counter;
    nvshmem_reduce_scatter_t(int my_pe, int n_pes, nvshmem_vector_t<uint64_t> rs_flags);

public:
    
    static  std::unique_ptr<nvshmem_reduce_scatter_t> create(int my_rank, int num_ranks);
    template<typename T> nvshmem_comm_t::error_t      reduce_scatter(const T* src, size_t src_rows, size_t src_cols, size_t src_ld, T* dst, size_t dst_ld, cudaStream_t stream);
    ~nvshmem_reduce_scatter_t() {};
    
};

class cublasmp_split_overlap_t {

private:
    cublasmp_split_overlap_t(std::unique_ptr<nvshmem_p2p_t> p2p, size_t m, size_t n, size_t k,
                                  std::vector<stream_t> compute, stream_t send, stream_t recv,
                                  event_t start_comms, event_t start_compute, event_t stop_compute, event_t stop_send, event_t stop_recv);
public: 
    static std::unique_ptr<cublasmp_split_overlap_t> create(int my_rank, int num_ranks, size_t m, size_t n, size_t k);
    ~cublasmp_split_overlap_t();

    // Same on all PEs
    const size_t m;
    const size_t n;
    const size_t k;
    const std::unique_ptr<nvshmem_p2p_t> p2p;

    const std::vector<stream_t> compute;
    const stream_t send;
    const stream_t recv;

    const event_t start_comms;
    const event_t start_compute;
    const event_t stop_compute;
    const event_t stop_send;
    const event_t stop_recv;

    // At the beginning - wait all steams on main
    nvshmem_comm_t::error_t wait_all_on(cudaStream_t main);

    // At the end - main waits on all streams
    nvshmem_comm_t::error_t  wait_on_all(cudaStream_t main);

    // Returns the ith compute stream, looping back at the end
    cudaStream_t compute_cyclic(size_t i);
};

template<typename TA, typename TB, typename TC>
class cublasmp_ag_gemm_t {

private:

    std::unique_ptr<cublasmp_split_overlap_t> overlap;
    cublasmp_ag_gemm_t(std::unique_ptr<cublasmp_split_overlap_t>  overlap, gemm_t<TA, TB, TC> gemm);
    const gemm_t<TA, TB, TC> gemm;

public:

    static std::unique_ptr<cublasmp_ag_gemm_t<TA, TB, TC>> create(int my_rank, int num_ranks, size_t m, size_t n, size_t k);
    nvshmem_comm_t::error_t execute(const TA* A, TB* B, TC* C, cudaStream_t main) const;
    nvshmem_p2p_t* p2p() { return overlap->p2p.get(); }
    ~cublasmp_ag_gemm_t();

};


template<typename TA, typename TB, typename TC>
class cublasmp_gemm_rs_t {

private:

    std::unique_ptr<cublasmp_split_overlap_t> overlap;
    cublasmp_gemm_rs_t(std::unique_ptr<cublasmp_split_overlap_t>  overlap, gemm_t<TA, TB, TC> gemm);
    const gemm_t<TA, TB, TC> gemm;

public:

    static  std::unique_ptr<cublasmp_gemm_rs_t<TA, TB, TC>> create(int my_rank, int num_ranks, size_t m, size_t n, size_t k);
    nvshmem_comm_t::error_t execute(const TA* A, const TB* B, void* workspace, TC* C, cudaStream_t main) const;
    nvshmem_p2p_t* p2p() { return overlap->p2p.get(); }
    size_t workspace_size() const { return 2 * overlap->m * overlap-> n * sizeof(TC); }
    ~cublasmp_gemm_rs_t();

};

template<typename TA, typename TB, typename TC>
class cublasmp_gemm_rs_atomic_t {

private:

    std::unique_ptr<cublasmp_split_overlap_t> overlap;
    cublasmp_gemm_rs_atomic_t(std::unique_ptr<cublasmp_split_overlap_t> overlap, device_vector_t<int32_t> counters, gemm_t<TA, TB, TC> gemm);
    device_vector_t<int32_t> counters;
    const gemm_t<TA, TB, TC> gemm;

public:

    static std::unique_ptr<cublasmp_gemm_rs_atomic_t<TA, TB, TC>> create(int my_rank, int num_ranks, size_t m, size_t n, size_t k);
    nvshmem_comm_t::error_t execute(const TA* A, const TB* B, void* workspace, TC* C, cudaStream_t main) const;
    nvshmem_p2p_t* p2p() { return overlap->p2p.get(); }
    size_t workspace_size() const { return 2 * overlap->m * overlap-> n * sizeof(TC); }
    ~cublasmp_gemm_rs_atomic_t() {};

};

}

#endif // __CUBLASMPLITE_H__