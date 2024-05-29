#include <nvshmem.h>
#include <cstdio>
#include <cuda_bf16.h>
#include <string>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>

#include "cublasmplite.h"

#include "macros.hpp.inc"

using namespace cublasmplite;

static const bool TE_NVSHMEM_DEBUG = (std::getenv("TE_NVSHMEM_DEBUG") != nullptr && std::string(std::getenv("TE_NVSHMEM_DEBUG")) == "1");

nvshmem_comm_t::nvshmem_comm_t(int my_pe, int n_pes) : 
    my_pe(my_pe), n_pes(n_pes) {};

std::unique_ptr<nvshmem_comm_t> nvshmem_comm_t::create(int my_rank, int num_ranks) {
    const int my_pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();
    if(TE_NVSHMEM_DEBUG) {
        printf("NVSHMEM initialized, PE %d/%d\n", my_pe, n_pes);
    }
    CUBLASMPLITE_ASSERT_EQ(my_rank, my_pe);
    CUBLASMPLITE_ASSERT_EQ(n_pes, num_ranks);
    CUBLASMPLITE_ASSERT(my_pe >= 0);
    CUBLASMPLITE_ASSERT(n_pes > 0);
    return std::unique_ptr<nvshmem_comm_t>(new nvshmem_comm_t(my_rank, num_ranks));
}

nvshmem_comm_t::~nvshmem_comm_t() {
    nvshmem_finalize();
}

void* nvshmem_comm_t::malloc(size_t size) {
    if(size == 0) {
        size = 1;
    }
    void* ptr = nvshmem_malloc(size);
    CUBLASMPLITE_ASSERT(ptr != nullptr);
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] nvshmem_malloc returned %p\n", my_pe, ptr);
    }
    return ptr;
}

void nvshmem_comm_t::free(void* ptr) {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] nvshmem_free %p\n", my_pe, ptr);
    }
    nvshmem_free(ptr);
}

nvshmem_comm_t::error_t nvshmem_comm_t::barrier_all() {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] barrier_all\n", my_pe);
    }
    nvshmem_barrier_all();
    return nvshmem_comm_t::error_t::SUCCESS; 
}

nvshmem_comm_t::error_t nvshmem_comm_t::sync_all_on_stream(cudaStream_t stream) {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] sync_all_on_stream stream %p\n", my_pe, (void*)stream);
    }
    nvshmemx_sync_all_on_stream(stream);
    return nvshmem_comm_t::error_t::SUCCESS; 
}

nvshmem_comm_t::error_t nvshmem_comm_t::barrier_all_on_stream(cudaStream_t stream) {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] barrier_all_on_stream stream %p\n", my_pe, (void*)stream);
    }
    nvshmemx_barrier_all_on_stream(stream);
    return nvshmem_comm_t::error_t::SUCCESS; 
}

int nvshmem_comm_t::this_pe() const {
    return my_pe;
}

int nvshmem_comm_t::num_pes() const {
    return n_pes;
}

template<> 
__device__ __forceinline__ int4 nvshmem_comm_t::nvshmem_g<int4>(const int4* ptr, int pe) {
    static_assert(sizeof(int4) == 2 * sizeof(uint64_t));
    int4 v;
    uint64_t* p = (uint64_t*)(&v);
    p[0] = nvshmem_uint64_g(((uint64_t*)ptr) + 0, pe);
    p[1] = nvshmem_uint64_g(((uint64_t*)ptr) + 1, pe);
    return v;
}

__global__ void set_kernel(int *flag, int value) {
    *flag = value;
    asm volatile("fence.sc.gpu;\n");
}

nvshmem_comm_t::error_t nvshmem_comm_t::set(int* flag, int value, cudaStream_t stream) {
    set_kernel<<<1, 1, 0, stream>>>(flag, value);
    CUBLASMPLITE_CUDA_CHECK(cudaGetLastError());
    return nvshmem_comm_t::error_t::SUCCESS;
}

// For producer consumer: wait_on_atomic_and_set_kernel(flag, signal=0, value=1)
__global__ void wait_on_atomic_and_set_kernel(int *flag, int signal, int value) {
    while (signal != (atomicCAS(flag, signal, signal))) {
        // spin
    }
    *flag = value;
    // fence, to ensure results are visible to following kernel
    asm volatile("fence.sc.gpu;\n");
}

nvshmem_comm_t::error_t nvshmem_comm_t::wait_on_atomic_and_set(int* flag, int signal, int value, cudaStream_t stream) {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] wait_on_atomic_and_set flag %p signal %d set %d stream %p\n", my_pe, flag, signal, value, (void*)stream);
    }
    wait_on_atomic_and_set_kernel<<<1, 1, 0, stream>>>(flag, signal, value);
    CUBLASMPLITE_CUDA_CHECK(cudaGetLastError());
    return nvshmem_comm_t::error_t::SUCCESS;
}

nvshmem_pipelined_p2p_t::nvshmem_pipelined_p2p_t(int my_pe, int n_pes, int pipeline_depth, signal_kind signal) :
    nvshmem_comm_t(my_pe, n_pes), 
    signal(signal),
    pipeline_depth(pipeline_depth), 
    flags(n_pes * pipeline_depth, (uint64_t)0),
    signals_step(n_pes, (uint64_t)0),
    waits_step(n_pes, (uint64_t)0), 
    signals(n_pes * pipeline_depth, (uint64_t)0),
    waits(n_pes * pipeline_depth, (uint64_t)0) {}

std::unique_ptr<nvshmem_pipelined_p2p_t> nvshmem_pipelined_p2p_t::create(int my_rank, int num_ranks, int pipeline_depth, nvshmem_pipelined_p2p_t::signal_kind signal, std::function<void(void*, size_t, int, int)> broadcast) {

    nvshmemx_init_attr_t attr = {};
    nvshmemx_uniqueid_t id = {};
    if (my_rank == 0) {
       nvshmemx_get_uniqueid(&id);
    }

    broadcast((void*)&id, sizeof(nvshmemx_uniqueid_t), 0, num_ranks);

    if(TE_NVSHMEM_DEBUG) {
        std::stringstream ss;
        ss << std::hex;
        for(size_t b = 0; b < sizeof(nvshmemx_uniqueid_t); b++) {
            ss << (int)((char*)(&id))[b];
        }
        printf("[%d] NVSHMEM initialized with UID PE %d/%d, UID = %s\n", my_rank, my_rank, num_ranks, ss.str().c_str());
    }

    nvshmemx_set_attr_uniqueid_args(my_rank, num_ranks, &id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    const int my_pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    CUBLASMPLITE_ASSERT(my_pe == my_rank);
    CUBLASMPLITE_ASSERT(num_ranks == n_pes);
    CUBLASMPLITE_ASSERT(my_pe >= 0);
    CUBLASMPLITE_ASSERT(n_pes > 0);
    return std::unique_ptr<nvshmem_pipelined_p2p_t>(new nvshmem_pipelined_p2p_t(my_pe, n_pes, pipeline_depth, signal));
}

std::unique_ptr<nvshmem_pipelined_p2p_t> nvshmem_pipelined_p2p_t::create(int my_rank, int num_ranks, int pipeline_depth, nvshmem_pipelined_p2p_t::signal_kind signal) {
    nvshmem_init();
    const int my_pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();
    if(TE_NVSHMEM_DEBUG) {
        printf("NVSHMEM initialized, PE %d/%d\n", my_pe, n_pes);
    }
    CUBLASMPLITE_ASSERT(my_pe == my_rank);
    CUBLASMPLITE_ASSERT(num_ranks == n_pes);
    CUBLASMPLITE_ASSERT(my_pe >= 0);
    CUBLASMPLITE_ASSERT(n_pes > 0);
    return std::unique_ptr<nvshmem_pipelined_p2p_t>(new nvshmem_pipelined_p2p_t(my_pe, n_pes, pipeline_depth, signal));
}

size_t nvshmem_pipelined_p2p_t::idx(int step, int pe) {
    CUBLASMPLITE_ASSERT(step >= 0 && step < this->pipeline_depth);
    CUBLASMPLITE_ASSERT(pe >= 0 && pe < this->n_pes);
    return (size_t)step * (size_t)this->n_pes + (size_t)pe;
}

std::tuple<int, uint64_t, uint64_t*> nvshmem_pipelined_p2p_t::next_signal(int dst_pe) {
    CUBLASMPLITE_ASSERT(dst_pe >= 0 && dst_pe < this->n_pes);
    int step = signals_step[dst_pe];
    CUBLASMPLITE_ASSERT(step >= 0 && step < this->pipeline_depth);
    signals_step[dst_pe]++;
    uint64_t* flag = flags.data() + idx(step, my_pe);
    uint64_t signal_value = 0;
    int sig_op = 0;
    if(this->signal == signal_kind::set) {
        signal_value = signals[idx(step, dst_pe)] + 1;
        sig_op = NVSHMEM_SIGNAL_SET;
        signals[idx(step, dst_pe)]++;
    } else if (this->signal == signal_kind::add) {
        signal_value = 1;
        sig_op = NVSHMEM_SIGNAL_ADD;
        signals[idx(step, dst_pe)]++;
    } else {
        CUBLASMPLITE_ASSERT(false);
    }
    return {sig_op, signal_value, flag};
}

std::tuple<uint64_t, uint64_t*> nvshmem_pipelined_p2p_t::next_wait(int src_pe) {
    CUBLASMPLITE_ASSERT(src_pe >= 0 && src_pe < this->n_pes);
    int step = waits_step[src_pe];
    CUBLASMPLITE_ASSERT(step >= 0 && step < this->pipeline_depth);
    waits_step[src_pe]++;
    uint64_t* f = flags.data() + idx(step, src_pe);
    uint64_t w = waits[idx(step, src_pe)] + 1;
    waits[idx(step, src_pe)] = w;
    return {w, f};
}

nvshmem_comm_t::error_t nvshmem_pipelined_p2p_t::send_and_signal(const void* src, void* dst, size_t size, int peer, cudaStream_t stream) {
    // Push-send mode
    auto [sig_op, signal, flag] = next_signal(peer);
    char* ptr_dst = (char*)dst;
    const char* ptr_src = (const char*)src;
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] putmem %p -> %p (pe %d/%d, flag %p, signal %d, op %d, stream %p)\n", my_pe, ptr_src, ptr_dst, peer, this->n_pes, flag, (int)signal, sig_op, (void*)stream);
    }
    nvshmemx_putmem_signal_on_stream(ptr_dst, ptr_src, size, flag, signal, sig_op, peer, stream);
    return nvshmem_comm_t::error_t::SUCCESS;
}

nvshmem_comm_t::error_t nvshmem_pipelined_p2p_t::wait(int peer, cudaStream_t stream) {
    // Push-send mode
    auto [signal, flag] = next_wait(peer);
    int sig_op = NVSHMEM_CMP_EQ;
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] wait until (pe %d/%d, flag %p, signal %d, stream %p)\n", my_pe, peer, this->n_pes, flag, (int)signal, (void*)stream);
    }
    nvshmemx_uint64_wait_until_on_stream(flag, sig_op, signal, stream);
    return nvshmem_comm_t::error_t::SUCCESS;
}

nvshmem_comm_t::error_t nvshmem_pipelined_p2p_t::start_pipeline() {
    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] Start pipeline\n", my_pe);
    }
    std::fill(this->signals_step.begin(), this->signals_step.end(), 0);
    std::fill(this->waits_step.begin(), this->waits_step.end(), 0);
    return nvshmem_comm_t::error_t::SUCCESS;
}

////////

template<typename T> 
nvshmem_vector_t<T> nvshmem_comm_t::make_vector(size_t size) {
    return nvshmem_vector_t<T>(size);
}

template<typename T> 
nvshmem_vector_t<T> nvshmem_comm_t::make_vector(const std::vector<T>& data) {
    return nvshmem_vector_t<T>(data);
}

////////

nvshmem_reduce_scatter_t::nvshmem_reduce_scatter_t(int my_pe, int n_pes, nvshmem_vector_t<uint64_t> flags) :
    nvshmem_comm_t(my_pe, n_pes), flags(std::move(flags)), counter(0) {};

std::unique_ptr<nvshmem_reduce_scatter_t> nvshmem_reduce_scatter_t::create(int my_rank, int num_ranks) {
    nvshmem_init();
    const int my_pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();
    if(TE_NVSHMEM_DEBUG) {
        printf("NVSHMEM initialized, PE %d/%d\n", my_pe, n_pes);
    }
    CUBLASMPLITE_ASSERT_EQ(my_pe, my_rank);
    CUBLASMPLITE_ASSERT_EQ(num_ranks, n_pes);
    CUBLASMPLITE_ASSERT(my_pe >= 0);
    CUBLASMPLITE_ASSERT(n_pes > 0);
    nvshmem_vector_t<uint64_t> rs_flags(n_pes);
    return std::unique_ptr<nvshmem_reduce_scatter_t>(new nvshmem_reduce_scatter_t(my_pe, n_pes, std::move(rs_flags)));
}

// Reduce (add) the matrices and scatter the rows accross PEs + sync PEs. 
// This syncs PEs at the beginning of the kernel: this means kernels on different PEs will wait on each other before starting
// to read data from each other.
//
// Example with 2 PEs and 4x8 matrices
// 
// Input
// -----
// Inputs are row-major, with leading dimension src_ld, shape src_rows x src_cols
//
// On PE0:
// src = [  0  1  2  3  4  5  6  7]
//       [  8  9 10 11 12 13 14 15]
//       [ 16 17 18 19 20 21 22 23]
//       [ 24 25 26 27 28 29 30 31] 
// 
//
// On PE1:
// src = [ 32 33 34 35 36 37 38 39]
//       [ 40 41 42 43 44 45 46 47]
//       [ 48 49 50 51 52 53 54 55]
//       [ 56 57 58 59 60 61 62 63]
// 
// Output
// -----
// Outputs are row-major, with leading dimension dst_ld, shape dst_rows x dst_cols
//
// On PE0:
// dst = [32 34 36 38 40 42 44 46]
//       [48 50 52 54 56 58 60 62]
//
// On PE1:
// dst = [64 66 68 70 72 74 76 78]
//       [80 82 84 86 88 90 92 94]
//
template<typename T, typename F, int num_pes>
__global__ void reduce_scatter_kernel(const T* src, 
                                      size_t src_rows, 
                                      size_t src_cols, 
                                      size_t src_ld, 
                                      T* dst, 
                                      size_t dst_rows, 
                                      size_t dst_cols, 
                                      size_t dst_ld,
                                      uint64_t* flags,
                                      uint64_t signal) {

    const int my_pe = nvshmem_my_pe();
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Increment flags[my_pe] on remote PE dest_pe
    if(threadIdx.x < num_pes && blockIdx.x == 0) {
        const int dest_pe = threadIdx.x;
        uint64_t* flag = flags + my_pe; // == &flags[my_pe];
        // printf("[%d %d] Incrementing %p (%p + %d) on %d\n", blockIdx.x, threadIdx.x, flag, flags, my_pe, dest_pe);
        nvshmem_uint64_atomic_inc(flag, dest_pe);
    }
    // Wait for flags[pe] to be incremented, for all pe
    // (each thread waits for one pe)
    if(threadIdx.x < num_pes) {
        const int source_pe = threadIdx.x;
        uint64_t* flag = flags + source_pe; // == &flags[source_pe];
        // printf("[%d %d] Waiting on %p (%p + %d) to reach %d on %d\n", blockIdx.x, threadIdx.x, flag, flags, source_pe, (int)signal, my_pe);
        nvshmem_uint64_wait_until(flag, NVSHMEM_CMP_GE, signal);
    }
    __syncthreads();

    if(tid >= dst_rows * dst_cols) {
        return;
    }

    const size_t dst_col = tid % dst_cols;
    const size_t dst_row = tid / dst_cols;
    const size_t dst_mem_idx = dst_col + dst_row * dst_ld;

    T reduced = F::init();

    // rows
    const size_t src_col = dst_col; // cols: my_pe * dst_cols + dst_col;
    const size_t src_row = my_pe * dst_rows + dst_row; // cols: dst_row
    const size_t src_mem_idx = src_col + src_row * src_ld;

    #pragma unroll
    for(int peid = 0; peid < num_pes; peid++) {

        // Shuffle PEs
        int warp = blockIdx.x + (threadIdx.x >> 5);
        int pe = (peid + my_pe + warp) & (num_pes - 1);

        T v = nvshmem_comm_t::nvshmem_g<T>(src + src_mem_idx, pe);

        // if(threadIdx.x == 0 && blockIdx.x == 0) printf("[%d] loading %p (base %p, idx %d, pe %d, src %d %d, dst %d %d, tid %d, src(s) %d %d, dst(s) %d %d) %f %f %f %f %f %f %f %f\n", my_pe, src + src_mem_idx, src, 
        //     (int)src_mem_idx, pe, (int)src_row, (int)src_col, (int)dst_row, (int)dst_col, (int)tid, (int)src_rows, (int)src_cols, (int)dst_rows, (int)dst_cols,
        //     (float)((nv_bfloat16*)&v)[0],
        //     (float)((nv_bfloat16*)&v)[1],
        //     (float)((nv_bfloat16*)&v)[2],
        //     (float)((nv_bfloat16*)&v)[3],
        //     (float)((nv_bfloat16*)&v)[4],
        //     (float)((nv_bfloat16*)&v)[5],
        //     (float)((nv_bfloat16*)&v)[6],
        //     (float)((nv_bfloat16*)&v)[7]);

        reduced = F::reduce(reduced, v);

        // if(threadIdx.x == 0 && blockIdx.x == 0) printf("[%d] reduced %f %f %f %f %f %f %f %f\n", my_pe, 
        //     (float)((nv_bfloat16*)&reduced)[0],
        //     (float)((nv_bfloat16*)&reduced)[1],
        //     (float)((nv_bfloat16*)&reduced)[2],
        //     (float)((nv_bfloat16*)&reduced)[3],
        //     (float)((nv_bfloat16*)&reduced)[4],
        //     (float)((nv_bfloat16*)&reduced)[5],
        //     (float)((nv_bfloat16*)&reduced)[6],
        //     (float)((nv_bfloat16*)&reduced)[7]);
    }

    // if(threadIdx.x == 0 && blockIdx.x == 0) printf("[%d] storing %p (base %p, idx %d) %f %f %f %f %f %f %f %f\n", my_pe, &dst[dst_mem_idx], dst, (int)dst_mem_idx,
    //         (float)((nv_bfloat16*)&reduced)[0],
    //         (float)((nv_bfloat16*)&reduced)[1],
    //         (float)((nv_bfloat16*)&reduced)[2],
    //         (float)((nv_bfloat16*)&reduced)[3],
    //         (float)((nv_bfloat16*)&reduced)[4],
    //         (float)((nv_bfloat16*)&reduced)[5],
    //         (float)((nv_bfloat16*)&reduced)[6],
    //         (float)((nv_bfloat16*)&reduced)[7]);

    dst[dst_mem_idx] = reduced;

}

template<typename T>
struct adder_int4 {

    static_assert(sizeof(int4) % sizeof(T) == 0);

    __forceinline__ __device__ static int4 init() {
        int4 out;
        for(int i = 0; i < sizeof(int4) / sizeof(T); i++) {
            ((T*)(&out))[i] = 0;
        }
        return out;
    }

    __forceinline__ __device__ static int4 reduce(int4 lhs, int4 rhs) {
        int4 out;
        for(int i = 0; i < sizeof(int4) / sizeof(T); i++) {
            ((T*)(&out))[i] = ((T*)(&lhs))[i] + ((T*)(&rhs))[i];
        }
        return out;
    }

};

/**
 * Matrices are rows major
 * 
 * Inputs: on each GPU, a `src` of size src_rows x src_cols, row-major, leading dimension src_ld
 * 
 * Outputs: with nPEs GPUs, a `dst` of size (src_rows / nPEs) x src_cols, row-major, leading dimension dst_ld
 * 
 */
template<typename T> 
nvshmem_comm_t::error_t nvshmem_reduce_scatter_t::reduce_scatter(const T* src, size_t src_rows, size_t src_cols, size_t src_ld, T* dst, size_t dst_ld, cudaStream_t stream) {

    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] reduce_scatter %p (%zu %zu %zu) -> %p (%zu) |T| %zu, stream %p\n", my_pe, src, src_rows, src_cols, src_ld, dst, dst_ld, sizeof(T), (void*)stream);
    }

    const int npes = nvshmem_n_pes();

    using Tv = int4;
    constexpr unsigned vec_size = sizeof(Tv) / sizeof(T);

    // Check it's OKay to vectorize & that source lds are >= cols
    CUBLASMPLITE_ASSERT(src_cols >= vec_size && src_cols % vec_size == 0);
    const size_t src_cols_v = src_cols / vec_size;
    const size_t src_rows_v = src_rows;
    CUBLASMPLITE_ASSERT(src_ld >= src_cols && src_ld % vec_size == 0);
    const size_t src_ld_v = src_ld / vec_size;

    // Distribute cols
    // const size_t dst_cols_v = src_cols_v / npes;
    // const size_t dst_rows_v = src_rows_v;

    // CUBLASMPLITE_ASSERT(src_cols % npes == 0);
    // const size_t dst_cols = src_cols / npes;
    // CUBLASMPLITE_ASSERT(dst_ld >= dst_cols && dst_ld % vec_size == 0);
    // const size_t dst_ld_v = dst_ld / vec_size;

    // Distribute rows
    const size_t dst_cols_v = src_cols_v;
    CUBLASMPLITE_ASSERT(src_rows_v % npes == 0);
    const size_t dst_rows_v = src_rows_v / npes;

    const size_t dst_cols = src_cols;
    CUBLASMPLITE_ASSERT(dst_ld >= dst_cols && dst_ld % vec_size == 0);
    const size_t dst_ld_v = dst_ld / vec_size;

    // Launch
    const size_t grid_size = dst_rows_v * dst_cols_v;
    const size_t block_size = 128;
    const size_t num_blocks = (grid_size + block_size - 1) / block_size;
    CUBLASMPLITE_ASSERT(block_size >= (size_t)npes);

    const Tv* src_v = (const Tv*) src;
          Tv* dst_v =       (Tv*) dst;

    CUBLASMPLITE_ASSERT(this->flags.size() == (size_t)npes);
    uint64_t* flags = this->flags.data();
    uint64_t  signal = counter + 1;
    counter += 1;

    if(TE_NVSHMEM_DEBUG) {
        printf("[%d] reduce_scatter_kernel<<<%zu %zu>>> %p (%zu x %zu ld %zu) -> %p  (%zu x %zu ld %zu), |T| %zu, flags %p signal %zu\n", my_pe, num_blocks, block_size, src_v, src_rows_v, src_cols_v, src_ld_v, dst_v, dst_rows_v, dst_cols_v, dst_ld_v, sizeof(Tv), flags, signal);
    }

    switch(npes) {
        case 2:
            reduce_scatter_kernel<Tv, adder_int4<__nv_bfloat16>, 2> <<<num_blocks, block_size, 0, stream>>> (src_v, src_rows_v, src_cols_v, src_ld_v, dst_v, dst_rows_v, dst_cols_v, dst_ld_v, flags, signal);
            break;
        case 4:
            reduce_scatter_kernel<Tv, adder_int4<__nv_bfloat16>, 4> <<<num_blocks, block_size, 0, stream>>> (src_v, src_rows_v, src_cols_v, src_ld_v, dst_v, dst_rows_v, dst_cols_v, dst_ld_v, flags, signal);
            break;
        case 8:
            reduce_scatter_kernel<Tv, adder_int4<__nv_bfloat16>, 8> <<<num_blocks, block_size, 0, stream>>> (src_v, src_rows_v, src_cols_v, src_ld_v, dst_v, dst_rows_v, dst_cols_v, dst_ld_v, flags, signal);
            break;
        default:
            printf("Unsupported npes, got %d\n", npes);
            CUBLASMPLITE_ASSERT(false);
    }
    
    CUBLASMPLITE_CUDA_CHECK(cudaGetLastError());

    return nvshmem_comm_t::error_t::SUCCESS;
}

///////

template nvshmem_vector_t<nv_bfloat16> nvshmem_comm_t::make_vector<nv_bfloat16>(size_t size);
template nvshmem_vector_t<nv_bfloat16> nvshmem_comm_t::make_vector<nv_bfloat16>(const std::vector<nv_bfloat16>& data);

template nvshmem_vector_t<char> nvshmem_comm_t::make_vector<char>(size_t size);
template nvshmem_vector_t<char> nvshmem_comm_t::make_vector<char>(const std::vector<char>& data);

template nvshmem_comm_t::error_t nvshmem_reduce_scatter_t::reduce_scatter<nv_bfloat16>(const nv_bfloat16* src, size_t rows, size_t cols, size_t src_ld, nv_bfloat16* dst, size_t dst_ld, cudaStream_t stream);