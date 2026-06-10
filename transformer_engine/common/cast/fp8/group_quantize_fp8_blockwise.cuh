/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8_blockwise.cuh
 *  \brief CUDA kernels to quantize grouped tensors to FP8 block-scaled formats.
 *
 *  Grouped MXFP8 design reuse: this path uses the existing nvte_group_quantize
 *  dispatch, GroupedTensor split metadata, one grouped kernel launch, tensor-local
 *  tile boundaries, rowwise/columnwise compile-time output switches, and no-op
 *  handling. FP8 block scaling deliberately keeps a different scale contract:
 *  FP32 scale-inverse buffers are laid out as per-member padded blockwise slices,
 *  and columnwise data is written as a per-member transpose rather than copying
 *  MXFP8 E8M0 scale swizzles or same-shape columnwise storage.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCKWISE_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCKWISE_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/cuda_runtime.h"
#include "../../util/math.h"
#include "../../utils.cuh"

// The TMA transpose variant uses tensormap replacement and PTX helpers that require
// architecture/family-specific Blackwell code generation. This header is included by
// the generic sm_100 translation unit, so keep that path out unless a future
// arch-specific TU explicitly opts in.
#ifdef NVTE_GROUPED_FP8_BLOCK_ENABLE_TMA
#include "../../util/ptx.cuh"
#include "../core/common.cuh"
#define NVTE_GROUPED_FP8_BLOCK_TMA_HW_SUPPORTED
#endif

namespace transformer_engine {
namespace dispatch {
namespace fp8_blockwise {
namespace group_quantize_kernel {

constexpr size_t THREADS_PER_BLOCK_1D = 128;
constexpr size_t THREADS_PER_BLOCK_2D = 256;
constexpr int kTileDim = 128;
constexpr int kNVecIn = 8;
constexpr int kNVecOut = 16;
constexpr int kNVecSMem = 2;
constexpr int kSMemRow = kTileDim;
constexpr int kSMemCol = (kTileDim / kNVecSMem) + 1;
constexpr int kSMemSize = kSMemRow * kSMemCol * kNVecSMem;
constexpr int kNumThreadsLoad = kTileDim / kNVecIn;
constexpr int kNumThreadsStore = kTileDim / kNVecOut;
static_assert(kNVecIn % kNVecSMem == 0, "kNVecIn must be divisible by kNVecSMem");
static_assert(kNVecOut % kNVecSMem == 0, "kNVecOut must be divisible by kNVecSMem");
static_assert(kNumThreadsLoad <= THREADS_PER_WARP,
              "kNumThreadsLoad must be <= THREADS_PER_WARP");
static_assert(kNumThreadsStore <= THREADS_PER_WARP,
              "kNumThreadsStore must be <= THREADS_PER_WARP");

constexpr size_t kWarpTileDimX2D = 32;
constexpr size_t kWarpTileDimY2D = 64;
constexpr size_t kThreadTileDimX2D = 16;
constexpr size_t kThreadTileDimY2D = 4;
constexpr size_t kElemsPerThread2D = kThreadTileDimX2D * kThreadTileDimY2D;
constexpr size_t kThreadsPerBlockTuned2D = kTileDim * kTileDim / kElemsPerThread2D;
constexpr size_t kNumWarpsXInBlock2D = kTileDim / kWarpTileDimX2D;
constexpr size_t kNumWarpsYInBlock2D = kTileDim / kWarpTileDimY2D;
constexpr size_t kNumWarpsInBlock2D = kNumWarpsXInBlock2D * kNumWarpsYInBlock2D;
constexpr size_t kNumThreadsXInWarp2D = kWarpTileDimX2D / kThreadTileDimX2D;
constexpr size_t kNumThreadsYInWarp2D = THREADS_PER_WARP / kNumThreadsXInWarp2D;
static_assert(kThreadsPerBlockTuned2D == THREADS_PER_BLOCK_2D);

#ifdef NVTE_GROUPED_FP8_BLOCK_TMA_HW_SUPPORTED
constexpr size_t kTmaNumBytesPerBank = 4;
constexpr size_t kTmaNumBanksPerSharedElem2D = kThreadTileDimY2D / kTmaNumBytesPerBank;
constexpr size_t kTmaSharedBlockTileDimY2D = kTileDim;
constexpr size_t kTmaSharedBlockTileDimXBanks2D =
    kTileDim / (kTmaNumBytesPerBank * kTmaNumBanksPerSharedElem2D);
constexpr size_t kTmaNumBanksYInWarp2D = kWarpTileDimY2D / kTmaNumBytesPerBank;
#endif

constexpr __device__ __host__ __forceinline__ size_t block_len() { return 128; }

constexpr __device__ __host__ __forceinline__ size_t divup_block_len(const size_t value) {
  return (value + block_len() - 1) / block_len();
}

__device__ __forceinline__ size_t round_up_to_multiple(const size_t value,
                                                       const size_t multiple) {
  return DIVUP(value, multiple) * multiple;
}

__device__ __forceinline__ size_t grouped_tensor_rows(const size_t tensor_idx,
                                                      const size_t num_tensors,
                                                      const size_t first_logical_dim,
                                                      const int64_t *first_dims) {
  return first_dims == nullptr ? first_logical_dim / num_tensors
                               : static_cast<size_t>(first_dims[tensor_idx]);
}

__device__ __forceinline__ size_t grouped_tensor_data_offset(const size_t tensor_idx,
                                                            const size_t num_tensors,
                                                            const size_t first_logical_dim,
                                                            const size_t last_logical_dim,
                                                            const int64_t *tensor_offsets) {
  return tensor_offsets == nullptr
             ? tensor_idx * (first_logical_dim / num_tensors) * last_logical_dim
             : static_cast<size_t>(tensor_offsets[tensor_idx]);
}

__device__ __forceinline__ size_t scale_elements_1d(const size_t rows, const size_t cols,
                                                    const bool columnwise) {
  if (columnwise) {
    return divup_block_len(rows) * round_up_to_multiple(cols, static_cast<size_t>(4));
  }
  return divup_block_len(cols) * round_up_to_multiple(rows, static_cast<size_t>(4));
}

__device__ __forceinline__ size_t scale_elements_2d(const size_t rows, const size_t cols,
                                                    const bool columnwise) {
  if (columnwise) {
    return divup_block_len(cols) *
           round_up_to_multiple(divup_block_len(rows), static_cast<size_t>(4));
  }
  return divup_block_len(rows) *
         round_up_to_multiple(divup_block_len(cols), static_cast<size_t>(4));
}

template <bool IS_2D>
__device__ __forceinline__ size_t scale_base_offset(const size_t tensor_idx,
                                                    const size_t num_tensors,
                                                    const size_t first_logical_dim,
                                                    const size_t last_logical_dim,
                                                    const int64_t *first_dims,
                                                    const bool columnwise) {
  size_t base = 0;
#pragma unroll 1
  for (size_t i = 0; i < tensor_idx; ++i) {
    const size_t rows = grouped_tensor_rows(i, num_tensors, first_logical_dim, first_dims);
    if constexpr (IS_2D) {
      base += scale_elements_2d(rows, last_logical_dim, columnwise);
    } else {
      base += scale_elements_1d(rows, last_logical_dim, columnwise);
    }
  }
  return base;
}

template <bool IS_2D>
__device__ __forceinline__ size_t scale_base_offset_fast(const size_t tensor_idx,
                                                         const size_t num_tensors,
                                                         const size_t first_logical_dim,
                                                         const size_t last_logical_dim,
                                                         const int64_t *first_dims,
                                                         const bool columnwise) {
  if (first_dims == nullptr) {
    const size_t rows = first_logical_dim / num_tensors;
    if constexpr (IS_2D) {
      return tensor_idx * scale_elements_2d(rows, last_logical_dim, columnwise);
    }
    return tensor_idx * scale_elements_1d(rows, last_logical_dim, columnwise);
  }
  return scale_base_offset<IS_2D>(tensor_idx, num_tensors, first_logical_dim, last_logical_dim,
                                  first_dims, columnwise);
}

template <typename IType>
__device__ __forceinline__ float load_input(const IType *input, const size_t idx) {
  return static_cast<float>(input[idx]);
}

#ifdef NVTE_GROUPED_FP8_BLOCK_TMA_HW_SUPPORTED
template <typename OType>
__global__ void __launch_bounds__(1) update_columnwise_transpose_tma_descriptors(
    const __grid_constant__ CUtensorMap base_tensor_map_output_t, OType *output_t,
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *tensor_offsets, const int64_t *first_dims) {
  const size_t tensor_idx = blockIdx.x;
  if (tensor_idx >= num_tensors) {
    return;
  }

  const size_t rows = grouped_tensor_rows(tensor_idx, num_tensors, first_logical_dim, first_dims);
  if (rows == 0 || last_logical_dim == 0) {
    return;
  }

  const size_t data_offset =
      grouped_tensor_data_offset(tensor_idx, num_tensors, first_logical_dim, last_logical_dim,
                                 tensor_offsets);
  common::modify_base_tensor_map(
      base_tensor_map_output_t, &common::g_tensor_maps.output_colwise[tensor_idx],
      reinterpret_cast<uintptr_t>(output_t + data_offset), last_logical_dim, rows, sizeof(OType));
}
#endif

template <bool kAligned, typename IType, typename OType, bool RETURN_ROWWISE,
          bool RETURN_COLUMNWISE>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_2D) vector_1d_tuned_kernel(
    const IType *input, OType *output, OType *output_t, float *scale_inv, float *scale_inv_t,
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *tensor_offsets, const int64_t *first_dims, const float epsilon,
    const bool force_pow_2_scales, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t tensor_idx = blockIdx.z;
  if (tensor_idx >= num_tensors) {
    return;
  }

  const size_t rows = grouped_tensor_rows(tensor_idx, num_tensors, first_logical_dim, first_dims);
  const size_t tile_m = blockIdx.y;
  const size_t tile_n = blockIdx.x;
  const size_t local_m_begin = tile_m * block_len();
  const size_t local_n_begin = tile_n * block_len();
  if (local_m_begin >= rows || local_n_begin >= last_logical_dim) {
    return;
  }

  const size_t data_offset =
      grouped_tensor_data_offset(tensor_idx, num_tensors, first_logical_dim, last_logical_dim,
                                 tensor_offsets);
  const size_t padded_rows = round_up_to_multiple(rows, static_cast<size_t>(4));
  const size_t padded_cols = round_up_to_multiple(last_logical_dim, static_cast<size_t>(4));
  const size_t rowwise_scale_base =
      RETURN_ROWWISE ? scale_base_offset_fast<false>(tensor_idx, num_tensors, first_logical_dim,
                                                     last_logical_dim, first_dims, false)
                     : 0;
  const size_t columnwise_scale_base =
      RETURN_COLUMNWISE ? scale_base_offset_fast<false>(tensor_idx, num_tensors,
                                                        first_logical_dim, last_logical_dim,
                                                        first_dims, true)
                        : 0;

  using SMemVec = Vec<IType, kNVecSMem>;
  union IVec {
    Vec<IType, kNVecIn> input_type;
    Vec<SMemVec, kNVecIn / kNVecSMem> smem_type;
  };
  using OVec = Vec<OType, kNVecOut>;

  extern __shared__ char smem_base[];
  SMemVec *smem = reinterpret_cast<SMemVec *>(&smem_base[0]);

  {
    constexpr int r_stride = THREADS_PER_BLOCK_2D / kNumThreadsLoad;
    constexpr int num_iterations = kTileDim / r_stride;
    const int c_s = (threadIdx.x % kNumThreadsLoad) * (kNVecIn / kNVecSMem);
    int r_s = threadIdx.x / kNumThreadsLoad;
    const size_t c_g = local_n_begin + static_cast<size_t>(c_s) * kNVecSMem;
    size_t r_g = local_m_begin + r_s;
    const size_t stride_g = static_cast<size_t>(r_stride) * last_logical_dim;
    const size_t num_ele = c_g < last_logical_dim
                               ? min(static_cast<size_t>(kNVecIn), last_logical_dim - c_g)
                               : 0;
    const IType *input_g = input + data_offset + r_g * last_logical_dim + c_g;
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      IVec input_vec;
      if constexpr (kAligned) {
        input_vec.input_type.load_from(input_g);
      } else {
        if (r_g < rows && num_ele > 0) {
          input_vec.input_type.load_from_elts(input_g, 0, num_ele);
        } else {
          input_vec.input_type.clear();
        }
      }
#pragma unroll
      for (int i = 0; i < kNVecIn / kNVecSMem; ++i) {
        const int c = c_s + i;
        const int r = r_s;
        smem[r * kSMemCol + c] = input_vec.smem_type.data.elt[i];
      }
      input_g += stride_g;
      r_s += r_stride;
      if constexpr (!kAligned) {
        r_g += r_stride;
      }
    }
  }

  __syncthreads();

  if constexpr (RETURN_ROWWISE) {
    constexpr int r_stride = THREADS_PER_BLOCK_2D / kNumThreadsStore;
    constexpr int num_iterations = kTileDim / r_stride;
    const int c_s = (threadIdx.x % kNumThreadsStore) * (kNVecOut / kNVecSMem);
    int r_s = threadIdx.x / kNumThreadsStore;
    const size_t c_g = local_n_begin + static_cast<size_t>(c_s) * kNVecSMem;
    size_t r_g = local_m_begin + r_s;
    const size_t stride_g = static_cast<size_t>(r_stride) * last_logical_dim;
    const size_t num_ele = c_g < last_logical_dim
                               ? min(static_cast<size_t>(kNVecOut), last_logical_dim - c_g)
                               : 0;
    OType *output_g = output + data_offset + r_g * last_logical_dim + c_g;
    const unsigned src_lane =
        (threadIdx.x % THREADS_PER_WARP) / kNumThreadsStore * kNumThreadsStore;
    const unsigned mask = ((1u << kNumThreadsStore) - 1u) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsStore) == 0;
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      SMemVec smem_vec[kNVecOut / kNVecSMem];
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
        const int c = c_s + i;
        const int r = r_s;
        smem_vec[i] = smem[r * kSMemCol + c];
      }
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
#pragma unroll
        for (int j = 0; j < kNVecSMem; ++j) {
          __builtin_assume(amax >= 0);
          amax = fmaxf(amax, fabsf(static_cast<float>(smem_vec[i].data.elt[j])));
        }
      }
#pragma unroll
      for (int delta = kNumThreadsStore / 2; delta > 0; delta /= 2) {
        const float other_amax = __shfl_down_sync(mask, amax, delta);
        __builtin_assume(amax >= 0);
        __builtin_assume(other_amax >= 0);
        amax = fmaxf(amax, other_amax);
      }
      amax = __shfl_sync(mask, amax, src_lane);
      const float scale =
          compute_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
      bool write_scale_inv = is_src_lane;
      if constexpr (!kAligned) {
        write_scale_inv &= r_g < rows;
      }
      if (write_scale_inv) {
        scale_inv[rowwise_scale_base + tile_n * padded_rows + r_g] = 1.0f / scale;
      }

      OVec output_vec;
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
#pragma unroll
        for (int j = 0; j < kNVecSMem; ++j) {
          output_vec.data.elt[i * kNVecSMem + j] =
              static_cast<OType>(static_cast<float>(smem_vec[i].data.elt[j]) * scale);
        }
      }
      if constexpr (kAligned) {
        output_vec.store_to(output_g);
      } else {
        if (r_g < rows && num_ele > 0) {
          output_vec.store_to_elts(output_g, 0, num_ele);
        }
      }
      output_g += stride_g;
      r_s += r_stride;
      r_g += r_stride;
    }
  }

  if constexpr (RETURN_COLUMNWISE) {
    constexpr int c_stride = THREADS_PER_BLOCK_2D / kNumThreadsStore;
    constexpr int num_iterations = kTileDim / (c_stride * kNVecSMem);
    const int r_s = (threadIdx.x % kNumThreadsStore) * kNVecOut;
    int c_s = threadIdx.x / kNumThreadsStore;
    size_t r_g = local_n_begin + static_cast<size_t>(c_s) * kNVecSMem;
    const size_t c_g = local_m_begin + r_s;
    const size_t stride_g = static_cast<size_t>(c_stride) * kNVecSMem * rows;
    const size_t num_ele =
        c_g < rows ? min(static_cast<size_t>(kNVecOut), rows - c_g) : 0;
    OType *output_g = output_t + data_offset + r_g * rows + c_g;
    const unsigned src_lane =
        (threadIdx.x % THREADS_PER_WARP) / kNumThreadsStore * kNumThreadsStore;
    const unsigned mask = ((1u << kNumThreadsStore) - 1u) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsStore) == 0;
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      SMemVec smem_vec[kNVecOut];
#pragma unroll
      for (int i = 0; i < kNVecOut; ++i) {
        const int r = r_s + i;
        const int c = c_s;
        smem_vec[i] = smem[r * kSMemCol + c];
      }
#pragma unroll
      for (int smem_idx = 0; smem_idx < kNVecSMem; ++smem_idx) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < kNVecOut; ++i) {
          __builtin_assume(amax >= 0);
          amax = fmaxf(amax, fabsf(static_cast<float>(smem_vec[i].data.elt[smem_idx])));
        }
#pragma unroll
        for (int delta = kNumThreadsStore / 2; delta > 0; delta /= 2) {
          const float other_amax = __shfl_down_sync(mask, amax, delta);
          __builtin_assume(amax >= 0);
          __builtin_assume(other_amax >= 0);
          amax = fmaxf(amax, other_amax);
        }
        amax = __shfl_sync(mask, amax, src_lane);
        const float scale =
            compute_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
        bool write_scale_inv = is_src_lane;
        if constexpr (!kAligned) {
          write_scale_inv &= (r_g + smem_idx) < last_logical_dim;
        }
        if (write_scale_inv) {
          scale_inv_t[columnwise_scale_base + tile_m * padded_cols + r_g + smem_idx] =
              1.0f / scale;
        }

        OVec output_vec;
#pragma unroll
        for (int i = 0; i < kNVecOut; ++i) {
          output_vec.data.elt[i] =
              static_cast<OType>(static_cast<float>(smem_vec[i].data.elt[smem_idx]) * scale);
        }
        if constexpr (kAligned) {
          output_vec.store_to(output_g + smem_idx * rows);
        } else {
          if ((r_g + smem_idx) < last_logical_dim && num_ele > 0) {
            output_vec.store_to_elts(output_g + smem_idx * rows, 0, num_ele);
          }
        }
      }
      output_g += stride_g;
      c_s += c_stride;
      r_g += c_stride * kNVecSMem;
    }
  }
}

template <typename IType, typename OType, bool RETURN_ROWWISE, bool RETURN_COLUMNWISE,
          bool USE_TMA_TRANSPOSE = false>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_2D) square_2d_tuned_kernel(
    const IType *input, OType *output, OType *output_t, float *scale_inv, float *scale_inv_t,
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *tensor_offsets, const int64_t *first_dims, const float epsilon,
    const bool force_pow_2_scales, const float *noop_ptr) {
  static_assert(!USE_TMA_TRANSPOSE || RETURN_COLUMNWISE,
                "TMA transpose path requires columnwise output.");
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t tensor_idx = blockIdx.z;
  if (tensor_idx >= num_tensors) {
    return;
  }

  const size_t rows = grouped_tensor_rows(tensor_idx, num_tensors, first_logical_dim, first_dims);
  const size_t tile_m = blockIdx.y;
  const size_t tile_n = blockIdx.x;
  const size_t block_tile_start_row_idx = tile_m * kTileDim;
  const size_t block_tile_start_col_idx = tile_n * kTileDim;
  if (block_tile_start_row_idx >= rows || block_tile_start_col_idx >= last_logical_dim) {
    return;
  }

  using IVec = Vec<IType, kThreadTileDimX2D>;
  using OVecCast = Vec<OType, kThreadTileDimX2D>;
  using OVecTrans = Vec<OType, kThreadTileDimY2D>;
  constexpr int kThreadTileDimXMaybe = RETURN_COLUMNWISE ? kThreadTileDimX2D : 1;

  __shared__ float block_tile_amax_shared[kNumWarpsInBlock2D];

  IVec thrd_tile_input[kThreadTileDimY2D];
  OVecTrans thrd_tile_out_trans[kThreadTileDimXMaybe];

  const int tid_in_warp = threadIdx.x % THREADS_PER_WARP;
  const int tid_in_warp_x = tid_in_warp % kNumThreadsXInWarp2D;
  const int tid_in_warp_y = tid_in_warp / kNumThreadsXInWarp2D;
  const int warp_id_in_block = threadIdx.x / THREADS_PER_WARP;
  const int warp_id_in_block_x = warp_id_in_block % kNumWarpsXInBlock2D;
  const int warp_id_in_block_y = warp_id_in_block / kNumWarpsXInBlock2D;

  const size_t thread_tile_start_row_idx =
      block_tile_start_row_idx +
      warp_id_in_block_y * kThreadTileDimY2D * kNumThreadsYInWarp2D +
      tid_in_warp_y * kThreadTileDimY2D;
  const size_t thread_tile_start_col_idx =
      block_tile_start_col_idx +
      warp_id_in_block_x * kThreadTileDimX2D * kNumThreadsXInWarp2D +
      tid_in_warp_x * kThreadTileDimX2D;
  const size_t thread_tile_end_row_idx = thread_tile_start_row_idx + kThreadTileDimY2D - 1;
  const size_t thread_tile_end_col_idx = thread_tile_start_col_idx + kThreadTileDimX2D - 1;

  const bool full_thrd_tile =
      (thread_tile_end_row_idx < rows) && (thread_tile_end_col_idx < last_logical_dim);
  const bool empty_thrd_tile =
      (thread_tile_start_row_idx >= rows) || (thread_tile_start_col_idx >= last_logical_dim);
  const bool nonfull_thrd_tile = (!full_thrd_tile) && (!empty_thrd_tile);

  const size_t thread_tile_ncols =
      empty_thrd_tile
          ? 0
          : min(static_cast<size_t>(kThreadTileDimX2D),
                min(thread_tile_end_col_idx, last_logical_dim - 1) - thread_tile_start_col_idx +
                    1);
  const size_t thread_tile_nrows =
      empty_thrd_tile
          ? 0
          : min(static_cast<size_t>(kThreadTileDimY2D),
                min(thread_tile_end_row_idx, rows - 1) - thread_tile_start_row_idx + 1);

  const size_t data_offset =
      grouped_tensor_data_offset(tensor_idx, num_tensors, first_logical_dim, last_logical_dim,
                                 tensor_offsets);
  const size_t thread_tile_start_idx =
      data_offset + thread_tile_start_row_idx * last_logical_dim + thread_tile_start_col_idx;

  float amax = 0.0f;
  if (!empty_thrd_tile) {
    if (nonfull_thrd_tile) {
#pragma unroll
      for (int i = 0; i < kThreadTileDimY2D; ++i) {
        if (static_cast<size_t>(i) >= thread_tile_nrows) {
          thrd_tile_input[i].clear();
        } else {
          thrd_tile_input[i].load_from_elts(input + thread_tile_start_idx + i * last_logical_dim,
                                            0, thread_tile_ncols);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kThreadTileDimY2D; ++i) {
        thrd_tile_input[i].load_from_elts(input + thread_tile_start_idx + i * last_logical_dim, 0,
                                          kThreadTileDimX2D);
      }
    }

#pragma unroll
    for (int i = 0; i < kThreadTileDimY2D; ++i) {
#pragma unroll
      for (int j = 0; j < kThreadTileDimX2D; ++j) {
        __builtin_assume(amax >= 0);
        amax = fmaxf(amax, fabsf(static_cast<float>(thrd_tile_input[i].data.elt[j])));
      }
    }
  }

  float warp_tile_amax = warp_reduce_max<THREADS_PER_WARP>(amax);
  warp_tile_amax = __shfl_sync(0xFFFFFFFF, warp_tile_amax, 0);
  if (tid_in_warp == 0) {
    block_tile_amax_shared[warp_id_in_block_y * kNumWarpsXInBlock2D + warp_id_in_block_x] =
        warp_tile_amax;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float blk_amax = block_tile_amax_shared[0];
#pragma unroll
    for (int idx = 1; idx < static_cast<int>(kNumWarpsInBlock2D); ++idx) {
      blk_amax = fmaxf(blk_amax, block_tile_amax_shared[idx]);
    }
    block_tile_amax_shared[0] = blk_amax;
  }
  __syncthreads();
  const float block_tile_scale =
      compute_scale_from_types<IType, OType>(block_tile_amax_shared[0], epsilon,
                                             force_pow_2_scales);

  if (threadIdx.x == 0) {
    const float scale_inv_value = 1.0f / block_tile_scale;
    if constexpr (RETURN_ROWWISE) {
      const size_t padded_scale_cols =
          round_up_to_multiple(divup_block_len(last_logical_dim), static_cast<size_t>(4));
      const size_t scale_base = scale_base_offset_fast<true>(
          tensor_idx, num_tensors, first_logical_dim, last_logical_dim, first_dims, false);
      scale_inv[scale_base + tile_m * padded_scale_cols + tile_n] = scale_inv_value;
    }
    if constexpr (RETURN_COLUMNWISE) {
      const size_t padded_scale_cols =
          round_up_to_multiple(divup_block_len(rows), static_cast<size_t>(4));
      const size_t scale_base = scale_base_offset_fast<true>(
          tensor_idx, num_tensors, first_logical_dim, last_logical_dim, first_dims, true);
      scale_inv_t[scale_base + tile_n * padded_scale_cols + tile_m] = scale_inv_value;
    }
  }

  if constexpr (RETURN_COLUMNWISE) {
#pragma unroll
    for (int j = 0; j < kThreadTileDimX2D; ++j) {
      thrd_tile_out_trans[j].clear();
    }
  }

  if (!empty_thrd_tile) {
    OVecCast tmp_output_c;
#pragma unroll
    for (int i = 0; i < kThreadTileDimY2D; ++i) {
      if (static_cast<size_t>(i) >= thread_tile_nrows) {
        continue;
      }
#pragma unroll
      for (int j = 0; j < kThreadTileDimX2D; ++j) {
        const OType scaled_elt =
            static_cast<OType>(static_cast<float>(thrd_tile_input[i].data.elt[j]) *
                                block_tile_scale);
        tmp_output_c.data.elt[j] = scaled_elt;
        if constexpr (RETURN_COLUMNWISE) {
          thrd_tile_out_trans[j].data.elt[i] = scaled_elt;
        }
      }
      if constexpr (RETURN_ROWWISE) {
        tmp_output_c.store_to_elts(output + thread_tile_start_idx + i * last_logical_dim, 0,
                                   thread_tile_ncols);
      }
    }

    if constexpr (RETURN_COLUMNWISE) {
#pragma unroll
      for (int j = 0; j < kThreadTileDimX2D; ++j) {
        if (static_cast<size_t>(j) >= thread_tile_ncols) {
          continue;
        }
        if constexpr (!USE_TMA_TRANSPOSE) {
          thrd_tile_out_trans[j].store_to_elts(
              output_t + data_offset + (thread_tile_start_col_idx + j) * rows +
                  thread_tile_start_row_idx,
              0, thread_tile_nrows);
        }
      }
    }
  }

  if constexpr (USE_TMA_TRANSPOSE) {
#ifdef NVTE_GROUPED_FP8_BLOCK_TMA_HW_SUPPORTED
    __shared__ alignas(128) OVecTrans
        block_tile_trans_shared[kTmaSharedBlockTileDimY2D][kTmaSharedBlockTileDimXBanks2D];
    OType(*block_tile_trans_shared_otype_ptr)[kTileDim] =
        reinterpret_cast<OType(*)[kTileDim]>(block_tile_trans_shared);

#pragma unroll
    for (int i = 0; i < kThreadTileDimX2D; ++i) {
      const int trans_warp_id_x = warp_id_in_block_y;
      const int trans_warp_id_y = warp_id_in_block_x;
      const int row_idx =
          trans_warp_id_y * kThreadTileDimX2D * kNumThreadsXInWarp2D +
          tid_in_warp_x * kThreadTileDimX2D + i;
      const int col_idx =
          trans_warp_id_x * (kTmaNumBanksYInWarp2D / kTmaNumBanksPerSharedElem2D) +
          tid_in_warp_y;
      block_tile_trans_shared[row_idx][col_idx] = thrd_tile_out_trans[i];
    }

    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    const CUtensorMap &tensor_map_output_t = common::g_tensor_maps.output_colwise[tensor_idx];
    if (threadIdx.x == 0) {
      common::fence_acquire_tensormap(&tensor_map_output_t);
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output_t), tile_m * kTileDim,
          tile_n * kTileDim, reinterpret_cast<uint64_t *>(block_tile_trans_shared_otype_ptr));
      ptx::cp_async_bulk_commit_group();
      ptx::cp_async_bulk_wait_group_read<0>();
    }
#else
    NVTE_DEVICE_ERROR("Grouped FP8 block TMA transpose requires SM 9.0+ code generation.");
#endif
  }
}

template <typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_1D) rowwise_1d_kernel(
    const IType *input, OType *output, float *scale_inv, const size_t num_tensors,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *tensor_offsets, const int64_t *first_dims, const float epsilon,
    const bool force_pow_2_scales, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t tensor_idx = blockIdx.z;
  if (tensor_idx >= num_tensors) {
    return;
  }

  const size_t rows = grouped_tensor_rows(tensor_idx, num_tensors, first_logical_dim, first_dims);
  const size_t tile_m = blockIdx.y;
  const size_t local_m = tile_m * block_len() + threadIdx.x;
  if (local_m >= rows) {
    return;
  }

  const size_t tile_n = blockIdx.x;
  const size_t local_n_begin = tile_n * block_len();
  if (local_n_begin >= last_logical_dim) {
    return;
  }

  const size_t data_offset =
      grouped_tensor_data_offset(tensor_idx, num_tensors, first_logical_dim, last_logical_dim,
                                 tensor_offsets);
  const size_t input_row = data_offset + local_m * last_logical_dim;
  const size_t cols_this_tile = min(block_len(), last_logical_dim - local_n_begin);

  float amax = 0.0f;
#pragma unroll
  for (size_t k = 0; k < block_len(); ++k) {
    if (k < cols_this_tile) {
      const float value = load_input(input, input_row + local_n_begin + k);
      amax = fmaxf(amax, fabsf(value));
    }
  }

  const float scale =
      compute_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
  const float scale_inv_value = 1.0f / scale;

  const size_t padded_rows = round_up_to_multiple(rows, static_cast<size_t>(4));
  const size_t scale_base = scale_base_offset<false>(tensor_idx, num_tensors, first_logical_dim,
                                                     last_logical_dim, first_dims, false);
  scale_inv[scale_base + tile_n * padded_rows + local_m] = scale_inv_value;

#pragma unroll
  for (size_t k = 0; k < block_len(); ++k) {
    if (k < cols_this_tile) {
      const float value = load_input(input, input_row + local_n_begin + k);
      output[input_row + local_n_begin + k] = static_cast<OType>(value * scale);
    }
  }
}

template <typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_1D) columnwise_1d_kernel(
    const IType *input, OType *output_t, float *scale_inv_t, const size_t num_tensors,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *tensor_offsets, const int64_t *first_dims, const float epsilon,
    const bool force_pow_2_scales, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t tensor_idx = blockIdx.z;
  if (tensor_idx >= num_tensors) {
    return;
  }

  const size_t rows = grouped_tensor_rows(tensor_idx, num_tensors, first_logical_dim, first_dims);
  const size_t tile_m = blockIdx.y;
  const size_t local_m_begin = tile_m * block_len();
  if (local_m_begin >= rows) {
    return;
  }

  const size_t tile_n = blockIdx.x;
  const size_t local_n = tile_n * block_len() + threadIdx.x;
  if (local_n >= last_logical_dim) {
    return;
  }

  const size_t rows_this_tile = min(block_len(), rows - local_m_begin);
  const size_t data_offset =
      grouped_tensor_data_offset(tensor_idx, num_tensors, first_logical_dim, last_logical_dim,
                                 tensor_offsets);

  float amax = 0.0f;
#pragma unroll
  for (size_t r = 0; r < block_len(); ++r) {
    if (r < rows_this_tile) {
      const size_t input_idx = data_offset + (local_m_begin + r) * last_logical_dim + local_n;
      const float value = load_input(input, input_idx);
      amax = fmaxf(amax, fabsf(value));
    }
  }

  const float scale =
      compute_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
  const float scale_inv_value = 1.0f / scale;

  const size_t padded_cols = round_up_to_multiple(last_logical_dim, static_cast<size_t>(4));
  const size_t scale_base = scale_base_offset<false>(tensor_idx, num_tensors, first_logical_dim,
                                                     last_logical_dim, first_dims, true);
  scale_inv_t[scale_base + tile_m * padded_cols + local_n] = scale_inv_value;

#pragma unroll
  for (size_t r = 0; r < block_len(); ++r) {
    if (r < rows_this_tile) {
      const size_t local_m = local_m_begin + r;
      const size_t input_idx = data_offset + local_m * last_logical_dim + local_n;
      const size_t output_idx = data_offset + local_n * rows + local_m;
      const float value = load_input(input, input_idx);
      output_t[output_idx] = static_cast<OType>(value * scale);
    }
  }
}

template <typename IType, typename OType, bool RETURN_ROWWISE, bool RETURN_COLUMNWISE>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_2D) square_2d_kernel(
    const IType *input, OType *output, OType *output_t, float *scale_inv, float *scale_inv_t,
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *tensor_offsets, const int64_t *first_dims, const float epsilon,
    const bool force_pow_2_scales, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t tensor_idx = blockIdx.z;
  if (tensor_idx >= num_tensors) {
    return;
  }

  const size_t rows = grouped_tensor_rows(tensor_idx, num_tensors, first_logical_dim, first_dims);
  const size_t tile_m = blockIdx.y;
  const size_t tile_n = blockIdx.x;
  const size_t local_m_begin = tile_m * block_len();
  const size_t local_n_begin = tile_n * block_len();
  if (local_m_begin >= rows || local_n_begin >= last_logical_dim) {
    return;
  }

  const size_t rows_this_tile = min(block_len(), rows - local_m_begin);
  const size_t cols_this_tile = min(block_len(), last_logical_dim - local_n_begin);
  const size_t data_offset =
      grouped_tensor_data_offset(tensor_idx, num_tensors, first_logical_dim, last_logical_dim,
                                 tensor_offsets);

  float thread_amax = 0.0f;
  for (size_t idx = threadIdx.x; idx < rows_this_tile * cols_this_tile;
       idx += THREADS_PER_BLOCK_2D) {
    const size_t r = idx / cols_this_tile;
    const size_t c = idx - r * cols_this_tile;
    const size_t input_idx =
        data_offset + (local_m_begin + r) * last_logical_dim + local_n_begin + c;
    const float value = load_input(input, input_idx);
    thread_amax = fmaxf(thread_amax, fabsf(value));
  }

  __shared__ float block_amax_storage[THREADS_PER_BLOCK_2D / THREADS_PER_WARP];
  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  float block_amax = reduce_max<THREADS_PER_BLOCK_2D / THREADS_PER_WARP>(thread_amax, warp_id);
  if (threadIdx.x == 0) {
    block_amax_storage[0] = block_amax;
  }
  __syncthreads();
  block_amax = block_amax_storage[0];

  const float scale =
      compute_scale_from_types<IType, OType>(block_amax, epsilon, force_pow_2_scales);

  if (threadIdx.x == 0) {
    const float scale_inv_value = 1.0f / scale;
    if constexpr (RETURN_ROWWISE) {
      const size_t padded_scale_cols =
          round_up_to_multiple(divup_block_len(last_logical_dim), static_cast<size_t>(4));
      const size_t scale_base = scale_base_offset<true>(tensor_idx, num_tensors, first_logical_dim,
                                                        last_logical_dim, first_dims, false);
      scale_inv[scale_base + tile_m * padded_scale_cols + tile_n] = scale_inv_value;
    }
    if constexpr (RETURN_COLUMNWISE) {
      const size_t padded_scale_cols =
          round_up_to_multiple(divup_block_len(rows), static_cast<size_t>(4));
      const size_t scale_base = scale_base_offset<true>(tensor_idx, num_tensors, first_logical_dim,
                                                        last_logical_dim, first_dims, true);
      scale_inv_t[scale_base + tile_n * padded_scale_cols + tile_m] = scale_inv_value;
    }
  }

  for (size_t idx = threadIdx.x; idx < rows_this_tile * cols_this_tile;
       idx += THREADS_PER_BLOCK_2D) {
    const size_t r = idx / cols_this_tile;
    const size_t c = idx - r * cols_this_tile;
    const size_t local_m = local_m_begin + r;
    const size_t local_n = local_n_begin + c;
    const size_t input_idx = data_offset + local_m * last_logical_dim + local_n;
    const float value = load_input(input, input_idx);
    const OType qvalue = static_cast<OType>(value * scale);
    if constexpr (RETURN_ROWWISE) {
      output[input_idx] = qvalue;
    }
    if constexpr (RETURN_COLUMNWISE) {
      output_t[data_offset + local_n * rows + local_m] = qvalue;
    }
  }
}

}  // namespace group_quantize_kernel

inline void check_grouped_fp8_blockwise_inputs(const GroupedTensor *input,
                                               const GroupedTensor *output,
                                               const Tensor *noop,
                                               const QuantizationConfig *quant_config) {
  CheckNoopTensor(*noop, "cast_noop");
  NVTE_CHECK(input->num_tensors > 0, "Grouped FP8 block scaling quantize expects num_tensors > 0.");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->all_same_shape() || output->all_same_last_dim(),
             "Grouped FP8 block scaling quantize only supports a common last dimension.");
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Grouped FP8 block scaling quantize expects 2D grouped tensors.");
  NVTE_CHECK(input->logical_shape.data[0] == output->logical_shape.data[0] &&
                 input->logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output grouped tensor logical shapes must match.");
  if (transformer_engine::cuda::sm_arch() >= 100) {
    NVTE_CHECK(quant_config == nullptr || quant_config->force_pow_2_scales,
               "On Blackwell and newer, the FP8 block scaling recipe is emulated ",
               "with MXFP8, which requires using power of two scaling factors.");
  }
}

template <bool IS_2D>
void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                    const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace group_quantize_kernel;

  checkCuDriverContext(stream);
  check_grouped_fp8_blockwise_inputs(input, output, noop, quant_config);

  const size_t num_tensors = input->num_tensors;
  const size_t first_logical_dim = input->logical_shape.data[0];
  const size_t last_logical_dim = input->logical_shape.data[1];
  if (first_logical_dim == 0 || last_logical_dim == 0) {
    return;
  }

  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const float *const noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const float epsilon = quant_config == nullptr ? 0.0f : quant_config->amax_epsilon;
  const bool force_pow_2_scales =
      quant_config == nullptr ? true : quant_config->force_pow_2_scales;

  const size_t uniform_tensor_rows = first_logical_dim / num_tensors;
  const size_t configured_max_tensor_rows =
      quant_config == nullptr ? 0 : quant_config->grouped_max_first_dim;
  // Preserve explicit first_dims metadata on the GroupedTensor, but use uniform launch
  // math when the host-provided max row count proves that all members have the same rows.
  const bool explicit_uniform_first_dims =
      first_dims_ptr != nullptr && first_logical_dim % num_tensors == 0 &&
      configured_max_tensor_rows == uniform_tensor_rows;
  const int64_t *const launch_first_dims_ptr =
      explicit_uniform_first_dims ? nullptr : first_dims_ptr;
  const size_t max_tensor_rows =
      launch_first_dims_ptr == nullptr
          ? uniform_tensor_rows
          : (configured_max_tensor_rows == 0 ? first_logical_dim : configured_max_tensor_rows);
  NVTE_CHECK(max_tensor_rows <= first_logical_dim,
             "Grouped FP8 block scaling max tensor rows must not exceed logical first dim.");
  const size_t tiles_m = divup_block_len(max_tensor_rows);
  const size_t tiles_n = divup_block_len(last_logical_dim);
  const bool full_uniform_tiles =
      launch_first_dims_ptr == nullptr && max_tensor_rows % block_len() == 0 &&
      last_logical_dim % block_len() == 0;
#ifdef NVTE_GROUPED_FP8_BLOCK_TMA_HW_SUPPORTED
  const bool use_dim2_tma_transpose =
      IS_2D && output->has_columnwise_data() && full_uniform_tiles &&
      transformer_engine::cuda::sm_arch() >= 100;
  if (use_dim2_tma_transpose) {
    NVTE_CHECK(num_tensors <= common::MAX_SUPPORTED_TENSOR_DESCRIPTORS,
               "Grouped FP8 block scaling TMA path supports at most ",
               common::MAX_SUPPORTED_TENSOR_DESCRIPTORS, " tensor descriptors.");
  }
#endif

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          if constexpr (IS_2D) {
            dim3 grid(tiles_n, tiles_m, num_tensors);
#ifdef NVTE_GROUPED_FP8_BLOCK_TMA_HW_SUPPORTED
            if (use_dim2_tma_transpose) {
              alignas(64) CUtensorMap tensor_map_output_t{};
              constexpr size_t output_type_bit_size = TypeInfo<OType>::size;
              create_2D_tensor_map(tensor_map_output_t, output->columnwise_data,
                                   last_logical_dim, max_tensor_rows, kTileDim, kTileDim,
                                   max_tensor_rows, 0, output_type_bit_size);
              update_columnwise_transpose_tma_descriptors<OType>
                  <<<num_tensors, 1, 0, stream>>>(
                      tensor_map_output_t,
                      reinterpret_cast<OType *>(output->columnwise_data.dptr), num_tensors,
                      first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr);
              NVTE_CHECK_CUDA(cudaGetLastError());
            }
#endif
            if (output->has_data() && output->has_columnwise_data()) {
#ifdef NVTE_GROUPED_FP8_BLOCK_TMA_HW_SUPPORTED
              if (use_dim2_tma_transpose) {
                square_2d_tuned_kernel<IType, OType, true, true, true>
                    <<<grid, THREADS_PER_BLOCK_2D, 0, stream>>>(
                    reinterpret_cast<const IType *>(input->data.dptr),
                    reinterpret_cast<OType *>(output->data.dptr),
                    reinterpret_cast<OType *>(output->columnwise_data.dptr),
                    reinterpret_cast<float *>(output->scale_inv.dptr),
                    reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                    first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr,
                    epsilon, force_pow_2_scales, noop_ptr);
              } else
#endif
              {
                square_2d_tuned_kernel<IType, OType, true, true>
                    <<<grid, THREADS_PER_BLOCK_2D, 0, stream>>>(
                    reinterpret_cast<const IType *>(input->data.dptr),
                    reinterpret_cast<OType *>(output->data.dptr),
                    reinterpret_cast<OType *>(output->columnwise_data.dptr),
                    reinterpret_cast<float *>(output->scale_inv.dptr),
                    reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                    first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr,
                    epsilon, force_pow_2_scales, noop_ptr);
              }
            } else if (output->has_data()) {
              square_2d_tuned_kernel<IType, OType, true, false>
                  <<<grid, THREADS_PER_BLOCK_2D, 0, stream>>>(
                      reinterpret_cast<const IType *>(input->data.dptr),
                      reinterpret_cast<OType *>(output->data.dptr), nullptr,
                      reinterpret_cast<float *>(output->scale_inv.dptr), nullptr, num_tensors,
                      first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr,
                      epsilon, force_pow_2_scales, noop_ptr);
            } else {
#ifdef NVTE_GROUPED_FP8_BLOCK_TMA_HW_SUPPORTED
              if (use_dim2_tma_transpose) {
                square_2d_tuned_kernel<IType, OType, false, true, true>
                    <<<grid, THREADS_PER_BLOCK_2D, 0, stream>>>(
                        reinterpret_cast<const IType *>(input->data.dptr), nullptr,
                        reinterpret_cast<OType *>(output->columnwise_data.dptr), nullptr,
                        reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                        first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr,
                        epsilon, force_pow_2_scales, noop_ptr);
              } else
#endif
              {
                square_2d_tuned_kernel<IType, OType, false, true>
                    <<<grid, THREADS_PER_BLOCK_2D, 0, stream>>>(
                        reinterpret_cast<const IType *>(input->data.dptr), nullptr,
                        reinterpret_cast<OType *>(output->columnwise_data.dptr), nullptr,
                        reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                        first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr,
                        epsilon, force_pow_2_scales, noop_ptr);
              }
            }
          } else {
            dim3 grid(tiles_n, tiles_m, num_tensors);
            const size_t smem_bytes = kSMemSize * sizeof(IType);
            if (output->has_data() && output->has_columnwise_data()) {
              if (full_uniform_tiles) {
                vector_1d_tuned_kernel<true, IType, OType, true, true>
                    <<<grid, THREADS_PER_BLOCK_2D, smem_bytes, stream>>>(
                        reinterpret_cast<const IType *>(input->data.dptr),
                        reinterpret_cast<OType *>(output->data.dptr),
                        reinterpret_cast<OType *>(output->columnwise_data.dptr),
                        reinterpret_cast<float *>(output->scale_inv.dptr),
                        reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                        first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr,
                        epsilon, force_pow_2_scales, noop_ptr);
              } else {
                vector_1d_tuned_kernel<false, IType, OType, true, true>
                    <<<grid, THREADS_PER_BLOCK_2D, smem_bytes, stream>>>(
                        reinterpret_cast<const IType *>(input->data.dptr),
                        reinterpret_cast<OType *>(output->data.dptr),
                        reinterpret_cast<OType *>(output->columnwise_data.dptr),
                        reinterpret_cast<float *>(output->scale_inv.dptr),
                        reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                        first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr,
                        epsilon, force_pow_2_scales, noop_ptr);
              }
            } else if (output->has_data()) {
              if (full_uniform_tiles) {
                vector_1d_tuned_kernel<true, IType, OType, true, false>
                    <<<grid, THREADS_PER_BLOCK_2D, smem_bytes, stream>>>(
                        reinterpret_cast<const IType *>(input->data.dptr),
                        reinterpret_cast<OType *>(output->data.dptr), nullptr,
                        reinterpret_cast<float *>(output->scale_inv.dptr), nullptr, num_tensors,
                        first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr,
                        epsilon, force_pow_2_scales, noop_ptr);
              } else {
                vector_1d_tuned_kernel<false, IType, OType, true, false>
                    <<<grid, THREADS_PER_BLOCK_2D, smem_bytes, stream>>>(
                        reinterpret_cast<const IType *>(input->data.dptr),
                        reinterpret_cast<OType *>(output->data.dptr), nullptr,
                        reinterpret_cast<float *>(output->scale_inv.dptr), nullptr, num_tensors,
                        first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr,
                        epsilon, force_pow_2_scales, noop_ptr);
              }
            } else {
              if (full_uniform_tiles) {
                vector_1d_tuned_kernel<true, IType, OType, false, true>
                    <<<grid, THREADS_PER_BLOCK_2D, smem_bytes, stream>>>(
                        reinterpret_cast<const IType *>(input->data.dptr), nullptr,
                        reinterpret_cast<OType *>(output->columnwise_data.dptr), nullptr,
                        reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                        first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr,
                        epsilon, force_pow_2_scales, noop_ptr);
              } else {
                vector_1d_tuned_kernel<false, IType, OType, false, true>
                    <<<grid, THREADS_PER_BLOCK_2D, smem_bytes, stream>>>(
                        reinterpret_cast<const IType *>(input->data.dptr), nullptr,
                        reinterpret_cast<OType *>(output->columnwise_data.dptr), nullptr,
                        reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                        first_logical_dim, last_logical_dim, offsets_ptr, launch_first_dims_ptr,
                        epsilon, force_pow_2_scales, noop_ptr);
              }
            }
          })  // Output type
  )          // Input type

  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fp8_blockwise
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCKWISE_CUH_
