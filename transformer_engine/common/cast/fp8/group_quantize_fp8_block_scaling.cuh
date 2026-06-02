/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8_block_scaling.cuh
 *  \brief CUDA kernels to quantize grouped tensors to FP8 with 2D block scaling.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCK_SCALING_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCK_SCALING_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/cuda_runtime.h"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_block_scaling_2d {

constexpr size_t kThreadsPerWarp = 32;
constexpr size_t kBlockTileDim = 128;
constexpr size_t kWarpTileDimX = 64;
constexpr size_t kWarpTileDimY = 32;
constexpr size_t kThreadTileDimX = 8;
constexpr size_t kThreadTileDimY = 8;
constexpr size_t kElePerThread = kThreadTileDimX * kThreadTileDimY;
constexpr size_t kThreadsPerBlock = kBlockTileDim * kBlockTileDim / kElePerThread;
constexpr size_t kNumWarpsXInBlock = kBlockTileDim / kWarpTileDimX;
constexpr size_t kNumWarpsYInBlock = kBlockTileDim / kWarpTileDimY;
constexpr size_t kNumWarpsInBlock = kNumWarpsXInBlock * kNumWarpsYInBlock;
constexpr size_t kNumThreadsXInWarp = kWarpTileDimX / kThreadTileDimX;
constexpr size_t kNumThreadsYInWarp = kThreadsPerWarp / kNumThreadsXInWarp;

__device__ __host__ __forceinline__ size_t min_size(const size_t a, const size_t b) {
  return a < b ? a : b;
}

__device__ __forceinline__ size_t scale_tiles_y(const size_t rows) {
  return DIVUP(rows, kBlockTileDim);
}

__device__ __forceinline__ size_t scale_tiles_x_padded(const size_t cols) {
  return DIVUP_TO_MULTIPLE(DIVUP(cols, kBlockTileDim), 4);
}

template <bool kSameShape>
__device__ __forceinline__ size_t tensor_rows(const size_t tensor_id,
                                              const size_t first_logical_dim,
                                              const size_t num_tensors,
                                              const int64_t *const first_dims) {
  if constexpr (kSameShape) {
    return first_logical_dim / num_tensors;
  } else {
    return static_cast<size_t>(first_dims[tensor_id]);
  }
}

template <bool kSameShape>
__device__ __forceinline__ size_t tensor_offset(const size_t tensor_id,
                                                const size_t first_logical_dim,
                                                const size_t last_logical_dim,
                                                const size_t num_tensors,
                                                const int64_t *const offsets) {
  if constexpr (kSameShape) {
    return tensor_id * (first_logical_dim / num_tensors) * last_logical_dim;
  } else {
    return static_cast<size_t>(offsets[tensor_id]);
  }
}

template <bool kSameShape>
__device__ __forceinline__ size_t rowwise_scale_offset(const size_t tensor_id,
                                                       const size_t first_logical_dim,
                                                       const size_t last_logical_dim,
                                                       const size_t num_tensors,
                                                       const int64_t *const first_dims) {
  const size_t stride = scale_tiles_x_padded(last_logical_dim);
  if constexpr (kSameShape) {
    const size_t rows = first_logical_dim / num_tensors;
    return tensor_id * scale_tiles_y(rows) * stride;
  } else {
    size_t offset = 0;
    for (size_t i = 0; i < tensor_id; ++i) {
      offset += scale_tiles_y(static_cast<size_t>(first_dims[i])) * stride;
    }
    return offset;
  }
}

template <bool kSameShape>
__device__ __forceinline__ size_t columnwise_scale_offset(const size_t tensor_id,
                                                          const size_t first_logical_dim,
                                                          const size_t last_logical_dim,
                                                          const size_t num_tensors,
                                                          const int64_t *const first_dims) {
  const size_t scale_rows = DIVUP(last_logical_dim, kBlockTileDim);
  if constexpr (kSameShape) {
    const size_t rows = first_logical_dim / num_tensors;
    return tensor_id * scale_rows * DIVUP_TO_MULTIPLE(scale_tiles_y(rows), 4);
  } else {
    size_t offset = 0;
    for (size_t i = 0; i < tensor_id; ++i) {
      offset += scale_rows * DIVUP_TO_MULTIPLE(scale_tiles_y(static_cast<size_t>(first_dims[i])),
                                               4);
    }
    return offset;
  }
}

template <bool kReturnRowwise, bool kReturnTranspose, bool kSameShape, typename CType,
          typename IType, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock) grouped_block_scaled_cast_kernel(
    const IType *const input, OType *const output_c, OType *const output_t,
    CType *const tile_scales_inv_c, CType *const tile_scales_inv_t, const size_t num_tensors,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets, const int64_t *const __restrict__ first_dims,
    const float epsilon, const bool pow_2_scaling, const float *const noop_ptr) {
  using IVec = Vec<IType, kThreadTileDimX>;
  using OVecCast = Vec<OType, kThreadTileDimX>;
  using OVecTrans = Vec<OType, kThreadTileDimY>;

  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t tensor_id = blockIdx.z;
  if (tensor_id >= num_tensors) {
    return;
  }

  const size_t rows = tensor_rows<kSameShape>(tensor_id, first_logical_dim, num_tensors,
                                              first_dims);
  const size_t cols = last_logical_dim;
  if (rows == 0 || cols == 0) {
    return;
  }

  const size_t tile_id_x = blockIdx.x;
  const size_t tile_id_y = blockIdx.y;
  if (tile_id_x >= DIVUP(cols, kBlockTileDim) || tile_id_y >= DIVUP(rows, kBlockTileDim)) {
    return;
  }

  __shared__ CType block_tile_amax_shared[kNumWarpsInBlock];

  IVec thrd_tile_input[kThreadTileDimY];
  constexpr int kThreadTileDimXForTranspose = kReturnTranspose ? kThreadTileDimX : 1;
  OVecTrans thrd_tile_out_trans[kThreadTileDimXForTranspose];

  const int tid_in_warp = threadIdx.x % kThreadsPerWarp;
  const int tid_in_warp_x = tid_in_warp % kNumThreadsXInWarp;
  const int tid_in_warp_y = tid_in_warp / kNumThreadsXInWarp;
  const int warp_id_in_block = threadIdx.x / kThreadsPerWarp;
  const int warp_id_in_block_x = warp_id_in_block % kNumWarpsXInBlock;
  const int warp_id_in_block_y = warp_id_in_block / kNumWarpsXInBlock;

  const size_t base_offset =
      tensor_offset<kSameShape>(tensor_id, first_logical_dim, last_logical_dim, num_tensors,
                                offsets);
  const size_t thread_tile_start_row_idx =
      tile_id_y * kBlockTileDim +
      warp_id_in_block_y * kThreadTileDimY * kNumThreadsYInWarp +
      tid_in_warp_y * kThreadTileDimY;
  const size_t thread_tile_start_col_idx =
      tile_id_x * kBlockTileDim +
      warp_id_in_block_x * kThreadTileDimX * kNumThreadsXInWarp +
      tid_in_warp_x * kThreadTileDimX;
  const size_t thread_tile_start_idx =
      base_offset + thread_tile_start_row_idx * cols + thread_tile_start_col_idx;

  const size_t thread_tile_end_row_idx = thread_tile_start_row_idx + kThreadTileDimY - 1;
  const size_t thread_tile_end_col_idx = thread_tile_start_col_idx + kThreadTileDimX - 1;

  const bool full_thrd_tile =
      (thread_tile_end_row_idx < rows) && (thread_tile_end_col_idx < cols);
  const bool empty_thrd_tile =
      (thread_tile_start_row_idx >= rows) || (thread_tile_start_col_idx >= cols);
  const bool nonfull_thrd_tile = (!full_thrd_tile) && (!empty_thrd_tile);

  const size_t thread_tile_ncols =
      min_size(kThreadTileDimX,
               (min_size(thread_tile_end_col_idx, cols - 1) - thread_tile_start_col_idx + 1));
  const size_t thread_tile_nrows =
      min_size(kThreadTileDimY,
               (min_size(thread_tile_end_row_idx, rows - 1) - thread_tile_start_row_idx + 1));

  CType amax = 0;

  if (!empty_thrd_tile) {
    if (nonfull_thrd_tile) {
#pragma unroll
      for (int i = 0; i < kThreadTileDimY; i++) {
        if (static_cast<size_t>(i) >= thread_tile_nrows) {
          thrd_tile_input[i].clear();
        } else {
          thrd_tile_input[i].load_from_elts(input + thread_tile_start_idx + i * cols, 0,
                                            thread_tile_ncols);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kThreadTileDimY; i++) {
        thrd_tile_input[i].load_from_elts(input + thread_tile_start_idx + i * cols, 0,
                                          kThreadTileDimX);
      }
    }

    for (int i = 0; i < kThreadTileDimY; i++) {
#pragma unroll
      for (int j = 0; j < kThreadTileDimX; j++) {
        __builtin_assume(amax >= 0);
        amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.elt[j])));
      }
    }
  }

  CType warp_tile_amax = warp_reduce_max<kThreadsPerWarp>(amax);
  constexpr int lane_zero = 0;
  warp_tile_amax = __shfl_sync(0xFFFFFFFF, warp_tile_amax, lane_zero);

  if (tid_in_warp == 0) {
    block_tile_amax_shared[warp_id_in_block_y * kNumWarpsXInBlock + warp_id_in_block_x] =
        warp_tile_amax;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    CType blk_amax = block_tile_amax_shared[0];
#pragma unroll
    for (int idx = 1; idx < kNumWarpsInBlock; idx++) {
      blk_amax = fmaxf(blk_amax, block_tile_amax_shared[idx]);
    }
    block_tile_amax_shared[0] = blk_amax;
  }
  __syncthreads();

  const CType block_tile_scale =
      compute_scale_from_types<IType, OType>(block_tile_amax_shared[0], epsilon, pow_2_scaling);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    const CType scale_inv = 1.0f / block_tile_scale;
    if constexpr (kReturnRowwise) {
      const size_t scale_stride = scale_tiles_x_padded(cols);
      const size_t scale_offset =
          rowwise_scale_offset<kSameShape>(tensor_id, first_logical_dim, last_logical_dim,
                                           num_tensors, first_dims);
      tile_scales_inv_c[scale_offset + tile_id_y * scale_stride + tile_id_x] = scale_inv;
    }
    if constexpr (kReturnTranspose) {
      const size_t scale_stride = DIVUP_TO_MULTIPLE(DIVUP(rows, kBlockTileDim), 4);
      const size_t scale_offset =
          columnwise_scale_offset<kSameShape>(tensor_id, first_logical_dim, last_logical_dim,
                                              num_tensors, first_dims);
      tile_scales_inv_t[scale_offset + tile_id_x * scale_stride + tile_id_y] = scale_inv;
    }
  }

  if constexpr (kReturnTranspose) {
#pragma unroll
    for (int j = 0; j < kThreadTileDimX; j++) {
      thrd_tile_out_trans[j].clear();
    }
  }

  if (!empty_thrd_tile) {
    OVecCast tmp_output_c;
    for (int i = 0; i < kThreadTileDimY; i++) {
      if (static_cast<size_t>(i) >= thread_tile_nrows) {
        continue;
      }
#pragma unroll
      for (int j = 0; j < kThreadTileDimX; j++) {
        OType scaled_elt =
            static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.elt[j]) *
                               block_tile_scale);
        if constexpr (kReturnRowwise) {
          tmp_output_c.data.elt[j] = scaled_elt;
        }
        if constexpr (kReturnTranspose) {
          thrd_tile_out_trans[j].data.elt[i] = scaled_elt;
        }
      }
      if constexpr (kReturnRowwise) {
        tmp_output_c.store_to_elts(output_c + thread_tile_start_idx + i * cols, 0,
                                   thread_tile_ncols);
      }
    }

    if constexpr (kReturnTranspose) {
      const size_t thread_tile_t_start_idx =
          base_offset + thread_tile_start_col_idx * rows + thread_tile_start_row_idx;
#pragma unroll
      for (int i = 0; i < kThreadTileDimX; i++) {
        if (static_cast<size_t>(i) >= thread_tile_ncols) {
          continue;
        }
        thrd_tile_out_trans[i].store_to_elts(output_t + thread_tile_t_start_idx + i * rows, 0,
                                             thread_tile_nrows);
      }
    }
  }
}

inline ShapeRepresentation get_shape_representation(const GroupedTensor &tensor) {
  if (tensor.all_same_shape()) {
    return ShapeRepresentation::SAME_BOTH_DIMS;
  }
  if (tensor.all_same_first_dim()) {
    return ShapeRepresentation::VARYING_LAST_DIM;
  }
  if (tensor.all_same_last_dim()) {
    return ShapeRepresentation::VARYING_FIRST_DIM;
  }
  return ShapeRepresentation::VARYING_BOTH_DIMS;
}

inline void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                           const QuantizationConfig *quant_config, cudaStream_t stream) {
  NVTE_CHECK(input != nullptr, "Input grouped tensor must be provided.");
  NVTE_CHECK(output != nullptr, "Output grouped tensor must be provided.");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "FP8 2D block grouped quantize requires 2D grouped logical shapes.");
  NVTE_CHECK(input->logical_shape.data[0] == output->logical_shape.data[0] &&
                 input->logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output grouped logical shapes must match.");
  NVTE_CHECK(input->has_data(), "Cannot quantize grouped tensor without rowwise input data.");
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise grouped output data need to be allocated.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->scaling_mode == NVTE_BLOCK_SCALING_2D,
             "Grouped FP8 block scaling kernel only supports NVTE_BLOCK_SCALING_2D.");

  const ShapeRepresentation shape_rep = get_shape_representation(*output);
  NVTE_CHECK(shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                 shape_rep == ShapeRepresentation::VARYING_FIRST_DIM,
             "Grouped FP8 2D block quantize currently supports uniform shapes or varying first "
             "dimensions with a common last dimension.");
  NVTE_CHECK(input->all_same_last_dim() && output->all_same_last_dim(),
             "Grouped FP8 2D block quantize requires a common last dimension.");

  const size_t num_tensors = output->num_tensors;
  const size_t first_logical_dim = output->logical_shape.data[0];
  const size_t last_logical_dim = output->logical_shape.data[1];
  NVTE_CHECK(last_logical_dim > 0, "Grouped FP8 2D block quantize requires nonzero last dim.");

  const bool same_shape = shape_rep == ShapeRepresentation::SAME_BOTH_DIMS;
  if (same_shape) {
    NVTE_CHECK(first_logical_dim % num_tensors == 0,
               "Uniform grouped FP8 2D block quantize logical first dim must divide num_tensors.");
  } else {
    NVTE_CHECK(output->first_dims.has_data(), "Varying-first grouped output needs first_dims.");
    NVTE_CHECK(output->tensor_offsets.has_data(),
               "Varying-first grouped output needs tensor_offsets.");
  }

  const bool return_rowwise = output->has_data();
  const bool return_transpose = output->has_columnwise_data();
  if (return_rowwise) {
    NVTE_CHECK(output->scale_inv.has_data(), "Rowwise scale_inv must be allocated.");
    NVTE_CHECK(output->scale_inv.dtype == DType::kFloat32,
               "Rowwise scale_inv must have Float32 dtype.");
  }
  if (return_transpose) {
    NVTE_CHECK(output->columnwise_scale_inv.has_data(),
               "Columnwise scale_inv must be allocated.");
    NVTE_CHECK(output->columnwise_scale_inv.dtype == DType::kFloat32,
               "Columnwise scale_inv must have Float32 dtype.");
  }

  QuantizationConfig local_config;
  if (quant_config != nullptr) {
    local_config = *quant_config;
  }
  if (local_config.stochastic_rounding) {
    NVTE_ERROR("Stochastic rounding is not supported for grouped FP8 2D block scaling.");
  }
  if (transformer_engine::cuda::sm_arch() >= 100) {
    NVTE_CHECK(local_config.force_pow_2_scales,
               "On Blackwell and newer, the FP8 block scaling recipe is emulated with MXFP8, "
               "which requires using power of two scaling factors.");
  }

  const int64_t *const offsets_ptr =
      reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr =
      reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const float *const noop_ptr =
      noop == nullptr ? nullptr : reinterpret_cast<const float *>(noop->data.dptr);

  const size_t blocks_x = DIVUP(last_logical_dim, kBlockTileDim);
  const size_t rows_per_tensor =
      same_shape ? (first_logical_dim / num_tensors) : first_logical_dim;
  const size_t blocks_y = DIVUP(rows_per_tensor, kBlockTileDim);
  NVTE_CHECK(blocks_x > 0 && blocks_y > 0, "Grouped FP8 2D block quantize has no work.");
  NVTE_CHECK(num_tensors <= 65535 && blocks_y <= 65535 && blocks_x <= 2147483647,
             "Grouped FP8 2D block quantize grid is too large.");
  const dim3 grid(blocks_x, blocks_y, num_tensors);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input->dtype(), InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OutputType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_rowwise, kReturnRowwise,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  return_transpose, kReturnTranspose,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      same_shape, kSameShape,
                      grouped_block_scaled_cast_kernel<kReturnRowwise, kReturnTranspose,
                                                       kSameShape, float, InputType, OutputType>
                          <<<grid, kThreadsPerBlock, 0, stream>>>(
                              reinterpret_cast<const InputType *>(input->data.dptr),
                              kReturnRowwise ? reinterpret_cast<OutputType *>(output->data.dptr)
                                             : nullptr,
                              kReturnTranspose
                                  ? reinterpret_cast<OutputType *>(output->columnwise_data.dptr)
                                  : nullptr,
                              kReturnRowwise ? reinterpret_cast<float *>(output->scale_inv.dptr)
                                             : nullptr,
                              kReturnTranspose
                                  ? reinterpret_cast<float *>(output->columnwise_scale_inv.dptr)
                                  : nullptr,
                              num_tensors, first_logical_dim, last_logical_dim, offsets_ptr,
                              first_dims_ptr, local_config.amax_epsilon,
                              local_config.force_pow_2_scales, noop_ptr);)))))
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace group_block_scaling_2d
}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCK_SCALING_CUH_
