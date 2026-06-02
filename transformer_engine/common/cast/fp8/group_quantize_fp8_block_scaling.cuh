/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8_block_scaling.cuh
 *  \brief CUDA kernels to quantize grouped tensors to FP8 block-scaling formats.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCK_SCALING_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCK_SCALING_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/cuda_runtime.h"
#include "../../utils.cuh"
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_block_scaling {

constexpr size_t kBlockLen = 128;
constexpr size_t kThreadsPerBlock = 256;

__device__ __forceinline__ size_t round_up_to_multiple(const size_t value,
                                                       const size_t multiple) {
  return DIVUP(value, multiple) * multiple;
}

template <bool kIs2DScaling>
__device__ __forceinline__ size_t rowwise_scale_elements(const size_t rows, const size_t cols) {
  const size_t row_blocks = DIVUP(rows, kBlockLen);
  const size_t col_blocks = DIVUP(cols, kBlockLen);
  if constexpr (kIs2DScaling) {
    return row_blocks * round_up_to_multiple(col_blocks, 4);
  } else {
    return col_blocks * round_up_to_multiple(rows, 4);
  }
}

template <bool kIs2DScaling>
__device__ __forceinline__ size_t columnwise_scale_elements(const size_t rows, const size_t cols) {
  const size_t row_blocks = DIVUP(rows, kBlockLen);
  const size_t col_blocks = DIVUP(cols, kBlockLen);
  if constexpr (kIs2DScaling) {
    return col_blocks * round_up_to_multiple(row_blocks, 4);
  } else {
    return row_blocks * round_up_to_multiple(cols, 4);
  }
}

template <bool kIs2DScaling>
__device__ __forceinline__ size_t scale_offset_for_tensor(
    const size_t tensor_id, const size_t rows_per_tensor, const size_t cols,
    const int64_t *const __restrict__ first_dims, const bool has_first_dims,
    const bool columnwise) {
  if (!has_first_dims) {
    const size_t scale_elements =
        columnwise ? columnwise_scale_elements<kIs2DScaling>(rows_per_tensor, cols)
                   : rowwise_scale_elements<kIs2DScaling>(rows_per_tensor, cols);
    return tensor_id * scale_elements;
  }

  size_t offset = 0;
  for (size_t i = 0; i < tensor_id; ++i) {
    const size_t rows = static_cast<size_t>(first_dims[i]);
    offset += columnwise ? columnwise_scale_elements<kIs2DScaling>(rows, cols)
                         : rowwise_scale_elements<kIs2DScaling>(rows, cols);
  }
  return offset;
}

struct TileDescriptor {
  size_t tensor_id = 0;
  size_t tile_y = 0;
  size_t rows = 0;
  size_t tensor_base = 0;
  bool valid = false;
};

__device__ __forceinline__ TileDescriptor decode_tile(
    size_t packed_tile_y, const size_t num_tensors, const size_t logical_first_dim,
    const size_t cols, const int64_t *const __restrict__ first_dims,
    const int64_t *const __restrict__ tensor_offsets, const bool has_first_dims) {
  TileDescriptor desc;

  if (!has_first_dims) {
    const size_t rows_per_tensor = logical_first_dim / num_tensors;
    const size_t tile_rows_per_tensor = DIVUP(rows_per_tensor, kBlockLen);
    if (tile_rows_per_tensor == 0) {
      return desc;
    }
    desc.tensor_id = packed_tile_y / tile_rows_per_tensor;
    if (desc.tensor_id >= num_tensors) {
      return desc;
    }
    desc.tile_y = packed_tile_y % tile_rows_per_tensor;
    desc.rows = rows_per_tensor;
    desc.tensor_base = desc.tensor_id * rows_per_tensor * cols;
    desc.valid = true;
    return desc;
  }

  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    const size_t rows = static_cast<size_t>(first_dims[tensor_id]);
    const size_t tile_rows = DIVUP(rows, kBlockLen);
    if (packed_tile_y < tile_rows) {
      desc.tensor_id = tensor_id;
      desc.tile_y = packed_tile_y;
      desc.rows = rows;
      desc.tensor_base = static_cast<size_t>(tensor_offsets[tensor_id]);
      desc.valid = true;
      return desc;
    }
    packed_tile_y -= tile_rows;
  }
  return desc;
}

__device__ __forceinline__ void shared_atomic_max_abs(unsigned int *addr, const float value) {
  atomicMax(addr, __float_as_uint(fabsf(value)));
}

template <bool kIs2DScaling, typename IType, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock) group_quantize_fp8_block_scaling_kernel(
    const IType *const __restrict__ input, OType *const __restrict__ output,
    OType *const __restrict__ output_t, float *const __restrict__ scale_inv,
    float *const __restrict__ scale_inv_t, const size_t num_tensors,
    const size_t logical_first_dim, const size_t cols,
    const int64_t *const __restrict__ first_dims,
    const int64_t *const __restrict__ tensor_offsets, const bool has_first_dims,
    const bool return_rowwise, const bool return_columnwise, const float epsilon,
    const bool force_pow_2_scales, const float *const __restrict__ noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const TileDescriptor tile =
      decode_tile(blockIdx.y, num_tensors, logical_first_dim, cols, first_dims, tensor_offsets,
                  has_first_dims);
  if (!tile.valid || tile.rows == 0 || cols == 0) {
    return;
  }

  const size_t tile_x = blockIdx.x;
  const size_t row_start = tile.tile_y * kBlockLen;
  const size_t col_start = tile_x * kBlockLen;
  if (row_start >= tile.rows || col_start >= cols) {
    return;
  }

  const size_t rows_per_tensor = has_first_dims ? 0 : logical_first_dim / num_tensors;
  const size_t rowwise_scale_offset =
      scale_offset_for_tensor<kIs2DScaling>(tile.tensor_id, rows_per_tensor, cols, first_dims,
                                            has_first_dims, false);
  const size_t columnwise_scale_offset =
      scale_offset_for_tensor<kIs2DScaling>(tile.tensor_id, rows_per_tensor, cols, first_dims,
                                            has_first_dims, true);

  __shared__ unsigned int tile_amax_bits;
  __shared__ unsigned int row_amax_bits[kBlockLen];
  __shared__ unsigned int col_amax_bits[kBlockLen];
  __shared__ float tile_scale;
  __shared__ float row_scale[kBlockLen];
  __shared__ float col_scale[kBlockLen];

  if (threadIdx.x == 0) {
    tile_amax_bits = 0;
  }
  if (threadIdx.x < kBlockLen) {
    row_amax_bits[threadIdx.x] = 0;
    col_amax_bits[threadIdx.x] = 0;
  }
  __syncthreads();

  for (size_t idx = threadIdx.x; idx < kBlockLen * kBlockLen; idx += blockDim.x) {
    const size_t local_row = idx / kBlockLen;
    const size_t local_col = idx % kBlockLen;
    const size_t row = row_start + local_row;
    const size_t col = col_start + local_col;
    if (row >= tile.rows || col >= cols) {
      continue;
    }
    const float value = static_cast<float>(input[tile.tensor_base + row * cols + col]);
    if constexpr (kIs2DScaling) {
      shared_atomic_max_abs(&tile_amax_bits, value);
    } else {
      if (return_rowwise) {
        shared_atomic_max_abs(&row_amax_bits[local_row], value);
      }
      if (return_columnwise) {
        shared_atomic_max_abs(&col_amax_bits[local_col], value);
      }
    }
  }
  __syncthreads();

  if constexpr (kIs2DScaling) {
    if (threadIdx.x == 0) {
      const float amax = __uint_as_float(tile_amax_bits);
      tile_scale = compute_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
      const float inv_scale = 1.0f / tile_scale;
      const size_t row_blocks = DIVUP(tile.rows, kBlockLen);
      const size_t col_blocks = DIVUP(cols, kBlockLen);
      if (return_rowwise) {
        const size_t rowwise_stride = round_up_to_multiple(col_blocks, 4);
        scale_inv[rowwise_scale_offset + tile.tile_y * rowwise_stride + tile_x] = inv_scale;
      }
      if (return_columnwise) {
        const size_t columnwise_stride = round_up_to_multiple(row_blocks, 4);
        scale_inv_t[columnwise_scale_offset + tile_x * columnwise_stride + tile.tile_y] =
            inv_scale;
      }
    }
  } else {
    if (threadIdx.x < kBlockLen) {
      const size_t local_row = threadIdx.x;
      const size_t row = row_start + local_row;
      if (return_rowwise && row < tile.rows) {
        const float amax = __uint_as_float(row_amax_bits[local_row]);
        row_scale[local_row] =
            compute_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
        const size_t rowwise_stride = round_up_to_multiple(tile.rows, 4);
        scale_inv[rowwise_scale_offset + tile_x * rowwise_stride + row] =
            1.0f / row_scale[local_row];
      }

      const size_t local_col = threadIdx.x;
      const size_t col = col_start + local_col;
      if (return_columnwise && col < cols) {
        const float amax = __uint_as_float(col_amax_bits[local_col]);
        col_scale[local_col] =
            compute_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
        const size_t columnwise_stride = round_up_to_multiple(cols, 4);
        scale_inv_t[columnwise_scale_offset + tile.tile_y * columnwise_stride + col] =
            1.0f / col_scale[local_col];
      }
    }
  }
  __syncthreads();

  for (size_t idx = threadIdx.x; idx < kBlockLen * kBlockLen; idx += blockDim.x) {
    const size_t local_row = idx / kBlockLen;
    const size_t local_col = idx % kBlockLen;
    const size_t row = row_start + local_row;
    const size_t col = col_start + local_col;
    if (row >= tile.rows || col >= cols) {
      continue;
    }

    const float value = static_cast<float>(input[tile.tensor_base + row * cols + col]);
    if (return_rowwise) {
      const float scale = kIs2DScaling ? tile_scale : row_scale[local_row];
      output[tile.tensor_base + row * cols + col] = static_cast<OType>(value * scale);
    }
    if (return_columnwise) {
      const float scale = kIs2DScaling ? tile_scale : col_scale[local_col];
      output_t[tile.tensor_base + col * tile.rows + row] = static_cast<OType>(value * scale);
    }
  }
}

template <bool kIs2DScaling>
inline size_t host_rowwise_scale_elements(const size_t rows, const size_t cols) {
  const size_t row_blocks = DIVUP(rows, kBlockLen);
  const size_t col_blocks = DIVUP(cols, kBlockLen);
  if constexpr (kIs2DScaling) {
    return row_blocks * DIVUP_TO_MULTIPLE(col_blocks, static_cast<size_t>(4));
  } else {
    return col_blocks * DIVUP_TO_MULTIPLE(rows, static_cast<size_t>(4));
  }
}

template <bool kIs2DScaling>
inline size_t host_columnwise_scale_elements(const size_t rows, const size_t cols) {
  const size_t row_blocks = DIVUP(rows, kBlockLen);
  const size_t col_blocks = DIVUP(cols, kBlockLen);
  if constexpr (kIs2DScaling) {
    return col_blocks * DIVUP_TO_MULTIPLE(row_blocks, static_cast<size_t>(4));
  } else {
    return row_blocks * DIVUP_TO_MULTIPLE(cols, static_cast<size_t>(4));
  }
}

template <bool kIs2DScaling>
void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                    const QuantizationConfig *quant_config, cudaStream_t stream) {
  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");

  const bool use_rowwise = output->has_data();
  const bool use_columnwise = output->has_columnwise_data();
  NVTE_CHECK(use_rowwise || use_columnwise,
             "Either rowwise or columnwise output data need to be allocated.");

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Grouped FP8 block-scaling quantize expects 2D logical shapes.");
  NVTE_CHECK(input->all_same_last_dim() && output->all_same_last_dim(),
             "Grouped FP8 block-scaling quantize currently supports only a common last dim.");
  NVTE_CHECK(input->all_same_first_dim() == output->all_same_first_dim(),
             "Input and output grouped tensors must use matching first-dim metadata.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->scale_inv.dtype == DType::kFloat32 || !use_rowwise,
             "FP8 block-scaling rowwise scale_inv must be FP32.");
  NVTE_CHECK(output->columnwise_scale_inv.dtype == DType::kFloat32 || !use_columnwise,
             "FP8 block-scaling columnwise scale_inv must be FP32.");

  const size_t num_tensors = input->num_tensors;
  const size_t logical_first_dim = input->logical_shape.data[0];
  const size_t cols = input->logical_shape.data[1];
  if (logical_first_dim == 0 || cols == 0) {
    return;
  }

  const bool has_first_dims = !input->all_same_first_dim();
  const int64_t *const first_dims_ptr =
      has_first_dims ? reinterpret_cast<const int64_t *>(input->first_dims.dptr) : nullptr;
  const int64_t *const tensor_offsets_ptr =
      has_first_dims ? reinterpret_cast<const int64_t *>(input->tensor_offsets.dptr) : nullptr;
  if (has_first_dims) {
    NVTE_CHECK(first_dims_ptr != nullptr && tensor_offsets_ptr != nullptr,
               "Grouped FP8 block-scaling quantize requires first_dims and tensor_offsets.");
  } else {
    NVTE_CHECK(logical_first_dim % num_tensors == 0,
               "Uniform grouped tensors require logical first dim divisible by num_tensors.");
  }

  if (transformer_engine::cuda::sm_arch() >= 100) {
    NVTE_CHECK(quant_config->force_pow_2_scales,
               "On Blackwell and newer, the FP8 block scaling recipe is emulated with MXFP8, "
               "which requires using power of two scaling factors.");
  }

  const size_t rows_per_tensor = has_first_dims ? 0 : logical_first_dim / num_tensors;
  const size_t work_tiles_y =
      has_first_dims ? (DIVUP(logical_first_dim, kBlockLen) + num_tensors)
                     : (num_tensors * DIVUP(rows_per_tensor, kBlockLen));
  const size_t work_tiles_x = DIVUP(cols, kBlockLen);
  if (work_tiles_x == 0 || work_tiles_y == 0) {
    return;
  }

  if (use_rowwise) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Rowwise scale_inv must be allocated.");
    const size_t min_elements =
        has_first_dims
            ? 0
            : num_tensors * host_rowwise_scale_elements<kIs2DScaling>(rows_per_tensor, cols);
    NVTE_CHECK(has_first_dims || output->scale_inv.numel() >= min_elements,
               "Rowwise scale_inv buffer is too small for grouped FP8 block-scaling quantize.");
  }
  if (use_columnwise) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Columnwise scale_inv must be allocated.");
    const size_t min_elements =
        has_first_dims
            ? 0
            : num_tensors * host_columnwise_scale_elements<kIs2DScaling>(rows_per_tensor, cols);
    NVTE_CHECK(has_first_dims || output->columnwise_scale_inv.numel() >= min_elements,
               "Columnwise scale_inv buffer is too small for grouped FP8 block-scaling quantize.");
  }

  const float epsilon = quant_config->amax_epsilon;
  const bool force_pow_2_scales = quant_config->force_pow_2_scales;
  const float *const noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          const dim3 grid(work_tiles_x, work_tiles_y, 1);
          group_quantize_fp8_block_scaling_kernel<kIs2DScaling, IType, OType>
              <<<grid, kThreadsPerBlock, 0, stream>>>(
                  reinterpret_cast<const IType *>(input->data.dptr),
                  use_rowwise ? reinterpret_cast<OType *>(output->data.dptr) : nullptr,
                  use_columnwise ? reinterpret_cast<OType *>(output->columnwise_data.dptr)
                                 : nullptr,
                  use_rowwise ? reinterpret_cast<float *>(output->scale_inv.dptr) : nullptr,
                  use_columnwise ? reinterpret_cast<float *>(output->columnwise_scale_inv.dptr)
                                 : nullptr,
                  num_tensors, logical_first_dim, cols, first_dims_ptr, tensor_offsets_ptr,
                  has_first_dims, use_rowwise, use_columnwise, epsilon, force_pow_2_scales,
                  noop_ptr);););
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace group_block_scaling
}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCK_SCALING_CUH_
