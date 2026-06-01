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
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/cuda_runtime.h"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_quantize_block_scaling_2d_kernel {

// Use enum integral constants (instead of namespace-scope constexpr variables) to avoid
// NVCC device lookup regressions in header-only kernels.
enum : size_t {
  BLOCK_DIM = 128,
  THREADS_PER_BLOCK = 256,
  WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP,
};

__device__ __forceinline__ size_t get_group_rows(
    const size_t tensor_id, const size_t num_tensors, const size_t logical_first_dim,
    const int64_t *const __restrict__ first_dims_ptr) {
  if (first_dims_ptr == nullptr) {
    return logical_first_dim / num_tensors;
  }
  return static_cast<size_t>(first_dims_ptr[tensor_id]);
}

__device__ __forceinline__ size_t get_group_data_offset(
    const size_t tensor_id, const size_t rows, const size_t cols,
    const int64_t *const __restrict__ offsets_ptr) {
  if (offsets_ptr == nullptr) {
    return tensor_id * rows * cols;
  }
  return static_cast<size_t>(offsets_ptr[tensor_id]);
}

__device__ __forceinline__ size_t scale_tiles_for_rows(const size_t rows) {
  return DIVUP(rows, static_cast<size_t>(BLOCK_DIM));
}

__device__ __forceinline__ size_t rowwise_scale_stride(const size_t cols) {
  return DIVUP_TO_MULTIPLE(DIVUP(cols, static_cast<size_t>(BLOCK_DIM)), static_cast<size_t>(4));
}

__device__ __forceinline__ size_t columnwise_scale_stride(const size_t rows) {
  return DIVUP_TO_MULTIPLE(DIVUP(rows, static_cast<size_t>(BLOCK_DIM)), static_cast<size_t>(4));
}

template <bool ROWWISE, bool COLWISE>
__device__ __forceinline__ void get_scale_offsets(
    const size_t tensor_id, const size_t num_tensors, const size_t logical_first_dim,
    const size_t cols, const int64_t *const __restrict__ first_dims_ptr,
    size_t *const rowwise_scale_base, size_t *const columnwise_scale_base) {
  const size_t row_stride = rowwise_scale_stride(cols);
  const size_t col_tiles = DIVUP(cols, static_cast<size_t>(BLOCK_DIM));

  size_t row_base = 0;
  size_t col_base = 0;
  if (first_dims_ptr == nullptr) {
    const size_t rows = logical_first_dim / num_tensors;
    const size_t row_tiles = scale_tiles_for_rows(rows);
    if constexpr (ROWWISE) {
      row_base = tensor_id * row_tiles * row_stride;
    }
    if constexpr (COLWISE) {
      col_base = tensor_id * col_tiles * columnwise_scale_stride(rows);
    }
  } else {
    for (size_t i = 0; i < tensor_id; ++i) {
      const size_t rows = static_cast<size_t>(first_dims_ptr[i]);
      const size_t row_tiles = scale_tiles_for_rows(rows);
      if constexpr (ROWWISE) {
        row_base += row_tiles * row_stride;
      }
      if constexpr (COLWISE) {
        col_base += col_tiles * columnwise_scale_stride(rows);
      }
    }
  }

  if constexpr (ROWWISE) {
    *rowwise_scale_base = row_base;
  }
  if constexpr (COLWISE) {
    *columnwise_scale_base = col_base;
  }
}

__device__ __forceinline__ float block_reduce_max(float val) {
  constexpr int warp_size = THREADS_PER_WARP;
  const int lane = threadIdx.x % warp_size;
  const int warp_id = threadIdx.x / warp_size;
  val = warp_reduce_max<warp_size>(val);

  __shared__ float warp_amax[WARPS_PER_BLOCK];
  if (lane == 0) {
    warp_amax[warp_id] = val;
  }
  __syncthreads();

  float block_amax = 0.0f;
  if (threadIdx.x < WARPS_PER_BLOCK) {
    block_amax = warp_amax[threadIdx.x];
  }
  if (warp_id == 0) {
    block_amax = warp_reduce_max<WARPS_PER_BLOCK>(block_amax);
  }
  if (threadIdx.x == 0) {
    warp_amax[0] = block_amax;
  }
  __syncthreads();
  return warp_amax[0];
}

template <typename IType, typename OType, bool ROWWISE, bool COLWISE>
__global__ void __launch_bounds__(THREADS_PER_BLOCK) group_quantize_fp8_block_scaling_2d_kernel(
    const IType *const __restrict__ input, OType *const __restrict__ output_rowwise,
    OType *const __restrict__ output_colwise, float *const __restrict__ scale_inv_rowwise,
    float *const __restrict__ scale_inv_colwise, const size_t num_tensors,
    const size_t logical_first_dim, const size_t cols,
    const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ offsets_ptr, const size_t max_row_tiles,
    const float epsilon, const bool force_pow_2_scales, const float *const __restrict__ noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const size_t col_tile = blockIdx.x;
  const size_t packed_y = blockIdx.y;
  const size_t tensor_id = packed_y / max_row_tiles;
  const size_t row_tile = packed_y - tensor_id * max_row_tiles;
  if (tensor_id >= num_tensors) {
    return;
  }

  const size_t rows = get_group_rows(tensor_id, num_tensors, logical_first_dim, first_dims_ptr);
  if (rows == 0 || cols == 0) {
    return;
  }

  const size_t row_tiles = scale_tiles_for_rows(rows);
  if (row_tile >= row_tiles) {
    return;
  }

  const size_t col_tiles = DIVUP(cols, static_cast<size_t>(BLOCK_DIM));
  if (col_tile >= col_tiles) {
    return;
  }

  const size_t tensor_base = get_group_data_offset(tensor_id, rows, cols, offsets_ptr);
  const size_t row_start = row_tile * BLOCK_DIM;
  const size_t col_start = col_tile * BLOCK_DIM;
  const size_t valid_rows = (rows - row_start < BLOCK_DIM) ? rows - row_start : BLOCK_DIM;
  const size_t valid_cols = (cols - col_start < BLOCK_DIM) ? cols - col_start : BLOCK_DIM;

  float thread_amax = 0.0f;
  for (size_t idx = threadIdx.x; idx < BLOCK_DIM * BLOCK_DIM; idx += blockDim.x) {
    const size_t local_row = idx / BLOCK_DIM;
    const size_t local_col = idx - local_row * BLOCK_DIM;
    if (local_row < valid_rows && local_col < valid_cols) {
      const size_t input_idx = tensor_base + (row_start + local_row) * cols + col_start + local_col;
      const float elt = static_cast<float>(input[input_idx]);
      thread_amax = fmaxf(thread_amax, fabsf(elt));
    }
  }

  const float tile_amax = block_reduce_max(thread_amax);
  const float scale = compute_scale_from_types<IType, OType>(tile_amax, epsilon, force_pow_2_scales);
  const float scale_inv = 1.0f / scale;

  size_t rowwise_scale_base = 0;
  size_t columnwise_scale_base = 0;
  get_scale_offsets<ROWWISE, COLWISE>(tensor_id, num_tensors, logical_first_dim, cols,
                                      first_dims_ptr, &rowwise_scale_base,
                                      &columnwise_scale_base);

  if (threadIdx.x == 0) {
    if constexpr (ROWWISE) {
      const size_t row_stride = rowwise_scale_stride(cols);
      scale_inv_rowwise[rowwise_scale_base + row_tile * row_stride + col_tile] = scale_inv;
    }
    if constexpr (COLWISE) {
      const size_t col_stride = columnwise_scale_stride(rows);
      scale_inv_colwise[columnwise_scale_base + col_tile * col_stride + row_tile] = scale_inv;
    }
  }

  for (size_t idx = threadIdx.x; idx < BLOCK_DIM * BLOCK_DIM; idx += blockDim.x) {
    const size_t local_row = idx / BLOCK_DIM;
    const size_t local_col = idx - local_row * BLOCK_DIM;
    if (local_row < valid_rows && local_col < valid_cols) {
      const size_t row = row_start + local_row;
      const size_t col = col_start + local_col;
      const size_t input_idx = tensor_base + row * cols + col;
      const OType out = static_cast<OType>(static_cast<float>(input[input_idx]) * scale);
      if constexpr (ROWWISE) {
        output_rowwise[input_idx] = out;
      }
      if constexpr (COLWISE) {
        output_colwise[tensor_base + col * rows + row] = out;
      }
    }
  }
}

}  // namespace group_quantize_block_scaling_2d_kernel

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void group_quantize_block_scaling_2d(const GroupedTensor *input, const GroupedTensor *activations,
                                     const Tensor *noop, GroupedTensor *output,
                                     GroupedTensor *dbias, Tensor *workspace,
                                     const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace group_quantize_block_scaling_2d_kernel;
  (void)activations;
  (void)dbias;
  (void)workspace;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");

  NVTE_CHECK(!IS_ACT, "IS_ACT is not implemented for grouped NVTE_BLOCK_SCALING_2D");
  NVTE_CHECK((!IS_DBIAS && !IS_DACT),
             "IS_DBIAS and IS_DACT are not implemented for grouped NVTE_BLOCK_SCALING_2D");

  const float epsilon = quant_config == nullptr ? 0.0f : quant_config->amax_epsilon;
  const bool force_pow_2_scales =
      quant_config == nullptr ? false : quant_config->force_pow_2_scales;

  if (transformer_engine::cuda::sm_arch() >= 100) {
    NVTE_CHECK(force_pow_2_scales,
               "On Blackwell and newer, the FP8 block scaling recipe is emulated with MXFP8, ",
               "which requires using power of two scaling factors.");
  }

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");
  NVTE_CHECK(input->all_same_last_dim() && output->all_same_last_dim(),
             "Grouped FP8 2D block-scaling quantize supports only a uniform last dimension.");
  NVTE_CHECK(!input->last_dims.has_data() && !output->last_dims.has_data(),
             "Grouped FP8 2D block-scaling quantize does not support varying last dimensions.");
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Grouped FP8 2D block-scaling quantize requires 2D logical shapes.");
  NVTE_CHECK(input->logical_shape.data[0] == output->logical_shape.data[0] &&
                 input->logical_shape.data[1] == output->logical_shape.data[1],
             "Input and output grouped tensors must have matching logical shapes.");
  NVTE_CHECK(input->first_dims.has_data() == output->first_dims.has_data(),
             "Input and output grouped tensors must have matching first_dims metadata.");
  NVTE_CHECK(input->tensor_offsets.has_data() == output->tensor_offsets.has_data(),
             "Input and output grouped tensors must have matching tensor_offsets metadata.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");

  const size_t num_tensors = input->num_tensors;
  const size_t logical_first_dim = input->logical_shape.data[0];
  const size_t logical_last_dim = input->logical_shape.data[1];
  NVTE_CHECK(num_tensors > 0, "Grouped FP8 2D block-scaling quantize requires num_tensors > 0.");

  const bool varying_first_dim = output->first_dims.has_data();

  if (!varying_first_dim) {
    NVTE_CHECK(logical_first_dim % num_tensors == 0,
               "Logical first dimension must be divisible by num_tensors for uniform grouped ",
               "FP8 2D block-scaling quantize.");
  }

  const size_t max_row_tiles =
      varying_first_dim ? DIVUP(logical_first_dim, static_cast<size_t>(BLOCK_DIM))
                        : DIVUP(logical_first_dim / num_tensors, static_cast<size_t>(BLOCK_DIM));
  const size_t col_tiles = DIVUP(logical_last_dim, static_cast<size_t>(BLOCK_DIM));
  if (max_row_tiles == 0 || col_tiles == 0) {
    return;
  }

  const int64_t *const first_dims_ptr =
      varying_first_dim ? reinterpret_cast<const int64_t *>(output->first_dims.dptr) : nullptr;
  const int64_t *const offsets_ptr =
      output->tensor_offsets.has_data()
          ? reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr)
          : nullptr;

  const float *const noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);

  const bool use_rowwise = output->has_data();
  const bool use_colwise = output->has_columnwise_data();
  if (use_rowwise) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
    NVTE_CHECK(output->scale_inv.dtype == DType::kFloat32,
               "FP8 2D block-scaling rowwise scale_inv must be Float32.");
  }
  if (use_colwise) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Columnwise scaling tensor must be allocated.");
    NVTE_CHECK(output->columnwise_scale_inv.dtype == DType::kFloat32,
               "FP8 2D block-scaling columnwise scale_inv must be Float32.");
  }

  const dim3 grid(col_tiles, num_tensors * max_row_tiles);
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              use_rowwise, ROWWISE,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  use_colwise, COLWISE,
                  auto kernel =
                      group_quantize_fp8_block_scaling_2d_kernel<IType, OType, ROWWISE, COLWISE>;
                  kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
                      reinterpret_cast<const IType *>(input->data.dptr),
                      reinterpret_cast<OType *>(output->data.dptr),
                      reinterpret_cast<OType *>(output->columnwise_data.dptr),
                      reinterpret_cast<float *>(output->scale_inv.dptr),
                      reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                      logical_first_dim, logical_last_dim, first_dims_ptr, offsets_ptr,
                      max_row_tiles, epsilon, force_pow_2_scales, noop_ptr);))));

  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCK_SCALING_CUH_
