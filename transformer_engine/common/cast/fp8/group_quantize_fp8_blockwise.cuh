/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8_blockwise.cuh
 *  \brief CUDA kernels to quantize grouped tensors to FP8 block-scaled formats.
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
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8_blockwise {
namespace group_quantize_kernel {

constexpr size_t BLOCK_LEN = 128;
constexpr size_t THREADS_PER_BLOCK_1D = 128;
constexpr size_t THREADS_PER_BLOCK_2D = 256;

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
    return DIVUP(rows, BLOCK_LEN) * round_up_to_multiple(cols, static_cast<size_t>(4));
  }
  return DIVUP(cols, BLOCK_LEN) * round_up_to_multiple(rows, static_cast<size_t>(4));
}

__device__ __forceinline__ size_t scale_elements_2d(const size_t rows, const size_t cols,
                                                    const bool columnwise) {
  if (columnwise) {
    return DIVUP(cols, BLOCK_LEN) *
           round_up_to_multiple(DIVUP(rows, BLOCK_LEN), static_cast<size_t>(4));
  }
  return DIVUP(rows, BLOCK_LEN) *
         round_up_to_multiple(DIVUP(cols, BLOCK_LEN), static_cast<size_t>(4));
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

template <typename IType>
__device__ __forceinline__ float load_input(const IType *input, const size_t idx) {
  return static_cast<float>(input[idx]);
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
  const size_t local_m = tile_m * BLOCK_LEN + threadIdx.x;
  if (local_m >= rows) {
    return;
  }

  const size_t tile_n = blockIdx.x;
  const size_t local_n_begin = tile_n * BLOCK_LEN;
  if (local_n_begin >= last_logical_dim) {
    return;
  }

  const size_t data_offset =
      grouped_tensor_data_offset(tensor_idx, num_tensors, first_logical_dim, last_logical_dim,
                                 tensor_offsets);
  const size_t input_row = data_offset + local_m * last_logical_dim;
  const size_t cols_this_tile = min(BLOCK_LEN, last_logical_dim - local_n_begin);

  float amax = 0.0f;
#pragma unroll
  for (size_t k = 0; k < BLOCK_LEN; ++k) {
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
  for (size_t k = 0; k < BLOCK_LEN; ++k) {
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
  const size_t local_m_begin = tile_m * BLOCK_LEN;
  if (local_m_begin >= rows) {
    return;
  }

  const size_t tile_n = blockIdx.x;
  const size_t local_n = tile_n * BLOCK_LEN + threadIdx.x;
  if (local_n >= last_logical_dim) {
    return;
  }

  const size_t rows_this_tile = min(BLOCK_LEN, rows - local_m_begin);
  const size_t data_offset =
      grouped_tensor_data_offset(tensor_idx, num_tensors, first_logical_dim, last_logical_dim,
                                 tensor_offsets);

  float amax = 0.0f;
#pragma unroll
  for (size_t r = 0; r < BLOCK_LEN; ++r) {
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
  for (size_t r = 0; r < BLOCK_LEN; ++r) {
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
  const size_t local_m_begin = tile_m * BLOCK_LEN;
  const size_t local_n_begin = tile_n * BLOCK_LEN;
  if (local_m_begin >= rows || local_n_begin >= last_logical_dim) {
    return;
  }

  const size_t rows_this_tile = min(BLOCK_LEN, rows - local_m_begin);
  const size_t cols_this_tile = min(BLOCK_LEN, last_logical_dim - local_n_begin);
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
          round_up_to_multiple(DIVUP(last_logical_dim, BLOCK_LEN), static_cast<size_t>(4));
      const size_t scale_base = scale_base_offset<true>(tensor_idx, num_tensors, first_logical_dim,
                                                        last_logical_dim, first_dims, false);
      scale_inv[scale_base + tile_m * padded_scale_cols + tile_n] = scale_inv_value;
    }
    if constexpr (RETURN_COLUMNWISE) {
      const size_t padded_scale_cols =
          round_up_to_multiple(DIVUP(rows, BLOCK_LEN), static_cast<size_t>(4));
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

  const size_t max_tensor_rows =
      first_dims_ptr == nullptr ? first_logical_dim / num_tensors : first_logical_dim;
  const size_t tiles_m = DIVUP(max_tensor_rows, BLOCK_LEN);
  const size_t tiles_n = DIVUP(last_logical_dim, BLOCK_LEN);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          if constexpr (IS_2D) {
            dim3 grid(tiles_n, tiles_m, num_tensors);
            if (output->has_data() && output->has_columnwise_data()) {
              square_2d_kernel<IType, OType, true, true><<<grid, THREADS_PER_BLOCK_2D, 0, stream>>>(
                  reinterpret_cast<const IType *>(input->data.dptr),
                  reinterpret_cast<OType *>(output->data.dptr),
                  reinterpret_cast<OType *>(output->columnwise_data.dptr),
                  reinterpret_cast<float *>(output->scale_inv.dptr),
                  reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                  first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr, epsilon,
                  force_pow_2_scales, noop_ptr);
            } else if (output->has_data()) {
              square_2d_kernel<IType, OType, true, false>
                  <<<grid, THREADS_PER_BLOCK_2D, 0, stream>>>(
                      reinterpret_cast<const IType *>(input->data.dptr),
                      reinterpret_cast<OType *>(output->data.dptr), nullptr,
                      reinterpret_cast<float *>(output->scale_inv.dptr), nullptr, num_tensors,
                      first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr, epsilon,
                      force_pow_2_scales, noop_ptr);
            } else {
              square_2d_kernel<IType, OType, false, true>
                  <<<grid, THREADS_PER_BLOCK_2D, 0, stream>>>(
                      reinterpret_cast<const IType *>(input->data.dptr), nullptr,
                      reinterpret_cast<OType *>(output->columnwise_data.dptr), nullptr,
                      reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                      first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr, epsilon,
                      force_pow_2_scales, noop_ptr);
            }
          } else {
            if (output->has_data()) {
              dim3 row_grid(tiles_n, tiles_m, num_tensors);
              rowwise_1d_kernel<IType, OType><<<row_grid, THREADS_PER_BLOCK_1D, 0, stream>>>(
                  reinterpret_cast<const IType *>(input->data.dptr),
                  reinterpret_cast<OType *>(output->data.dptr),
                  reinterpret_cast<float *>(output->scale_inv.dptr), num_tensors,
                  first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr, epsilon,
                  force_pow_2_scales, noop_ptr);
            }
            if (output->has_columnwise_data()) {
              dim3 col_grid(tiles_n, tiles_m, num_tensors);
              columnwise_1d_kernel<IType, OType><<<col_grid, THREADS_PER_BLOCK_1D, 0, stream>>>(
                  reinterpret_cast<const IType *>(input->data.dptr),
                  reinterpret_cast<OType *>(output->columnwise_data.dptr),
                  reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                  first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr, epsilon,
                  force_pow_2_scales, noop_ptr);
            }
          })  // Output type
  )          // Input type

  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fp8_blockwise
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCKWISE_CUH_
