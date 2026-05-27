/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8_block_scaling.cuh
 *  \brief Grouped FP8 block-scaling quantization helpers.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCK_SCALING_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCK_SCALING_CUH_

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "../../common.h"
#include "../../transpose/cast_transpose.h"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {

namespace {

inline std::vector<int64_t> copy_i64_from_device(const SimpleTensor &tensor, const size_t n,
                                                 cudaStream_t stream) {
  std::vector<int64_t> host(n);
  NVTE_CHECK(tensor.dptr != nullptr, "Expected device tensor metadata to be allocated.");
  NVTE_CHECK(tensor.dtype == DType::kInt64, "Expected int64 metadata tensor.");
  NVTE_CHECK_CUDA(cudaMemcpyAsync(host.data(), tensor.dptr, n * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  return host;
}

inline size_t div_up_128(const size_t x) { return DIVUP(x, static_cast<size_t>(128)); }

inline void *ptr_with_elem_offset(void *base, const size_t elem_offset, const DType dtype) {
  auto *base_u8 = reinterpret_cast<uint8_t *>(base);
  return static_cast<void *>(base_u8 + get_buffer_size_bytes(elem_offset, dtype));
}

}  // namespace

inline void group_quantize_2d_block_scaling(const GroupedTensor *input, const Tensor *noop,
                                            GroupedTensor *output,
                                            const QuantizationConfig *quant_config,
                                            cudaStream_t stream) {
  using transformer_engine::detail::quantize_transpose_square_blockwise;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must match.");
  NVTE_CHECK(input->has_data(), "Grouped input tensor must have rowwise data.");
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data must be allocated.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must be FP8.");
  NVTE_CHECK(output->all_same_last_dim(),
             "Grouped FP8 block-scaling only supports constant last dimension.");

  const size_t num_tensors = input->num_tensors;
  const size_t first_logical_dim = input->logical_shape.data[0];
  const size_t last_logical_dim = input->logical_shape.data[1];

  // Derive per-tensor row counts and flattened element offsets.
  std::vector<int64_t> first_dims_host;
  std::vector<int64_t> offsets_host;
  first_dims_host.resize(num_tensors, 0);
  offsets_host.resize(num_tensors + 1, 0);

  if (!output->all_same_first_dim()) {
    NVTE_CHECK(output->first_dims.has_data(),
               "Expected first_dims metadata for varying-first-dim grouped tensor.");
    NVTE_CHECK(output->tensor_offsets.has_data(),
               "Expected tensor_offsets metadata for varying-first-dim grouped tensor.");
    first_dims_host = copy_i64_from_device(output->first_dims, num_tensors, stream);
    offsets_host = copy_i64_from_device(output->tensor_offsets, num_tensors + 1, stream);
  } else {
    const size_t common_first_dim = output->get_common_first_dim();
    for (size_t i = 0; i < num_tensors; ++i) {
      first_dims_host[i] = static_cast<int64_t>(common_first_dim);
      offsets_host[i] = static_cast<int64_t>(i * common_first_dim * last_logical_dim);
    }
    offsets_host[num_tensors] = static_cast<int64_t>(first_logical_dim * last_logical_dim);
  }

  const bool return_transpose = output->has_columnwise_data();
  const bool force_pow_2_scales = (quant_config != nullptr) ? quant_config->force_pow_2_scales
                                                             : false;
  const float epsilon = (quant_config != nullptr) ? quant_config->amax_epsilon : 0.0f;

  const size_t rowwise_scale_capacity = output->scale_inv.numel();
  const size_t colwise_scale_capacity = output->columnwise_scale_inv.numel();
  size_t rowwise_scale_offset = 0;
  size_t colwise_scale_offset = 0;

  for (size_t i = 0; i < num_tensors; ++i) {
    const size_t rows = static_cast<size_t>(std::max<int64_t>(first_dims_host[i], 0));
    const size_t cols = last_logical_dim;
    const size_t elem_offset = static_cast<size_t>(std::max<int64_t>(offsets_host[i], 0));

    const size_t row_scale_rows = div_up_128(rows);
    const size_t row_scale_cols = div_up_128(cols);
    const size_t row_scale_elems = row_scale_rows * row_scale_cols;
    const size_t col_scale_rows = div_up_128(cols);
    const size_t col_scale_cols = div_up_128(rows);
    const size_t col_scale_elems = col_scale_rows * col_scale_cols;

    if (output->has_data()) {
      NVTE_CHECK(rowwise_scale_offset + row_scale_elems <= rowwise_scale_capacity,
                 "Grouped FP8 block-scaling rowwise scale buffer is too small.");
    }
    if (return_transpose) {
      NVTE_CHECK(colwise_scale_offset + col_scale_elems <= colwise_scale_capacity,
                 "Grouped FP8 block-scaling columnwise scale buffer is too small.");
    }

    if (rows == 0 || cols == 0) {
      rowwise_scale_offset += row_scale_elems;
      colwise_scale_offset += col_scale_elems;
      continue;
    }

    const SimpleTensor input_view(ptr_with_elem_offset(input->data.dptr, elem_offset,
                                                       input->data.dtype),
                                  {rows, cols}, input->data.dtype);
    SimpleTensor output_rowwise_view;
    if (output->has_data()) {
      output_rowwise_view =
          SimpleTensor(ptr_with_elem_offset(output->data.dptr, elem_offset, output->data.dtype),
                       {rows, cols}, output->data.dtype);
    }
    SimpleTensor output_colwise_view;
    if (return_transpose) {
      output_colwise_view = SimpleTensor(
          ptr_with_elem_offset(output->columnwise_data.dptr, elem_offset, output->columnwise_data.dtype),
          {cols, rows}, output->columnwise_data.dtype);
    }

    SimpleTensor rowwise_scale_view;
    if (output->has_data()) {
      rowwise_scale_view =
          SimpleTensor(ptr_with_elem_offset(output->scale_inv.dptr, rowwise_scale_offset,
                                            output->scale_inv.dtype),
                       {row_scale_rows, row_scale_cols}, output->scale_inv.dtype);
    }
    SimpleTensor colwise_scale_view;
    if (return_transpose) {
      colwise_scale_view = SimpleTensor(
          ptr_with_elem_offset(output->columnwise_scale_inv.dptr, colwise_scale_offset,
                               output->columnwise_scale_inv.dtype),
          {col_scale_rows, col_scale_cols}, output->columnwise_scale_inv.dtype);
    }

    quantize_transpose_square_blockwise(
        input_view, rowwise_scale_view, colwise_scale_view, output_rowwise_view,
        output_colwise_view, epsilon, return_transpose, force_pow_2_scales, noop->data, stream);

    rowwise_scale_offset += row_scale_elems;
    colwise_scale_offset += col_scale_elems;
  }
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCK_SCALING_CUH_
