/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

constexpr size_t kBlockLen = 128;

size_t round_up4(size_t value) { return ((value + 3u) / 4u) * 4u; }

std::vector<size_t> scale_shape(size_t rows, size_t cols, size_t block_scaling_dim,
                                bool columnwise) {
  const size_t row_blocks = (rows + kBlockLen - 1) / kBlockLen;
  const size_t col_blocks = (cols + kBlockLen - 1) / kBlockLen;
  if (block_scaling_dim == 2) {
    if (columnwise) {
      return {col_blocks, round_up4(row_blocks)};
    }
    return {row_blocks, round_up4(col_blocks)};
  }
  if (columnwise) {
    return {row_blocks, round_up4(cols)};
  }
  return {col_blocks, round_up4(rows)};
}

size_t numel(const std::vector<size_t> &shape) {
  size_t ret = 1;
  for (const auto dim : shape) {
    ret *= dim;
  }
  return ret;
}

template <typename T>
void expect_equal_buffer(const std::string &name, const T *got, const T *ref, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    ASSERT_EQ(static_cast<float>(got[i]), static_cast<float>(ref[i]))
        << name << " mismatch at " << i;
  }
}

void expect_equal_scales(const std::string &name, const float *got, const float *ref, size_t rows,
                         size_t cols, size_t block_scaling_dim, bool columnwise) {
  const auto shape = scale_shape(rows, cols, block_scaling_dim, columnwise);
  if (block_scaling_dim == 2) {
    const size_t valid_rows = columnwise ? ((cols + kBlockLen - 1) / kBlockLen)
                                         : ((rows + kBlockLen - 1) / kBlockLen);
    const size_t valid_cols = columnwise ? ((rows + kBlockLen - 1) / kBlockLen)
                                         : ((cols + kBlockLen - 1) / kBlockLen);
    for (size_t i = 0; i < valid_rows; ++i) {
      for (size_t j = 0; j < valid_cols; ++j) {
        ASSERT_EQ(got[i * shape[1] + j], ref[i * shape[1] + j])
            << name << " mismatch at " << i << "," << j;
      }
    }
    return;
  }

  const size_t valid_rows = columnwise ? ((rows + kBlockLen - 1) / kBlockLen)
                                       : ((cols + kBlockLen - 1) / kBlockLen);
  const size_t valid_cols = columnwise ? cols : rows;
  for (size_t i = 0; i < valid_rows; ++i) {
    for (size_t j = 0; j < valid_cols; ++j) {
      ASSERT_EQ(got[i * shape[1] + j], ref[i * shape[1] + j])
          << name << " mismatch at " << i << "," << j;
    }
  }
}

template <typename InputType, typename OutputType>
void run_grouped_fp8_block_test(size_t block_scaling_dim, bool rowwise, bool columnwise,
                                const std::vector<size_t> &splits, size_t cols) {
  const DType itype = TypeInfo<InputType>::dtype;
  const DType otype = TypeInfo<OutputType>::dtype;
  const NVTEScalingMode mode =
      block_scaling_dim == 2 ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D;

  std::vector<Tensor> inputs;
  std::vector<Tensor> refs;
  std::vector<Tensor *> input_ptrs;
  inputs.reserve(splits.size());
  refs.reserve(splits.size());
  input_ptrs.reserve(splits.size());
  for (size_t i = 0; i < splits.size(); ++i) {
    inputs.emplace_back("input", std::vector<size_t>{splits[i], cols}, itype);
    refs.emplace_back("ref", std::vector<size_t>{splits[i], cols}, otype, rowwise, columnwise,
                      mode);
    fillUniform(&inputs.back());
    input_ptrs.push_back(&inputs.back());
  }

  auto grouped_input = build_grouped_tensor(input_ptrs, NVTE_DELAYED_TENSOR_SCALING);
  NVTEGroupedTensor grouped_output =
      nvte_create_grouped_tensor(mode, splits.size(), grouped_input.logical_shape);

  const size_t total_elems = static_cast<size_t>(grouped_input.offsets_host.back());
  const size_t data_bytes = total_elems * sizeof(OutputType);
  OutputType *rowwise_data = nullptr;
  OutputType *columnwise_data = nullptr;
  float *rowwise_scales = nullptr;
  float *columnwise_scales = nullptr;

  size_t total_rowwise_scales = 0;
  size_t total_columnwise_scales = 0;
  std::vector<size_t> rowwise_scale_offsets(splits.size() + 1, 0);
  std::vector<size_t> columnwise_scale_offsets(splits.size() + 1, 0);
  for (size_t i = 0; i < splits.size(); ++i) {
    total_rowwise_scales += numel(scale_shape(splits[i], cols, block_scaling_dim, false));
    total_columnwise_scales += numel(scale_shape(splits[i], cols, block_scaling_dim, true));
    rowwise_scale_offsets[i + 1] = total_rowwise_scales;
    columnwise_scale_offsets[i + 1] = total_columnwise_scales;
  }

  size_t flat_data_shape_data = total_elems;
  NVTEShape flat_data_shape = nvte_make_shape(&flat_data_shape_data, 1);
  if (rowwise) {
    NVTE_CHECK_CUDA(cudaMalloc(&rowwise_data, data_bytes));
    NVTE_CHECK_CUDA(cudaMalloc(&rowwise_scales, total_rowwise_scales * sizeof(float)));
    NVTE_CHECK_CUDA(cudaMemset(rowwise_data, 0, data_bytes));
    NVTE_CHECK_CUDA(cudaMemset(rowwise_scales, 0, total_rowwise_scales * sizeof(float)));
    NVTEBasicTensor data{rowwise_data, static_cast<NVTEDType>(otype), flat_data_shape};
    size_t scale_shape_data = total_rowwise_scales;
    NVTEShape scale_flat_shape = nvte_make_shape(&scale_shape_data, 1);
    NVTEBasicTensor scales{rowwise_scales, kNVTEFloat32, scale_flat_shape};
    nvte_set_grouped_tensor_param(grouped_output, kNVTEGroupedRowwiseData, &data, sizeof(data));
    nvte_set_grouped_tensor_param(grouped_output, kNVTEGroupedRowwiseScaleInv, &scales,
                                  sizeof(scales));
  }
  if (columnwise) {
    NVTE_CHECK_CUDA(cudaMalloc(&columnwise_data, data_bytes));
    NVTE_CHECK_CUDA(cudaMalloc(&columnwise_scales, total_columnwise_scales * sizeof(float)));
    NVTE_CHECK_CUDA(cudaMemset(columnwise_data, 0, data_bytes));
    NVTE_CHECK_CUDA(cudaMemset(columnwise_scales, 0, total_columnwise_scales * sizeof(float)));
    NVTEBasicTensor data{columnwise_data, static_cast<NVTEDType>(otype), flat_data_shape};
    size_t scale_shape_data = total_columnwise_scales;
    NVTEShape scale_flat_shape = nvte_make_shape(&scale_shape_data, 1);
    NVTEBasicTensor scales{columnwise_scales, kNVTEFloat32, scale_flat_shape};
    nvte_set_grouped_tensor_param(grouped_output, kNVTEGroupedColumnwiseData, &data, sizeof(data));
    nvte_set_grouped_tensor_param(grouped_output, kNVTEGroupedColumnwiseScaleInv, &scales,
                                  sizeof(scales));
  }

  if (grouped_input.first_dims_dev) {
    size_t dims_shape_data = splits.size();
    NVTEShape dims_shape = nvte_make_shape(&dims_shape_data, 1);
    NVTEBasicTensor first_dims{grouped_input.first_dims_dev.get(), kNVTEInt64, dims_shape};
    nvte_set_grouped_tensor_param(grouped_output, kNVTEGroupedFirstDims, &first_dims,
                                  sizeof(first_dims));
  }
  if (grouped_input.offsets_dev) {
    size_t offsets_shape_data = splits.size() + 1;
    NVTEShape offsets_shape = nvte_make_shape(&offsets_shape_data, 1);
    NVTEBasicTensor offsets{grouped_input.offsets_dev.get(), kNVTEInt64, offsets_shape};
    nvte_set_grouped_tensor_param(grouped_output, kNVTEGroupedTensorOffsets, &offsets,
                                  sizeof(offsets));
  }

  QuantizationConfigWrapper config;
  config.set_force_pow_2_scales(true);
  config.set_amax_epsilon(0.0f);

  nvte_group_quantize(grouped_input.get_handle(), grouped_output, config, nullptr);
  for (size_t i = 0; i < splits.size(); ++i) {
    nvte_quantize_v2(inputs[i].data(), refs[i].data(), config, nullptr);
  }
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<OutputType> grouped_rowwise(total_elems);
  std::vector<OutputType> grouped_columnwise(total_elems);
  std::vector<float> grouped_rowwise_scales(total_rowwise_scales);
  std::vector<float> grouped_columnwise_scales(total_columnwise_scales);
  if (rowwise) {
    NVTE_CHECK_CUDA(cudaMemcpy(grouped_rowwise.data(), rowwise_data, data_bytes,
                               cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(cudaMemcpy(grouped_rowwise_scales.data(), rowwise_scales,
                               total_rowwise_scales * sizeof(float), cudaMemcpyDeviceToHost));
  }
  if (columnwise) {
    NVTE_CHECK_CUDA(cudaMemcpy(grouped_columnwise.data(), columnwise_data, data_bytes,
                               cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(cudaMemcpy(grouped_columnwise_scales.data(), columnwise_scales,
                               total_columnwise_scales * sizeof(float), cudaMemcpyDeviceToHost));
  }

  for (size_t i = 0; i < splits.size(); ++i) {
    refs[i].to_cpu();
    if (splits[i] == 0) {
      continue;
    }
    const size_t data_offset = static_cast<size_t>(grouped_input.offsets_host[i]);
    const size_t elements = splits[i] * cols;
    if (rowwise) {
      expect_equal_buffer("rowwise_data", grouped_rowwise.data() + data_offset,
                          refs[i].rowwise_cpu_dptr<OutputType>(), elements);
      expect_equal_scales("rowwise_scales",
                          grouped_rowwise_scales.data() + rowwise_scale_offsets[i],
                          refs[i].rowwise_cpu_scale_inv_ptr<float>(), splits[i], cols,
                          block_scaling_dim, false);
    }
    if (columnwise) {
      expect_equal_buffer("columnwise_data", grouped_columnwise.data() + data_offset,
                          refs[i].columnwise_cpu_dptr<OutputType>(), elements);
      expect_equal_scales("columnwise_scales",
                          grouped_columnwise_scales.data() + columnwise_scale_offsets[i],
                          refs[i].columnwise_cpu_scale_inv_ptr<float>(), splits[i], cols,
                          block_scaling_dim, true);
    }
  }

  cudaFree(rowwise_data);
  cudaFree(columnwise_data);
  cudaFree(rowwise_scales);
  cudaFree(columnwise_scales);
  nvte_destroy_grouped_tensor(grouped_output);
}

class GroupedFP8BlockScalingTest
    : public ::testing::TestWithParam<std::tuple<size_t, bool, bool, std::vector<size_t>, size_t>> {
};

TEST_P(GroupedFP8BlockScalingTest, MatchesNonGroupedLoop) {
  if (getDeviceComputeCapability() < hopperComputeCapability) {
    GTEST_SKIP();
  }
  const auto block_scaling_dim = std::get<0>(GetParam());
  const auto rowwise = std::get<1>(GetParam());
  const auto columnwise = std::get<2>(GetParam());
  if (!rowwise && !columnwise) {
    GTEST_SKIP();
  }
  const auto splits = std::get<3>(GetParam());
  const auto cols = std::get<4>(GetParam());
  run_grouped_fp8_block_test<bf16, fp8e4m3>(block_scaling_dim, rowwise, columnwise, splits, cols);
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, GroupedFP8BlockScalingTest,
    ::testing::Combine(::testing::Values(1u, 2u), ::testing::Values(true, false),
                       ::testing::Values(true, false),
                       ::testing::Values(std::vector<size_t>{128, 256, 384, 512},
                                         std::vector<size_t>{129, 0, 255, 303}),
                       ::testing::Values(300u)));

}  // namespace
