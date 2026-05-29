/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include <transformer_engine/cast.h>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

constexpr size_t kBlockLen = 128;

size_t ceildiv(const size_t x, const size_t y) { return (x + y - 1) / y; }

size_t roundup_to_multiple(const size_t x, const size_t y) { return ceildiv(x, y) * y; }

size_t rowwise_scale_stride(const size_t cols) {
  return roundup_to_multiple(ceildiv(cols, kBlockLen), 4);
}

size_t columnwise_scale_stride(const size_t rows) {
  return roundup_to_multiple(ceildiv(rows, kBlockLen), 4);
}

struct GroupedCase {
  std::vector<int64_t> first_dims;
  size_t cols;
  bool rowwise;
  bool columnwise;
  bool set_first_dims;
};

template <typename T>
void fill_random(std::vector<T> *data) {
  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (auto &x : *data) {
    x = static_cast<T>(dist(gen));
  }
}

template <typename OutputType>
void expect_equal_bytes(const std::vector<OutputType> &actual,
                        const std::vector<OutputType> &expected, const char *name) {
  ASSERT_EQ(actual.size(), expected.size());
  const auto *actual_bytes = reinterpret_cast<const uint8_t *>(actual.data());
  const auto *expected_bytes = reinterpret_cast<const uint8_t *>(expected.data());
  for (size_t i = 0; i < actual.size() * sizeof(OutputType); ++i) {
    ASSERT_EQ(actual_bytes[i], expected_bytes[i]) << name << " byte mismatch at " << i;
  }
}

void expect_equal_float(const std::vector<float> &actual, const std::vector<float> &expected,
                        const char *name) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_EQ(actual[i], expected[i]) << name << " mismatch at " << i;
  }
}

template <typename InputType, typename OutputType>
void run_grouped_fp8_block_scaling_case(const GroupedCase &tc) {
  const size_t num_tensors = tc.first_dims.size();
  const size_t cols = tc.cols;
  size_t total_rows = 0;
  std::vector<int64_t> data_offsets(num_tensors + 1, 0);
  std::vector<int64_t> rowwise_scale_offsets(num_tensors + 1, 0);
  std::vector<int64_t> columnwise_scale_offsets(num_tensors + 1, 0);
  const size_t col_tiles = ceildiv(cols, kBlockLen);
  const size_t row_stride = rowwise_scale_stride(cols);

  for (size_t i = 0; i < num_tensors; ++i) {
    const size_t rows = static_cast<size_t>(tc.first_dims[i]);
    total_rows += rows;
    data_offsets[i + 1] = data_offsets[i] + rows * cols;
    rowwise_scale_offsets[i + 1] =
        rowwise_scale_offsets[i] + ceildiv(rows, kBlockLen) * row_stride;
    columnwise_scale_offsets[i + 1] =
        columnwise_scale_offsets[i] + col_tiles * columnwise_scale_stride(rows);
  }

  const size_t total_elements = total_rows * cols;
  const size_t rowwise_scale_elements = rowwise_scale_offsets.back();
  const size_t columnwise_scale_elements = columnwise_scale_offsets.back();

  std::vector<InputType> input_h(total_elements);
  fill_random(&input_h);

  auto input_d = cuda_alloc<InputType>(total_elements * sizeof(InputType));
  auto output_rowwise_d =
      cuda_alloc<OutputType>(tc.rowwise ? total_elements * sizeof(OutputType) : 1);
  auto output_colwise_d =
      cuda_alloc<OutputType>(tc.columnwise ? total_elements * sizeof(OutputType) : 1);
  auto scale_rowwise_d = cuda_alloc<float>(tc.rowwise ? rowwise_scale_elements * sizeof(float) : 1);
  auto scale_colwise_d =
      cuda_alloc<float>(tc.columnwise ? columnwise_scale_elements * sizeof(float) : 1);
  auto ref_output_rowwise_d =
      cuda_alloc<OutputType>(tc.rowwise ? total_elements * sizeof(OutputType) : 1);
  auto ref_output_colwise_d =
      cuda_alloc<OutputType>(tc.columnwise ? total_elements * sizeof(OutputType) : 1);
  auto ref_scale_rowwise_d =
      cuda_alloc<float>(tc.rowwise ? rowwise_scale_elements * sizeof(float) : 1);
  auto ref_scale_colwise_d =
      cuda_alloc<float>(tc.columnwise ? columnwise_scale_elements * sizeof(float) : 1);
  auto first_dims_d = cuda_alloc<int64_t>(num_tensors * sizeof(int64_t));
  auto offsets_d = cuda_alloc<int64_t>((num_tensors + 1) * sizeof(int64_t));

  NVTE_CHECK_CUDA(cudaMemcpy(input_d.get(), input_h.data(), total_elements * sizeof(InputType),
                             cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(first_dims_d.get(), tc.first_dims.data(),
                             num_tensors * sizeof(int64_t), cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(offsets_d.get(), data_offsets.data(),
                             (num_tensors + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  if (tc.rowwise) {
    NVTE_CHECK_CUDA(cudaMemset(output_rowwise_d.get(), 0, total_elements * sizeof(OutputType)));
    NVTE_CHECK_CUDA(cudaMemset(ref_output_rowwise_d.get(), 0, total_elements * sizeof(OutputType)));
    NVTE_CHECK_CUDA(cudaMemset(scale_rowwise_d.get(), 0, rowwise_scale_elements * sizeof(float)));
    NVTE_CHECK_CUDA(
        cudaMemset(ref_scale_rowwise_d.get(), 0, rowwise_scale_elements * sizeof(float)));
  }
  if (tc.columnwise) {
    NVTE_CHECK_CUDA(cudaMemset(output_colwise_d.get(), 0, total_elements * sizeof(OutputType)));
    NVTE_CHECK_CUDA(
        cudaMemset(ref_output_colwise_d.get(), 0, total_elements * sizeof(OutputType)));
    NVTE_CHECK_CUDA(
        cudaMemset(scale_colwise_d.get(), 0, columnwise_scale_elements * sizeof(float)));
    NVTE_CHECK_CUDA(
        cudaMemset(ref_scale_colwise_d.get(), 0, columnwise_scale_elements * sizeof(float)));
  }

  const std::vector<size_t> logical_shape = {total_rows, cols};
  GroupedTensorWrapper input_group(num_tensors, logical_shape, NVTE_DELAYED_TENSOR_SCALING);
  input_group.set_rowwise_data(input_d.get(), TypeInfo<InputType>::dtype,
                               std::vector<size_t>{total_elements});

  GroupedTensorWrapper output_group(num_tensors, logical_shape, NVTE_BLOCK_SCALING_2D);
  if (tc.rowwise) {
    output_group.set_rowwise_data(output_rowwise_d.get(), TypeInfo<OutputType>::dtype,
                                  std::vector<size_t>{total_elements});
    output_group.set_rowwise_scale_inv(scale_rowwise_d.get(), DType::kFloat32,
                                       std::vector<size_t>{rowwise_scale_elements});
  }
  if (tc.columnwise) {
    output_group.set_columnwise_data(output_colwise_d.get(), TypeInfo<OutputType>::dtype,
                                     std::vector<size_t>{total_elements});
    output_group.set_columnwise_scale_inv(scale_colwise_d.get(), DType::kFloat32,
                                          std::vector<size_t>{columnwise_scale_elements});
  }
  if (tc.set_first_dims) {
    input_group.set_first_dims(first_dims_d.get(), DType::kInt64, std::vector<size_t>{num_tensors});
    input_group.set_tensor_offsets(offsets_d.get(), DType::kInt64,
                                   std::vector<size_t>{num_tensors + 1});
    output_group.set_first_dims(first_dims_d.get(), DType::kInt64,
                                std::vector<size_t>{num_tensors});
    output_group.set_tensor_offsets(offsets_d.get(), DType::kInt64,
                                    std::vector<size_t>{num_tensors + 1});
  }

  QuantizationConfigWrapper quant_config;
  quant_config.set_force_pow_2_scales(true);
  quant_config.set_amax_epsilon(0.0f);
  nvte_group_quantize(input_group.data(), output_group.data(), quant_config, 0);

  for (size_t i = 0; i < num_tensors; ++i) {
    const size_t rows = static_cast<size_t>(tc.first_dims[i]);
    if (rows == 0) {
      continue;
    }
    const std::vector<size_t> tensor_shape = {rows, cols};
    TensorWrapper ref_input(NVTE_DELAYED_TENSOR_SCALING);
    ref_input.set_rowwise_data(input_d.get() + data_offsets[i], TypeInfo<InputType>::dtype,
                               tensor_shape);

    TensorWrapper ref_output(NVTE_BLOCK_SCALING_2D);
    if (tc.rowwise) {
      ref_output.set_rowwise_data(ref_output_rowwise_d.get() + data_offsets[i],
                                  TypeInfo<OutputType>::dtype, tensor_shape);
      ref_output.set_rowwise_scale_inv(ref_scale_rowwise_d.get() + rowwise_scale_offsets[i],
                                       DType::kFloat32,
                                       std::vector<size_t>{ceildiv(rows, kBlockLen), row_stride});
    }
    if (tc.columnwise) {
      ref_output.set_columnwise_data(ref_output_colwise_d.get() + data_offsets[i],
                                     TypeInfo<OutputType>::dtype,
                                     std::vector<size_t>{cols, rows});
      ref_output.set_columnwise_scale_inv(
          ref_scale_colwise_d.get() + columnwise_scale_offsets[i], DType::kFloat32,
          std::vector<size_t>{col_tiles, columnwise_scale_stride(rows)});
    }
    nvte_quantize_v2(ref_input.data(), ref_output.data(), quant_config, 0);
  }

  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  if (tc.rowwise) {
    std::vector<OutputType> actual(total_elements), expected(total_elements);
    std::vector<float> actual_scale(rowwise_scale_elements), expected_scale(rowwise_scale_elements);
    NVTE_CHECK_CUDA(cudaMemcpy(actual.data(), output_rowwise_d.get(),
                               total_elements * sizeof(OutputType), cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(cudaMemcpy(expected.data(), ref_output_rowwise_d.get(),
                               total_elements * sizeof(OutputType), cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(cudaMemcpy(actual_scale.data(), scale_rowwise_d.get(),
                               rowwise_scale_elements * sizeof(float), cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(cudaMemcpy(expected_scale.data(), ref_scale_rowwise_d.get(),
                               rowwise_scale_elements * sizeof(float), cudaMemcpyDeviceToHost));
    expect_equal_bytes(actual, expected, "rowwise_data");
    expect_equal_float(actual_scale, expected_scale, "rowwise_scale_inv");
  }
  if (tc.columnwise) {
    std::vector<OutputType> actual(total_elements), expected(total_elements);
    std::vector<float> actual_scale(columnwise_scale_elements),
        expected_scale(columnwise_scale_elements);
    NVTE_CHECK_CUDA(cudaMemcpy(actual.data(), output_colwise_d.get(),
                               total_elements * sizeof(OutputType), cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(cudaMemcpy(expected.data(), ref_output_colwise_d.get(),
                               total_elements * sizeof(OutputType), cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(cudaMemcpy(actual_scale.data(), scale_colwise_d.get(),
                               columnwise_scale_elements * sizeof(float), cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(cudaMemcpy(expected_scale.data(), ref_scale_colwise_d.get(),
                               columnwise_scale_elements * sizeof(float), cudaMemcpyDeviceToHost));
    expect_equal_bytes(actual, expected, "columnwise_data");
    expect_equal_float(actual_scale, expected_scale, "columnwise_scale_inv");
  }
}

}  // namespace

TEST(GroupedFP8BlockScalingQuantize, UniformShapeRowwiseAndColumnwise) {
  run_grouped_fp8_block_scaling_case<bf16, fp8e4m3>(
      GroupedCase{{128, 128, 128, 128}, 256, true, true, false});
}

TEST(GroupedFP8BlockScalingQuantize, VaryingFirstDimWithEdgesAndZeroGroup) {
  run_grouped_fp8_block_scaling_case<fp32, fp8e4m3>(
      GroupedCase{{129, 0, 255, 303}, 300, true, true, true});
}

TEST(GroupedFP8BlockScalingQuantize, RowwiseOnlyE5M2) {
  run_grouped_fp8_block_scaling_case<fp16, fp8e5m2>(
      GroupedCase{{17, 128, 129, 1, 256}, 129, true, false, true});
}

TEST(GroupedFP8BlockScalingQuantize, ColumnwiseOnly) {
  run_grouped_fp8_block_scaling_case<bf16, fp8e4m3>(
      GroupedCase{{128, 257, 64}, 272, false, true, true});
}
