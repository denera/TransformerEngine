/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <transformer_engine/cast.h>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

constexpr size_t kBlockLen = 128;

enum class ScalingDirection {
  ROWWISE,
  COLUMNWISE,
  BOTH,
};

struct GroupedCase {
  std::vector<size_t> first_dims;
  size_t k_dim;
};

struct GroupedFp8BlockOutput {
  GroupedTensorHandle handle;
  CudaPtr<> rowwise_data;
  CudaPtr<> columnwise_data;
  CudaPtr<> rowwise_scale_inv;
  CudaPtr<> columnwise_scale_inv;
  std::vector<int64_t> tensor_offsets;
  std::vector<size_t> rowwise_scale_offsets;
  std::vector<size_t> columnwise_scale_offsets;
  size_t total_elements{0};
  size_t rowwise_scale_elements{0};
  size_t columnwise_scale_elements{0};
  size_t element_size{0};
};

size_t ceildiv_size(const size_t numerator, const size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

size_t roundup_size(const size_t value, const size_t multiple) {
  return ceildiv_size(value, multiple) * multiple;
}

bool all_same_value(const std::vector<size_t> &values) {
  return std::all_of(values.begin(), values.end(),
                     [&](const size_t value) { return value == values[0]; });
}

std::vector<size_t> scale_offsets_rowwise(const std::vector<size_t> &first_dims,
                                          const size_t k_dim) {
  const size_t scale_stride = roundup_size(ceildiv_size(k_dim, kBlockLen), 4);
  std::vector<size_t> offsets;
  offsets.reserve(first_dims.size() + 1);
  offsets.push_back(0);
  for (const size_t m_dim : first_dims) {
    offsets.push_back(offsets.back() + ceildiv_size(m_dim, kBlockLen) * scale_stride);
  }
  return offsets;
}

std::vector<size_t> scale_offsets_columnwise(const std::vector<size_t> &first_dims,
                                             const size_t k_dim) {
  const size_t scale_rows = ceildiv_size(k_dim, kBlockLen);
  std::vector<size_t> offsets;
  offsets.reserve(first_dims.size() + 1);
  offsets.push_back(0);
  for (const size_t m_dim : first_dims) {
    offsets.push_back(offsets.back() +
                      scale_rows * roundup_size(ceildiv_size(m_dim, kBlockLen), 4));
  }
  return offsets;
}

void compare_device_bytes(const std::string &name, const void *device_ptr, const void *ref_cpu_ptr,
                          const size_t num_bytes) {
  if (num_bytes == 0) {
    return;
  }
  std::vector<uint8_t> actual(num_bytes);
  NVTE_CHECK_CUDA(cudaMemcpy(actual.data(), device_ptr, num_bytes, cudaMemcpyDeviceToHost));
  const auto *expected = static_cast<const uint8_t *>(ref_cpu_ptr);
  for (size_t i = 0; i < num_bytes; ++i) {
    ASSERT_EQ(actual[i], expected[i]) << name << " byte mismatch at index " << i;
  }
}

void expect_device_bytes_filled(const std::string &name, const void *device_ptr,
                                const size_t num_bytes, const uint8_t expected) {
  if (num_bytes == 0) {
    return;
  }
  std::vector<uint8_t> actual(num_bytes);
  NVTE_CHECK_CUDA(cudaMemcpy(actual.data(), device_ptr, num_bytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < num_bytes; ++i) {
    ASSERT_EQ(actual[i], expected) << name << " byte changed at index " << i;
  }
}

GroupedFp8BlockOutput make_grouped_fp8_block_output(const GroupedBuffers &grouped_input,
                                                    const std::vector<size_t> &first_dims,
                                                    const size_t k_dim, const DType output_dtype,
                                                    const bool rowwise, const bool columnwise,
                                                    const uint8_t fill_byte = 0) {
  GroupedFp8BlockOutput output;
  const size_t num_tensors = first_dims.size();
  output.element_size = typeToNumBits(output_dtype) / 8;
  output.tensor_offsets = grouped_input.offsets_host;
  output.total_elements = static_cast<size_t>(output.tensor_offsets[num_tensors - 1]) +
                          first_dims.back() * k_dim;
  output.rowwise_scale_offsets = scale_offsets_rowwise(first_dims, k_dim);
  output.columnwise_scale_offsets = scale_offsets_columnwise(first_dims, k_dim);
  output.rowwise_scale_elements = output.rowwise_scale_offsets.back();
  output.columnwise_scale_elements = output.columnwise_scale_offsets.back();

  output.handle.reset(
      nvte_create_grouped_tensor(NVTE_BLOCK_SCALING_2D, num_tensors, grouped_input.logical_shape));
  NVTEGroupedTensor handle = output.handle.get();

  const size_t data_bytes = output.total_elements * output.element_size;
  const NVTEShape flat_data_shape = nvte_make_shape(&output.total_elements, 1);
  if (rowwise) {
    output.rowwise_data = cuda_alloc(data_bytes);
    NVTE_CHECK_CUDA(cudaMemset(output.rowwise_data.get(), fill_byte, data_bytes));
    NVTEBasicTensor data_tensor{output.rowwise_data.get(), static_cast<NVTEDType>(output_dtype),
                                flat_data_shape};
    nvte_set_grouped_tensor_param(handle, kNVTEGroupedRowwiseData, &data_tensor,
                                  sizeof(data_tensor));
  }
  if (columnwise) {
    output.columnwise_data = cuda_alloc(data_bytes);
    NVTE_CHECK_CUDA(cudaMemset(output.columnwise_data.get(), fill_byte, data_bytes));
    NVTEBasicTensor data_tensor{output.columnwise_data.get(),
                                static_cast<NVTEDType>(output_dtype), flat_data_shape};
    nvte_set_grouped_tensor_param(handle, kNVTEGroupedColumnwiseData, &data_tensor,
                                  sizeof(data_tensor));
  }

  if (rowwise) {
    const size_t scale_bytes = output.rowwise_scale_elements * sizeof(float);
    output.rowwise_scale_inv = cuda_alloc(scale_bytes);
    NVTE_CHECK_CUDA(cudaMemset(output.rowwise_scale_inv.get(), fill_byte, scale_bytes));
    const NVTEShape scale_shape = nvte_make_shape(&output.rowwise_scale_elements, 1);
    NVTEBasicTensor scale_tensor{output.rowwise_scale_inv.get(), kNVTEFloat32, scale_shape};
    nvte_set_grouped_tensor_param(handle, kNVTEGroupedRowwiseScaleInv, &scale_tensor,
                                  sizeof(scale_tensor));
  }
  if (columnwise) {
    const size_t scale_bytes = output.columnwise_scale_elements * sizeof(float);
    output.columnwise_scale_inv = cuda_alloc(scale_bytes);
    NVTE_CHECK_CUDA(cudaMemset(output.columnwise_scale_inv.get(), fill_byte, scale_bytes));
    const NVTEShape scale_shape = nvte_make_shape(&output.columnwise_scale_elements, 1);
    NVTEBasicTensor scale_tensor{output.columnwise_scale_inv.get(), kNVTEFloat32, scale_shape};
    nvte_set_grouped_tensor_param(handle, kNVTEGroupedColumnwiseScaleInv, &scale_tensor,
                                  sizeof(scale_tensor));
  }

  if (!all_same_value(first_dims)) {
    const size_t first_dims_len = num_tensors;
    const NVTEShape first_dims_shape = nvte_make_shape(&first_dims_len, 1);
    NVTEBasicTensor first_dims_tensor{grouped_input.first_dims_dev.get(), kNVTEInt64,
                                      first_dims_shape};
    nvte_set_grouped_tensor_param(handle, kNVTEGroupedFirstDims, &first_dims_tensor,
                                  sizeof(first_dims_tensor));

    const size_t offsets_len = num_tensors + 1;
    const NVTEShape offsets_shape = nvte_make_shape(&offsets_len, 1);
    NVTEBasicTensor offsets_tensor{grouped_input.offsets_dev.get(), kNVTEInt64, offsets_shape};
    nvte_set_grouped_tensor_param(handle, kNVTEGroupedTensorOffsets, &offsets_tensor,
                                  sizeof(offsets_tensor));
  }

  return output;
}

template <typename InputType, typename OutputType>
void compare_grouped_to_looped_quantize(const GroupedCase &grouped_case,
                                        const DType input_dtype, const DType output_dtype,
                                        const ScalingDirection direction) {
  const bool rowwise =
      direction == ScalingDirection::ROWWISE || direction == ScalingDirection::BOTH;
  const bool columnwise =
      direction == ScalingDirection::COLUMNWISE || direction == ScalingDirection::BOTH;
  const size_t num_tensors = grouped_case.first_dims.size();
  const size_t k_dim = grouped_case.k_dim;

  std::vector<std::unique_ptr<Tensor>> inputs;
  std::vector<Tensor *> input_ptrs;
  inputs.reserve(num_tensors);
  input_ptrs.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    inputs.emplace_back(std::make_unique<Tensor>("input_" + std::to_string(i),
                                                 std::vector<size_t>{grouped_case.first_dims[i],
                                                                     k_dim},
                                                 input_dtype));
    fillCase<fp32>(inputs.back().get(), InputsFillCase::uniform);
    input_ptrs.push_back(inputs.back().get());
  }

  GroupedBuffers grouped_input = build_grouped_tensor(input_ptrs, NVTE_DELAYED_TENSOR_SCALING);
  GroupedFp8BlockOutput grouped_output =
      make_grouped_fp8_block_output(grouped_input, grouped_case.first_dims, k_dim, output_dtype,
                                    rowwise, columnwise);

  QuantizationConfigWrapper quant_config;
  quant_config.set_force_pow_2_scales(true);
  quant_config.set_amax_epsilon(0.0f);
  nvte_group_quantize(grouped_input.get_handle(), grouped_output.handle.get(), quant_config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  for (size_t i = 0; i < num_tensors; ++i) {
    const size_t m_dim = grouped_case.first_dims[i];
    if (m_dim == 0) {
      continue;
    }

    Tensor reference("reference_" + std::to_string(i), std::vector<size_t>{m_dim, k_dim},
                     output_dtype, rowwise, columnwise, NVTE_BLOCK_SCALING_2D);
    nvte_quantize_v2(inputs[i]->data(), reference.data(), quant_config, 0);
    NVTE_CHECK_CUDA(cudaDeviceSynchronize());
    reference.to_cpu();

    const size_t data_offset = static_cast<size_t>(grouped_output.tensor_offsets[i]);
    const size_t data_bytes = m_dim * k_dim * grouped_output.element_size;
    if (rowwise) {
      const auto *candidate_ptr =
          static_cast<const uint8_t *>(grouped_output.rowwise_data.get()) +
          data_offset * grouped_output.element_size;
      compare_device_bytes("rowwise_data", candidate_ptr,
                           reference.rowwise_cpu_dptr<OutputType>(), data_bytes);

      const size_t scale_offset = grouped_output.rowwise_scale_offsets[i];
      const size_t scale_bytes = bytes(reference.rowwise_scale_inv_shape(), DType::kFloat32);
      const auto *scale_ptr =
          static_cast<const uint8_t *>(grouped_output.rowwise_scale_inv.get()) +
          scale_offset * sizeof(float);
      compare_device_bytes("rowwise_scale_inv", scale_ptr,
                           reference.rowwise_cpu_scale_inv_ptr<float>(), scale_bytes);
    }
    if (columnwise) {
      const auto *candidate_ptr =
          static_cast<const uint8_t *>(grouped_output.columnwise_data.get()) +
          data_offset * grouped_output.element_size;
      compare_device_bytes("columnwise_data", candidate_ptr,
                           reference.columnwise_cpu_dptr<OutputType>(), data_bytes);

      const size_t scale_offset = grouped_output.columnwise_scale_offsets[i];
      const size_t scale_bytes = bytes(reference.columnwise_scale_inv_shape(), DType::kFloat32);
      const auto *scale_ptr =
          static_cast<const uint8_t *>(grouped_output.columnwise_scale_inv.get()) +
          scale_offset * sizeof(float);
      compare_device_bytes("columnwise_scale_inv", scale_ptr,
                           reference.columnwise_cpu_scale_inv_ptr<float>(), scale_bytes);
    }
  }
}

template <typename InputType, typename OutputType>
void run_noop_test() {
  const GroupedCase grouped_case{{129, 0, 255, 303}, 300};
  const size_t num_tensors = grouped_case.first_dims.size();
  std::vector<std::unique_ptr<Tensor>> inputs;
  std::vector<Tensor *> input_ptrs;
  for (size_t i = 0; i < num_tensors; ++i) {
    inputs.emplace_back(std::make_unique<Tensor>("noop_input_" + std::to_string(i),
                                                 std::vector<size_t>{grouped_case.first_dims[i],
                                                                     grouped_case.k_dim},
                                                 TypeInfo<InputType>::dtype));
    fillCase<fp32>(inputs.back().get(), InputsFillCase::uniform);
    input_ptrs.push_back(inputs.back().get());
  }

  GroupedBuffers grouped_input = build_grouped_tensor(input_ptrs, NVTE_DELAYED_TENSOR_SCALING);
  constexpr uint8_t kSentinel = 0xA5;
  GroupedFp8BlockOutput grouped_output =
      make_grouped_fp8_block_output(grouped_input, grouped_case.first_dims, grouped_case.k_dim,
                                    TypeInfo<OutputType>::dtype, true, true, kSentinel);

  Tensor noop("noop", std::vector<size_t>{1}, DType::kFloat32);
  noop.rowwise_cpu_dptr<float>()[0] = 1.0f;
  noop.from_cpu();

  QuantizationConfigWrapper quant_config;
  quant_config.set_force_pow_2_scales(true);
  quant_config.set_amax_epsilon(0.0f);
  quant_config.set_noop_tensor(noop.data());
  nvte_group_quantize(grouped_input.get_handle(), grouped_output.handle.get(), quant_config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  const size_t data_bytes = grouped_output.total_elements * grouped_output.element_size;
  expect_device_bytes_filled("rowwise_data_noop", grouped_output.rowwise_data.get(), data_bytes,
                             kSentinel);
  expect_device_bytes_filled("columnwise_data_noop", grouped_output.columnwise_data.get(),
                             data_bytes, kSentinel);
  expect_device_bytes_filled("rowwise_scale_inv_noop", grouped_output.rowwise_scale_inv.get(),
                             grouped_output.rowwise_scale_elements * sizeof(float), kSentinel);
  expect_device_bytes_filled("columnwise_scale_inv_noop",
                             grouped_output.columnwise_scale_inv.get(),
                             grouped_output.columnwise_scale_elements * sizeof(float), kSentinel);
}

std::string direction_name(const ScalingDirection direction) {
  switch (direction) {
    case ScalingDirection::ROWWISE:
      return "rowwise";
    case ScalingDirection::COLUMNWISE:
      return "columnwise";
    case ScalingDirection::BOTH:
      return "both";
  }
  return "unknown";
}

std::string case_name(const GroupedCase &grouped_case) {
  std::ostringstream name;
  name << "G" << grouped_case.first_dims.size() << "K" << grouped_case.k_dim << "M";
  for (const size_t m_dim : grouped_case.first_dims) {
    name << "_" << m_dim;
  }
  return name.str();
}

std::vector<GroupedCase> grouped_cases = {
    {{129}, 129},
    {{255, 303}, 272},
    {{129, 0, 255, 303}, 300},
    {{64, 128, 0, 129, 255, 303, 1, 512}, 129},
};

std::vector<ScalingDirection> scaling_directions = {
    ScalingDirection::ROWWISE,
    ScalingDirection::COLUMNWISE,
    ScalingDirection::BOTH,
};

}  // namespace

class GroupedFp8BlockScalingQuantizeTestSuite
    : public ::testing::TestWithParam<
          std::tuple<GroupedCase, transformer_engine::DType, transformer_engine::DType,
                     ScalingDirection>> {};

TEST_P(GroupedFp8BlockScalingQuantizeTestSuite, MatchesLoopedNvteQuantizeV2) {
  if (getDeviceComputeCapability() < hopperComputeCapability) {
    GTEST_SKIP();
  }

  const GroupedCase grouped_case = std::get<0>(GetParam());
  const DType input_dtype = std::get<1>(GetParam());
  const DType output_dtype = std::get<2>(GetParam());
  const ScalingDirection direction = std::get<3>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
      input_dtype, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(
          output_dtype, OutputType,
          compare_grouped_to_looped_quantize<InputType, OutputType>(
              grouped_case, input_dtype, output_dtype, direction);););
}

TEST(GroupedFp8BlockScalingQuantizeNoopTest, NoopLeavesOutputsUnchanged) {
  if (getDeviceComputeCapability() < hopperComputeCapability) {
    GTEST_SKIP();
  }

  run_noop_test<bf16, fp8e4m3>();
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, GroupedFp8BlockScalingQuantizeTestSuite,
    ::testing::Combine(::testing::ValuesIn(grouped_cases),
                       ::testing::Values(DType::kFloat32, DType::kFloat16, DType::kBFloat16),
                       ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
                       ::testing::ValuesIn(scaling_directions)),
    [](const testing::TestParamInfo<GroupedFp8BlockScalingQuantizeTestSuite::ParamType> &info) {
      return case_name(std::get<0>(info.param)) + "_" + test::typeName(std::get<1>(info.param)) +
             "_" + test::typeName(std::get<2>(info.param)) + "_" +
             direction_name(std::get<3>(info.param));
    });
