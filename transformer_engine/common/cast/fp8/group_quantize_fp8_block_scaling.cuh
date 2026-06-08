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

#include <cstdint>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/cuda_runtime.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_block_scaling {

constexpr size_t kBlockLen = 128;
constexpr size_t kThreadsPerBlock = 256;
constexpr size_t kWarpSize = 32;
constexpr size_t kThreadsPerScale = 8;
constexpr size_t kScalesPerIteration = kThreadsPerBlock / kThreadsPerScale;
constexpr size_t kElemsPerScaleThread = kBlockLen / kThreadsPerScale;
constexpr size_t kLoadVec = 8;
constexpr size_t kLoadThreadsPerRow = kBlockLen / kLoadVec;
constexpr size_t kLoadRowStride = kThreadsPerBlock / kLoadThreadsPerRow;
constexpr size_t kSharedVec = 2;
constexpr size_t kSharedCols = kBlockLen / kSharedVec;
constexpr size_t kAligned1DThreadsPerScale = 16;
constexpr size_t kAligned1DScalesPerIteration = kThreadsPerBlock / kAligned1DThreadsPerScale;
constexpr size_t kAligned1DOutputVec = kBlockLen / kAligned1DThreadsPerScale;
constexpr size_t kRegTileRows = 4;
constexpr size_t kRegTileCols = 16;
constexpr size_t kRegThreadsXInWarp = 2;
constexpr size_t kRegThreadsYInWarp = kWarpSize / kRegThreadsXInWarp;
constexpr size_t kRegWarpsX = 4;
constexpr size_t kRegWarpsY = (kThreadsPerBlock / kWarpSize) / kRegWarpsX;
constexpr size_t kAligned2DRegTileRows = 2;
constexpr size_t kAligned2DRowBandRows =
    kRegWarpsY * kRegThreadsYInWarp * kAligned2DRegTileRows;
constexpr size_t kAligned2DRowBands = kBlockLen / kAligned2DRowBandRows;
static_assert(kBlockLen % kLoadVec == 0, "kLoadVec must divide kBlockLen.");
static_assert(kThreadsPerBlock % kLoadThreadsPerRow == 0,
              "kLoadThreadsPerRow must divide kThreadsPerBlock.");
static_assert(kBlockLen % kLoadRowStride == 0, "kLoadRowStride must divide kBlockLen.");
static_assert(kLoadVec % kSharedVec == 0, "kSharedVec must divide kLoadVec.");
static_assert(kAligned1DOutputVec % kSharedVec == 0,
              "kSharedVec must divide the aligned 1D output vector.");
static_assert(kBlockLen % kAligned1DThreadsPerScale == 0,
              "Aligned 1D scale groups must tile the block.");
static_assert(kThreadsPerBlock % kAligned1DThreadsPerScale == 0,
              "Aligned 1D scale groups must tile the thread block.");
static_assert(kBlockLen % (kAligned1DScalesPerIteration * kSharedVec) == 0,
              "Aligned 1D columnwise iterations must cover the full tile.");
static_assert(kBlockLen % kAligned2DRowBandRows == 0,
              "Aligned 2D row bands must tile the block rows.");

constexpr __device__ __host__ __forceinline__ size_t block_len() { return 128; }

constexpr __device__ __host__ __forceinline__ size_t divup_by_block_len(const size_t value) {
  return (value + block_len() - 1) / block_len();
}

__device__ __forceinline__ size_t round_up_to_multiple(const size_t value,
                                                       const size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

template <bool kIs2DScaling>
__device__ __forceinline__ size_t rowwise_scale_elements(const size_t rows, const size_t cols) {
  const size_t row_blocks = divup_by_block_len(rows);
  const size_t col_blocks = divup_by_block_len(cols);
  if constexpr (kIs2DScaling) {
    return row_blocks * round_up_to_multiple(col_blocks, 4);
  } else {
    return col_blocks * round_up_to_multiple(rows, 4);
  }
}

template <bool kIs2DScaling>
__device__ __forceinline__ size_t columnwise_scale_elements(const size_t rows, const size_t cols) {
  const size_t row_blocks = divup_by_block_len(rows);
  const size_t col_blocks = divup_by_block_len(cols);
  if constexpr (kIs2DScaling) {
    return col_blocks * round_up_to_multiple(row_blocks, 4);
  } else {
    return row_blocks * round_up_to_multiple(cols, 4);
  }
}

struct TileDescriptor {
  size_t tensor_id = 0;
  size_t tile_y = 0;
  size_t rows = 0;
  size_t tensor_base = 0;
  size_t rowwise_scale_offset = 0;
  size_t columnwise_scale_offset = 0;
  bool valid = false;
};

template <size_t kGroupSize>
__device__ __forceinline__ float warp_group_reduce_max(float value) {
  static_assert(kGroupSize > 0 && kGroupSize < 32, "Warp group size must be in [1, 31].");
  const int lane = threadIdx.x % kWarpSize;
  const int group_lane_base = (lane / kGroupSize) * kGroupSize;
  const unsigned mask = ((1u << kGroupSize) - 1u) << group_lane_base;
#pragma unroll
  for (int delta = kGroupSize / 2; delta > 0; delta >>= 1) {
    const float other = __shfl_down_sync(mask, value, delta);
    value = fmaxf(value, other);
  }
  return __shfl_sync(mask, value, group_lane_base);
}

template <size_t kGroupSize>
__device__ __forceinline__ float warp_group_broadcast(const float value) {
  static_assert(kGroupSize > 0 && kGroupSize < 32, "Warp group size must be in [1, 31].");
  const int lane = threadIdx.x % kWarpSize;
  const int group_lane_base = (lane / kGroupSize) * kGroupSize;
  const unsigned mask = ((1u << kGroupSize) - 1u) << group_lane_base;
  return __shfl_sync(mask, value, group_lane_base);
}

__device__ __forceinline__ float reciprocal_scale(const float scale,
                                                  const bool force_pow_2_scales) {
  if (force_pow_2_scales) {
    constexpr uint32_t kSignMantissaMask = 0x807FFFFFu;
    constexpr uint32_t kExponentMask = 0x7F800000u;
    constexpr uint32_t kMaxReciprocalNormalExponent = 0x7E800000u;
    constexpr uint32_t kReciprocalExponentSum = 0x7F000000u;
    const uint32_t scale_bits = __float_as_uint(scale);
    const uint32_t exponent_bits = scale_bits & kExponentMask;
    if ((scale_bits & kSignMantissaMask) == 0 && exponent_bits != 0 &&
        exponent_bits <= kMaxReciprocalNormalExponent) {
      return __uint_as_float(kReciprocalExponentSum - exponent_bits);
    }
  }
  return 1.0f / scale;
}

template <typename IType, typename OType>
__device__ __forceinline__ float compute_pow2_scale_from_types(float amax, const float epsilon) {
  if constexpr (std::is_same_v<IType, fp16>) {
    // FP16 amax values can have finite FP32 pow2 scales above the FP16 finite exponent range.
    return compute_scale_from_types<IType, OType>(amax, epsilon, true);
  }

  if (amax < epsilon) {
    amax = epsilon;
  }
  if (isinf(amax) || amax == 0.0f || isnan(amax)) {
    return 1.0f;
  }

  constexpr uint32_t kExponentMask = 0x7F800000u;
  constexpr uint32_t kMantissaMask = 0x007FFFFFu;
  constexpr uint32_t kHiddenBit = 0x00800000u;
  const uint32_t fp8_max_bits = __float_as_uint(TypeInfo<OType>::max_finite_value);
  const uint32_t value_for_inf_bits =
      __float_as_uint(TypeInfo<IType>::max_finite_value) & kExponentMask;
  const uint32_t amax_bits = __float_as_uint(amax);
  const uint32_t amax_exponent_bits = amax_bits & kExponentMask;

  if (amax_exponent_bits == 0) {
    return __uint_as_float(value_for_inf_bits);
  }

  const int fp8_max_exponent = static_cast<int>((fp8_max_bits & kExponentMask) >> 23);
  const int amax_exponent = static_cast<int>(amax_exponent_bits >> 23);
  const uint32_t fp8_max_significand = (fp8_max_bits & kMantissaMask) | kHiddenBit;
  const uint32_t amax_significand = (amax_bits & kMantissaMask) | kHiddenBit;
  // Force-pow2 scaling keeps only the exponent of max_fp8 / amax. Compare significands to
  // compute floor(log2(max_fp8 / amax)) without issuing an FP32 divide in each scale group.
  int scale_exponent = fp8_max_exponent - amax_exponent + 127;
  if (fp8_max_significand < amax_significand) {
    --scale_exponent;
  }

  const int max_scale_exponent = static_cast<int>(value_for_inf_bits >> 23);
  if (scale_exponent > max_scale_exponent) {
    scale_exponent = max_scale_exponent;
  }
  if (scale_exponent <= 0) {
    return 0.0f;
  }
  return __uint_as_float(static_cast<uint32_t>(scale_exponent) << 23);
}

template <typename IType, typename OType, bool kForcePow2Scales>
__device__ __forceinline__ float compute_group_scale_from_types(const float amax,
                                                                const float epsilon) {
  if constexpr (kForcePow2Scales) {
    return compute_pow2_scale_from_types<IType, OType>(amax, epsilon);
  } else {
    return compute_scale_from_types<IType, OType>(amax, epsilon, false);
  }
}

template <typename IType, typename OType>
__device__ __forceinline__ float compute_group_scale_from_types(const float amax,
                                                                const float epsilon,
                                                                const bool force_pow_2_scales) {
  if (force_pow_2_scales) {
    return compute_pow2_scale_from_types<IType, OType>(amax, epsilon);
  }
  return compute_scale_from_types<IType, OType>(amax, epsilon, false);
}

template <typename IType, typename OType>
__device__ __forceinline__ void scaled_cast_pair(OType *const __restrict__ output,
                                                 const IType *const __restrict__ input,
                                                 const float scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if constexpr ((std::is_same_v<IType, bf16> || std::is_same_v<IType, fp16> ||
                 std::is_same_v<IType, fp32>) &&
                (std::is_same_v<OType, fp8e4m3> || std::is_same_v<OType, fp8e5m2>)) {
    const ptx::floatx2 scale_pair{scale, scale};
    if constexpr (std::is_same_v<IType, bf16>) {
      const ptx::bf16x2 input_pair{input[0], input[1]};
      if constexpr (std::is_same_v<OType, fp8e4m3>) {
        ptx::fp8e4m3x2 output_pair;
        ptx::mul_cvt_2x(output_pair, input_pair, scale_pair);
        output[0] = output_pair.x;
        output[1] = output_pair.y;
      } else {
        ptx::fp8e5m2x2 output_pair;
        ptx::mul_cvt_2x(output_pair, input_pair, scale_pair);
        output[0] = output_pair.x;
        output[1] = output_pair.y;
      }
    } else if constexpr (std::is_same_v<IType, fp16>) {
      const ptx::fp16x2 input_pair{input[0], input[1]};
      if constexpr (std::is_same_v<OType, fp8e4m3>) {
        ptx::fp8e4m3x2 output_pair;
        ptx::mul_cvt_2x(output_pair, input_pair, scale_pair);
        output[0] = output_pair.x;
        output[1] = output_pair.y;
      } else {
        ptx::fp8e5m2x2 output_pair;
        ptx::mul_cvt_2x(output_pair, input_pair, scale_pair);
        output[0] = output_pair.x;
        output[1] = output_pair.y;
      }
    } else {
      const ptx::floatx2 input_pair{input[0], input[1]};
      if constexpr (std::is_same_v<OType, fp8e4m3>) {
        ptx::fp8e4m3x2 output_pair;
        ptx::mul_cvt_2x(output_pair, input_pair, scale_pair);
        output[0] = output_pair.x;
        output[1] = output_pair.y;
      } else {
        ptx::fp8e5m2x2 output_pair;
        ptx::mul_cvt_2x(output_pair, input_pair, scale_pair);
        output[0] = output_pair.x;
        output[1] = output_pair.y;
      }
    }
  } else
#endif
  {
    output[0] = static_cast<OType>(static_cast<float>(input[0]) * scale);
    output[1] = static_cast<OType>(static_cast<float>(input[1]) * scale);
  }
}

template <typename IType, typename OType, size_t kVecElems>
__device__ __forceinline__ Vec<OType, kVecElems> scaled_cast_vec(
    const Vec<IType, kVecElems> &input, const float scale) {
  static_assert(kVecElems % 2 == 0, "Scaled FP8 vector casts require an even vector length.");
  Vec<OType, kVecElems> output;
#pragma unroll
  for (size_t i = 0; i < kVecElems; i += 2) {
    scaled_cast_pair<IType, OType>(&output.data.elt[i], &input.data.elt[i], scale);
  }
  return output;
}

__device__ __forceinline__ float warp_group_reduce_max(float value) {
  return warp_group_reduce_max<kThreadsPerScale>(value);
}

__device__ __forceinline__ size_t aligned_1d_shared_col_vec(const size_t local_row,
                                                            const size_t local_col_vec) {
  return local_col_vec ^ (local_row / kAligned1DOutputVec);
}

template <bool kIs2DScaling>
__device__ __forceinline__ TileDescriptor decode_uniform_grid_z_tile(
    const size_t tensor_id, const size_t tile_y, const size_t num_tensors,
    const size_t rows_per_tensor, const size_t cols) {
  TileDescriptor desc;
  if (tensor_id >= num_tensors || rows_per_tensor == 0) {
    return desc;
  }
  desc.tensor_id = tensor_id;
  desc.tile_y = tile_y;
  desc.rows = rows_per_tensor;
  desc.tensor_base = tensor_id * rows_per_tensor * cols;
  desc.rowwise_scale_offset =
      tensor_id * rowwise_scale_elements<kIs2DScaling>(rows_per_tensor, cols);
  desc.columnwise_scale_offset =
      tensor_id * columnwise_scale_elements<kIs2DScaling>(rows_per_tensor, cols);
  desc.valid = true;
  return desc;
}

template <bool kIs2DScaling>
__device__ __forceinline__ TileDescriptor decode_tile(
    size_t packed_tile_y, const size_t num_tensors, const size_t logical_first_dim,
    const size_t cols, const int64_t *const __restrict__ first_dims,
    const int64_t *const __restrict__ tensor_offsets,
    const int64_t *const __restrict__ row_block_offsets,
    const int64_t *const __restrict__ rowwise_scale_inv_offsets,
    const int64_t *const __restrict__ columnwise_scale_inv_offsets,
    const bool has_first_dims) {
  TileDescriptor desc;

  if (!has_first_dims) {
    const size_t rows_per_tensor = logical_first_dim / num_tensors;
    const size_t tile_rows_per_tensor = divup_by_block_len(rows_per_tensor);
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
    desc.rowwise_scale_offset =
        desc.tensor_id * rowwise_scale_elements<kIs2DScaling>(rows_per_tensor, cols);
    desc.columnwise_scale_offset =
        desc.tensor_id * columnwise_scale_elements<kIs2DScaling>(rows_per_tensor, cols);
    desc.valid = true;
    return desc;
  }

  if (row_block_offsets != nullptr) {
    for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
      const size_t next_tile = static_cast<size_t>(row_block_offsets[tensor_id + 1]);
      if (packed_tile_y < next_tile) {
        const size_t first_tile = static_cast<size_t>(row_block_offsets[tensor_id]);
        desc.tensor_id = tensor_id;
        desc.tile_y = packed_tile_y - first_tile;
        desc.rows = static_cast<size_t>(first_dims[tensor_id]);
        desc.tensor_base = static_cast<size_t>(tensor_offsets[tensor_id]);
        desc.rowwise_scale_offset = rowwise_scale_inv_offsets == nullptr
                                        ? 0
                                        : static_cast<size_t>(rowwise_scale_inv_offsets[tensor_id]);
        desc.columnwise_scale_offset =
            columnwise_scale_inv_offsets == nullptr
                ? 0
                : static_cast<size_t>(columnwise_scale_inv_offsets[tensor_id]);
        desc.valid = true;
        return desc;
      }
    }
    return desc;
  }

  size_t rowwise_scale_offset = 0;
  size_t columnwise_scale_offset = 0;
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    const size_t rows = static_cast<size_t>(first_dims[tensor_id]);
    const size_t tile_rows = divup_by_block_len(rows);
    if (packed_tile_y < tile_rows) {
      desc.tensor_id = tensor_id;
      desc.tile_y = packed_tile_y;
      desc.rows = rows;
      desc.tensor_base = static_cast<size_t>(tensor_offsets[tensor_id]);
      desc.rowwise_scale_offset = rowwise_scale_offset;
      desc.columnwise_scale_offset = columnwise_scale_offset;
      desc.valid = true;
      return desc;
    }
    packed_tile_y -= tile_rows;
    rowwise_scale_offset += rowwise_scale_elements<kIs2DScaling>(rows, cols);
    columnwise_scale_offset += columnwise_scale_elements<kIs2DScaling>(rows, cols);
  }
  return desc;
}

template <typename IType, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    group_quantize_fp8_2d_block_scaling_register_kernel(
        const IType *const __restrict__ input, OType *const __restrict__ output,
        OType *const __restrict__ output_t, float *const __restrict__ scale_inv,
        float *const __restrict__ scale_inv_t, const size_t num_tensors,
        const size_t logical_first_dim, const size_t cols,
        const int64_t *const __restrict__ first_dims,
        const int64_t *const __restrict__ tensor_offsets,
        const int64_t *const __restrict__ row_block_offsets,
        const int64_t *const __restrict__ rowwise_scale_inv_offsets,
        const int64_t *const __restrict__ columnwise_scale_inv_offsets, const bool return_rowwise,
        const bool return_columnwise, const float epsilon, const bool force_pow_2_scales,
        const float *const __restrict__ noop_ptr) {
  using IVec = Vec<IType, kRegTileCols>;
  using OVec = Vec<OType, kRegTileCols>;
  using OVecTrans = Vec<OType, kRegTileRows>;

  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  __shared__ TileDescriptor shared_tile;
  if (threadIdx.x == 0) {
    shared_tile = decode_tile<true>(blockIdx.y, num_tensors, logical_first_dim, cols, first_dims,
                                    tensor_offsets, row_block_offsets, rowwise_scale_inv_offsets,
                                    columnwise_scale_inv_offsets, true);
  }
  __syncthreads();
  const TileDescriptor tile = shared_tile;
  if (!tile.valid || tile.rows == 0 || cols == 0) {
    return;
  }

  const size_t tile_x = blockIdx.x;
  const size_t row_start = tile.tile_y * block_len();
  const size_t col_start = tile_x * block_len();
  if (row_start >= tile.rows || col_start >= cols) {
    return;
  }

  const size_t lane = threadIdx.x % kWarpSize;
  const size_t warp_id = threadIdx.x / kWarpSize;
  const size_t lane_x = lane % kRegThreadsXInWarp;
  const size_t lane_y = lane / kRegThreadsXInWarp;
  const size_t warp_x = warp_id % kRegWarpsX;
  const size_t warp_y = warp_id / kRegWarpsX;
  const size_t local_row_start = warp_y * kRegThreadsYInWarp * kRegTileRows +
                                 lane_y * kRegTileRows;
  const size_t local_col_start = warp_x * kRegThreadsXInWarp * kRegTileCols +
                                 lane_x * kRegTileCols;
  const size_t global_col_start = col_start + local_col_start;

  IVec input_vec[kRegTileRows];
  float local_amax = 0.0f;
#pragma unroll
  for (size_t i = 0; i < kRegTileRows; ++i) {
    const size_t local_row = local_row_start + i;
    const size_t row = row_start + local_row;
    const size_t valid_cols =
        global_col_start < cols ? min(static_cast<size_t>(kRegTileCols), cols - global_col_start)
                                : 0;
    if (row < tile.rows && valid_cols > 0) {
      input_vec[i].load_from_elts(input + tile.tensor_base + row * cols + global_col_start, 0,
                                  valid_cols);
    } else {
      input_vec[i].clear();
    }
#pragma unroll
    for (size_t j = 0; j < kRegTileCols; ++j) {
      local_amax = fmaxf(local_amax, fabsf(static_cast<float>(input_vec[i].data.elt[j])));
    }
  }

  __shared__ float warp_amax[kThreadsPerBlock / kWarpSize];
  __shared__ float tile_scale;
  const float reduced_warp_amax = warp_reduce_max<kWarpSize>(local_amax);
  if (lane == 0) {
    warp_amax[warp_id] = reduced_warp_amax;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float amax = warp_amax[0];
#pragma unroll
    for (size_t i = 1; i < kThreadsPerBlock / kWarpSize; ++i) {
      amax = fmaxf(amax, warp_amax[i]);
    }
    tile_scale =
        compute_group_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
    const float inv_scale = reciprocal_scale(tile_scale, force_pow_2_scales);
    const size_t row_blocks = divup_by_block_len(tile.rows);
    const size_t col_blocks = divup_by_block_len(cols);
    if (return_rowwise) {
      const size_t rowwise_stride = round_up_to_multiple(col_blocks, 4);
      scale_inv[tile.rowwise_scale_offset + tile.tile_y * rowwise_stride + tile_x] = inv_scale;
    }
    if (return_columnwise) {
      const size_t columnwise_stride = round_up_to_multiple(row_blocks, 4);
      scale_inv_t[tile.columnwise_scale_offset + tile_x * columnwise_stride + tile.tile_y] =
          inv_scale;
    }
  }
  __syncthreads();

  const float scale = tile_scale;
  if (return_rowwise) {
#pragma unroll
    for (size_t i = 0; i < kRegTileRows; ++i) {
      const size_t local_row = local_row_start + i;
      const size_t row = row_start + local_row;
      const size_t valid_cols =
          global_col_start < cols ? min(static_cast<size_t>(kRegTileCols), cols - global_col_start)
                                  : 0;
      if (row < tile.rows && valid_cols > 0) {
        OVec output_vec;
#pragma unroll
        for (size_t j = 0; j < kRegTileCols; ++j) {
          output_vec.data.elt[j] =
              static_cast<OType>(static_cast<float>(input_vec[i].data.elt[j]) * scale);
        }
        output_vec.store_to_elts(output + tile.tensor_base + row * cols + global_col_start, 0,
                                 valid_cols);
      }
    }
  }

  if (return_columnwise) {
    const size_t row = row_start + local_row_start;
    const size_t valid_rows =
        row < tile.rows ? min(static_cast<size_t>(kRegTileRows), tile.rows - row) : 0;
#pragma unroll
    for (size_t j = 0; j < kRegTileCols; ++j) {
      const size_t col = global_col_start + j;
      if (col < cols && valid_rows > 0) {
        OVecTrans output_vec_t;
#pragma unroll
        for (size_t i = 0; i < kRegTileRows; ++i) {
          output_vec_t.data.elt[i] =
              static_cast<OType>(static_cast<float>(input_vec[i].data.elt[j]) * scale);
        }
        output_vec_t.store_to_elts(output_t + tile.tensor_base + col * tile.rows + row, 0,
                                   valid_rows);
      }
    }
  }
}

template <typename IType, typename OType, bool kReturnRowwise, bool kReturnColumnwise,
          bool kForcePow2Scales>
__global__ void __launch_bounds__(kThreadsPerBlock)
    group_quantize_fp8_1d_block_scaling_aligned_kernel(
        const IType *const __restrict__ input, OType *const __restrict__ output,
        OType *const __restrict__ output_t, float *const __restrict__ scale_inv,
        float *const __restrict__ scale_inv_t, const size_t num_tensors,
        const size_t logical_first_dim, const size_t rows_per_tensor, const size_t cols,
        const int64_t *const __restrict__ first_dims,
        const int64_t *const __restrict__ tensor_offsets,
        const int64_t *const __restrict__ row_block_offsets,
        const int64_t *const __restrict__ rowwise_scale_inv_offsets,
        const int64_t *const __restrict__ columnwise_scale_inv_offsets, const bool has_first_dims,
        const float epsilon, const float *const __restrict__ noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  TileDescriptor tile;
  if (!has_first_dims) {
    tile = decode_uniform_grid_z_tile<false>(blockIdx.z, blockIdx.y, num_tensors,
                                             rows_per_tensor, cols);
  } else {
    __shared__ TileDescriptor shared_tile;
    if (threadIdx.x == 0) {
      shared_tile = decode_tile<false>(blockIdx.y, num_tensors, logical_first_dim, cols,
                                       first_dims, tensor_offsets, row_block_offsets,
                                       rowwise_scale_inv_offsets,
                                       columnwise_scale_inv_offsets, has_first_dims);
    }
    __syncthreads();
    tile = shared_tile;
  }
  if (!tile.valid || tile.rows == 0 || cols == 0) {
    return;
  }

  const size_t tile_x = blockIdx.x;
  const size_t row_start = tile.tile_y * block_len();
  const size_t col_start = tile_x * block_len();
  if (row_start + block_len() > tile.rows || col_start + block_len() > cols) {
    return;
  }
  const size_t rowwise_stride = tile.rows;
  const size_t columnwise_stride = cols;

  extern __shared__ char input_tile_base[];
  using SMemVec = Vec<IType, kSharedVec>;
  SMemVec *const input_tile = reinterpret_cast<SMemVec *>(input_tile_base);

  union ILoad {
    Vec<IType, kLoadVec> input_type;
    Vec<SMemVec, kLoadVec / kSharedVec> smem_type;
  };

#pragma unroll
  for (size_t iter = 0; iter < kBlockLen / kLoadRowStride; ++iter) {
    const size_t local_row = iter * kLoadRowStride + threadIdx.x / kLoadThreadsPerRow;
    const size_t local_col_vec =
        (threadIdx.x % kLoadThreadsPerRow) * (kLoadVec / kSharedVec);
    const size_t row = row_start + local_row;
    const size_t col = col_start + local_col_vec * kSharedVec;
    ILoad input_vec;
    input_vec.input_type.load_from(input + tile.tensor_base + row * cols + col);
    if constexpr (kReturnColumnwise) {
#pragma unroll
      for (size_t i = 0; i < kLoadVec / kSharedVec; ++i) {
        input_tile[local_row * kSharedCols +
                   aligned_1d_shared_col_vec(local_row, local_col_vec + i)] =
            input_vec.smem_type.data.elt[i];
      }
    }
    if constexpr (kReturnRowwise) {
      float local_amax = 0.0f;
#pragma unroll
      for (size_t i = 0; i < kLoadVec; ++i) {
        const float value = static_cast<float>(input_vec.input_type.data.elt[i]);
        local_amax = fmaxf(local_amax, fabsf(value));
      }

      const float amax = warp_group_reduce_max<kLoadThreadsPerRow>(local_amax);
      const size_t group_lane = threadIdx.x % kLoadThreadsPerRow;
      float scale = 1.0f;
      if (group_lane == 0) {
        scale = compute_group_scale_from_types<IType, OType, kForcePow2Scales>(amax, epsilon);
        scale_inv[tile.rowwise_scale_offset + tile_x * rowwise_stride + row] =
            reciprocal_scale(scale, kForcePow2Scales);
      }
      scale = warp_group_broadcast<kLoadThreadsPerRow>(scale);

      const Vec<OType, kLoadVec> output_vec =
          scaled_cast_vec<IType, OType, kLoadVec>(input_vec.input_type, scale);
      output_vec.store_to(output + tile.tensor_base + row * cols + col);
    }
  }
  if constexpr (kReturnColumnwise) {
    __syncthreads();
  }

  if constexpr (kReturnColumnwise) {
    using OVec = Vec<OType, kAligned1DOutputVec>;
    const size_t group_id = threadIdx.x / kAligned1DThreadsPerScale;
    const size_t group_lane = threadIdx.x % kAligned1DThreadsPerScale;
    constexpr size_t kColumnwiseIterations =
        kBlockLen / (kAligned1DScalesPerIteration * kSharedVec);
#pragma unroll
    for (size_t iter = 0; iter < kColumnwiseIterations; ++iter) {
      const size_t local_col_vec = iter * kAligned1DScalesPerIteration + group_id;
      const size_t col = col_start + local_col_vec * kSharedVec;
      const size_t local_row = group_lane * kAligned1DOutputVec;
      const size_t row = row_start + local_row;
      SMemVec smem_vec[kAligned1DOutputVec];

#pragma unroll
      for (size_t e = 0; e < kAligned1DOutputVec; ++e) {
        const size_t row_vec = local_row + e;
        smem_vec[e] = input_tile[row_vec * kSharedCols +
                                 aligned_1d_shared_col_vec(row_vec, local_col_vec)];
      }

      for (size_t smem_idx = 0; smem_idx < kSharedVec; ++smem_idx) {
        float local_amax = 0.0f;
#pragma unroll
        for (size_t e = 0; e < kAligned1DOutputVec; ++e) {
          const float value = static_cast<float>(smem_vec[e].data.elt[smem_idx]);
          local_amax = fmaxf(local_amax, fabsf(value));
        }

        const float amax = warp_group_reduce_max<kAligned1DThreadsPerScale>(local_amax);
        float scale = 1.0f;
        if (group_lane == 0) {
          scale = compute_group_scale_from_types<IType, OType, kForcePow2Scales>(amax, epsilon);
          scale_inv_t[tile.columnwise_scale_offset + tile.tile_y * columnwise_stride + col +
                      smem_idx] = reciprocal_scale(scale, kForcePow2Scales);
        }
        scale = warp_group_broadcast<kAligned1DThreadsPerScale>(scale);

        Vec<IType, kAligned1DOutputVec> input_col_vec;
#pragma unroll
        for (size_t e = 0; e < kAligned1DOutputVec; ++e) {
          input_col_vec.data.elt[e] = smem_vec[e].data.elt[smem_idx];
        }
        const OVec output_vec =
            scaled_cast_vec<IType, OType, kAligned1DOutputVec>(input_col_vec, scale);
        output_vec.store_to(output_t + tile.tensor_base + (col + smem_idx) * tile.rows + row);
      }
    }
  }
}

template <typename IType, typename OType, bool kReturnRowwise, bool kReturnColumnwise,
          bool kForcePow2Scales>
__global__ void __launch_bounds__(kThreadsPerBlock, 3)
    group_quantize_fp8_2d_block_scaling_aligned_register_kernel(
        const IType *const __restrict__ input, OType *const __restrict__ output,
        OType *const __restrict__ output_t, float *const __restrict__ scale_inv,
        float *const __restrict__ scale_inv_t, const size_t num_tensors,
        const size_t logical_first_dim, const size_t rows_per_tensor, const size_t cols,
        const int64_t *const __restrict__ first_dims,
        const int64_t *const __restrict__ tensor_offsets,
        const int64_t *const __restrict__ row_block_offsets,
        const int64_t *const __restrict__ rowwise_scale_inv_offsets,
        const int64_t *const __restrict__ columnwise_scale_inv_offsets, const bool has_first_dims,
        const float epsilon, const float *const __restrict__ noop_ptr) {
  using IVec = Vec<IType, kRegTileCols>;
  using OVec = Vec<OType, kRegTileCols>;

  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  TileDescriptor tile;
  if (!has_first_dims) {
    tile = decode_uniform_grid_z_tile<true>(blockIdx.z, blockIdx.y, num_tensors,
                                            rows_per_tensor, cols);
  } else {
    __shared__ TileDescriptor shared_tile;
    if (threadIdx.x == 0) {
      shared_tile = decode_tile<true>(blockIdx.y, num_tensors, logical_first_dim, cols,
                                      first_dims, tensor_offsets, row_block_offsets,
                                      rowwise_scale_inv_offsets,
                                      columnwise_scale_inv_offsets, has_first_dims);
    }
    __syncthreads();
    tile = shared_tile;
  }
  if (!tile.valid || tile.rows == 0 || cols == 0) {
    return;
  }

  const size_t tile_x = blockIdx.x;
  const size_t row_start = tile.tile_y * block_len();
  const size_t col_start = tile_x * block_len();
  if (row_start + block_len() > tile.rows || col_start + block_len() > cols) {
    return;
  }

  const size_t lane = threadIdx.x % kWarpSize;
  const size_t warp_id = threadIdx.x / kWarpSize;
  const size_t lane_x = lane % kRegThreadsXInWarp;
  const size_t lane_y = lane / kRegThreadsXInWarp;
  const size_t warp_x = warp_id % kRegWarpsX;
  const size_t warp_y = warp_id / kRegWarpsX;
  const size_t local_col_start = warp_x * kRegThreadsXInWarp * kRegTileCols +
                                 lane_x * kRegTileCols;
  const size_t global_col_start = col_start + local_col_start;

  float local_amax = 0.0f;
#pragma unroll
  for (size_t band = 0; band < kAligned2DRowBands; ++band) {
    const size_t local_row_start =
        band * kAligned2DRowBandRows +
        warp_y * kRegThreadsYInWarp * kAligned2DRegTileRows +
        lane_y * kAligned2DRegTileRows;
#pragma unroll
    for (size_t i = 0; i < kAligned2DRegTileRows; ++i) {
      IVec input_vec;
      const size_t row = row_start + local_row_start + i;
      input_vec.load_from(input + tile.tensor_base + row * cols + global_col_start);
#pragma unroll
      for (size_t j = 0; j < kRegTileCols; ++j) {
        local_amax = fmaxf(local_amax, fabsf(static_cast<float>(input_vec.data.elt[j])));
      }
    }
  }

  __shared__ float warp_amax[kThreadsPerBlock / kWarpSize];
  __shared__ float tile_scale;
  const float reduced_warp_amax = warp_reduce_max<kWarpSize>(local_amax);
  if (lane == 0) {
    warp_amax[warp_id] = reduced_warp_amax;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float amax = warp_amax[0];
#pragma unroll
    for (size_t i = 1; i < kThreadsPerBlock / kWarpSize; ++i) {
      amax = fmaxf(amax, warp_amax[i]);
    }
    tile_scale = compute_group_scale_from_types<IType, OType, kForcePow2Scales>(amax, epsilon);
    const float inv_scale = reciprocal_scale(tile_scale, kForcePow2Scales);
    const size_t row_blocks = divup_by_block_len(tile.rows);
    const size_t col_blocks = divup_by_block_len(cols);
    if constexpr (kReturnRowwise) {
      const size_t rowwise_stride = round_up_to_multiple(col_blocks, 4);
      scale_inv[tile.rowwise_scale_offset + tile.tile_y * rowwise_stride + tile_x] = inv_scale;
    }
    if constexpr (kReturnColumnwise) {
      const size_t columnwise_stride = round_up_to_multiple(row_blocks, 4);
      scale_inv_t[tile.columnwise_scale_offset + tile_x * columnwise_stride + tile.tile_y] =
          inv_scale;
    }
  }
  __syncthreads();

  const float scale = tile_scale;
#pragma unroll
  for (size_t band = 0; band < kAligned2DRowBands; ++band) {
    const size_t local_row_start =
        band * kAligned2DRowBandRows +
        warp_y * kRegThreadsYInWarp * kAligned2DRegTileRows +
        lane_y * kAligned2DRegTileRows;
    IVec input_vec[kAligned2DRegTileRows];
    // This aligned 2D path intentionally reloads the tile after the amax pass.
    // Keeping all band values live across the block-wide reduction would raise
    // register pressure enough to reduce occupancy. The benchmark HBM model
    // charges one input pass and reports the duplicate pass as global-load
    // instruction/cache traffic.
#pragma unroll
    for (size_t i = 0; i < kAligned2DRegTileRows; ++i) {
      const size_t row = row_start + local_row_start + i;
      input_vec[i].load_from(input + tile.tensor_base + row * cols + global_col_start);
    }

    if constexpr (kReturnRowwise) {
#pragma unroll
      for (size_t i = 0; i < kAligned2DRegTileRows; ++i) {
        const size_t row = row_start + local_row_start + i;
        const OVec output_vec = scaled_cast_vec<IType, OType, kRegTileCols>(input_vec[i], scale);
        output_vec.store_to(output + tile.tensor_base + row * cols + global_col_start);
      }
    }

    if constexpr (kReturnColumnwise) {
      using OVecTrans = Vec<OType, kAligned2DRegTileRows>;
      const size_t row = row_start + local_row_start;
#pragma unroll
      for (size_t j = 0; j < kRegTileCols; ++j) {
        const size_t col = global_col_start + j;
        OVecTrans output_vec_t;
#pragma unroll
        for (size_t i = 0; i < kAligned2DRegTileRows; ++i) {
          output_vec_t.data.elt[i] =
              static_cast<OType>(static_cast<float>(input_vec[i].data.elt[j]) * scale);
        }
        output_vec_t.store_to(output_t + tile.tensor_base + col * tile.rows + row);
      }
    }
  }
}

template <bool kIs2DScaling, typename IType, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock) group_quantize_fp8_block_scaling_kernel(
    const IType *const __restrict__ input, OType *const __restrict__ output,
    OType *const __restrict__ output_t, float *const __restrict__ scale_inv,
    float *const __restrict__ scale_inv_t, const size_t num_tensors,
    const size_t logical_first_dim, const size_t cols,
    const int64_t *const __restrict__ first_dims,
    const int64_t *const __restrict__ tensor_offsets,
    const int64_t *const __restrict__ row_block_offsets,
    const int64_t *const __restrict__ rowwise_scale_inv_offsets,
    const int64_t *const __restrict__ columnwise_scale_inv_offsets, const bool has_first_dims,
    const bool return_rowwise, const bool return_columnwise, const float epsilon,
    const bool force_pow_2_scales, const float *const __restrict__ noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  TileDescriptor tile;
  if (has_first_dims) {
    __shared__ TileDescriptor shared_tile;
    if (threadIdx.x == 0) {
      shared_tile = decode_tile<kIs2DScaling>(blockIdx.y, num_tensors, logical_first_dim, cols,
                                              first_dims, tensor_offsets, row_block_offsets,
                                              rowwise_scale_inv_offsets,
                                              columnwise_scale_inv_offsets, has_first_dims);
    }
    __syncthreads();
    tile = shared_tile;
  } else {
    tile = decode_tile<kIs2DScaling>(blockIdx.y, num_tensors, logical_first_dim, cols, first_dims,
                                     tensor_offsets, row_block_offsets,
                                     rowwise_scale_inv_offsets,
                                     columnwise_scale_inv_offsets, has_first_dims);
  }
  if (!tile.valid || tile.rows == 0 || cols == 0) {
    return;
  }

  const size_t tile_x = blockIdx.x;
  const size_t row_start = tile.tile_y * block_len();
  const size_t col_start = tile_x * block_len();
  if (row_start >= tile.rows || col_start >= cols) {
    return;
  }

  __shared__ float tile_amax[kThreadsPerBlock];
  __shared__ float row_amax[kBlockLen];
  __shared__ float col_amax[kBlockLen];
  __shared__ float tile_scale;
  __shared__ float row_scale[kBlockLen];
  __shared__ float col_scale[kBlockLen];
  extern __shared__ char input_tile_base[];
  IType *const input_tile = reinterpret_cast<IType *>(input_tile_base);

  using ILoadVec = Vec<IType, kLoadVec>;
  float tile_load_amax = 0.0f;
#pragma unroll
  for (size_t iter = 0; iter < kBlockLen / kLoadRowStride; ++iter) {
    const size_t local_row = iter * kLoadRowStride + threadIdx.x / kLoadThreadsPerRow;
    const size_t local_col = (threadIdx.x % kLoadThreadsPerRow) * kLoadVec;
    const size_t row = row_start + local_row;
    const size_t col = col_start + local_col;
    const size_t valid_cols =
        col < cols ? min(static_cast<size_t>(kLoadVec), cols - col) : 0;
    ILoadVec input_vec;
    if (row < tile.rows && col < cols) {
      input_vec.load_from_elts(input + tile.tensor_base + row * cols + col, 0, valid_cols);
      input_vec.store_to_elts(input_tile + local_row * (block_len() + 1) + local_col, 0,
                              valid_cols);
    } else {
      input_vec.clear();
    }
    if constexpr (kIs2DScaling) {
#pragma unroll
      for (size_t e = 0; e < kLoadVec; ++e) {
        tile_load_amax = fmaxf(tile_load_amax, fabsf(static_cast<float>(input_vec.data.elt[e])));
      }
    }
  }
  __syncthreads();

  if constexpr (!kIs2DScaling) {
    using OVec = Vec<OType, kElemsPerScaleThread>;
    const size_t group_id = threadIdx.x / kThreadsPerScale;
    const size_t group_lane = threadIdx.x % kThreadsPerScale;

    if (return_rowwise) {
#pragma unroll
      for (size_t iter = 0; iter < kBlockLen / kScalesPerIteration; ++iter) {
        const size_t local_row = iter * kScalesPerIteration + group_id;
        const size_t row = row_start + local_row;
        const size_t local_col = group_lane * kElemsPerScaleThread;
        const size_t col = col_start + local_col;
        const size_t valid_cols =
            col < cols ? min(static_cast<size_t>(kElemsPerScaleThread), cols - col) : 0;

        float local_amax = 0.0f;
#pragma unroll
        for (size_t e = 0; e < kElemsPerScaleThread; ++e) {
          if (row < tile.rows && e < valid_cols) {
            const float value =
                static_cast<float>(input_tile[local_row * (block_len() + 1) + local_col + e]);
            local_amax = fmaxf(local_amax, fabsf(value));
          }
        }

        const float amax = warp_group_reduce_max(local_amax);
        const float scale =
            compute_group_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
        if (group_lane == 0 && row < tile.rows) {
          const size_t rowwise_stride = round_up_to_multiple(tile.rows, 4);
          scale_inv[tile.rowwise_scale_offset + tile_x * rowwise_stride + row] =
              reciprocal_scale(scale, force_pow_2_scales);
        }
        if (row < tile.rows && valid_cols > 0) {
          OVec output_vec;
#pragma unroll
          for (size_t e = 0; e < kElemsPerScaleThread; ++e) {
            const float value =
                static_cast<float>(input_tile[local_row * (block_len() + 1) + local_col + e]);
            output_vec.data.elt[e] = static_cast<OType>(value * scale);
          }
          output_vec.store_to_elts(output + tile.tensor_base + row * cols + col, 0, valid_cols);
        }
      }
    }

    if (return_columnwise) {
#pragma unroll
      for (size_t iter = 0; iter < kBlockLen / kScalesPerIteration; ++iter) {
        const size_t local_col = iter * kScalesPerIteration + group_id;
        const size_t col = col_start + local_col;
        const size_t local_row = group_lane * kElemsPerScaleThread;
        const size_t row = row_start + local_row;
        const size_t valid_rows =
            row < tile.rows ? min(static_cast<size_t>(kElemsPerScaleThread), tile.rows - row) : 0;

        float local_amax = 0.0f;
#pragma unroll
        for (size_t e = 0; e < kElemsPerScaleThread; ++e) {
          if (col < cols && e < valid_rows) {
            const float value =
                static_cast<float>(input_tile[(local_row + e) * (block_len() + 1) + local_col]);
            local_amax = fmaxf(local_amax, fabsf(value));
          }
        }

        const float amax = warp_group_reduce_max(local_amax);
        const float scale =
            compute_group_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
        if (group_lane == 0 && col < cols) {
          const size_t columnwise_stride = round_up_to_multiple(cols, 4);
          scale_inv_t[tile.columnwise_scale_offset + tile.tile_y * columnwise_stride + col] =
              reciprocal_scale(scale, force_pow_2_scales);
        }
        if (col < cols && valid_rows > 0) {
          OVec output_vec;
#pragma unroll
          for (size_t e = 0; e < kElemsPerScaleThread; ++e) {
            const float value =
                static_cast<float>(input_tile[(local_row + e) * (block_len() + 1) + local_col]);
            output_vec.data.elt[e] = static_cast<OType>(value * scale);
          }
          output_vec.store_to_elts(output_t + tile.tensor_base + col * tile.rows + row, 0,
                                   valid_rows);
        }
      }
    }
    return;
  }

  if constexpr (kIs2DScaling) {
    tile_amax[threadIdx.x] = tile_load_amax;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        tile_amax[threadIdx.x] = fmaxf(tile_amax[threadIdx.x], tile_amax[threadIdx.x + stride]);
      }
      __syncthreads();
    }
  } else if (threadIdx.x < block_len()) {
    const size_t lane = threadIdx.x;
    if (return_rowwise) {
      const size_t row = row_start + lane;
      float local_amax = 0.0f;
      if (row < tile.rows) {
        for (size_t local_col = 0; local_col < block_len(); ++local_col) {
          const size_t col = col_start + local_col;
          if (col < cols) {
            const float value =
                static_cast<float>(input_tile[lane * (block_len() + 1) + local_col]);
            local_amax = fmaxf(local_amax, fabsf(value));
          }
        }
      }
      row_amax[lane] = local_amax;
    }
    if (return_columnwise) {
      const size_t col = col_start + lane;
      float local_amax = 0.0f;
      if (col < cols) {
        for (size_t local_row = 0; local_row < block_len(); ++local_row) {
          const size_t row = row_start + local_row;
          if (row < tile.rows) {
            const float value =
                static_cast<float>(input_tile[local_row * (block_len() + 1) + lane]);
            local_amax = fmaxf(local_amax, fabsf(value));
          }
        }
      }
      col_amax[lane] = local_amax;
    }
  }
  __syncthreads();

  if constexpr (kIs2DScaling) {
    if (threadIdx.x == 0) {
      const float amax = tile_amax[0];
      tile_scale =
          compute_group_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
      const float inv_scale = reciprocal_scale(tile_scale, force_pow_2_scales);
      const size_t row_blocks = divup_by_block_len(tile.rows);
      const size_t col_blocks = divup_by_block_len(cols);
      if (return_rowwise) {
        const size_t rowwise_stride = round_up_to_multiple(col_blocks, 4);
        scale_inv[tile.rowwise_scale_offset + tile.tile_y * rowwise_stride + tile_x] = inv_scale;
      }
      if (return_columnwise) {
        const size_t columnwise_stride = round_up_to_multiple(row_blocks, 4);
        scale_inv_t[tile.columnwise_scale_offset + tile_x * columnwise_stride + tile.tile_y] =
            inv_scale;
      }
    }
  } else {
    if (threadIdx.x < block_len()) {
      const size_t local_row = threadIdx.x;
      const size_t row = row_start + local_row;
      if (return_rowwise && row < tile.rows) {
        const float amax = row_amax[local_row];
        row_scale[local_row] =
            compute_group_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
        const size_t rowwise_stride = round_up_to_multiple(tile.rows, 4);
        scale_inv[tile.rowwise_scale_offset + tile_x * rowwise_stride + row] =
            reciprocal_scale(row_scale[local_row], force_pow_2_scales);
      }

      const size_t local_col = threadIdx.x;
      const size_t col = col_start + local_col;
      if (return_columnwise && col < cols) {
        const float amax = col_amax[local_col];
        col_scale[local_col] =
            compute_group_scale_from_types<IType, OType>(amax, epsilon, force_pow_2_scales);
        const size_t columnwise_stride = round_up_to_multiple(cols, 4);
        scale_inv_t[tile.columnwise_scale_offset + tile.tile_y * columnwise_stride + col] =
            reciprocal_scale(col_scale[local_col], force_pow_2_scales);
      }
    }
  }
  __syncthreads();

  for (size_t idx = threadIdx.x; idx < block_len() * block_len(); idx += blockDim.x) {
    const size_t local_row = idx / block_len();
    const size_t local_col = idx % block_len();
    const size_t row = row_start + local_row;
    const size_t col = col_start + local_col;
    if (row >= tile.rows || col >= cols) {
      continue;
    }

    const float value =
        static_cast<float>(input_tile[local_row * (block_len() + 1) + local_col]);
    if (return_rowwise) {
      const float scale = kIs2DScaling ? tile_scale : row_scale[local_row];
      output[tile.tensor_base + row * cols + col] = static_cast<OType>(value * scale);
    }
  }

  if (return_columnwise) {
    for (size_t idx = threadIdx.x; idx < block_len() * block_len(); idx += blockDim.x) {
      const size_t local_col = idx / block_len();
      const size_t local_row = idx % block_len();
      const size_t row = row_start + local_row;
      const size_t col = col_start + local_col;
      if (row >= tile.rows || col >= cols) {
        continue;
      }
      const float value =
          static_cast<float>(input_tile[local_row * (block_len() + 1) + local_col]);
      const float scale = kIs2DScaling ? tile_scale : col_scale[local_col];
      output_t[tile.tensor_base + col * tile.rows + row] =
          static_cast<OType>(value * scale);
    }
  }
}

template <typename IType, typename OType, bool kReturnRowwise, bool kReturnColumnwise,
          bool kForcePow2Scales>
void launch_group_quantize_fp8_2d_block_scaling_aligned_register(
    const dim3 &grid, cudaStream_t stream, const IType *const input, OType *const output,
    OType *const output_t, float *const scale_inv, float *const scale_inv_t,
    const size_t num_tensors, const size_t logical_first_dim, const size_t rows_per_tensor,
    const size_t cols, const int64_t *const first_dims, const int64_t *const tensor_offsets,
    const int64_t *const row_block_offsets, const int64_t *const rowwise_scale_inv_offsets,
    const int64_t *const columnwise_scale_inv_offsets, const bool has_first_dims,
    const float epsilon, const float *const noop_ptr) {
  group_quantize_fp8_2d_block_scaling_aligned_register_kernel<
      IType, OType, kReturnRowwise, kReturnColumnwise, kForcePow2Scales>
      <<<grid, kThreadsPerBlock, 0, stream>>>(input, output, output_t, scale_inv, scale_inv_t,
                                              num_tensors, logical_first_dim, rows_per_tensor,
                                              cols, first_dims, tensor_offsets, row_block_offsets,
                                              rowwise_scale_inv_offsets,
                                              columnwise_scale_inv_offsets, has_first_dims,
                                              epsilon, noop_ptr);
}

template <typename IType, typename OType, bool kReturnRowwise, bool kReturnColumnwise>
void launch_group_quantize_fp8_2d_block_scaling_aligned_register(
    const bool force_pow_2_scales, const dim3 &grid, cudaStream_t stream,
    const IType *const input, OType *const output, OType *const output_t,
    float *const scale_inv, float *const scale_inv_t, const size_t num_tensors,
    const size_t logical_first_dim, const size_t rows_per_tensor, const size_t cols,
    const int64_t *const first_dims, const int64_t *const tensor_offsets,
    const int64_t *const row_block_offsets, const int64_t *const rowwise_scale_inv_offsets,
    const int64_t *const columnwise_scale_inv_offsets, const bool has_first_dims,
    const float epsilon, const float *const noop_ptr) {
  if (force_pow_2_scales) {
    launch_group_quantize_fp8_2d_block_scaling_aligned_register<
        IType, OType, kReturnRowwise, kReturnColumnwise, true>(
        grid, stream, input, output, output_t, scale_inv, scale_inv_t, num_tensors,
        logical_first_dim, rows_per_tensor, cols, first_dims, tensor_offsets, row_block_offsets,
        rowwise_scale_inv_offsets, columnwise_scale_inv_offsets, has_first_dims, epsilon,
        noop_ptr);
  } else {
    launch_group_quantize_fp8_2d_block_scaling_aligned_register<
        IType, OType, kReturnRowwise, kReturnColumnwise, false>(
        grid, stream, input, output, output_t, scale_inv, scale_inv_t, num_tensors,
        logical_first_dim, rows_per_tensor, cols, first_dims, tensor_offsets, row_block_offsets,
        rowwise_scale_inv_offsets, columnwise_scale_inv_offsets, has_first_dims, epsilon,
        noop_ptr);
  }
}

template <typename IType, typename OType, bool kReturnRowwise, bool kReturnColumnwise,
          bool kForcePow2Scales>
void launch_group_quantize_fp8_1d_block_scaling_aligned(
    const dim3 &grid, const size_t smem_bytes, cudaStream_t stream,
    const IType *const input, OType *const output, OType *const output_t,
    float *const scale_inv, float *const scale_inv_t, const size_t num_tensors,
    const size_t logical_first_dim, const size_t rows_per_tensor, const size_t cols,
    const int64_t *const first_dims, const int64_t *const tensor_offsets,
    const int64_t *const row_block_offsets, const int64_t *const rowwise_scale_inv_offsets,
    const int64_t *const columnwise_scale_inv_offsets, const bool has_first_dims,
    const float epsilon, const float *const noop_ptr) {
  if (smem_bytes >= 48 * 1024) {
    cudaError_t err = cudaFuncSetAttribute(
        &group_quantize_fp8_1d_block_scaling_aligned_kernel<
            IType, OType, kReturnRowwise, kReturnColumnwise, kForcePow2Scales>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    NVTE_CHECK(err == cudaSuccess,
               "Failed to set grouped FP8 block-scaling shared memory size.");
  }
  group_quantize_fp8_1d_block_scaling_aligned_kernel<
      IType, OType, kReturnRowwise, kReturnColumnwise, kForcePow2Scales>
      <<<grid, kThreadsPerBlock, smem_bytes, stream>>>(
          input, output, output_t, scale_inv, scale_inv_t, num_tensors, logical_first_dim,
          rows_per_tensor, cols, first_dims, tensor_offsets, row_block_offsets,
          rowwise_scale_inv_offsets, columnwise_scale_inv_offsets, has_first_dims, epsilon,
          noop_ptr);
}

template <typename IType, typename OType, bool kReturnRowwise, bool kReturnColumnwise>
void launch_group_quantize_fp8_1d_block_scaling_aligned(
    const bool force_pow_2_scales, const dim3 &grid, const size_t smem_bytes,
    cudaStream_t stream, const IType *const input, OType *const output,
    OType *const output_t, float *const scale_inv, float *const scale_inv_t,
    const size_t num_tensors, const size_t logical_first_dim, const size_t rows_per_tensor,
    const size_t cols, const int64_t *const first_dims, const int64_t *const tensor_offsets,
    const int64_t *const row_block_offsets, const int64_t *const rowwise_scale_inv_offsets,
    const int64_t *const columnwise_scale_inv_offsets, const bool has_first_dims,
    const float epsilon, const float *const noop_ptr) {
  if (force_pow_2_scales) {
    launch_group_quantize_fp8_1d_block_scaling_aligned<
        IType, OType, kReturnRowwise, kReturnColumnwise, true>(
        grid, smem_bytes, stream, input, output, output_t, scale_inv, scale_inv_t, num_tensors,
        logical_first_dim, rows_per_tensor, cols, first_dims, tensor_offsets, row_block_offsets,
        rowwise_scale_inv_offsets, columnwise_scale_inv_offsets, has_first_dims, epsilon,
        noop_ptr);
  } else {
    launch_group_quantize_fp8_1d_block_scaling_aligned<
        IType, OType, kReturnRowwise, kReturnColumnwise, false>(
        grid, smem_bytes, stream, input, output, output_t, scale_inv, scale_inv_t, num_tensors,
        logical_first_dim, rows_per_tensor, cols, first_dims, tensor_offsets, row_block_offsets,
        rowwise_scale_inv_offsets, columnwise_scale_inv_offsets, has_first_dims, epsilon,
        noop_ptr);
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

  const bool has_row_block_offsets = has_first_dims && output->row_block_offsets.has_data();
  const bool has_rowwise_scale_inv_offsets =
      !use_rowwise || output->rowwise_scale_inv_offsets.has_data();
  const bool has_columnwise_scale_inv_offsets =
      !use_columnwise || output->columnwise_scale_inv_offsets.has_data();
  const bool use_offset_metadata =
      has_row_block_offsets && has_rowwise_scale_inv_offsets && has_columnwise_scale_inv_offsets;
  const int64_t *const row_block_offsets_ptr =
      use_offset_metadata ? reinterpret_cast<const int64_t *>(output->row_block_offsets.dptr)
                          : nullptr;
  const int64_t *const rowwise_scale_inv_offsets_ptr =
      use_offset_metadata && use_rowwise
          ? reinterpret_cast<const int64_t *>(output->rowwise_scale_inv_offsets.dptr)
          : nullptr;
  const int64_t *const columnwise_scale_inv_offsets_ptr =
      use_offset_metadata && use_columnwise
          ? reinterpret_cast<const int64_t *>(output->columnwise_scale_inv_offsets.dptr)
          : nullptr;

  if (use_offset_metadata) {
    NVTE_CHECK(output->row_block_offsets.dtype == DType::kInt64,
               "Grouped FP8 row-block offsets must be Int64.");
    NVTE_CHECK(output->row_block_offsets.shape.size() == 2 &&
                   output->row_block_offsets.shape[0] >= num_tensors + 1,
               "Grouped FP8 row-block offsets must have shape [num_tensors + 1, total_tiles].");
    if (use_rowwise) {
      NVTE_CHECK(output->rowwise_scale_inv_offsets.dtype == DType::kInt64,
                 "Grouped FP8 rowwise scale_inv offsets must be Int64.");
      NVTE_CHECK(output->rowwise_scale_inv_offsets.shape.size() == 1 &&
                     output->rowwise_scale_inv_offsets.shape[0] >= num_tensors + 1,
                 "Grouped FP8 rowwise scale_inv offsets must have length num_tensors + 1.");
    }
    if (use_columnwise) {
      NVTE_CHECK(output->columnwise_scale_inv_offsets.dtype == DType::kInt64,
                 "Grouped FP8 columnwise scale_inv offsets must be Int64.");
      NVTE_CHECK(output->columnwise_scale_inv_offsets.shape.size() == 1 &&
                     output->columnwise_scale_inv_offsets.shape[0] >= num_tensors + 1,
                 "Grouped FP8 columnwise scale_inv offsets must have length num_tensors + 1.");
    }
  }

  const size_t rows_per_tensor = has_first_dims ? 0 : logical_first_dim / num_tensors;
  const size_t work_tiles_y =
      has_first_dims ? (use_offset_metadata
                            ? output->row_block_offsets.shape[1]
                            : (DIVUP(logical_first_dim, kBlockLen) + num_tensors - 1))
                     : (num_tensors * DIVUP(rows_per_tensor, kBlockLen));
  const size_t work_tiles_x = DIVUP(cols, kBlockLen);
  if (work_tiles_x == 0 || work_tiles_y == 0) {
    return;
  }
  const bool use_aligned_first_dim_kernel =
      has_first_dims && use_offset_metadata && (cols % kBlockLen == 0) &&
      (logical_first_dim % kBlockLen == 0) &&
      (output->row_block_offsets.shape[1] == logical_first_dim / kBlockLen);
  const bool use_aligned_uniform_kernel =
      !has_first_dims && (cols % kBlockLen == 0) && (rows_per_tensor % kBlockLen == 0);
  const bool use_aligned_2d_kernel = use_aligned_first_dim_kernel || use_aligned_uniform_kernel;
  const bool use_aligned_1d_kernel = use_aligned_first_dim_kernel || use_aligned_uniform_kernel;
  const dim3 grid(work_tiles_x, work_tiles_y, 1);
  const dim3 aligned_grid(work_tiles_x,
                          use_aligned_uniform_kernel ? DIVUP(rows_per_tensor, kBlockLen)
                                                     : work_tiles_y,
                          use_aligned_uniform_kernel ? num_tensors : 1);

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
          if constexpr (kIs2DScaling) {
            if (use_aligned_2d_kernel) {
              if (use_rowwise && use_columnwise) {
                launch_group_quantize_fp8_2d_block_scaling_aligned_register<IType, OType, true,
                                                                            true>(
                    force_pow_2_scales, aligned_grid, stream,
                    reinterpret_cast<const IType *>(input->data.dptr),
                    reinterpret_cast<OType *>(output->data.dptr),
                    reinterpret_cast<OType *>(output->columnwise_data.dptr),
                    reinterpret_cast<float *>(output->scale_inv.dptr),
                    reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                    logical_first_dim, rows_per_tensor, cols, first_dims_ptr, tensor_offsets_ptr,
                    row_block_offsets_ptr, rowwise_scale_inv_offsets_ptr,
                    columnwise_scale_inv_offsets_ptr, has_first_dims, epsilon, noop_ptr);
              } else if (use_rowwise) {
                launch_group_quantize_fp8_2d_block_scaling_aligned_register<IType, OType, true,
                                                                            false>(
                    force_pow_2_scales, aligned_grid, stream,
                    reinterpret_cast<const IType *>(input->data.dptr),
                    reinterpret_cast<OType *>(output->data.dptr), nullptr,
                    reinterpret_cast<float *>(output->scale_inv.dptr), nullptr, num_tensors,
                    logical_first_dim, rows_per_tensor, cols, first_dims_ptr, tensor_offsets_ptr,
                    row_block_offsets_ptr, rowwise_scale_inv_offsets_ptr, nullptr,
                    has_first_dims, epsilon, noop_ptr);
              } else {
                launch_group_quantize_fp8_2d_block_scaling_aligned_register<IType, OType, false,
                                                                            true>(
                    force_pow_2_scales, aligned_grid, stream,
                    reinterpret_cast<const IType *>(input->data.dptr), nullptr,
                    reinterpret_cast<OType *>(output->columnwise_data.dptr), nullptr,
                    reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                    logical_first_dim, rows_per_tensor, cols, first_dims_ptr, tensor_offsets_ptr,
                    row_block_offsets_ptr, nullptr, columnwise_scale_inv_offsets_ptr,
                    has_first_dims, epsilon, noop_ptr);
              }
            } else if (has_first_dims && use_offset_metadata) {
              // The register first-dims specialization assumes explicit packed row-block and
              // per-member scale offsets. Metadata-free C API calls use the generic kernel below,
              // which derives compact scale offsets while preserving per-group tile boundaries.
              group_quantize_fp8_2d_block_scaling_register_kernel<IType, OType>
                  <<<grid, kThreadsPerBlock, 0, stream>>>(
                      reinterpret_cast<const IType *>(input->data.dptr),
                      use_rowwise ? reinterpret_cast<OType *>(output->data.dptr) : nullptr,
                      use_columnwise ? reinterpret_cast<OType *>(output->columnwise_data.dptr)
                                     : nullptr,
                      use_rowwise ? reinterpret_cast<float *>(output->scale_inv.dptr) : nullptr,
                      use_columnwise
                          ? reinterpret_cast<float *>(output->columnwise_scale_inv.dptr)
                          : nullptr,
                      num_tensors, logical_first_dim, cols, first_dims_ptr, tensor_offsets_ptr,
                      row_block_offsets_ptr, rowwise_scale_inv_offsets_ptr,
                      columnwise_scale_inv_offsets_ptr, use_rowwise, use_columnwise, epsilon,
                      force_pow_2_scales, noop_ptr);
            } else {
              const size_t smem_bytes = kBlockLen * (kBlockLen + 1) * sizeof(IType);
              if (smem_bytes >= 48 * 1024) {
                cudaError_t err =
                    cudaFuncSetAttribute(
                        &group_quantize_fp8_block_scaling_kernel<kIs2DScaling, IType, OType>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
                NVTE_CHECK(err == cudaSuccess,
                           "Failed to set grouped FP8 block-scaling shared memory size.");
              }
              group_quantize_fp8_block_scaling_kernel<kIs2DScaling, IType, OType>
                  <<<grid, kThreadsPerBlock, smem_bytes, stream>>>(
                      reinterpret_cast<const IType *>(input->data.dptr),
                      use_rowwise ? reinterpret_cast<OType *>(output->data.dptr) : nullptr,
                      use_columnwise ? reinterpret_cast<OType *>(output->columnwise_data.dptr)
                                     : nullptr,
                      use_rowwise ? reinterpret_cast<float *>(output->scale_inv.dptr) : nullptr,
                      use_columnwise
                          ? reinterpret_cast<float *>(output->columnwise_scale_inv.dptr)
                          : nullptr,
                      num_tensors, logical_first_dim, cols, first_dims_ptr, tensor_offsets_ptr,
                      row_block_offsets_ptr, rowwise_scale_inv_offsets_ptr,
                      columnwise_scale_inv_offsets_ptr, has_first_dims, use_rowwise,
                      use_columnwise, epsilon, force_pow_2_scales, noop_ptr);
            }
          } else {
            const size_t smem_bytes = kBlockLen * (kBlockLen + 1) * sizeof(IType);
            using SMemVec = Vec<IType, kSharedVec>;
            const size_t aligned_smem_bytes =
                use_columnwise ? kBlockLen * kSharedCols * sizeof(SMemVec) : 0;
            if (use_aligned_1d_kernel) {
              if (use_rowwise && use_columnwise) {
                launch_group_quantize_fp8_1d_block_scaling_aligned<IType, OType, true, true>(
                    force_pow_2_scales, aligned_grid, aligned_smem_bytes, stream,
                    reinterpret_cast<const IType *>(input->data.dptr),
                    reinterpret_cast<OType *>(output->data.dptr),
                    reinterpret_cast<OType *>(output->columnwise_data.dptr),
                    reinterpret_cast<float *>(output->scale_inv.dptr),
                    reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                    logical_first_dim, rows_per_tensor, cols, first_dims_ptr, tensor_offsets_ptr,
                    row_block_offsets_ptr, rowwise_scale_inv_offsets_ptr,
                    columnwise_scale_inv_offsets_ptr, has_first_dims, epsilon, noop_ptr);
              } else if (use_rowwise) {
                launch_group_quantize_fp8_1d_block_scaling_aligned<IType, OType, true, false>(
                    force_pow_2_scales, aligned_grid, aligned_smem_bytes, stream,
                    reinterpret_cast<const IType *>(input->data.dptr),
                    reinterpret_cast<OType *>(output->data.dptr), nullptr,
                    reinterpret_cast<float *>(output->scale_inv.dptr), nullptr, num_tensors,
                    logical_first_dim, rows_per_tensor, cols, first_dims_ptr, tensor_offsets_ptr,
                    row_block_offsets_ptr, rowwise_scale_inv_offsets_ptr, nullptr,
                    has_first_dims, epsilon, noop_ptr);
              } else {
                launch_group_quantize_fp8_1d_block_scaling_aligned<IType, OType, false, true>(
                    force_pow_2_scales, aligned_grid, aligned_smem_bytes, stream,
                    reinterpret_cast<const IType *>(input->data.dptr), nullptr,
                    reinterpret_cast<OType *>(output->columnwise_data.dptr), nullptr,
                    reinterpret_cast<float *>(output->columnwise_scale_inv.dptr), num_tensors,
                    logical_first_dim, rows_per_tensor, cols, first_dims_ptr, tensor_offsets_ptr,
                    row_block_offsets_ptr, nullptr, columnwise_scale_inv_offsets_ptr,
                    has_first_dims, epsilon, noop_ptr);
              }
            } else {
              if (smem_bytes >= 48 * 1024) {
                cudaError_t err =
                    cudaFuncSetAttribute(
                        &group_quantize_fp8_block_scaling_kernel<kIs2DScaling, IType, OType>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
                NVTE_CHECK(err == cudaSuccess,
                           "Failed to set grouped FP8 block-scaling shared memory size.");
              }
              group_quantize_fp8_block_scaling_kernel<kIs2DScaling, IType, OType>
                  <<<grid, kThreadsPerBlock, smem_bytes, stream>>>(
                      reinterpret_cast<const IType *>(input->data.dptr),
                      use_rowwise ? reinterpret_cast<OType *>(output->data.dptr) : nullptr,
                      use_columnwise ? reinterpret_cast<OType *>(output->columnwise_data.dptr)
                                     : nullptr,
                      use_rowwise ? reinterpret_cast<float *>(output->scale_inv.dptr) : nullptr,
                      use_columnwise
                          ? reinterpret_cast<float *>(output->columnwise_scale_inv.dptr)
                          : nullptr,
                      num_tensors, logical_first_dim, cols, first_dims_ptr, tensor_offsets_ptr,
                      row_block_offsets_ptr, rowwise_scale_inv_offsets_ptr,
                      columnwise_scale_inv_offsets_ptr, has_first_dims, use_rowwise,
                      use_columnwise, epsilon, force_pow_2_scales, noop_ptr);
            }
          }
          ););
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace group_block_scaling
}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCK_SCALING_CUH_
