/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8_blockwise.cu
 *  \brief Generic-architecture instantiation for grouped FP8 block-scaled quantize kernels.
 */

#include "group_quantize_fp8_blockwise.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8_blockwise {

void group_quantize_1d(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                       const QuantizationConfig *quant_config, cudaStream_t stream) {
  group_quantize</*IS_2D=*/false>(input, noop, output, quant_config, stream);
}

void group_quantize_2d(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                       const QuantizationConfig *quant_config, cudaStream_t stream) {
  group_quantize</*IS_2D=*/true>(input, noop, output, quant_config, stream);
}

}  // namespace fp8_blockwise
}  // namespace dispatch
}  // namespace transformer_engine
