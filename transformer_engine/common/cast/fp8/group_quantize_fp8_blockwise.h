/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8_blockwise.h
 *  \brief Host dispatch entry points for grouped FP8 block-scaled quantize kernels.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCKWISE_H_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCKWISE_H_

#include <cuda_runtime.h>

#include "../../common.h"

namespace transformer_engine {
namespace dispatch {
namespace fp8_blockwise {

void group_quantize_1d(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                       const QuantizationConfig *quant_config, cudaStream_t stream);

void group_quantize_2d(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                       const QuantizationConfig *quant_config, cudaStream_t stream);

}  // namespace fp8_blockwise
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCKWISE_H_
