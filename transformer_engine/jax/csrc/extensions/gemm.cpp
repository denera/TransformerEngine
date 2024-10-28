/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"
#include "transformer_engine/gemm.h"

namespace transformer_engine {

namespace jax {

void GemmImpl(
    cudaStream_t stream, void *A, const std::vector<size_t> &A_shape, DType A_dtype,
    float *A_scale_inv, bool A_trans, void *B, const std::vector<size_t> &B_shape, DType B_dtype,
    float *B_scale_inv, bool B_trans, void *bias, DType bias_dtype, void *pre_gelu_out, void *D,
    DType D_dtype, float *D_amax, float *D_scale, void *workspace, size_t workspace_size, bool grad,
    bool accumulate, bool use_split_accumulator, int sm_count) {
  std::vector<size_t> D_shape(2, 0);
  D_shape[0] = (B_trans) ? B_shape[1] : B_shape[0];
  D_shape[1] = (A_trans) ? A_shape[0] : A_shape[1];
  auto A_ = TensorWrapper(A, A_shape, A_dtype, nullptr, nullptr, A_scale_inv);
  auto B_ = TensorWrapper(B, B_shape, B_dtype, nullptr, nullptr, B_scale_inv);
  auto D_ = TensorWrapper(D, D_shape, D_dtype, D_amax, D_scale, nullptr);
  std::vector<size_t> bias_shape = (bias == nullptr) ? std::vector<size_t>{0}
                                                     : std::vector<size_t>{D_shape[1]};
  auto bias_ = TensorWrapper(bias, bias_shape, bias_dtype);
  std::vector<size_t> pre_gelu_shape = (pre_gelu_out == nullptr) ? std::vector<size_t>{0}
                                                                 : D_shape;
  auto pre_gelu_out_ = TensorWrapper(pre_gelu_out, pre_gelu_shape, bias_dtype);
  auto workspace_ = TensorWrapper(workspace, std::vector<size_t>{workspace_size}, DType::kByte);
  nvte_cublas_gemm(A_.data(), B_.data(), D_.data(), bias_.data(), pre_gelu_out_.data(),
                   (A_trans) ? CUBLAS_OP_T : CUBLAS_OP_N, (B_trans) ? CUBLAS_OP_T : CUBLAS_OP_N,
                   grad, workspace_.data(), accumulate, use_split_accumulator, sm_count, stream);
}

void Gemm(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  // Inputs
  auto *A = buffers[0];
  auto *B = buffers[1];
  auto *A_scale_inv = reinterpret_cast<float *>(buffers[2]);
  auto *B_scale_inv = reinterpret_cast<float *>(buffers[3]);
  auto *bias = buffers[4];

  // Outputs
  auto *pre_gelu_out = buffers[5];
  auto *D = buffers[6];
  auto *D_amax = reinterpret_cast<float *>(buffers[7]);
  auto *D_scale = reinterpret_cast<float *>(buffers[8]);
  auto *workspace = buffers[9];

  // GEMM sizing
  const auto &desc = *UnpackOpaque<CustomCallGemmDescriptor>(opaque, opaque_len);
  std::vector<size_t> A_shape = {(desc.A_trans) ? desc.k : desc.m,
                                 (desc.A_trans) ? desc.m : desc.k};
  std::vector<size_t> B_shape = {(desc.B_trans) ? desc.k : desc.n,
                                 (desc.B_trans) ? desc.n : desc.k};

  GemmImpl(stream, A, A_shape, desc.A_dtype, A_scale_inv, desc.A_trans, B, B_shape,
           desc.B_dtype, B_scale_inv, desc.B_trans, bias, desc.bias_dtype, pre_gelu_out, D,
           desc.D_dtype, D_amax, D_scale, workspace, desc.workspace_size, desc.grad,
           desc.accumulate, desc.use_split_accumulator, desc.sm_count);
}

Error_Type GemmFFI(
    cudaStream_t stream, Buffer_Type A, Buffer_Type A_scale_inv, Buffer_Type B,
    Buffer_Type B_scale_inv, Buffer_Type bias, Result_Type pre_gelu_out, Result_Type D,
    Result_Type D_amax, Result_Type D_scale, Result_Type workspace, int32_t workspace_size,
    bool A_trans, bool B_trans, bool grad, bool accumulate, bool use_split_accumulator,
    int32_t sm_count) {
  // Inputs
  auto A_ptr = A.untyped_data();
  auto A_dtype = convert_ffi_datatype_to_te_dtype(A.element_type());
  auto A_scale_inv_ptr = reinterpret_cast<float *>(A_scale_inv.untyped_data());
  auto B_ptr = B.untyped_data();
  auto B_dtype = convert_ffi_datatype_to_te_dtype(B.element_type());
  auto B_scale_inv_ptr = reinterpret_cast<float *>(B_scale_inv.untyped_data());
  auto bias_ptr = bias.untyped_data();
  auto bias_dtype = convert_ffi_datatype_to_te_dtype(bias.element_type());

  // Outputs
  auto pre_gelu_ptr = pre_gelu_out->untyped_data();
  auto D_ptr = D->untyped_data();
  auto D_dtype = convert_ffi_datatype_to_te_dtype(D->element_type());
  auto D_amax_ptr = reinterpret_cast<float *>(D_amax->untyped_data());
  auto D_scale_ptr = reinterpret_cast<float *>(D_scale->untyped_data());
  auto workspace_ptr = workspace->untyped_data();

  // GEMM sizing
  auto A_shape = std::vector<size_t>(A.dimensions().begin(), A.dimensions().end());
  auto B_shape = std::vector<size_t>(B.dimensions().begin(), B.dimensions().end());

  GemmImpl(stream, A_ptr, A_shape, A_dtype, A_scale_inv_ptr, A_trans, B_ptr, B_shape,
           B_dtype, B_scale_inv_ptr, B_trans, bias_ptr, bias_dtype, pre_gelu_ptr, D_ptr, D_dtype,
           D_amax_ptr, D_scale_ptr, workspace_ptr, workspace_size, grad, accumulate,
           use_split_accumulator, sm_count);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GemmHandler, GemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // A
                                  .Arg<Buffer_Type>()      // A_scale_inv
                                  .Arg<Buffer_Type>()      // B
                                  .Arg<Buffer_Type>()      // B_scale_inv
                                  .Ret<Buffer_Type>()      // bias
                                  .Ret<Result_Type>()      // pre_gelu_out
                                  .Ret<Result_Type>()      // D
                                  .Ret<Result_Type>()      // D_amax
                                  .Ret<Result_Type>()      // D_scale
                                  .Ret<Result_Type>()      // workspace
                                  .Attr<int32_t>("workspace_size")
                                  .Attr<bool>("A_trans")
                                  .Attr<bool>("B_trans")
                                  .Attr<bool>("grad")
                                  .Attr<bool>("accumulate")
                                  .Attr<bool>("use_split_accumulator")
                                  .Attr<int32_t>("sm_count"),
                              FFI_CudaGraph_Traits);

}  // namespace jax

}  // namespace transformer_engine
