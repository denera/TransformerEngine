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
    cudaStream_t stream, void *A, DType A_dtype, float *A_scale_inv, bool A_trans, void *B,
    DType B_dtype, float *B_scale_inv, bool B_trans, void *bias, DType bias_dtype,
    void *pre_gelu_out, void *D, DType D_type, float *D_amax, float *D_scale, void *workspace,
    size_t workspace_size, bool grad, bool accumulate, bool use_split_accumulator, int sm_count,
    size_t m, size_t k, size_t n) {
  auto A_ = TensorWrapper(A, {m, k}, A_dtype, nullptr, nullptr, A_scale_inv);
  auto B_ = TensorWrapper(B, {k, n}, B_dtype, nullptr, nullptr, B_scale_inv);
  auto D_ = TensorWrapper(D, {m, n}, D_dtype, D_amax, D_scale, nullptr);
  auto bias_ = TensorWrapper(bias, std::vector<size_t>{n}, bias_dtype);
  std::vector<size_t> pre_gelu_shape = (pre_gelu_out == nullptr) ? std::vector<size_t>{0}
                                                                 : {m, n};
  auto pre_gelu_out_ = TensorWrapper(pre_gelu_out_, pre_gelu_shape, bias_dtype);
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
  auto m = (desc.A_trans)
         ? desc.A_shape.back()
         : std::accumulate(desc.A_shape.begin(), desc.A_shape.end() - 1, 1, std::multiplies<>());
  auto k = (desc.A_trans) ? desc.B_shape.back() : desc.A_shape.back();
  auto n = (desc.B_trans)
         ? desc.B_shape.back()
         : std::accumulate(desc.B_shape.begin(), desc.B_shape.end() - 1, 1, std::multiplies<>());

  GemmImpl(stream, A, desc.A_dtype, A_scale_inv, desc.A_trans, B, desc.B_dtype, B_scale_inv,
           desc.B_trans, bias, desc.bias_dtype, pre_gelu_out, D, desc.D_dtype, D_amax, D_scale,
           workspace, desc.workspace_size, desc.grad, desc.accumulate, desc.use_split_accumulator,
           desc.sm_count, m, n, k);
}

Error_Type GemmFFI(
    cudaStream_t stream, Buffer_Type A, Buffer_Type A_scale_inv, Buffer_Type B,
    Buffer_Type B_scale_inv, Buffer_Type bias, Return_Type pre_gelu_out, Return_Type D,
    Return_Type D_amax, Return_Type D_scale, Return_Type workspace, bool A_trans, bool B_trans,
    bool grad, bool accumulate, bool use_split_accumulator, int sm_count) {
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
  auto A_dims = A.dimensions();
  auto B_dims = B.dimensions();
  auto m = (A_trans) ? A_dims.back()
                     : std::accumulate(A_dims.begin(), A_dims.end() - 1, 1, std::multiplies<>());
  auto k = (A_trans) ? A_dims.front() : A_dims.back();
  auto n = (B_trans) ? B_dims.back()
                     : std::accumulate(B_dims.begin(), B_dims.end() - 1, 1, std::multiplies<>());
  auto workspace_size = workspace_ptr.front();

  GemmImpl(stream, A_ptr, A_dtype, A_scale_inv_ptr, A_trans, B_ptr, B_dtype, B_scale_inv_ptr,
           B_trans, bias_ptr, bias_dtype, pre_gelu_ptr, D_ptr, D_dtype, D_amax_ptr, D_scale_ptr,
           workspace_ptr, workspace_size, grad, accumulate, use_split_accumulator, sm_count,
           m, n, k);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GemmHandler, GemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // A
                                  .Arg<Buffer_Type>()      // A_scale_inv
                                  .Arg<Buffer_Type>()      // B
                                  .Arg<Buffer_Type>()      // B_scale_inv
                                  .Ret<Buffer_Type>()      // bias
                                  .Ret<Return_Type>()      // pre_gelu_out
                                  .Ret<Return_Type>()      // D
                                  .Ret<Return_Type>()      // D_amax
                                  .Ret<Return_Type>()      // D_scale
                                  .Ret<Return_Type>()      // workspace
                                  .Attr<bool>("A_trans")
                                  .Attr<bool>("B_trans")
                                  .Attr<bool>("grad")
                                  .Attr<bool>("accumulate")
                                  .Attr<bool>("use_split_accumulator")
                                  .Attr<int32_t>("sm_count"),
                              FFI_CudaGraph_Traits);

}  // namespace jax

}  // namespace transformer_engine
