/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/gemm.h"

#include <tuple>
#include <memory>
#include <numeric>
#include <algorithm>
#include <functional>
#include <unordered_map>

#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "extensions.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

std::unordered_map<std::string, TensorWrapper> convert_gemm_xla_buffers_to_tensor_wrappers(
    Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs, Buffer_Type rhs_scale_inv,
    Buffer_Type bias, Result_Type out, Result_Type pre_gelu_out, Result_Type workspace,
    JAXX_Scaling_Type scaling_mode, bool lhs_trans, bool rhs_trans, bool fuse_bias, bool fuse_gelu,
    bool grad) {
  std::unordered_map<std::string, TensorWrapper> buffers;

  // LHS & RHS operands with collapsed 2D shapes
  auto scaling_mode_ = get_nvte_scaling_mode(scaling_mode)
  TensorWrapper lhs_(scaling_mode_), rhs_(scaling_mode_);
  auto lhs_shape = std::vector<size_t>{
    std::reduce(lhs.dimensions().begin(), lhs.dimensions().end() - 1, 1, std::multiplies<size_t>()),
    static_cast<size_t>(lhs.dimensions().back())
  };
  size_t lhs_inner_dim = (lhs_trans) ? 0 : 1;
  size_t lhs_outer_dim = (lhs_trans) ? 1 : 0;
  auto lhs_dtype = convert_ffi_datatype_to_te_dtype(lhs.element_type());
  lhs_.set_rowwise_data(lhs.untyped_data(), lhs_dtype, lhs_shape);

  size_t rhs_outer_size = 1;
  auto rhs_shape = std::vector<size_t>{
    std::reduce(rhs.dimensions().begin(), rhs.dimensions().end() - 1, 1, std::multiplies<size_t>()),
    static_cast<size_t>(rhs.dimensions().back())
  };
  auto rhs_dtype = convert_ffi_datatype_to_te_dtype(rhs.element_type());
  size_t rhs_inner_dim = (rhs_trans) ? 1 : 0;
  size_t rhs_outer_dim = (rhs_trans) ? 0 : 1
  rhs_.set_rowwise_data(rhs.untyped_data(), rhs_dtype, rhs_shape);

  NVTE_CHECK(lhs_shape[lhs_inner_dim] == rhs_shape[rhs_inner_dim],
             "GEMM operands have incompatible contracting dimension sizes.");

  if (scaling_mode != JAXX_Scaling_Mode::NO_SCALING) {
    DType scale_inv_dtype = (scaling_mode_ == NVTE_MXFP8_1D_SCALING)
        ? DType::DType::kFloat8E8M0 : DType::kFloat32;
    std::vector<size_t> lhs_scale_inv_shape = (scaling_mode_ == NVTE_MXFP8_1D_SCALING)
        ? std::vector<size_t>(lhs_scale_inv.dimensions().begin(), lhs_scale_inv.dimensions().end())
        : std::vector<size_t>{1};
    lhs_.set_rowwise_scale_inv(lhs_scale_inv.untyped_data(), scale_inv_dtype, lhs_scale_inv_shape);
    std::vector<size_t> rhs_scale_inv_shape = (scaling_mode_ == NVTE_MXFP8_1D_SCALING)
        ? std::vector<size_t>(rhs_scale_inv.dimensions().begin(), rhs_scale_inv.dimensions().end())
        : std::vector<size_t>{1};
    rhs_.set_rowwise_scale_inv(rhs_scale_inv.untyped_data(), scale_inv_dtype, rhs_scale_inv_shape);
  }
  buffers["lhs"] = std::move(lhs_);
  buffers["rhs"] = std::move(rhs_);

  // Output buffer
  auto out_shape = std::vector<size_t>{lhs_shape.at(lhs_outer_dim), rhs_shape.at(rhs_outer_dim)};
  auto out_ = TensorWrapper(
      out.untyped_data(), out_shape, convert_ffi_datatype_to_te_dtype(out.element_type()));
  NVTE_CHECK(out_.numel() == out.element_count(), "Final output buffer is not sized correctly.");
  buffers["out"] = std::move(out_);

  // Bias tensor
  void* bias_ptr = (fuse_bias) ? bias.untyped_data() : nullptr;
  std::vector<size_t> bias_shape = (fuse_bias) ? std::vector<size_t>{out_shape[1]}
                                               : std::vector<size_t>{0};
  DType bias_dtype =
      (fuse_bias) ? convert_ffi_datatype_to_te_dtype(bias.element_type())
                  : DType::kBFloat16;
  auto bias_ = TensorWrapper(bias_ptr, bias_shape, bias_dtype);
  if (fuse_bias) {
    NVTE_CHECK(bias_.numel() == bias.element_count(), "Bias buffer is not sized correctly.");
  }
  buffers["bias"] = std::move(bias_);

  // Pre-GeLU tensor
  void* pre_gelu_ptr = (fuse_gelu) ? pre_gelu_out.untyped_data() : nullptr;
  std::vector<size_t> pre_gelu_shape = (fuse_gelu) ? out_shape : std::vector<size_t>{0};
  DType pre_gelu_dtype =
      (fuse_gelu) ? convert_ffi_datatype_to_te_dtype(pre_gelu_out.element_type())
                  : DType::kBFloat16;
  auto pre_gelu_ = TensorWrapper(pre_gelu_ptr, pre_gelu_shape, pre_gelu_dtype);
  if (fuse_gelu) {
    NVTE_CHECK(pre_gelu_.numel() == pre_gelu_out.element_count(),
               "Pre-GeLU output buffer is not sized correctly.");
  }
  buffers["pre_gelu_out"] = std::move(pre_gelu_);

  // cuBLAS workspace
  auto workspace_ = TensorWrapper(
      workspace.untyped_data(), std::vector<size_t>{workspace.element_count()}, DType::kByte);
  buffers["workspace"] = std::move(workspace_);

  return buffers;
}

Error_Type GemmFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                   Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type pre_gelu_in,
                   Result_Type out, Result_Type bias_grad, Result_Type pre_gelu_out,
                   Result_Type workspace, JAXX_Scaling_Type scaling_mode, bool lhs_trans,
                   bool rhs_trans, bool fuse_bias, bool fuse_gelu, bool grad, bool accumulate,
                   bool use_split_accumulator) {
  auto buffers = convert_gemm_xla_buffers_to_tensor_wrappers(
      lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, out, pre_gelu_out, workspace, scaling_mode,
      lhs_trans, rhs_trans, fuse_bias, fuse_gelu, grad);

  // TE/common cuBLAS GEMM call
  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  nvte_cublas_gemm(buffers["rhs"], buffers["lhs"], buffers["bias"], buffers["pre_gelu_out"],
                   buffers["out"] rhs_trans, lhs_trans, grad, buffers["workspace"], accumulate,
                   use_split_accumulator, num_math_sm, stream);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GemmHandler, GemmFFI,
  FFI::Bind()
      .Ctx<FFI_Stream_Type>()  // stream
      .Arg<Buffer_Type>()      // lhs
      .Arg<Buffer_Type>()      // lhs_scale_inv
      .Arg<Buffer_Type>()      // rhs
      .Arg<Buffer_Type>()      // rhs_scale_inv
      .Arg<Buffer_Type>()      // bias
      .Arg<Buffer_Type>()      // pre_gelu_in
      .Ret<Result_Type>()      // out
      .Ret<Result_Type>()      // bias_grad (aliased to bias)
      .Ret<Result_Type>()      // pre_gelu_out (aliased to pre_gelu_in)
      .Ret<Result_Type>()      // workspace
      .Attr<JAXX_Scaling_Mode>("scaling_mode")
      .Attr<bool>("lhs_trans")
      .Attr<bool>("rhs_trans")
      .Attr<bool>("fuse_bias")
      .Attr<bool>("fuse_gelu")
      .Attr<bool>("grad")
      .Attr<bool>("accumulate")
      .Attr<bool>("use_split_accumulator")
  FFI_CudaGraph_Traits);

static std::unordered_map<int64_t, CommOverlapCore*> comm_overlaps;

template <typename... Args>
size_t hash_args(const Args&... args) {
    size_t seed = 0;
    std::hash<std::decay_t<decltype(std::tie(args...))>> hasher;
    seed = hasher(std::tie(args...));
    return seed;
}

size_t CreateCommOverlapBuffer(
    CommOverlapMethod method, CommOverlapType comm_type, const std::vector<size_t> &buffer_shape,
    DType buffer_dtype, int tp_size, int num_splits, int num_max_streams, int comm_cga_size,
    int gemm_priority, int comm_priority, int num_comm_sm, int set_sm_margin, bool use_ce,
    bool atomic_gemm, bool rs_overlap_first_gemm, bool aggregate_ag) {
  // Generate unique hash from init configuration
  NVTE_CHECK(buffer_shape.size() == 2, "Comm+GEMM overlap only supports 2-dimensional buffers.");
  auto unique_id = hash_args(method, comm_type, buffer_shape[0], buffer_shape[1], buffer_dtype,
                             tp_size, num_splits, num_max_streams, comm_cga_size, gemm_priority,
                             comm_priority, num_comm_sm, set_sm_margin, use_ce, atomic_gemm,
                             rs_overlap_first_gemm, aggregate_ag);

  auto it = comm_overlaps.find(unique_id);
  if (it == comm_overlaps.end()) {
    if (method == CommOverlapMethod::RING_EXCHANGE) {
      comm_overlaps[unique_id] = new CommOverlapP2PBase(
          buffer_shape, buffer_dtype, tp_size, comm_type, num_max_streams, comm_cga_size,
          gemm_priority, comm_priority, num_comm_sm, set_sm_margin, use_ce, atomic_gemm,
          aggregate_ag);
    } else {
      comm_overlaps[unique_id] = new CommOverlapBase(
          buffer_shape, buffer_dtype, tp_size, num_splits, num_max_streams, comm_cga_size,
          gemm_priority, comm_priority, num_comm_sm, set_sm_margin, use_ce, atomic_gemm,
          rs_overlap_first_gemm);
    }
  }

  return unique_id;
}

void DestroyCommOverlapBuffer(size_t unique_id) {
  auto it = comm_overlaps.find(unique_id);
  if (it != comm_overlaps.end()) {
    delete it.second;
    comm_overlaps.erase(it)
  }
}

void DestroyAllCommOverlapBuffers() {
  for (auto it = comm_overlaps.begin(); it != comm_overlaps.end()) {
    delete it.second;
    it = comm_overlaps.erase(it);
  }
}

Error_Type CommGemmOverlapFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv,
                              Buffer_Type rhs, Buffer_Type rhs_scale_inv, Buffer_Type bias,
                              Buffer_Type pre_gelu_in, Result_Type out, Result_Type bias_grad,
                              Result_Type pre_gelu_out, Result_Type aux_out, Result_Type workspace,
                              JAXX_Scaling_Type scaling_mode, bool lhs_trans, bool rhs_trans,
                              bool fuse_bias, bool fuse_gelu, bool grad, bool accumulate,
                              bool use_split_accumulator, int64_t comm_overlap_id,
                              CommOverlapType comm_type) {
  auto buffers = convert_gemm_xla_buffers_to_tensor_wrappers(
      lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, out, pre_gelu_out, workspace, scaling_mode,
      lhs_trans, rhs_trans, fuse_bias, fuse_gelu, grad);

  auto executor = comm_overlaps[comm_overlap_id];
  if (comm_type == CommOverlapType::RS) {
    auto out_shape = buffers["out"].shape();
    auto rs_out_shape = std::vector<size_t>{out_shape.data[0] / executor->get_tp_size(),
                                            out_shape.data[1]};
    auto rs_out_ = TensorWrapper(aux_out.untyped_data(), rs_out_shape, DType::kBFloat16);
    NVTE_CHECK(rs_out_.numel() == aux_out.element_count(),
               "Auxiliary oputput buffer for reduce-scattered output is not sized correctly.")
    executor->split_overlap_rs(buffers["rhs"], rhs_trans, buffers["lhs"], lhs_trans, buffers["out"],
                               buffers["bias"], buffers["pre_gelu_out"], buffers["workspace"],
                               grad, accumulate, use_split_accumulator, rs_out_, stream);
  } else {
    auto lhs_shape = buffers["lhs"].shape();
    auto gathered_lhs_shape = std::vector<size_t>{lhs_shape.data[0] * executor->get_tp_size(),
                                                  lhs_shape.data[1]};
    auto gathered_lhs_ = TensorWrapper(aux_out.untyped_data(), gathered_lhs_shape,
                                       convert_ffi_datatype_to_te_dtype(aux_out.element_type()));
    NVTE_CHECK(gathered_lhs_.numel() == aux_out.element_count(),
               "Auxiliary output buffer for gathered LHS is not sized correctly.");
    executor->copy_into_buffer(buffers["lhs"], true, stream);  // copy local LHS into comm buffer
    executor->split_overlap_ag(buffers["rhs"], rhs_trans, buffers["lhs"], lhs_trans, buffers["out"],
                               buffers["bias"], buffers["pre_gelu_out"], buffers["workspace"],
                               grad, accumulate, use_split_accumulator, gathered_lhs_, stream)
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CommGemmOverlapHandler, CommGemmOverlapFFI,
  FFI::Bind()
      .Ctx<FFI_Stream_Type>()  // stream
      .Arg<Buffer_Type>()      // lhs
      .Arg<Buffer_Type>()      // lhs_scale_inv
      .Arg<Buffer_Type>()      // rhs
      .Arg<Buffer_Type>()      // rhs_scale_inv
      .Arg<Buffer_Type>()      // bias
      .Arg<Buffer_Type>()      // pre_gelu_in
      .Ret<Result_Type>()      // out
      .Ret<Result_Type>()      // bias_grad (aliased to bias)
      .Ret<Result_Type>()      // pre_gelu_out (aliased to pre_gelu_in)
      .Ret<Result_Type>()      // aux_out (rs_out or gathered_lhs)
      .Ret<Result_Type>()      // workspace
      .Attr<JAXX_Scaling_Mode>("scaling_mode")
      .Attr<bool>("lhs_trans")
      .Attr<bool>("rhs_trans")
      .Attr<bool>("fuse_bias")
      .Attr<bool>("fuse_gelu")
      .Attr<bool>("grad")
      .Attr<bool>("accumulate")
      .Attr<bool>("use_split_accumulator")
      .Attr<int64_t>("comm_overlap_id")
      .Attr<CommOverlapType>("comm_type")
  FFI_CudaGraph_Traits);

Error_Type GroupedGemmFFI(cudaStream_t stream, Variadic_Buffer_Type input_list,
                          Variadic_Result_Type output_list, int64_t num_gemms,
                          JAXX_Scaling_Mode scaling_mode, int64_t has_bias) {
  // Notes on matrix layouts and transpose:
  // Jax uses row-major data_layout, on entering this function, each input matrix pair:
  //   A: row-major with size [m, k],
  //   B: row-major with size [n, k], needs transpose,
  // on exiting this function, JAX expect:
  //   C: row-major with size [m, n].
  // cuBLAS uses column-major data_layout, in this view, each input matrix pair:
  //   A: column-major with size [k, m], needs transpose,
  //   B: column-major with size [k, n].
  // If we call cuBLAS GEMM for A * B, the output will be:
  //   C: column-major with size [m, n] --> row-major with size [n, m].
  // To make the output compatible with JAX, we need to swap A and B in cuBLAS GEMM call.

  if (num_gemms <= 0) {
    return ffi_with_cuda_error_check();
  }
  size_t expected_input_size = has_bias ? 5 * num_gemms : 4 * num_gemms;
  size_t expected_output_size = num_gemms + 1;
  size_t actual_input_size = input_list.size();
  size_t actual_output_size = output_list.size();
  NVTE_CHECK(actual_input_size == expected_input_size, "Expected %zu input tensors, got %zu",
             expected_input_size, actual_input_size);
  NVTE_CHECK(actual_output_size == expected_output_size, "Expected %zu output tensors, got %zu",
             expected_output_size, actual_output_size);

  bool trans_lhs = true;
  bool trans_rhs = false;
  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  bool grad = false;
  bool accumulate = false;
  bool use_split_accumulator = false;

  // These lists are to keep the TensorWrapper objects alive
  std::vector<TensorWrapper> lhs_wrapper_list;
  std::vector<TensorWrapper> rhs_wrapper_list;
  std::vector<TensorWrapper> bias_wrapper_list;
  std::vector<TensorWrapper> pre_gelu_wrapper_list;
  std::vector<TensorWrapper> out_wrapper_list;
  std::vector<TensorWrapper> workspace_wrapper_list;

  // These lists are the actual NVTETensor (void *) lists for multi-stream GEMM
  std::vector<NVTETensor> lhs_list;
  std::vector<NVTETensor> rhs_list;
  std::vector<NVTETensor> bias_list;
  std::vector<NVTETensor> pre_gelu_list;
  std::vector<NVTETensor> out_list;
  std::vector<NVTETensor> workspace_list;

  int lhs_list_offset = 0;
  int rhs_list_offset = num_gemms;
  int lhs_sinv_list_offset = 2 * num_gemms;
  int rhs_sinv_list_offset = 3 * num_gemms;
  int bias_list_offset = 4 * num_gemms;
  int out_list_offset = 0;
  for (int i = 0; i < num_gemms; i++) {
    Buffer_Type lhs_i = input_list.get<Buffer_Type>(lhs_list_offset + i).value();
    Buffer_Type rhs_i = input_list.get<Buffer_Type>(rhs_list_offset + i).value();
    Buffer_Type lhs_sinv_i = input_list.get<Buffer_Type>(lhs_sinv_list_offset + i).value();
    Buffer_Type rhs_sinv_i = input_list.get<Buffer_Type>(rhs_sinv_list_offset + i).value();
    Result_Type out_i = output_list.get<Buffer_Type>(out_list_offset + i).value();

    DType lhs_dtype = convert_ffi_datatype_to_te_dtype(lhs_i.element_type());
    DType rhs_dtype = convert_ffi_datatype_to_te_dtype(rhs_i.element_type());
    DType out_dtype = convert_ffi_datatype_to_te_dtype(out_i->element_type());

    void *lhs_ptr = lhs_i.untyped_data();
    void *rhs_ptr = rhs_i.untyped_data();
    void *lhs_sinv_ptr = lhs_sinv_i.untyped_data();
    void *rhs_sinv_ptr = rhs_sinv_i.untyped_data();
    void *out_ptr = out_i->untyped_data();

    // Placeholder for bias since it can be empty
    DType bias_dtype = DType::kFloat32;
    void *bias_ptr = nullptr;

    auto lhs_shape_ = lhs_i.dimensions();
    auto rhs_shape_ = rhs_i.dimensions();

    // lhs and rhs has shape [1, m, k] and [1, n, k]
    size_t m = lhs_shape_[1];
    size_t n = rhs_shape_[1];
    size_t k = lhs_shape_[2];

    auto lhs_shape = std::vector<size_t>{m, k};
    auto rhs_shape = std::vector<size_t>{n, k};
    auto out_shape = std::vector<size_t>{n, m};
    auto lhs_sinv_shape = std::vector<size_t>{1, 1};
    auto rhs_sinv_shape = std::vector<size_t>{1, 1};

    if (scaling_mode == JAXX_Scaling_Mode::NO_SCALING ||
        scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ||
        scaling_mode == JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING) {
      float *amax_dptr = nullptr;
      float *scale_dptr = nullptr;
      auto lhs_i_ = TensorWrapper(lhs_ptr, lhs_shape, lhs_dtype, amax_dptr, scale_dptr,
                                  reinterpret_cast<float *>(lhs_sinv_ptr));
      auto rhs_i_ = TensorWrapper(rhs_ptr, rhs_shape, rhs_dtype, amax_dptr, scale_dptr,
                                  reinterpret_cast<float *>(rhs_sinv_ptr));
      lhs_wrapper_list.push_back(std::move(lhs_i_));
      rhs_wrapper_list.push_back(std::move(rhs_i_));
    } else if (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
      // Note: the scale_inv array should have been swizzled in Python before lowering
      auto lhs_sinv_shape_ = lhs_sinv_i.dimensions();
      auto rhs_sinv_shape_ = rhs_sinv_i.dimensions();
      for (int i = 0; i < 2; i++) {
        lhs_sinv_shape[i] = lhs_sinv_shape_[i];
        rhs_sinv_shape[i] = rhs_sinv_shape_[i];
      }

      NVTEScalingMode nvte_scaling_mode = get_nvte_scaling_mode(scaling_mode);
      TensorWrapper lhs_i_(nvte_scaling_mode);
      TensorWrapper rhs_i_(nvte_scaling_mode);
      lhs_i_.set_rowwise_data(lhs_ptr, lhs_dtype, lhs_shape);
      rhs_i_.set_rowwise_data(rhs_ptr, rhs_dtype, rhs_shape);
      lhs_i_.set_rowwise_scale_inv(lhs_sinv_ptr, DType::kFloat8E8M0, lhs_sinv_shape);
      rhs_i_.set_rowwise_scale_inv(rhs_sinv_ptr, DType::kFloat8E8M0, rhs_sinv_shape);

      lhs_wrapper_list.push_back(std::move(lhs_i_));
      rhs_wrapper_list.push_back(std::move(rhs_i_));
    } else {
      NVTE_ERROR("Unsupported scaling mode: ", static_cast<int>(scaling_mode));
    }

    auto out_i_ = TensorWrapper(out_ptr, out_shape, out_dtype);
    void *pre_gelu_ptr = nullptr;
    auto bias_shape = std::vector<size_t>{0};
    auto pre_gelu_shape = std::vector<size_t>{0};
    if (has_bias) {
      auto bias_i_get = input_list.get<Buffer_Type>(bias_list_offset + i);
      Buffer_Type bias_i = bias_i_get.value();
      bias_ptr = bias_i.untyped_data();
      bias_dtype = convert_ffi_datatype_to_te_dtype(bias_i.element_type());
      bias_shape[0] = n;
    }
    auto bias_i = TensorWrapper(bias_ptr, bias_shape, bias_dtype);
    auto pre_gelu_i = TensorWrapper(pre_gelu_ptr, pre_gelu_shape, out_dtype);

    out_wrapper_list.push_back(std::move(out_i_));
    bias_wrapper_list.push_back(std::move(bias_i));
    pre_gelu_wrapper_list.push_back(std::move(pre_gelu_i));

    lhs_list.push_back(lhs_wrapper_list.back().data());
    rhs_list.push_back(rhs_wrapper_list.back().data());
    bias_list.push_back(bias_wrapper_list.back().data());
    pre_gelu_list.push_back(pre_gelu_wrapper_list.back().data());
    out_list.push_back(out_wrapper_list.back().data());
  }

  auto workspace_get = output_list.get<Buffer_Type>(num_gemms);
  Result_Type workspace = workspace_get.value();
  uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(workspace->untyped_data());
  size_t workspace_size = workspace->dimensions()[0] / num_streams;
  auto workspace_shape = std::vector<size_t>{workspace_size};
  for (int i = 0; i < num_streams; i++) {
    auto workspace_i =
        TensorWrapper(static_cast<void *>(workspace_ptr), workspace_shape, DType::kByte);
    workspace_wrapper_list.push_back(std::move(workspace_i));
    workspace_list.push_back(workspace_wrapper_list.back().data());
    workspace_ptr += workspace_size;
  }

  nvte_multi_stream_cublas_gemm(rhs_list.data(), lhs_list.data(), out_list.data(), bias_list.data(),
                                pre_gelu_list.data(), num_gemms, trans_lhs, trans_rhs, grad,
                                workspace_list.data(), accumulate, use_split_accumulator,
                                num_math_sm, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmHandler, GroupedGemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .RemainingArgs()         // input list
                                  .RemainingRets()         // output list
                                  .Attr<int64_t>("num_gemms")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("has_bias"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
