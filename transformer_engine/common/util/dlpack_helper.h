/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_DLPACK_HELPER_H
#define TRANSFORMER_ENGINE_COMMON_UTIL_DLPACK_HELPER_H

#include <pybind11/pybind11.h>
#include <dlpack/dlpack.h>

#include <transformer_engine/transformer_engine.h>

#include "cuda_runtime.h"

namespace transformer_engine {

DLDataType * nvte_dtype_to_dldtype(DType dtype);

DType dldtype_to_nvte_dtype(const DLDataType &dtype);

class DLPackWrapper : public TensorWrapper {
 public:
  DLManagedTensor managed_tensor;

  DLPackWrapper() : TensorWrapper() {}

  DLPackWrapper(void *dptr, const std::vector<size_t> &shape, const DType dtype)
      : TensorWrapper(dptr, shape, dtype) {}

  DLPackWrapper(TensorWrapper &&other) : TensorWrapper(other) {}

  DLPackWrapper(pybind11::object obj) {
    NVTE_CHECK(PyCapsule_CheckExact(obj.ptr()), "Expected DLPack capsule");

    DLManagedTensor* dlMTensor =
        (DLManagedTensor*)PyCapsule_GetPointer(obj.ptr(), "dltensor");
    NVTE_CHECK(dlMTensor, "Invalid DLPack capsule.");

    DLTensor* dlTensor = &dlMTensor->dl_tensor;
    NVTE_CHECK(dlTensor->device.device_type == DLDeviceType::kDLCUDA,
               "DLPack tensor is not on a CUDA device.", dlTensor->device.device_type);
    NVTE_CHECK(dlTensor->device.device_id == cuda::current_device(),
               "DLPack tensor resides on a different device.");

    if (dlTensor->strides) {
      for (int idx = dlTensor->ndim - 1; idx >= 0; ++idx) {
        NVTE_CHECK(dlTensor->strides[idx] == 1,
                   "DLPack tensors with non-standard strides are not supported.");
      }
    }

    NVTEShape shape;
    shape.data = const_cast<size_t*>(dlTensor->shape);
    shape.ndim = static_cast<size_t>(dlTensor->ndim);
    tensor_ = nvte_create_tensor(dlTensor->data, shape, dldtype_to_nvte_dtype(dlTensor->dtype),
                                 nullptr, nullptr, nullptr);
  }

  pybind11::object capsule() {
    DLDevice tensor_context;
    auto device = ;
    tensor_context.device_type = DLDeviceType::kDLCUDA;
    tensor_context.device_id = cuda::current_device();

    DLTensor dlTensor;
    dlTensor.data = dptr();
    dlTensor.device = tensor_context;
    dlTensor.ndim = ndim();
    dlTensor.dtype = nvte_dtype_to_dldtype(dtype());
    dlTensor.shape = const_cast<int64_t*>(shape().data);
    dlTensor.strides = nullptr;
    dlTensor.byte_offset = 0;

    managed_tensor.dl_tensor = dlTensor;
    managed_tensor.manager_ctx = nullptr;
    managed_tensor.deleter = [](DLManagedTensor*) {};

    return py::reinterpret_steal<py::object>(
        PyCapsule_New(&managed_tensor, "dltensor", nullptr));
  }

};

}

#endif

