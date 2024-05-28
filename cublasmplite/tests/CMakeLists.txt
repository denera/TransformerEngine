# Test drivers
foreach(test ag_gemm gemm_rs gemm_rs_atomic)
    add_executable(${test} "${test}.cu")
    target_include_directories(${test} PRIVATE ${NCCL_HOME}/include "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_directories(${test} PRIVATE ${NCCL_HOME}/lib)
    target_link_libraries(${test} PRIVATE MPI::MPI_CXX cublasmplite -static-libstdc++ CUDA::nvml CUDA::cublas CUDA::cuda_driver CUDA::cudart_static nccl)
    set_target_properties(${test} PROPERTIES
        CUDA_STANDARD 17
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()