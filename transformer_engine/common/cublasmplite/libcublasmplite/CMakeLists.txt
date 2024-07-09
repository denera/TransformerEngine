add_library(cublasmplite SHARED src/cublasmplite.cu src/nvshmem.cu src/cuda.cu)
set(CUBLASMPLITE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)
target_link_directories(cublasmplite PRIVATE ${NVSHMEM_HOME}/lib)
target_link_libraries(cublasmplite PRIVATE -static-libstdc++ nvshmem_device nvshmem_host CUDA::nvml CUDA::cublas)
target_include_directories(cublasmplite PRIVATE
                           ${NVSHMEM_HOME}/include/
                           "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(cublasmplite PUBLIC
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

set_target_properties(cublasmplite PROPERTIES
                      CUDA_STANDARD 17
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS cublasmplite DESTINATION lib)
install(DIRECTORY "include/" DESTINATION include)