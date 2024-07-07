cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cu" "*.cpp")
add_library(CudaClasses STATIC ${sources})

set_target_properties(CudaClasses PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(CudaClasses PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
