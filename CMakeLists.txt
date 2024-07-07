cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(cuvista LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

#find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES all)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
enable_language(CUDA)

add_subdirectory(Matrix)
add_subdirectory(CudaClasses)
add_subdirectory(Deshaker)
add_subdirectory(cuvistaCli)
