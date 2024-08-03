cmake_minimum_required(VERSION 3.24)

project(cuvista LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES all)
enable_language(CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512vl -mavx512bw")

find_package(PkgConfig REQUIRED)
pkg_check_modules(ffmpeg REQUIRED IMPORTED_TARGET
    libavcodec
    libavdevice
    libavfilter
    libavformat
    libavutil
    libswresample
    libswscale
)

add_subdirectory(Common)
add_subdirectory(Matrix)
add_subdirectory(CudaStuff)
add_subdirectory(Deshaker)
add_subdirectory(cuvistaCli)
add_subdirectory(cuvistaGui)
