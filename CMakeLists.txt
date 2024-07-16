cmake_minimum_required(VERSION 3.24)

project(cuvista LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES all)
enable_language(CUDA)

add_subdirectory(Matrix)
add_subdirectory(CudaClasses)
add_subdirectory(Deshaker)
add_subdirectory(cuvistaCli)


#sudo apt install -y ffmpeg
#sudo apt install -y libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev
#sudo apt install -y cmake
#cuda
#cuda path set
