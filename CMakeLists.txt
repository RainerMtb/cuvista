cmake_minimum_required(VERSION 3.24)

project(cuvista LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES all)
enable_language(CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CONFIGURATION_TYPES="Release;Debug")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install")

#settings specific to Linux ------------------------------
if (CMAKE_SYSTEM_NAME MATCHES Linux)
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
set(CORE_LIBS 
    OpenCL 
    cuda
    cpu_features
    PkgConfig::ffmpeg
    nvcuvid
    nvidia-encode
    nvidia-ml
)
endif()

#settings specific to Windows ------------------------------
if (CMAKE_SYSTEM_NAME MATCHES Windows)
link_directories("${PROJECT_SOURCE_DIR}/nvapi/amd64")
link_directories("${PROJECT_SOURCE_DIR}/lib/x64")
set(CORE_LIBS
OpenCL 
cuda
cpu_features
avcodec
avdevice
avfilter
avformat
avutil
swresample
swscale
nvapi64
nvcuvid
nvencodeapi
)
endif()

#settings specific to compilers ------------------------------
if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /arch:AVX512")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512vl -mavx512bw -ffp-contract off")
endif()

add_subdirectory(Common)
add_subdirectory(Matrix)
add_subdirectory(CudaStuff)
add_subdirectory(Deshaker)
add_subdirectory(cuvistaCli)
add_subdirectory(cuvistaGui)
