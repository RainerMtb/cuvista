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
message("Looking for ffmpeg components in folder ${FFMPEG_PATH}")
#check one exemplary shared ffmpeg file, assume others will be there
find_library(ffmpegLIB avcodec.lib REQUIRED HINTS "${FFMPEG_PATH}/lib")
find_path(ffmpegH avcodec.h REQUIRED HINTS "${FFMPEG_PATH}/include/libavcodec")
find_program(ffmpegDLL avcodec-61.dll REQUIRED HINTS "${FFMPEG_PATH}/bin")

link_directories("${PROJECT_SOURCE_DIR}/nvapi/amd64")
link_directories("${PROJECT_SOURCE_DIR}/lib/x64")
link_directories("${FFMPEG_PATH}/lib")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/nvapi")
include_directories("${FFMPEG_PATH}/include")

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
#-----------------------------------

#main stuff
add_subdirectory(Common)
add_subdirectory(Matrix)
add_subdirectory(CudaStuff)
add_subdirectory(AvxStuff)
add_subdirectory(Deshaker)
add_subdirectory(cuvistaCli)
add_subdirectory(cuvistaGui)

#compiler options
target_compile_options(CudaStuff PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -rdc=true -fmad=false>)

if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    target_compile_options(AvxStuff PRIVATE /MP /arch:AVX512)
    target_compile_options(Deshaker PRIVATE /MP)
    target_compile_options(cuvista PRIVATE /MP)
    target_compile_options(cuvistaGui PRIVATE /MP)
    target_link_options(cuvista PRIVATE /NODEFAULTLIB:libcmt.lib)
    target_link_options(cuvistaGui PRIVATE /NODEFAULTLIB:libcmt.lib)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    target_compile_options(Common PRIVATE -ffp-contract=off)
    target_compile_options(Matrix PRIVATE -ffp-contract=off)
    target_compile_options(AvxStuff PRIVATE -ffp-contract=off -mavx512f -mavx512vl -mavx512bw -mavx512dq)
    target_compile_options(Deshaker PRIVATE -ffp-contract=off)
    target_compile_options(cuvista PRIVATE -ffp-contract=off)
    target_compile_options(cuvistaGui PRIVATE -ffp-contract=off)
endif()
