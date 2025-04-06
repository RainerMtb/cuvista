cmake_minimum_required(VERSION 3.24)

if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(cuvista LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CONFIGURATION_TYPES="Release;Debug")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install")

# --------------------------------
# start the list of libraries
set(CORE_LIBS
    OpenCL 
    cpu_features
)

# --------------------------------
# check for cuda
if (BUILD_CUDA MATCHES 0)
    message(NOTICE "Cuda was deselected, Skipping Cuda Build...")
elseif (BUILD_CUDA MATCHES 1)
    message(NOTICE "Cuda was selected, Including Cuda Build...")
else()
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(BUILD_CUDA 1)
        message(NOTICE "Cuda was found, Including Cuda Build...") 
    else()
        set(BUILD_CUDA 0)
        message(NOTICE "Cuda was not found, Skipping Cuda Build...") 
    endif()
endif()
if (BUILD_CUDA MATCHES 1)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    set(CMAKE_CUDA_ARCHITECTURES native)
    enable_language(CUDA)
    list(APPEND CORE_LIBS 
        cuda
    )
endif()
    
# include cuda libraries for current platform
if (BUILD_CUDA MATCHES 1 AND CMAKE_SYSTEM_NAME MATCHES Linux)
    list(APPEND CORE_LIBS
        nvcuvid
        nvidia-encode
    )
endif()
if (BUILD_CUDA MATCHES 1 AND CMAKE_SYSTEM_NAME MATCHES Windows)
    list(APPEND CORE_LIBS
        nvcuvid
        nvencodeapi
    )
endif()


# --------------------------------
# settings specific to Linux
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

    list(APPEND CORE_LIBS 
        PkgConfig::ffmpeg
    )

    # for wsl use libraries at C:\Windows\System32\lxss\lib
    link_directories("/usr/lib/wsl/lib")
endif()


# --------------------------------
# settings specific to Windows
if (CMAKE_SYSTEM_NAME MATCHES Windows)
    message(NOTICE "Looking for ffmpeg components in folder '${FFMPEG_PATH}'...")
    #check one exemplary shared ffmpeg file, assume others will be there as well
    find_library(ffmpegLIB avcodec.lib REQUIRED NO_DEFAULT_PATH HINTS "${FFMPEG_PATH}/lib")
    find_path(ffmpegH avcodec.h REQUIRED NO_DEFAULT_PATH HINTS "${FFMPEG_PATH}/include/libavcodec")
    find_program(ffmpegDLL avcodec-61.dll REQUIRED NO_DEFAULT_PATH HINTS "${FFMPEG_PATH}/bin")

    link_directories("${PROJECT_SOURCE_DIR}/lib/x64")
    link_directories("${FFMPEG_PATH}/lib")

    include_directories("${PROJECT_SOURCE_DIR}/include")
    include_directories("${FFMPEG_PATH}/include")

    list(APPEND CORE_LIBS
        avcodec
        avdevice
        avfilter
        avformat
        avutil
        swresample
        swscale
    )
endif()

# do not build cllayerinfo
set(ENABLE_OPENCL_LAYERINFO OFF)


# --------------------------------
# main stuff, declare targets
add_subdirectory(OpenCL-Headers)
add_subdirectory(OpenCL-ICD-Loader)
add_subdirectory(cpu_features)
add_subdirectory(Common)
add_subdirectory(Matrix)
add_subdirectory(CudaStuff)
add_subdirectory(AvxStuff)
add_subdirectory(Deshaker)
add_subdirectory(cuvistaCli)


# check if Qt6 is available and add gui subdirectory
if (BUILD_GUI MATCHES 0)
    message(NOTICE "Gui was deselected, Skipping Gui Build...")
else()
    find_package(Qt6 QUIET COMPONENTS Core Gui Widgets Multimedia MultimediaWidgets)
    if (Qt6_FOUND)
        set(BUILD_GUI 1)
        find_package(QT NAMES Qt6) # only this line will populate the version numbers... man Qt is such a $!#*
        message(NOTICE "Qt6 version ${QT_VERSION_MAJOR}.${QT_VERSION_MINOR}.${QT_VERSION_PATCH} was found, Including Gui Build...")
        add_subdirectory(cuvistaGui)
    else()
        set(BUILD_GUI 0)
        message(NOTICE "Qt6 was not found, Skipping Gui Build...")
    endif()
endif()


# --------------------------------
# setting compiler specific options
if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    target_compile_options(Common PRIVATE /MP)
    target_compile_options(Matrix PRIVATE /MP)
    target_compile_options(AvxStuff PRIVATE -D BUILD_CUDA=${BUILD_CUDA} /MP /arch:AVX512)
    target_compile_options(Deshaker PRIVATE -D BUILD_CUDA=${BUILD_CUDA} /MP)
    target_link_options(cuvista PRIVATE /NODEFAULTLIB:libcmt.lib)
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES MSVC AND TARGET cuvistaGui)
    target_compile_options(cuvistaGui PRIVATE /MP)
    target_link_options(cuvistaGui PRIVATE /NODEFAULTLIB:libcmt.lib) 
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    target_compile_options(Common PRIVATE -ffp-contract=off)
    target_compile_options(Matrix PRIVATE -ffp-contract=off)
    target_compile_options(AvxStuff PRIVATE -ffp-contract=off -D BUILD_CUDA=${BUILD_CUDA} -mavx512f -mavx512vl -mavx512bw -mavx512dq)
    target_compile_options(Deshaker PRIVATE -ffp-contract=off -D BUILD_CUDA=${BUILD_CUDA} -Wno-deprecated-declarations)
    target_compile_options(cuvista PRIVATE -ffp-contract=off)
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES GNU AND TARGET cuvistaGui)
    target_compile_options(cuvistaGui PRIVATE -ffp-contract=off)
endif()


# copy ffmpeg dll files to install dir
install(DIRECTORY "${FFMPEG_PATH}/bin"
    DESTINATION "${CMAKE_INSTALL_PREFIX}"
    FILES_MATCHING PATTERN "*.dll"
)
