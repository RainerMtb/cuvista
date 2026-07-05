#project(cuvistaFFmpeg)

set(LIBS)

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

    list(APPEND LIBS 
        PkgConfig::ffmpeg
    )
endif()


# --------------------------------
# settings specific to Windows
if (CMAKE_SYSTEM_NAME MATCHES Windows)
    message(NOTICE "Looking for ffmpeg components in folder '${FFMPEG_PATH}'...")
    #check one exemplary shared ffmpeg file, assume others will be there as well
    find_library(ffmpegLIB avcodec.lib REQUIRED NO_DEFAULT_PATH HINTS "${FFMPEG_PATH}/lib")
    find_path(ffmpegH avcodec.h REQUIRED NO_DEFAULT_PATH HINTS "${FFMPEG_PATH}/include/libavcodec")
    find_program(ffmpegDLL avcodec-62.dll REQUIRED NO_DEFAULT_PATH HINTS "${FFMPEG_PATH}/bin")

    link_directories("${FFMPEG_PATH}/lib")
    include_directories("${FFMPEG_PATH}/include")

    list(APPEND LIBS
        avcodec
        avdevice
        avfilter
        avformat
        avutil
        swresample
        swscale
    )
endif()

add_library(cuvistaFFmpeg SHARED
    FFmpegMain.cpp
    Reader.cpp
    Writer.cpp
    CudaWriter.cpp
    FFmpegWriter.cpp
    FFmpegFormatWriter.cpp
    StackedWriter.cpp
)

target_include_directories(cuvistaFFmpeg PUBLIC 
    "${PROJECT_SOURCE_DIR}/Common"
    "${PROJECT_SOURCE_DIR}/Matrix"
    "${PROJECT_SOURCE_DIR}/Deshaker"
    "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(cuvistaFFmpeg PUBLIC 
    Common
    Deshaker
    "${LIBS}"
    "${CORE_LIBS}"
)

install(TARGETS 
    cuvistaFFmpeg
)
