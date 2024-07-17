#project(cuvistaCli)

add_executable(cuvista Main.cpp)
target_include_directories(cuvista PRIVATE "${PROJECT_SOURCE_DIR}/Matrix")
target_include_directories(cuvista PRIVATE "${PROJECT_SOURCE_DIR}/CudaClasses")
target_include_directories(cuvista PRIVATE "${PROJECT_SOURCE_DIR}/Deshaker")
target_link_directories(cuvista PRIVATE "${PROJECT_SOURCE_DIR}/lib/linux")

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
target_link_libraries(cuvista PRIVATE 
    Matrix 
    CudaClasses 
    Deshaker 
    OpenCL 
    PkgConfig::ffmpeg
    cuda
    cpu_features
    nvcuvid
    nvidia-encode
)
