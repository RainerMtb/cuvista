#project(cuvistaCli)

add_executable(cuvista Main.cpp)
target_include_directories(cuvista PRIVATE "${PROJECT_SOURCE_DIR}/Deshaker")

target_link_libraries(cuvista PRIVATE 
    Common
    Matrix 
    CudaStuff 
    Deshaker 
    OpenCL 
    PkgConfig::ffmpeg
    cuda
    cpu_features
    nvcuvid
    nvidia-encode
)
