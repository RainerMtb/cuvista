#project(CudaClasses)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cu")
add_library(CudaStuff STATIC ${sources})

set_target_properties(CudaStuff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(CudaStuff PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(CudaStuff PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(CudaStuff PUBLIC "${PROJECT_SOURCE_DIR}/Common")
target_include_directories(CudaStuff PUBLIC "${PROJECT_SOURCE_DIR}/Matrix")
target_compile_options(CudaStuff PRIVATE "-fmad=false")
target_link_libraries(CudaStuff PUBLIC Common)
target_link_libraries(CudaStuff PUBLIC Matrix)
