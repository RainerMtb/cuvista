#project(CudaClasses)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cu" "*.cpp")
add_library(CudaClasses STATIC ${sources})

set_target_properties(CudaClasses PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(CudaClasses PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(CudaClasses PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -rdc=true -fmad=false>)
target_include_directories(CudaClasses PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(CudaClasses PUBLIC "${PROJECT_SOURCE_DIR}/Matrix")
target_link_libraries(CudaClasses PUBLIC Matrix)
