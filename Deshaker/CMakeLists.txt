cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")
add_library(Deshaker STATIC ${sources})
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/Matrix")
target_link_libraries(Deshaker PUBLIC Matrix)
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/CudaClasses")
target_link_libraries(Deshaker PUBLIC CudaClasses)
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/include/ffmpeg")
