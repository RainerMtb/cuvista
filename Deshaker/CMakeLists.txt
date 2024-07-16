#project(Deshaker)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512vl -mavx512bw")

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")
add_library(Deshaker STATIC ${sources})
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/Matrix")
target_link_libraries(Deshaker PUBLIC Matrix)
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/CudaClasses")
target_link_libraries(Deshaker PUBLIC CudaClasses)
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/include/ffmpeg")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/include/CL")