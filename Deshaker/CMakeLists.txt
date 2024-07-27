#project(Deshaker)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512vl -mavx512bw")

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")
add_library(Deshaker STATIC ${sources})
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/Common")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/Matrix")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/CudaStuff")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/include/CL")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/include/nvenc")
target_link_libraries(Deshaker PUBLIC Common)
target_link_libraries(Deshaker PUBLIC Matrix)
target_link_libraries(Deshaker PUBLIC CudaStuff)