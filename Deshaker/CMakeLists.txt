#project(Deshaker)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")
add_library(Deshaker STATIC ${sources})
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/Common")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/Matrix")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/CudaStuff")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/include/CL")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/include/nvenc")

if (CMAKE_SYSTEM_NAME MATCHES Windows)
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/nvapi")
target_include_directories(Deshaker PUBLIC "${PROJECT_SOURCE_DIR}/include/ffmpeg")
endif()

target_link_libraries(Deshaker PUBLIC Common)
target_link_libraries(Deshaker PUBLIC Matrix)
target_link_libraries(Deshaker PUBLIC CudaStuff)