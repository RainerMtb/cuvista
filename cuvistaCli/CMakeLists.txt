#project(cuvistaCli)

add_executable(cuvista Main.cpp)
target_include_directories(cuvista PRIVATE "${PROJECT_SOURCE_DIR}/Deshaker")

target_link_libraries(cuvista PRIVATE 
    Common
    Matrix 
    CudaStuff 
    Deshaker 
    "${CORE_LIBS}"
)

install(TARGETS
    cuvista
)
install(DIRECTORY "${FFMPEG_DIR}/bin"
    DESTINATION "${CMAKE_INSTALL_PREFIX}"
    FILES_MATCHING PATTERN "*.dll"
)
