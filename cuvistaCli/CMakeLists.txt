#project(cuvistaCli)

add_executable(cuvista 
    Main.cpp
    Deshaker.cpp
    CMakeIcon.rc
)

target_include_directories(cuvista PRIVATE 
    "${PROJECT_SOURCE_DIR}/Deshaker"
)

target_link_libraries(cuvista PRIVATE 
    Deshaker
    "${CORE_LIBS}"
)

install(TARGETS
    cuvista
)
