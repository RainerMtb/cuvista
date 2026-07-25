#project Avx2

add_library(Avx2 STATIC 
    Avx2Wrapper.cpp
)

target_include_directories(Avx2 PRIVATE "${PROJECT_SOURCE_DIR}/Matrix")
target_link_libraries(Avx2 PRIVATE Matrix)
