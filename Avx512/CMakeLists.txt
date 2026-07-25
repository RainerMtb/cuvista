#project Avx512

add_library(Avx512 STATIC 
    AvxFrame.cpp
    AvxWrapper.cpp
)

target_include_directories(Avx512 PRIVATE "${PROJECT_SOURCE_DIR}/Avx2")
target_include_directories(Avx512 PRIVATE "${PROJECT_SOURCE_DIR}/Matrix")
target_link_libraries(Avx512 PRIVATE Matrix)
target_link_libraries(Avx512 PRIVATE Avx2)
