#project(Matrix)

add_library(Matrix STATIC 
    Affine2D.cpp
)

target_include_directories(Matrix PUBLIC "${PROJECT_SOURCE_DIR}/Common")
target_link_libraries(Matrix PUBLIC Common)