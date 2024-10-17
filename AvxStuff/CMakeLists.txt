#project AvxStuff

add_library(AvxStuff STATIC 
    AvxFrame.cpp
    AvxUtil.cpp
    AvxWrapper.cpp
)

target_include_directories(AvxStuff PUBLIC "${PROJECT_SOURCE_DIR}/CudaStuff")
target_link_libraries(AvxStuff PUBLIC CudaStuff)
