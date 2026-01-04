#project AvxStuff

add_library(AvxStuff STATIC 
    AvxFrame.cpp
    AvxUtil.cpp
    AvxWrapper.cpp
)

target_include_directories(AvxStuff PRIVATE "${PROJECT_SOURCE_DIR}/CudaStuff")
target_include_directories(AvxStuff PRIVATE "${PROJECT_SOURCE_DIR}/Matrix")
target_link_libraries(AvxStuff PRIVATE CudaStuff)
target_link_libraries(AvxStuff PRIVATE Matrix)
