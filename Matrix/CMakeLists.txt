project(Matrix)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")
add_library(Matrix STATIC ${sources})

target_include_directories(Matrix PUBLIC "${PROJECT_SOURCE_DIR}/Common")
target_link_libraries(Matrix PUBLIC Common)