project(Matrix)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")
add_library(Matrix STATIC ${sources})
