#project(Common)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")
add_library(Common STATIC ${sources})