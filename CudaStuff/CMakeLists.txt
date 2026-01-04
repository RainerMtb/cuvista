#project(CudaClasses)

if (BUILD_CUDA MATCHES 0)
	add_library(CudaStuff INTERFACE)
	target_include_directories(CudaStuff PUBLIC INTERFACE "${PROJECT_SOURCE_DIR}/Common")
else()
	add_library(CudaStuff STATIC
		cuDecompose.cu
		cuDeshaker.cu
		cuKernelCompute.cu
		cuNPP.cu
		cuTest.cu
		cuUtil.cu
	)

	set_target_properties(CudaStuff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_target_properties(CudaStuff PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	target_include_directories(CudaStuff PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
	target_include_directories(CudaStuff PUBLIC "${PROJECT_SOURCE_DIR}/Common")
	target_compile_options(CudaStuff PRIVATE "-fmad=false" "-Wno-deprecated-gpu-targets")
	target_link_libraries(CudaStuff PUBLIC Common)
endif()