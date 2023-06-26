#pragma once

#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

enum class OutputCodec {
	AUTO,
	H264,
	H265,
	JPEG,
};

//info about cuda devices
struct CudaInfo {
	int nvidiaDriverVersion = 0;
	int cudaRuntimeVersion = 0;
	int cudaDriverVersion = 0;

	int nppMajor = 0;
	int nppMinor = 0;
	int nppBuild = 0;

	int deviceCount = 0;
	std::vector<cudaDeviceProp> cudaProps;
	std::vector<std::vector<OutputCodec>> supportedCodecs;

	uint32_t nvencVersionApi;
	uint32_t nvencVersionDriver;

	std::string nvidiaDriver() const;
	std::string cudaRuntime() const;
	std::string cudaDriver() const;
	bool isSupported(int device, OutputCodec codec) const;
};