#include "clTest.hpp"
#include <iostream>

LoadResult cltest::loadKernels(std::initializer_list<std::string> kernelNames) {
	cl_int status;
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	cl::Platform::get(&platforms);
	status = platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	cl::Device dev = devices[0];
	cl::Context context(dev);
	cl::CommandQueue queue(context, dev);

	cl::Program::Sources sources;
	for (const std::string& str : kernelNames) {
		sources.emplace_back(str.c_str(), str.size());
	}

	cl::Program program(context, sources, &status);
	program.build();
	if (status != CL_SUCCESS) {
		std::cout << "error program " << clErrorStrings[-status] << std::endl;
		return { false };
	}
	cl::Kernel kernel(program, "luinvTest", &status);
	if (status != CL_SUCCESS) {
		std::cout << "error kernel " << clErrorStrings[-status] << std::endl;
		return { false };
	}
	return { status, context, queue, kernel };
}

bool cltest::cl_inv(LoadResult& res, double* input, double* invOut, size_t s) {
	size_t siz = sizeof(double) * s * s;
	cl::Buffer cl_input(res.context, CL_MEM_READ_ONLY, siz);
	cl::Buffer cl_inv(res.context, CL_MEM_WRITE_ONLY, siz);
	res.queue.enqueueWriteBuffer(cl_input, CL_TRUE, 0, siz, input);

	//kernel parameters
	cl_int status;
	status = res.kernel.setArg(0, cl_input);
	status = res.kernel.setArg(1, cl_inv);
	status = res.kernel.setArg(2, (int) s);
	//set local memory size, can only define ONE SINGLE local array
	status = res.kernel.setArg(3, sizeof(double) * s * 2, nullptr);

	status = res.queue.enqueueNDRangeKernel(res.kernel, 0, s, s);
	if (status != CL_SUCCESS) {
		std::cout << "error parameters " << clErrorStrings[-status] << std::endl;
		return false;
	}

	//retrieve results from device
	res.queue.enqueueReadBuffer(cl_inv, CL_TRUE, 0, siz, invOut);
	return status == CL_SUCCESS;
}

double cltest::cl_norm1(double* input, size_t s, int threads) {
	return 1.0;
}