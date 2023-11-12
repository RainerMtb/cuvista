#include "clTest.hpp"
#include <iostream>

LoadResult cltest::loadKernels(std::initializer_list<std::string> kernelNames, const std::string& startKernel) {
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

	std::cout << "OpenCL Device Name: " << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
	try {
		cl::Program program(context, sources, &status);
		//program.build("-cl-opt-disable");
		program.build();
		cl::Kernel kernel(program, startKernel.c_str(), &status);
		return { status, context, queue, kernel };

	} catch (const cl::BuildError& err) {
		for (auto& data : err.getBuildLog()) {
			cl::Device dev = data.first;
			std::string msg = data.second;
			std::cout << msg << std::endl;
		}

	} catch (const cl::Error& err) {
		std::cout << err.what() << std::endl;
	}

	return { CL_BUILD_ERROR, context, queue, {} };
}

bool cltest::cl_inv(LoadResult& res, double* input, double* invOut, size_t s) {
	size_t siz = sizeof(double) * s * s;
	cl::Buffer cl_input(res.context, CL_MEM_READ_ONLY, siz);
	cl::Buffer cl_inv(res.context, CL_MEM_WRITE_ONLY, siz);
	res.queue.enqueueWriteBuffer(cl_input, CL_TRUE, 0, siz, input);

	//kernel parameters
	try {
		res.kernel.setArg(0, cl_input);
		res.kernel.setArg(1, cl_inv);
		//set local memory size, can only define ONE SINGLE local array
		res.kernel.setArg(2, sizeof(double) * s * 2, nullptr);

		//start kernel
		res.queue.enqueueNDRangeKernel(res.kernel, 0, s, s);
		res.queue.finish();

		//retrieve results from device
		res.queue.enqueueReadBuffer(cl_inv, CL_TRUE, 0, siz, invOut);
		return true;

	} catch (cl::Error err) {
		std::cout << "error: " << err.what() << std::endl;
		//std::cout << clErrorStrings[-err.what] << std::endl;
	}
	return false;
}

bool cltest::cl_inv_group(LoadResult& res, double* input, double* invGroup, int groupWidth, size_t s) {
	size_t siz = sizeof(double) * s * s * groupWidth;
	cl::Buffer cl_input(res.context, CL_MEM_READ_ONLY, siz);
	cl::Buffer cl_inv(res.context, CL_MEM_WRITE_ONLY, siz);
	res.queue.enqueueWriteBuffer(cl_input, CL_TRUE, 0, siz, input);

	//kernel parameters
	try {
		size_t shdsiz = sizeof(double) * s * (s + 2);
		res.kernel.setArg(0, cl_input);
		res.kernel.setArg(1, cl_inv);
		res.kernel.setArg(2, shdsiz, nullptr); //local memory size

		//start kernel
		cl::NDRange ndglob(groupWidth * s, 3);
		cl::NDRange ndloc(s, 3);
		res.queue.enqueueNDRangeKernel(res.kernel, 0, ndglob, ndloc);
		res.queue.finish();

		//retrieve results from device
		res.queue.enqueueReadBuffer(cl_inv, CL_TRUE, 0, siz, invGroup);
		return true;

	} catch (cl::Error err) {
		std::cout << "error: " << err.what() << std::endl;
		//std::cout << clErrorStrings[-err.what] << std::endl;
	}
	return false;
}

double cltest::cl_norm1(LoadResult& res, double* input, int s) {
	size_t siz = sizeof(double) * s * s;
	cl::Buffer cl_input(res.context, CL_MEM_READ_ONLY, siz);
	cl::Buffer cl_retval(res.context, CL_MEM_WRITE_ONLY, sizeof(double) * 1);
	res.queue.enqueueWriteBuffer(cl_input, CL_TRUE, 0, siz, input);
	double norm1value = NAN;

	//kernel parameters
	try {
		res.kernel.setArg(0, cl_input);
		res.kernel.setArg(1, s);
		res.kernel.setArg(2, s);
		res.kernel.setArg(3, sizeof(double) * s, nullptr); //local memory size
		res.kernel.setArg(4, cl_retval);

		//start kernel
		cl::NDRange ndglob(s, 3);
		cl::NDRange ndloc(s, 3);
		res.queue.enqueueNDRangeKernel(res.kernel, 0, ndglob, ndloc);
		res.queue.finish();

		//retrieve results from device
		res.queue.enqueueReadBuffer(cl_retval, CL_TRUE, 0, sizeof(double) * 1, &norm1value);

	} catch (cl::Error err) {
		std::cout << "error: " << err.what() << std::endl;
		//std::cout << clErrorStrings[-err.what] << std::endl;
	}
	return norm1value;
}