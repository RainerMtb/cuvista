/*
 * This file is part of CUVISTA - Cuda Video Stabilizer
 * Copyright (c) 2023 Rainer Bitschi cuvista@a1.net
 *
 * This program is free software : you can redistribute it and /or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.If not, see < http://www.gnu.org/licenses/>.
 */

#include <CL/cl.hpp>
#undef min
#undef max

#include "clMain.hpp"
#include "AVException.hpp"

void cl::init(CoreData& data, ImageYuv& inputFrame) {

}

std::size_t cl::deviceCount() {
	return 0;
}

OpenClInfo cl::probeRuntime() {
	cl_int status;
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	status = cl::Platform::get(&platforms);
	if (status == CL_SUCCESS && platforms.size() > 0) {
		status = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
		for (cl::Device dev : devices) {
			cl::Context context(dev);
			cl::CommandQueue queue(context, dev);

			std::string devName = dev.getInfo<CL_DEVICE_NAME>(&status);
			//std::cout << devName << std::endl;

			cl_int prefWidth = dev.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>(&status);
			cl_int nativeWidth = dev.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>(&status);
			bool hasDouble = prefWidth != 0 && nativeWidth != 0;
			//std::cout << hasDouble << std::endl;
		}

	} else {
		std::cout << "No OpenCL Devices found" << std::endl;
	}
	return {};
}

ImageYuv cl::getInput(int64_t idx) {
	return {};
}

Matf cl::getTransformedOutput() {
	return Matf();
}

Matf getPyramid(std::size_t idx) {
	return Matf();
}