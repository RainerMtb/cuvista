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

#include "clFunctions.hpp"
#include <cassert>

void cl::scale_8u32f(cl::Image src, cl::Image dest, ClData& clData) {
	size_t w = src.getImageInfo<CL_IMAGE_WIDTH>();
	size_t h = src.getImageInfo<CL_IMAGE_HEIGHT>();
	assert(w == dest.getImageInfo<CL_IMAGE_WIDTH>() && "image width mismatch");
	assert(h == dest.getImageInfo<CL_IMAGE_HEIGHT>() && "image width mismatch");

	cl::Kernel& kernel = clData.scale_8u32f;
	//cl::Sampler sampler(clData.context, false, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST);
	kernel.setArg(0, src);
	kernel.setArg(1, dest);

	cl::NDRange dim(w, h);
	clData.queue.enqueueNDRangeKernel(kernel, cl::NullRange, dim);
}

void cl::filter_32f_func(cl::Kernel& kernel, cl::Image src, cl::Image dest, const float* filterKernel, int kernelSize, ClData& clData) {
	kernel.setArg(0, src);
	kernel.setArg(1, dest);
	kernel.setArg(2, clData.filterKernel);
	kernel.setArg(3, kernelSize);

	clData.queue.enqueueWriteBuffer(clData.filterKernel, CL_TRUE, 0, sizeof(float) * kernelSize, filterKernel);

	size_t w = src.getImageInfo<CL_IMAGE_WIDTH>();
	size_t h = src.getImageInfo<CL_IMAGE_HEIGHT>();
    cl::NDRange dim(w, h);
	clData.queue.enqueueNDRangeKernel(kernel, cl::NullRange, dim);
}

void cl::filter_32f_h(cl::Image src, cl::Image dest, const float* filterKernel, int kernelSize, ClData& clData) {
	filter_32f_func(clData.filter_32f_h, src, dest, filterKernel, kernelSize, clData);
}

void cl::filter_32f_v(cl::Image src, cl::Image dest, const float* filterKernel, int kernelSize, ClData& clData) {
	filter_32f_func(clData.filter_32f_v, src, dest, filterKernel, kernelSize, clData);
}
