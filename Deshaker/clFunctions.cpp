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

void cl::runKernel(cl::Kernel& kernel, cl::Image src, cl::Image dest, cl::CommandQueue queue, size_t w, size_t h) {
	kernel.setArg(0, src);
	kernel.setArg(1, dest);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(w, h));
}

void cl::runKernel(cl::Kernel& kernel, cl::Image src, cl::Image dest, cl::CommandQueue queue) {
	runKernel(kernel, src, dest, queue, dest.getImageInfo<CL_IMAGE_WIDTH>(), dest.getImageInfo<CL_IMAGE_HEIGHT>());
}

void cl::scale_8u32f_1(cl::Image src, cl::Image dest, ClData& clData) {
	//cl::Sampler sampler(clData.context, false, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST);
	assert(src.getImageInfo<CL_IMAGE_WIDTH>() == dest.getImageInfo<CL_IMAGE_WIDTH>() && "image width mismatch");
	assert(src.getImageInfo<CL_IMAGE_HEIGHT>() == dest.getImageInfo<CL_IMAGE_HEIGHT>() * 3 && "image width mismatch");
	runKernel(clData.kernel("scale_8u32f_1"), src, dest, clData.queue);
}

void cl::scale_8u32f_3(cl::Image src, cl::Image dest, ClData& clData) {
	runKernel(clData.kernel("scale_8u32f_3"), src, dest, clData.queue);
}

void cl::scale_32f8u_3(cl::Image src, cl::Image dest, ClData& clData) {
	runKernel(clData.kernel("scale_32f8u_3"), src, dest, clData.queue, src.getImageInfo<CL_IMAGE_WIDTH>(), src.getImageInfo<CL_IMAGE_HEIGHT>());
}

void cl::remap_downsize_32f(cl::Image src, cl::Image dest, ClData& clData) {
	runKernel(clData.kernel("remap_downsize_32f"), src, dest, clData.queue);
}

void cl::filter_32f_func(cl::Kernel& kernel, cl::Image src, cl::Image dest, int filterIndex, int dx, int dy, ClData& clData) {
	kernel.setArg(2, filterIndex);
	kernel.setArg(3, dx);
	kernel.setArg(4, dy);
	runKernel(kernel, src, dest, clData.queue);
}

void cl::filter_32f_h1(cl::Image src, cl::Image dest, int filterIndex, ClData& clData) {
	filter_32f_func(clData.kernel("filter_32f_1"), src, dest, filterIndex, 1, 0, clData);
}

void cl::filter_32f_h3(cl::Image src, cl::Image dest, ClData& clData) {
	filter_32f_func(clData.kernel("filter_32f_3"), src, dest, -1, 1, 0, clData);
}

void cl::filter_32f_v1(cl::Image src, cl::Image dest, int filterIndex, ClData& clData) {
	filter_32f_func(clData.kernel("filter_32f_1"), src, dest, filterIndex, 0, 1, clData);
}

void cl::filter_32f_v3(cl::Image src, cl::Image dest, ClData& clData) {
	filter_32f_func(clData.kernel("filter_32f_3"), src, dest, -1, 0, 1, clData);
}

void cl::warp_back(cl::Image src, cl::Image dest, ClData& clData, std::array<double, 6> trf) {
	cl_double8 cltrf = { trf[0], trf[1], trf[2], trf[3], trf[4], trf[5] };
	cl::Kernel& kernel = clData.kernel("warp_back");
	kernel.setArg(2, cltrf);
	runKernel(kernel, src, dest, clData.queue);
}

void cl::unsharp(cl::Image src, cl::Image dest, cl::Image gauss, ClData& clData, cl_float4 factor) {
	cl::Kernel& kernel = clData.kernel("unsharp");
	kernel.setArg(2, gauss);
	kernel.setArg(3, factor);
	runKernel(kernel, src, dest, clData.queue);
}

void cl::yuv_to_rgb(const std::string& kernelName, cl::Image src, unsigned char* imageData, ClData& clData, int w, int h) {
	cl::Kernel& kernel = clData.kernel(kernelName);
	kernel.setArg(0, src);
	kernel.setArg(1, clData.rgbOut);
	clData.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(w, h));
	clData.queue.enqueueReadBuffer(clData.rgbOut, CL_TRUE, 0, 3ull * w * h, imageData);
}