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
	cl::NDRange dim(w, h);
	queue.enqueueNDRangeKernel(kernel, cl::NDRange(), dim);
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

void cl::filter_32f_func(cl::Kernel& kernel, cl::Image src, cl::Image dest, FilterKernelData& filterData, cl_int8 ix, cl_int8 iy, ClData& clData) {
	clData.queue.enqueueWriteBuffer(clData.filterKernel, CL_TRUE, 0, sizeof(cl_float4) * filterData.siz, filterData.filterKernel.data());
	kernel.setArg(2, clData.filterKernel);
	kernel.setArg(3, ix);
	kernel.setArg(4, iy);
	kernel.setArg(5, filterData.siz);
	runKernel(kernel, src, dest, clData.queue);
}

void cl::filter_32f_h1(cl::Image src, cl::Image dest, FilterKernelData& filterData, ClData& clData) {
	filter_32f_func(clData.kernel("filter_32f_1"), src, dest, filterData, filterData.idx, {}, clData);
}

void cl::filter_32f_h3(cl::Image src, cl::Image dest, FilterKernelData& filterData, ClData& clData) {
	filter_32f_func(clData.kernel("filter_32f_3"), src, dest, filterData, filterData.idx, {}, clData);
}

void cl::filter_32f_v1(cl::Image src, cl::Image dest, FilterKernelData& filterData, ClData& clData) {
	filter_32f_func(clData.kernel("filter_32f_1"), src, dest, filterData, {}, filterData.idx, clData);
}

void cl::filter_32f_v3(cl::Image src, cl::Image dest, FilterKernelData& filterData, ClData& clData) {
	filter_32f_func(clData.kernel("filter_32f_3"), src, dest, filterData, {}, filterData.idx, clData);
}

void cl::warp_back(cl::Image src, cl::Image dest, ClData& clData, std::array<double, 6> trf) {
	cl_double8 cltrf = { trf[0], trf[1], trf[2], trf[3], trf[4], trf[5] };
	cl::Kernel& kernel = clData.kernel("warp_back");
	kernel.setArg(2, cltrf);
	runKernel(kernel, src, dest, clData.queue);
}

void cl::unsharp(cl::Image src, cl::Image dest, cl::Image gauss, ClData& clData, std::array<float, 3> factor) {
	cl_float4 clfactor = { factor[0], factor[1], factor[2] };
	cl::Kernel& kernel = clData.kernel("unsharp");
	kernel.setArg(2, gauss);
	kernel.setArg(3, clfactor);
	runKernel(kernel, src, dest, clData.queue);
}