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

static void runKernel(cl::Kernel& kernel, cl::Image src, cl::Image dest, cl::CommandQueue queue, size_t w, size_t h) {
	kernel.setArg(0, src);
	kernel.setArg(1, dest);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(w, h));
}

static void runKernel(cl::Kernel& kernel, cl::Image src, cl::Image dest, cl::CommandQueue queue) {
	runKernel(kernel, src, dest, queue, dest.getImageInfo<CL_IMAGE_WIDTH>(), dest.getImageInfo<CL_IMAGE_HEIGHT>());
}

static void filter_32f_func(cl::Kernel& kernel, cl::Image src, cl::Image dest, int filterIndex, int dx, int dy, cl::Data& clData, size_t w, size_t h) {
	kernel.setArg(2, filterIndex);
	kernel.setArg(3, dx);
	kernel.setArg(4, dy);
	runKernel(kernel, src, dest, clData.queue, w, h);
}

static void filter_32f_func(cl::Kernel& kernel, cl::Image src, cl::Image dest, int filterIndex, int dx, int dy, cl::Data& clData) {
	filter_32f_func(kernel, src, dest, filterIndex, dx, dy, clData, dest.getImageInfo<CL_IMAGE_WIDTH>(), dest.getImageInfo<CL_IMAGE_HEIGHT>());
}


//-------------------------------------------

void cl::scale_8u32f_1(Image src, Image dest, Data& clData) {
	assert(src.getImageInfo<CL_IMAGE_WIDTH>() == dest.getImageInfo<CL_IMAGE_WIDTH>() && "image width mismatch");
	assert(src.getImageInfo<CL_IMAGE_HEIGHT>() == dest.getImageInfo<CL_IMAGE_HEIGHT>() * 3 && "image width mismatch");
	runKernel(clData.kernels.scale_8u32f_1, src, dest, clData.queue);
}

void cl::scale_8u32f_3(Image src, Image dest, Data& clData) {
	runKernel(clData.kernels.scale_8u32f_3, src, dest, clData.queue);
}

void cl::scale_32f8u_3(Image src, Buffer dest, int pitch, const Data& clData) {
	Kernel kernel = clData.kernels.scale_32f8u_3;
	kernel.setArg(0, src);
	kernel.setArg(1, dest);
	kernel.setArg(2, pitch);
	size_t w = src.getImageInfo<CL_IMAGE_WIDTH>();
	size_t h = src.getImageInfo<CL_IMAGE_HEIGHT>();
	clData.queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(w, h));
}

void cl::remap_downsize_32f(Image src, Image dest, Data& clData) {
	runKernel(clData.kernels.remap_downsize_32f, src, dest, clData.queue);
}

void cl::filter_32f_h1(Image src, Image dest, int filterIndex, Data& clData) {
	filter_32f_func(clData.kernels.filter_32f_1, src, dest, filterIndex, 1, 0, clData);
}

void cl::filter_32f_h3(Image src, Image dest, Data& clData) {
	filter_32f_func(clData.kernels.filter_32f_3, src, dest, -1, 1, 0, clData);
}

void cl::filter_32f_v1(Image src, Image dest, int filterIndex, Data& clData) {
	filter_32f_func(clData.kernels.filter_32f_1, src, dest, filterIndex, 0, 1, clData);
}

void cl::filter_32f_v3(Image src, Image dest, Data& clData) {
	filter_32f_func(clData.kernels.filter_32f_3, src, dest, -1, 0, 1, clData);
}

void cl::warp_back(Image src, Image dest, Data& clData, cl_float8& trf) {
	Kernel& kernel = clData.kernels.warp_back;
	kernel.setArg(2, trf);
	runKernel(kernel, src, dest, clData.queue);
}

void cl::unsharp(Image src, Image dest, Image gauss, Data& clData, cl_float4 factor) {
	Kernel& kernel = clData.kernels.unsharp;
	kernel.setArg(2, gauss);
	kernel.setArg(3, factor);
	runKernel(kernel, src, dest, clData.queue);
}

void cl::yuv_to_rgba(Kernel kernel, Image src, unsigned char* imageData, const Data& clData, int w, int h, const std::vector<int>& index) {
	cl_int4 offset4 = { index[0], index[1], index[2], index[3] };
	kernel.setArg(0, src);
	kernel.setArg(1, clData.rgbaOut);
	kernel.setArg(2, offset4);
	clData.queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(w, h));
	clData.queue.enqueueReadBuffer(clData.rgbaOut, CL_TRUE, 0, 4ull * w * h, imageData);
}

void cl::yuv_to_nv12(Kernel kernel, Image src, unsigned char* imageData, const Data& clData, int w, int h, int stride) {
	kernel.setArg(0, src);
	kernel.setArg(1, clData.yuvOut);
	kernel.setArg(2, stride);
	kernel.setArg(3, h);
	clData.queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(w / 2, h / 2));
	clData.queue.enqueueReadBuffer(clData.yuvOut, CL_TRUE, 0, 3ull * stride * h / 2, imageData);
}