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

#pragma once

#include "clHeaders.hpp"
#include "ErrorLogger.hpp"
#include <map>

struct FilterKernelData {
	std::vector<cl_float4> filterKernel;  //vector holding filter kernel values
	cl_int8 idx;                          //vector holding index offset values
	int siz;                              //size of this kernel
	inline static int maxSize = 8;        //max size
};

struct ClData {
	cl::Context context;
	cl::CommandQueue queue;

	std::vector<cl::Image2D> yuv;
	std::vector<std::vector<std::vector<cl::Image2D>>> pyr;
	std::vector<std::vector<cl::Image2D>> pyrBuffer;

	cl::Buffer filterKernel;
	std::array<cl::Image2D, 5> out;
	cl::Image2D yuvOut;
	cl::Buffer rgbOut;

	std::map<std::string, cl::Kernel> kernelMap = {
		{"scale_8u32f_1", {}},
		{"scale_8u32f_3", {}},
		{"scale_32f8u_3", {}},
		{"filter_32f_1", {}},
		{"filter_32f_3", {}},
		{"remap_downsize_32f", {}},
		{"warp_back", {}},
		{"unsharp", {}},
		{"yuv8u_to_rgb", {}},
		{"yuv32f_to_rgb", {}},
		{"scrap", {}},
	};

	FilterKernelData filterGauss = {
		{
		{0.0625f, 0.0f, 0.0f},
		{0.25f, 0.25f, 0.25f},
		{0.375f, 0.5f, 0.5f},
		{0.25f, 0.25f, 0.25f},
		{0.0625f, 0.0f, 0.0f},
		},
		{-2, -1, 0, 1, 2},
		5,
	};
	FilterKernelData filterDifference = {
		{
		{-0.5f},
		{0.0f},
		{0.5f},
		},
		{-1, 0, 1},
		3,
	};

	cl::Kernel& kernel(const std::string& key) {
		return kernelMap.at(key);
	}
};

namespace cl {
	void scale_8u32f_1(cl::Image src, cl::Image dest, ClData& clData);
	void scale_8u32f_3(cl::Image src, cl::Image dest, ClData& clData);
	void scale_32f8u_3(cl::Image src, cl::Image dest, ClData& clData);

	void filter_32f_h1(cl::Image src, cl::Image dest, FilterKernelData& filterData, ClData& clData);
	void filter_32f_h3(cl::Image src, cl::Image dest, FilterKernelData& filterData, ClData& clData);
	void filter_32f_v1(cl::Image src, cl::Image dest, FilterKernelData& filterData, ClData& clData);
	void filter_32f_v3(cl::Image src, cl::Image dest, FilterKernelData& filterData, ClData& clData);
	void filter_32f_func(cl::Kernel& kernel, cl::Image src, cl::Image dest, FilterKernelData& filterData, cl_int8 ix, cl_int8 iy, ClData& clData);

	void remap_downsize_32f(cl::Image src, cl::Image dest, ClData& clData);
	void warp_back(cl::Image src, cl::Image dest, ClData& clData, std::array<double, 6> trf);
	void unsharp(cl::Image src, cl::Image dest, cl::Image gauss, ClData& clData, std::array<float, 3> factor);

	void yuv_to_rgb(const std::string& kernelName, cl::Image src, unsigned char* imageData, ClData& clData, int w, int h);

	void runKernel(cl::Kernel& kernel, cl::Image src, cl::Image dest, cl::CommandQueue queue);
	void runKernel(cl::Kernel& kernel, cl::Image src, cl::Image dest, cl::CommandQueue queue, size_t w, size_t h);
}