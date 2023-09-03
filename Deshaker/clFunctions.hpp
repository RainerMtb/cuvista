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

struct ClData {
	std::vector<cl::Device> devices;
	cl::Context context;
	cl::CommandQueue queue;

	std::vector<std::vector<cl::Image2D>> yuv;
	std::vector<std::vector<std::vector<cl::Image2D>>> pyr;
	std::vector<std::vector<cl::Image2D>> pyrBuffer;

	cl::Buffer filterKernel;

	cl::Kernel scale_8u32f;
	cl::Kernel filter_32f_h;
	cl::Kernel filter_32f_v;
	cl::Kernel remap_downsize_32f;
};

namespace cl {
	void scale_8u32f(cl::Image src, cl::Image dest, ClData& clData);
	void filter_32f_h(cl::Image src, cl::Image dest, const float* filterKernel, int kernelSize, ClData& clData);
	void filter_32f_v(cl::Image src, cl::Image dest, const float* filterKernel, int kernelSize, ClData& clData);
	void remap_downsize_32f(cl::Image src, cl::Image dest, ClData& clData);

	void filter_32f_func(cl::Kernel& kernel, cl::Image src, cl::Image dest, const float* filterKernel, int kernelSize, ClData& clData);
}