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
#include "clKernels.hpp"
#include "clMain.hpp"


struct LoadResult {
	cl_int status;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Kernel kernel;
};

namespace cltest {

	LoadResult loadKernels(std::initializer_list<std::string> kernelNames, const std::string& startKernel);

	bool cl_inv(LoadResult& res, double* input, double* invOut, size_t s);

	bool cl_inv_group(LoadResult& res, double* input, double* invGroup, int groupWidth, size_t s);

	std::vector<double> cl_norm1(LoadResult& res, double* input, int s);
}