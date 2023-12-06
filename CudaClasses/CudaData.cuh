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

#include "cuUtil.cuh"
#include "DeviceInfoCuda.cuh"
#include "CoreData.hpp"


struct DebugData {
	std::vector<double> debugData;
	ImageBGR kernelTimings;
};

struct CudaData : public CoreData {
	size_t cudaMemTotal = 0;
	size_t cudaUsedMem = 0;
	size_t computeSharedMem = 0;

	int3 computeBlocks = {};
	int3 computeThreads = {};

	int strideChar = 0 ;     //row length in bytes for char values
	int strideFloat = 0;	 //row lenhth in bytes for float values
	int strideFloatN = 0;	 //number of float values in a row including padding
	int strideFloat4 = 0;    //row length in bytes for float4 struct
	int strideFloat4N = 0;   //number of float4 values

	int outBufferCount = 6;     //number of images to hold as buffers for output generation
};
