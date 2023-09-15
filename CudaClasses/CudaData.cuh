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

	//numeric constants used in compute kernel, will be initialized once
	double dmin = 0.0, dmax = 0.0, deps = 0.0, dnan = 0.0;
};

//result type of one computed point
enum class PointResultType {
	FAIL_SINGULAR,
	FAIL_ITERATIONS,
	FAIL_ETA_NAN,
	RUNNING,
	SUCCESS_ABSOLUTE_ERR,
	SUCCESS_STABLE_ITER,
};

//result of one computed point in a frame
struct PointResult {

private:
	__device__ __host__ bool equal(double a, double b, double tol) const;

public:
	size_t idx, ix0, iy0;
	int px, py;
	int x, y;
	double u, v;
	PointResultType result;
	double distance;
	double length;
	double distanceRelative;

	//is valid when numeric stable result was found
	__device__ __host__ bool isValid() const;

	//numeric value for type of result
	__device__ __host__ int resultValue() const;

	__device__ __host__ bool equals(const PointResult& other, double tol = 0.0) const;

	__device__ __host__ bool operator == (const PointResult& other) const;

	__device__ __host__ bool operator != (const PointResult& other) const;

	friend __host__ std::ostream& operator << (std::ostream& out, const PointResult& res);
};
