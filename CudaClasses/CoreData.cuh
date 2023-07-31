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

#include "Image.hpp"
#include "Image.hpp"
#include "ErrorLogger.hpp"
#include "cuUtil.cuh"
#include "CudaInfo.hpp"

struct BlendInput {
	double percent = 0.0;
	int blendStart = 0;
	int blendWidth = 0;
	int separatorStart = 0;
	int separatorWidth = 0;
};

struct OutputContext {
	bool encodeCpu;
	bool encodeGpu;
	ImageYuv* outputFrame;
	unsigned char* cudaNv12ptr;
	int cudaPitch;
};

struct DebugData {
	std::vector<double> debugData;
	ImageBGR kernelTimings;
};

//how to deal with background when frame does not cover complete output canvas
enum class BackgroundMode {
	BLEND,
	COLOR
};

struct CoreData {
	//inside CoreData struct all values must be initialized
	//see __constant__ variables in device code

public:
	size_t BUFFER_COUNT = 21;	//number of buffer buffer frames to do filtering and output
	int MAX_POINTS_COUNT = 150;		//max number of points in x or y direction

	int pyramidLevels = 3;			//number of pyramid levels, not necessary starting at level 0
	int pitch = 0;					//alignment of image rows, will be overwritten in cuda setup
	int strideCount = 0;		    //number of float values in a row including padding
	int strideFloatBytes = 0;		//number of bytes in an image row including padding
	int compMaxIter = 20;			//max loop iterations
	double compMaxTol = 0.05;		//tolerance to stop window pattern matching

	std::array<float, 3> unsharp = { 0.6f, 0.3f, 0.3f };	//ffmpeg unsharp=5:5:0.6:3:3:0.3
	ColorNorm bgcol_yuv = {};								//background fill colors in yuv
	BlendInput blendInput = {};
	BackgroundMode bgmode = BackgroundMode::BLEND;

	int ir = 3;					//integration window, radius around point to integrate
	int iw = 7;					//integration window, 2 * ir + 1
	double imZoom = 1.05;		//additional zoom
	double radsec = 0.5;		//radius in senconds
	int radius = -1;			//number of frames before and after used for smoothing
	int zMin = -1;
	int zMax = -1;				//pyramid steps used for actual computing
	int div = -1;

	int w = 0;                  //frame width
	int h = 0;					//frame height
	int64_t frameCount = -1;

	int pyramidRows = -1;		//number of rows for one pyramid, for example all the rows of Y data
	int bufferCount = -1;		//number of frames to read before starting to average out trajectory
	size_t pyramidCount = 3;	//number of pyramids to allocate in memory

	int ixCount = -1;
	int iyCount = -1;
	int resultCount = 0;		//number of points to compute in a frame

	uint8_t crf = 22;
	char fileDelimiter = ';';

	int deviceNum = -1;
	int deviceNumBest = -1;
	size_t cudaUsedMem = 0;
	size_t maxPixel = cu::MAX_PIXEL;
	cudaDeviceProp cudaProps = {};
	size_t computeSharedMem = 0;

	size_t cudaMemTotal = 0;
	size_t cudaMemUsed = 0;

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
