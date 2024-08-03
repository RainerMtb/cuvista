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

#include "cuDecompose.cuh"
#include "CudaData.cuh"

#include <sstream>
#include <fstream>
#include <iostream>


class CudaPointResult {

public:
	int64_t timeStart; //nanos
	int64_t timeStop;  //nanos
	double u, v;
	int idx, ix0, iy0;
	int xm, ym;
	PointResultType result;
	int z;
	double err;
	bool computed;
};

//textures per pyramid
class ComputeTextures {

public:
	cudaTextureObject_t Ycur, Yprev;

	__host__ void create(int64_t idx, int64_t idxPrev, const CudaData& core);

	__host__ void destroy();
};

//parameters for kernel launch
struct ComputeKernelParam {
	double* debugData;
	size_t debugDataSize;
	int3 blk;
	int3 thr;
	size_t shdBytes;
	cudaStream_t stream;
	int64_t frameIdx;
	volatile char* d_interrupt;
};

void kernelComputeCall(ComputeKernelParam param, ComputeTextures& tex, CudaPointResult* d_results);

//----------------------------------
// interface to host callers
//----------------------------------

/*
brief probe present cuda devices and capabilities
*/
CudaProbeResult cudaProbeRuntime();

/*
@brief initialize cuda device
@param core: CudaData structure
@param yuvFrame: the object used later to transfer frame data
*/
void cudaInit(CudaData& core, int devIdx, const cudaDeviceProp& prop, ImageYuv& yuvFrame);

/*
@brief read a frame into device memory
@param frameIdx: frame index to read
@param core: CudaData structure
@param inputFrame: YUV pixel data to load into device for processing
*/
void cudaReadFrame(int64_t frameIdx, const CudaData& core, const ImageYuv& inputFrame);

/*
@brief create image pyramid
*/
void cudaCreatePyramid(int64_t frameIdx, const CudaData& core);

/*
@brief compute displacements between frame and previous frame in video for part of a frame
*/
void cudaCompute(int64_t frameIdx, const CudaData& core, const cudaDeviceProp& props);

/*
@brief return vector of results from async computation
*/
void cudaComputeTerminate(int64_t frameIdx, const CudaData& core, std::vector<PointResult>& results);

/*
@brief transform a frame and prepare for output of pixel data
*/
void cudaOutput(int64_t frameIdx, const CudaData& core, std::array<double, 6> trf);

/*
@brief output pixel data to host
*/
void cudaOutputCpu(int64_t frameIndex, ImageYuv& image, const CudaData& core);

/*
@brief output pixel data in ARGB format to host
*/
void cudaOutputCpu(int64_t frameIndex, ImageRGBA& image, const CudaData& core);

/*
@brief output pixel data to cuda encoding
*/
void cudaOutputCuda(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch, const CudaData& core);

/*
@brief only encode given nv12 data
*/
void encodeNvData(const std::vector<unsigned char>& nv12, unsigned char* nvencPtr);

/*
@brief get NV12 data prepared for cuda encoding
*/
void getNvData(std::vector<unsigned char>& nv12, unsigned char* cudaNv12ptr);

/*
@brief shutdown cuda device
*/
void cudaShutdown(const CudaData& core);

/*
@brief get debug data from device
@return timing values and debug data
*/
void getDebugData(const CudaData& core, const std::string& imageFile, std::function<void(size_t,size_t,double*)> fcn);

/*
@brief get transformed float output for testing
*/
void cudaGetTransformedOutput(float* warpedData, const CudaData& core);

/*
@brief get pyramid image for given index
*/
void cudaGetPyramid(float* pyramid, const CudaData& core, int64_t frameIndex);

/*
@brief get input iomage from buffers
*/
void cudaGetInput(ImageYuv& image, const CudaData& core, int64_t frameIndex);

/*
@brief get current input frame for progress display
*/
void cudaGetCurrentInputFrame(ImageRGBA& image, const CudaData& core, int64_t frameIndex);

/*
@brief get current output frame for progress display
*/
void cudaGetTransformedOutput(ImageRGBA& image, const CudaData& core);
