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
#include "FrameExecutor.hpp"
#include "AffineCore.hpp"

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

	__host__ void create(int64_t idx, int64_t idxPrev, const CudaData& core, float* pyrBase);

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

CudaProbeResult cudaProbeRuntime();


class CudaExecutor : public FrameExecutor {

private:
	unsigned char* d_yuvData;			     //continuous array of all pixel values in yuv format, allocated on device
	unsigned char** d_yuvRows;			     //index into rows of pixels, allocated on device
	unsigned char*** d_yuvPlanes;		     //index into Y-U-V planes of frames, allocated on device 

	unsigned char* d_yuvOut;   //image data for encoding on host
	unsigned char* d_rgba;     //image data for progress update

	struct {
		float4* data;
		float4* start;
		float4* warped;
		float4* filterH;
		float4* filterV;
		float4* final;
		float4* background;
	} out;

	float* d_bufferH;
	float* d_bufferV;

	float* d_pyrData;
	float** d_pyrRows;

	//results from compute kernel
	CudaPointResult* d_results;
	CudaPointResult* h_results;

	//init cuda streams
	std::vector<cudaStream_t> cs;

	//data output from kernels for later analysis
	cu::DebugData debugData = {};

	//registered memory
	void* registeredMemPtr = nullptr;

	//textures used in compute kernel
	ComputeTextures compTex;

	//signal to interrupt compute kernel
	char* d_interrupt;

	//keep track of frames in the buffer
	std::vector<int64_t> frameIndizes;

	//properties of device in use
	cudaDeviceProp props = {};

	void getDebugData(const CudaData& core, const std::string& imageFile, std::function<void(size_t, size_t, double*)> fcn);

public:
	CudaExecutor(CudaData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool);
	~CudaExecutor();

	void cudaInit(CudaData& core, int devIdx, const cudaDeviceProp& prop, ImageYuv& yuvFrame);
	void inputData(int64_t frameIndex, const ImageYuv& inputFrame) override;
	void createPyramid(int64_t frameIndex) override;
	void computeStart(int64_t frameIndex, std::vector<PointResult>& results) override;
	void computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) override;
	void cudaOutputData(int64_t frameIndex, const AffineCore& trf);
	void getOutput(int64_t frameIndex, ImageYuv& image) override;
	void getOutput(int64_t frameIndex, ImageRGBA& image) override;
	void getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) override;
	void cudaGetTransformedOutput(float* data) const;
	void cudaGetPyramid(int64_t frameIndex, float* data) const;
	void getInput(int64_t frameIndex, ImageYuv& image) const override;
	void getInput(int64_t frameIndex, ImageRGBA& image) const override;
	void getWarped(int64_t frameIndex, ImageRGBA& image) override;
};

/*
@brief only encode given nv12 data
*/
void encodeNvData(const std::vector<unsigned char>& nv12, unsigned char* nvencPtr);

/*
@brief get NV12 data prepared for cuda encoding
*/
void getNvData(std::vector<unsigned char>& nv12, unsigned char* cudaNv12ptr);
