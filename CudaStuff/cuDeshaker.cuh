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

#include "CoreData.hpp"
#include "FrameExecutor.hpp"
#include "AffineCore.hpp"

struct cudaDeviceProp;

struct CudaProbeResult {
	int runtimeVersion;
	int driverVersion;
	std::vector<cudaDeviceProp> props;
};

#if defined(BUILD_CUDA) && BUILD_CUDA == 0

struct cudaDeviceProp {
	char name[256];
	int major;
	int minor;
	int clockRate;
	size_t totalGlobalMem;
	int multiProcessorCount;
	int maxTexture2D[2];
	size_t sharedMemPerBlock;
};

class CudaExecutor : public FrameExecutor {
public:
	CudaExecutor(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
		FrameExecutor(data, deviceInfo, frame, pool) {}

	void inputData(int64_t frameIndex, const ImageYuv& inputFrame) override {}
	void createPyramidTransformed(int64_t frameIndex, const Affine2D& trf) override {};
	void createPyramid(int64_t frameIndex) override {};
	void computeStart(int64_t frameIndex, std::vector<PointResult>& results) override {}
	void computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) override {}
	void getOutputYuv(int64_t frameIndex, ImageYuvData& image) override {}
	void getOutputRgba(int64_t frameIndex, ImageRGBA& image) override {}
	void getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) override {}
	void getInput(int64_t frameIndex, ImageYuv& image) const override {}
	void getInput(int64_t frameIndex, ImageRGBA& image) const override {}
	void getWarped(int64_t frameIndex, ImageRGBA& image) override {}
	void outputData(int64_t frameIndex, const Affine2D& trf) override {}
	Matf getPyramid(int64_t frameIndex) const override { return Matf(); }
	Matf getTransformedOutput() const override { return Matf(); }

	void cudaInit(CoreData& core, int devIdx, const cudaDeviceProp& prop, ImageYuv& yuvFrame) {}
	void cudaOutputData(int64_t frameIndex, const AffineCore& trf) {}
	void cudaGetTransformedOutput(float* data) const {}
	void cudaGetPyramid(int64_t frameIndex, float* data) const {}
};

inline CudaProbeResult cudaProbeRuntime() { return { 0, 0 }; }
inline void encodeNvData(const std::vector<unsigned char>& nv12, unsigned char* nvencPtr) {}
inline void getNvData(std::vector<unsigned char>& nv12, unsigned char* cudaNv12ptr) {}

#else

#include "cuUtil.cuh"
#include "cuDecompose.cuh"

struct DebugData {
	std::vector<double> debugData;
	ImageBGR kernelTimings;
};

struct CudaData {
	size_t cudaMemTotal = 0;
	size_t cudaUsedMem = 0;
	size_t computeSharedMem = 0;

	int3 computeBlocks = {};
	int3 computeThreads = {};

	int strideChar = 0;      //row length in bytes for char values
	int strideFloat = 0;     //row lenhth in bytes for float values
	int strideFloatN = 0;    //number of float values in a row including padding
	int strideFloat4 = 0;    //row length in bytes for float4 struct
	int strideFloat4N = 0;   //number of float4 values

	int outBufferCount = 6;  //number of images to hold as buffers for output generation
};

class CudaPointResult {

public:
	int64_t timeStart; //nanos
	int64_t timeStop;  //nanos
	double u, v;
	int xm, ym;
	int idx, ix0, iy0;
	PointResultType result;
	int z;
	int direction;
	bool computed;
};

//textures per pyramid
class ComputeTextures {

public:
	cudaTextureObject_t Y[2];

	__host__ void create(int64_t idx, int64_t idxPrev, const CoreData& core, float* pyrBase);

	__host__ void destroy() const;
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


class CudaExecutor : public FrameExecutor {

private:
	unsigned char* d_yuvData = nullptr;	     //continuous array of all pixel values in yuv format, allocated on device
	unsigned char** d_yuvRows = nullptr;     //index into rows of pixels, allocated on device
	unsigned char*** d_yuvPlanes = nullptr;  //index into Y-U-V planes of frames, allocated on device 

	unsigned char* d_yuvOut = nullptr;       //image data for encoding on host
	unsigned char* d_rgba = nullptr;         //image data for progress update

	struct {
		float4* data;
		float4* start;
		float4* warped;
		float4* filterH;
		float4* filterV;
		float4* final;
		float4* background;
	} out = {};

	float* d_bufferH = nullptr;
	float* d_bufferV = nullptr;

	float* d_pyrData = nullptr;
	float** d_pyrRows = nullptr;

	//results from compute kernel
	CudaPointResult* d_results = nullptr;
	CudaPointResult* h_results = nullptr;

	//init cuda streams
	std::vector<cudaStream_t> cs;

	//data output from kernels for later analysis
	cu::DebugData debugData = {};

	//registered memory
	void* registeredMemPtr = nullptr;

	//textures used in compute kernel
	ComputeTextures computeTexture = {};

	//signal to interrupt compute kernel
	char* d_interrupt = nullptr;

	//keep track of frames in the buffer
	std::vector<int64_t> frameIndizes;

	//keep track of pyramids in buffer
	std::vector<int64_t> pyramidIndizes;

	//properties of device in use
	cudaDeviceProp props = {};

	void getDebugData(const CoreData& core, const std::string& imageFile, std::function<void(size_t, size_t, double*)> fcn);

public:
	CudaExecutor(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool);
	~CudaExecutor();

	void cudaInit(CoreData& core, int devIdx, const cudaDeviceProp& prop, ImageYuv& yuvFrame);
	void inputData(int64_t frameIndex, const ImageYuv& inputFrame) override;
	void createPyramid(int64_t frameIndex) override;
	void computeStart(int64_t frameIndex, std::vector<PointResult>& results) override;
	void computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) override;
	void cudaOutputData(int64_t frameIndex, const AffineCore& trf);
	void getOutputYuv(int64_t frameIndex, ImageYuvData& image) override;
	void getOutputRgba(int64_t frameIndex, ImageRGBA& image) override;
	void getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) override;
	void getInput(int64_t frameIndex, ImageYuv& image) const override;
	void getInput(int64_t frameIndex, ImageRGBA& image) const override;
	void getWarped(int64_t frameIndex, ImageRGBA& image) override;

	void cudaCreatePyramidTransformed(int64_t frameIndex, const AffineCore& trf);
	void cudaGetTransformedOutput(float* data) const;
	void cudaGetPyramid(int64_t frameIndex, float* data) const;
};

void kernelComputeCall(ComputeKernelParam param, ComputeTextures& tex, CudaPointResult* d_results);


//see if cuda is available
CudaProbeResult cudaProbeRuntime();

//only encode given nv12 data
void encodeNvData(const std::vector<unsigned char>& nv12, unsigned char* nvencPtr);

//get NV12 data prepared for cuda encoding
void getNvData(std::vector<unsigned char>& nv12, unsigned char* cudaNv12ptr);

#endif