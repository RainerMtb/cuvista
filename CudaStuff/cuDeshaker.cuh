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
#include "AffineData.hpp"
#include "cuUtil.cuh"
#include "cuDecompose.cuh"


struct CudaProbeResult {
	int runtimeVersion;
	int driverVersion;
	std::vector<cudaDeviceProp> props;
};

struct DebugData {
	std::vector<double> debugData;
	ImageBGR kernelTimings;
};

class CudaPointResult {

public:
	double u, v;
	double length;
	int xm, ym;
	int idx, ix0, iy0;
	PointResultType result;
	int z;
	int direction;
	bool computed;

	int64_t timeStart; //nanos
	int64_t timeStop;  //nanos
};

//textures per pyramid
class ComputeTextures {

public:
	cudaTextureObject_t Y[2];

	__host__ void create(int64_t idx, int64_t idxPrev, float* pyrBase, const CoreData& core);

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
		float4* output;
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

	void getDebugData(const std::string& imageFile, std::function<void(size_t, size_t, double*)> fcn);

	//size parameters for kernels
	int3 computeBlocks() const;
	int3 computeThreads() const;

	bool checkKernelParameters(int3 threads, int3 blocks, size_t shdsize) const;
	bool checkKernelParameters(int3 threads, int3 blocks) const;
	bool checkKernelParameters() const;

	void writeText(const std::string& text, int x0, int y0, int scaleX, int scaleY, float* deviceData);

public:
	CudaExecutor(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool);
	~CudaExecutor();

	void cudaInit(CoreData& coreData, int devIdx, const cudaDeviceProp& prop, ImageYuv& yuvFrame);
	void inputData(int64_t frameIndex, const ImageYuv& inputFrame) override;
	void createPyramid(int64_t frameIndex, AffineDataFloat trf = {}, bool warp = false) override;
	void computeStart(int64_t frameIndex, std::span<PointResult> results) override;
	void computeTerminate(int64_t frameIndex, std::span<PointResult> results) override;
	void outputData(int64_t frameIndex, AffineDataFloat trf) override;
	void getOutputYuv(int64_t frameIndex, ImageYuv& image) const override;
	void getOutputImage(int64_t frameIndex, ImageBaseRgb& image) const override;
	bool getOutputNvenc(int64_t frameIndex, ImageNV12& image, unsigned char* cudaNv12ptr) const override;
	void getInput(int64_t frameIndex, ImageYuv& image) const override;
	void getInput(int64_t frameIndex, ImageBaseRgb& image) const override;
	void getWarped(int64_t frameIndex, ImageBaseRgb& image) override;

	void cudaGetTransformedOutput(float* data) const;
	void cudaGetPyramid(int64_t frameIndex, float* data) const;
};

//interface to call cuda kernels
void kernelComputeCall(ComputeKernelParam param, ComputeTextures& tex, CudaPointResult* d_results);

//see if cuda is available
CudaProbeResult cudaProbeRuntime();

//encode given nv12 data
void encodeNvData(const ImageNV12& image, unsigned char* nvencPtr);

//size parameters for kernels
dim3 configThreads();
dim3 configBlocks(dim3 threads, int width, int height);
