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
#include "cuUtil.cuh"
#include "cuDecompose.cuh"


struct CudaProbeResult {
	int runtimeVersion = 0;
	int driverVersion = 0;
	std::vector<cudaDeviceProp> props;
};

struct DebugData {
	std::vector<double> debugData;
	ImageBgr kernelTimings;
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
	//host memory
	ImageYuv* h_input = nullptr;
	CudaPointResult* h_results = nullptr;

	//device memory
	float* d_pyrData = nullptr;
	float* d_bufferH = nullptr;
	float* d_bufferV = nullptr;

	unsigned char* d_input = nullptr;         //yuv input
	unsigned char* d_vuyxData = nullptr;      //continuous array of all pixel values in yuv format, allocated on device
	unsigned char* d_output = nullptr;        //image data for output, allocated as vuyx, but reused

	struct {
		float4* data;
		float4* start;
		float4* warped;
		float4* filterH;
		float4* filterV;
		float4* output;
		float4* background;
	} out = {};

	//luma sum
	int64_t* d_luma = nullptr;

	//results from compute kernel
	CudaPointResult* d_results = nullptr;

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

	void writeText(const std::string& text, int x0, int y0, float4* deviceData, int devicePitch);

public:
	CudaExecutor(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool);
	~CudaExecutor();

	void init() override;
	Image8& inputDestination(int64_t frameIndex) override;
	void inputData(int64_t frameIndex) override;
	int64_t createPyramid(int64_t frameIndex, AffineDataFloat trf = {}, bool warp = false) override;
	void computeStart(int64_t frameIndex, std::span<PointResult> results) override;
	void computeTerminate(int64_t frameIndex, std::span<PointResult> results) override;
	void outputData(int64_t frameIndex, AffineDataFloat trf) override;
	void getOutput(int64_t frameIndex, Image8& image) const override;
	bool getOutput(int64_t frameIndex, Image8& image, int cudaNv12stride, unsigned char* cudaNv12ptr) const override;
	void getInput(int64_t frameIndex, Image8& image) const override;

	Mat<float> getTransformedOutput() const = 0;
	void cudaGetTransformedOutput(float* warpedData, size_t h, size_t w) const;
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
