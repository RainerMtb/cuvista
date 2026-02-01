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

//replace cuda stuff with empty shell when cuda is not included in build
#if defined(BUILD_CUDA) && BUILD_CUDA == 0

//dummy code to replace cuda stuff
#include "MovieFrame.hpp"
#include "Mat.hpp"

struct NvPacket {};

struct CudaProbeResult {
	int runtimeVersion;
	int driverVersion;
	std::vector<cudaDeviceProp> props;
};

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

class NvEncoder {
public:
	NvEncoder(int cudaIndex) {}
	void probeEncoding(uint32_t* nvencVersionApi, uint32_t* nvencVersionDriver) {}
	void probeSupportedCodecs(DeviceInfoCuda& deviceInfoCuda) {}
};

class CudaExecutor : public FrameExecutor {
public:
	CudaExecutor(MainData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
		FrameExecutor(data, deviceInfo, frame, pool)
	{}

	void cudaInit(CoreData& core, int devIdx, const cudaDeviceProp& prop, ImageYuv& yuvFrame) {}
	void inputData(int64_t frameIndex, const ImageYuv& inputFrame) override {}
	void createPyramid(int64_t frameIndex, AffineDataFloat trf = {}, bool warp = false) override {}
	void computeStart(int64_t frameIndex, std::span<PointResult> results) override {}
	void computeTerminate(int64_t frameIndex, std::span<PointResult> results) override {}
	void outputData(int64_t frameIndex, AffineDataFloat trf) override {}
	void getOutputYuv(int64_t frameIndex, ImageYuv& image) const override {}
	void getOutputImage(int64_t frameIndex, ImageBaseRgb& image) const override {}
	bool getOutputNvenc(int64_t frameIndex, ImageNV12& image, unsigned char* cudaNv12ptr) const override { return true; }
	void getInput(int64_t frameIndex, ImageYuv& image) const override {}
	void getInput(int64_t frameIndex, ImageBaseRgb& image) const override {}
	void getWarped(int64_t frameIndex, ImageBaseRgb& image) override {}

	void cudaGetTransformedOutput(float* data) const {}
	void cudaGetPyramid(int64_t frameIndex, float* data) const {}
};

inline CudaProbeResult cudaProbeRuntime() { return { 0, 0 }; }

#else

#include "NvEncoder.hpp"
#include "cuDeshaker.cuh"

#endif