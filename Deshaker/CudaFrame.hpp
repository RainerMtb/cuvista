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

#include "MovieFrame.hpp"


class CudaFrame : public MovieFrame {

private:
	DeviceInfoCuda* device;

public:
	CudaFrame(MainData& data) : MovieFrame(data) {
		DeviceInfo* dev = data.deviceList[data.deviceSelected];
		assert(dev->type == DeviceType::CUDA && "device type must be CUDA here");
		device = static_cast<DeviceInfoCuda*>(dev);
		cudaInit(data, device->cudaIndex, device->props, inputFrame);
	}

	~CudaFrame() {
		auto fcn = [] (size_t h, size_t w, double* ptr) { Matd::fromArray(h, w, ptr, false).toConsole("", 16); };
		//getDebugData(mData, "f:/kernel.bmp", fcn);
		cudaShutdown(mData);
	}

	void inputData(ImageYuv& frame) override {
		cudaReadFrame(mStatus.frameInputIndex, mData, frame);
	}

	void createPyramid() override {
		cudaCreatePyramid(mStatus.frameInputIndex, mData);
	}

	void computeStart() override {
		cudaCompute(mStatus.frameInputIndex, mData, device->props);
	}

	void computeTerminate() override {
		cudaComputeTerminate(mStatus.frameInputIndex, mData, resultPoints);
	}

	void outputData(const AffineTransform& trf, OutputContext outCtx) override {
		cudaOutput(mStatus.frameWriteIndex, mData, outCtx, trf.toArray());
	}

	Mat<float> getTransformedOutput() const override {
		Mat<float> warped = Mat<float>::allocate(3LL * mData.h, mData.w);
		cudaGetTransformedOutput(warped.data(), mData);
		return warped;
	}

	Mat<float> getPyramid(size_t idx) const override {
		Mat<float> out = Mat<float>::allocate(mData.pyramidRowCount, mData.w);
		cudaGetPyramid(out.data(), idx, mData);
		return out;
	}

	ImageYuv getInput(int64_t index) const override {
		return cudaGetInput(index, mData);
	}

	void getCurrentInputFrame(ImagePPM& image) override {
		cudaGetCurrentInputFrame(image, mData, mStatus.frameReadIndex - 1);
	}

	void getTransformedOutput(ImagePPM& image) override {
		cudaGetTransformedOutput(image, mData);
	}

	std::string name() const override {
		return device->getName();
	}
};