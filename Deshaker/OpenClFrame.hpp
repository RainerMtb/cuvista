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

class OpenClFrame : public MovieFrame {

private:
	DeviceInfo* device;

public:
	OpenClFrame(MainData& data, MovieReader& reader, MovieWriter& writer) : 
		MovieFrame(data, reader, writer) 
	{
		device = data.deviceList[data.deviceSelected];
		cl::init(data, mBufferFrame, device);
	}

	~OpenClFrame() {
		cl::shutdown(mData);
	}

	void inputData() override {
		cl::inputData(mBufferFrame.index, mData, mBufferFrame);
	}

	void createPyramid(int64_t frameIndex) override {
		cl::createPyramid(frameIndex, mData);
	}

	void computeStart(int64_t frameIndex) override {
		cl::computeStart(frameIndex, mData);
	}

	void computeTerminate(int64_t frameIndex) override {
		cl::computeTerminate(frameIndex, mData, mResultPoints);
	}

	void outputData(const AffineTransform& trf) override {
		cl::outputData(trf.frameIndex, mData, trf.toArray());
	}

	void getOutput(int64_t frameIndex, ImageYuv& image) override {
		cl::outputDataCpu(frameIndex, mData, image);
	}

	void getOutput(int64_t frameIndex, ImageRGBA& argb) override {
		cl::outputDataCpu(frameIndex, mData, argb);
	}

	void getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) override {
		throw std::exception("not supported");
	}

	void getWarped(int64_t frameIndex, ImageRGBA& image) override {
		cl::getTransformedOutput(image);
	}

	Mat<float> getPyramid(size_t idx) const override {
		Mat<float> out = Mat<float>::zeros(mData.pyramidRowCount, mData.w);
		cl::getPyramid(out.data(), idx, mData);
		return out;
	}

	Mat<float> getTransformedOutput() const override {
		return cl::getTransformedOutput(mData);
	}

	void getInput(int64_t idx, ImageYuv& image) const override {
		return cl::getInput(idx, image, mData);
	}

	void getInput(int64_t frameIndex, ImageRGBA& image) override {
		cl::getCurrentInputFrame(image, frameIndex);
	}

	MovieFrameId getId() const override {
		return { "OpenCL", device->getName() };
	}
};
