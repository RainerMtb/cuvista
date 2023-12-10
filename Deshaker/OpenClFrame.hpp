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
	OpenClFrame(MainData& data) : MovieFrame(data) {
		device = data.deviceList[data.deviceSelected];
		cl::init(data, inputFrame, device);
	}

	~OpenClFrame() {
		cl::shutdown(mData);
	}

	void inputData(ImageYuv& frame) override {
		cl::inputData(mStatus.frameInputIndex, mData, frame);
	}

	void createPyramid() override {
		cl::createPyramid(mStatus.frameInputIndex, mData);
	}

	void computeStart() override {
		cl::compute(mStatus.frameInputIndex, mData);
	}

	void computeTerminate() override {
		cl::computeTerminate(mStatus.frameInputIndex, mData, resultPoints);
	}

	void outputData(const AffineTransform& trf, OutputContext outCtx) override {
		cl::outputData(mStatus.frameWriteIndex, mData, outCtx, trf.toArray());
	}

	Mat<float> getPyramid(size_t idx) const override {
		Mat<float> out = Mat<float>::zeros(mData.pyramidRowCount, mData.w);
		cl::getPyramid(out.data(), idx, mData);
		return out;
	}

	Mat<float> getTransformedOutput() const override {
		return cl::getTransformedOutput(mData);
	}

	ImageYuv getInput(int64_t idx) const override {
		return cl::getInput(idx, mData);
	}

	void getCurrentInputFrame(ImagePPM& image) override {
		cl::getCurrentInputFrame(image, mStatus.frameReadIndex - 1);
	}

	void getTransformedOutput(ImagePPM& image) override {
		cl::getTransformedOutput(image);
	}

	std::string name() const override {
		return device->getName();
	}
};
