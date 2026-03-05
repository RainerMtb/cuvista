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

#include "DummyFrame.hpp"
#include "MainData.hpp"
#include "MovieFrame.hpp"

DummyFrame::DummyFrame(MainData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
	FrameExecutor(data, deviceInfo, frame, pool)
{
	mFrames.resize(data.bufferCount);
	for (int i = 0; i < mFrames.size(); i++) mFrames[i] = ImageYuv(data.h, data.w, data.w);
}

Matf DummyFrame::getTransformedOutput() const { 
	return {}; 
};

Matf DummyFrame::getPyramid(int64_t frameIndex) const { 
	return {}; 
};

void DummyFrame::inputData(int64_t frameIndex, const ImageYuv& inputFrame) {
	size_t idx = frameIndex % mFrames.size();
	inputFrame.copyTo(mFrames[idx], mPool);
	debugLogger->format("input frame {} {}", frameIndex, idx);
	//mFrames[idx].writeText(std::to_string(frameIndex), 0, 0, 3, 3, im::TextAlign::TOP_LEFT);
}

void DummyFrame::outputData(int64_t frameIndex, AffineDataFloat trf) {}

void DummyFrame::getOutput(int64_t frameIndex, Image8& image) const {
	size_t idx = frameIndex % mFrames.size();
	if (image.colorBase() == ColorBase::YUV) {
		mFrames[idx].copyTo(image, mPool);

	} else if (image.colorBase() == ColorBase::RGB) {
		mFrames[idx].convertTo(image, mPool);
	}
	debugLogger->format("output frame {} {}", frameIndex, idx);
}

bool DummyFrame::getOutput(int64_t frameIndex, Image8& image, int cudaNv12stride, unsigned char* cudaNv12ptr) const {
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].convertTo(image, mPool);
	return true;
}

void DummyFrame::getWarped(int64_t frameIndex, Image8& image) {
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].convertTo(image, mPool);
}

void DummyFrame::getInput(int64_t frameIndex, Image8& image) const {
	size_t idx = frameIndex % mFrames.size();
	if (image.colorBase() == ColorBase::YUV) {
		mFrames[idx].copyTo(image);

	} else if (image.colorBase() == ColorBase::RGB) {
		mFrames[idx].convertTo(image, mPool);
	}
}
