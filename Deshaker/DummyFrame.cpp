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
#include "CudaFrame.hpp"
#include "MainData.hpp"
#include "MovieFrame.hpp"

DummyFrame::DummyFrame(MainData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
	FrameExecutor(data, deviceInfo, frame, pool),
	mFrames(data.bufferCount, { data.h, data.w, data.w }) {}

Matf DummyFrame::getTransformedOutput() const { 
	return {}; 
};

Matf DummyFrame::getPyramid(int64_t frameIndex) const { 
	return {}; 
};

void DummyFrame::init() {}

void DummyFrame::inputData(int64_t frameIndex, const ImageYuv& inputFrame) {
	size_t idx = frameIndex % mFrames.size();
	inputFrame.copyTo(mFrames[idx], mPool);
}

void DummyFrame::outputData(int64_t frameIndex, const Affine2D& trf) {
	size_t idx = frameIndex % mFrames.size();
}

void DummyFrame::getOutput(int64_t frameIndex, ImageYuv& image) {
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].copyTo(image, mPool);
}

void DummyFrame::getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) {
	static std::vector<unsigned char> nv12(cudaPitch * mData.h * 3 / 2);
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].toNV12(nv12, cudaPitch, mPool);
	encodeNvData(nv12, cudaNv12ptr);
}

void DummyFrame::getOutput(int64_t frameIndex, ImageRGBA& image) {
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].toRGBA(image, mPool);
}

void DummyFrame::getWarped(int64_t frameIndex, ImageRGBA& image) {
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].toRGBA(image, mPool);
}

void DummyFrame::getInput(int64_t frameIndex, ImageYuv& image) const {
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].copyTo(image);
}

void DummyFrame::getInput(int64_t frameIndex, ImageRGBA& image) const {
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].toRGBA(image, mPool);
}