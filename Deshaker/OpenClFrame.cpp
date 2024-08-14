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

#include "OpenClFrame.hpp"

OpenClFrame::OpenClFrame(MainData& data, MovieReader& reader, MovieWriter& writer) :
	MovieFrame(data, reader, writer) 
{
	device = data.deviceList[data.deviceSelected];
	cl::init(data, mBufferFrame, device);
}

OpenClFrame::~OpenClFrame() {
	cl::shutdown(mData);
}

void OpenClFrame::inputData() {
	cl::inputData(mBufferFrame.index, mData, mBufferFrame);
}

void OpenClFrame::createPyramid(int64_t frameIndex) {
	cl::createPyramid(frameIndex, mData);
}

void OpenClFrame::computeStart(int64_t frameIndex) {
	cl::computeStart(frameIndex, mData);
}

void OpenClFrame::computeTerminate(int64_t frameIndex) {
	cl::computeTerminate(frameIndex, mData, mResultPoints);
}

void OpenClFrame::outputData(const AffineTransform& trf) {
	cl::outputData(trf.frameIndex, mData, trf.toArray());
}

void OpenClFrame::getOutput(int64_t frameIndex, ImageYuv& image) {
	cl::outputDataCpu(frameIndex, mData, image);
}

void OpenClFrame::getOutput(int64_t frameIndex, ImageRGBA& image) {
	cl::outputDataCpu(frameIndex, mData, image);
}

void OpenClFrame::getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) {
	throw std::runtime_error("not supported");
}

void OpenClFrame::getWarped(int64_t frameIndex, ImageRGBA& image) {
	cl::getTransformedOutput(image);
}

Mat<float> OpenClFrame::getPyramid(int64_t index) const {
	Mat<float> out = Mat<float>::zeros(mData.pyramidRowCount, mData.w);
	cl::getPyramid(out.data(), index, mData);
	return out;
}

Mat<float> OpenClFrame::getTransformedOutput() const {
	return cl::getTransformedOutput(mData);
}

void OpenClFrame::getInput(int64_t idx, ImageYuv& image) const {
	return cl::getInput(idx, image, mData);
}

void OpenClFrame::getInput(int64_t frameIndex, ImageRGBA& image) {
	cl::getCurrentInputFrame(image, frameIndex);
}

MovieFrameId OpenClFrame::getId() const {
	return { "OpenCL", device->getName() };
}