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

#include "CudaFrame.hpp"
#include "DeviceInfo.hpp"

CudaFrame::CudaFrame(MainData& data, MovieReader& reader, MovieWriter& writer) :
	MovieFrame(data, reader, writer) 
{
	DeviceInfoBase* dev = data.deviceList[data.deviceSelected];
	assert(dev->type == DeviceType::CUDA && "device type must be CUDA here");
	device = static_cast<DeviceInfo<CudaFrame>*>(dev);
	cudaInit(data, device->cudaIndex, device->props, mBufferFrame);
}

CudaFrame::~CudaFrame() {
	auto fcn = [] (size_t h, size_t w, double* ptr) { Matd::fromArray(h, w, ptr, false).toConsole("", 16); };
	//getDebugData(mData, "f:/kernel.bmp", fcn);
	cudaShutdown(mData);
}

void CudaFrame::inputData() {
	cudaReadFrame(mBufferFrame.index, mData, mBufferFrame);
}

void CudaFrame::createPyramid(int64_t frameIndex) {
	cudaCreatePyramid(frameIndex, mData);
}

void CudaFrame::computeStart(int64_t frameIndex) {
	cudaCompute(frameIndex, mData, device->props);
}

void CudaFrame::computeTerminate(int64_t frameIndex) {
	cudaComputeTerminate(frameIndex, mData, mResultPoints);
}

void CudaFrame::outputData(const AffineTransform& trf) {
	cudaOutput(trf.frameIndex, mData, trf.toArray());
}

void CudaFrame::getOutput(int64_t frameIndex, ImageYuv& image) {
	cudaOutputCpu(frameIndex, image, mData);
}

void CudaFrame::getOutput(int64_t frameIndex, ImageRGBA& image) {
	cudaOutputCpu(frameIndex, image, mData);
}

void CudaFrame::getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) {
	cudaOutputCuda(frameIndex, cudaNv12ptr, cudaPitch, mData);
}

Mat<float> CudaFrame::getTransformedOutput() const {
	Mat<float> warped = Mat<float>::allocate(3LL * mData.h, mData.w);
	cudaGetTransformedOutput(warped.data(), mData);
	return warped;
}

Mat<float> CudaFrame::getPyramid(int64_t index) const {
	Mat<float> out = Mat<float>::allocate(mData.pyramidRowCount, mData.w);
	cudaGetPyramid(out.data(), mData, index);
	return out;
}

void CudaFrame::getInput(int64_t index, ImageYuv& image) const {
	return cudaGetInput(image, mData, index);
}

void CudaFrame::getInput(int64_t frameIndex, ImageRGBA& image) {
	cudaGetCurrentInputFrame(image, mData, frameIndex);
}

void CudaFrame::getWarped(int64_t frameIndex, ImageRGBA& image) {
	cudaGetTransformedOutput(image, mData);
}

MovieFrameId CudaFrame::getId() const {
	return { "Cuda", device->getName() };
}