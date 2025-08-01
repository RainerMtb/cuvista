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
#include "MovieFrame.hpp"

CudaFrame::CudaFrame(MainData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
	CudaExecutor(data, deviceInfo, frame, pool) 
{
	assert(deviceInfo.type == DeviceType::CUDA && "device type must be CUDA here");
	const DeviceInfoCuda* device = static_cast<const DeviceInfoCuda*>(&mDeviceInfo);
	cudaInit(mData, device->cudaIndex, *device->props, mFrame.mBufferFrame);
}

void CudaFrame::createPyramidTransformed(int64_t frameIndex, const Affine2D& trf) {
	cudaCreatePyramidTransformed(frameIndex, trf.toAffineCore());
}

void CudaFrame::outputData(int64_t frameIndex, const Affine2D& trf) {
	cudaOutputData(frameIndex, trf.toAffineCore());
}

Matf CudaFrame::getPyramid(int64_t frameIndex) const {
	Matf out = Matf::allocate(mData.pyramidRowCount, mData.w);
	cudaGetPyramid(frameIndex, out.data());
	return out;
}

Matf CudaFrame::getTransformedOutput() const {
	Matf warped = Mat<float>::allocate(3ull * mData.h, mData.w);
	cudaGetTransformedOutput(warped.data());
	return warped;
}
