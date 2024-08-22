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

#include "DeviceInfoBase.hpp"
#include "ThreadPoolBase.h"
#include "Image2.hpp"
#include "CoreData.hpp"

template <class T> class Mat;
class MovieFrame;
class Affine2D;
struct CudaData;

class FrameExecutor {

public:
	CudaData& mData;
	DeviceInfoBase& mDeviceInfo;
	MovieFrame& mFrame;
	ThreadPoolBase& mPool;

	FrameExecutor(CudaData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
		mData { data },
		mDeviceInfo { deviceInfo },
		mFrame { frame },
		mPool { pool } {}
	
	//get frame data from reader into frame object
	virtual void inputData(int64_t frameIndex, const ImageYuv& inputFrame) = 0;
	//set up image pyramid
	virtual void createPyramid(int64_t frameIndex) = 0;
	//start computation asynchronously for some part of a frame
	virtual void computeStart(int64_t frameIndex, std::vector<PointResult>& results) = 0;
	//start computation asynchronously for second part and get results
	virtual void computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) = 0;
	//prepare data for output to writer
	virtual void outputData(int64_t frameIndex, const Affine2D& trf) = 0;
	//prepare data for encoding on cpu
	virtual void getOutput(int64_t frameIndex, ImageYuv& image) = 0;
	//prepare data for encoding on cpu
	virtual void getOutput(int64_t frameIndex, ImageRGBA& image) = 0;
	//prepare data for encoding on cuda
	virtual void getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) = 0;
	//get transformed image as Mat<float> where YUV color planes are stacked vertically
	virtual Mat<float> getTransformedOutput() const = 0;
	//get image pyramid as single Mat<float> where images are stacked vertically from large to small
	virtual Mat<float> getPyramid(int64_t frameIndex) const = 0;
	//get input image as stored in frame buffers
	virtual void getInput(int64_t frameIndex, ImageYuv& image) const = 0;
	//get input image as stored in frame buffers
	virtual void getInput(int64_t frameIndex, ImageRGBA& image) const = 0;
	//output rgb data warped but not unsharped
	virtual void getWarped(int64_t frameIndex, ImageRGBA& image) = 0;
	//destructor
	virtual ~FrameExecutor() {}
};