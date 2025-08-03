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

#include "FrameExecutor.hpp"
#include "Affine2D.hpp"

//---------------------------------------------------------------------
//---------- CPU FRAME ------------------------------------------------
//---------------------------------------------------------------------

struct FilterKernel {
	static const int maxSize = 8;
	int siz;
	float k[maxSize];
};

class CpuFrame : public FrameExecutor {

public:
	CpuFrame(MainData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool);

	void inputData(int64_t frameIndex, const ImageYuv& inputFrame) override;
	void createPyramid(int64_t frameIndex, const Affine2D& trf, bool warp) override;
	void computeStart(int64_t frameIndex, std::vector<PointResult>& results) override;
	void computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) override;
	void outputData(int64_t frameIndex, const Affine2D& trf) override;
	void getOutputYuv(int64_t frameIndex, ImageYuvData& image) override;
	void getOutputRgba(int64_t frameIndex, ImageRGBA& image) override;
	void getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) override;
	Matf getTransformedOutput() const override;
	Matf getPyramid(int64_t frameIndex) const override;
	void getInput(int64_t frameIndex, ImageYuv& image) const override;
	void getInput(int64_t frameIndex, ImageRGBA& image) const override;
	void getWarped(int64_t frameIndex, ImageRGBA& image) override;

private:
	FilterKernel filterKernels[4] = {
		{5, {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f}},
		{3, {0.25f, 0.5f, 0.25f}},
		{3, {0.25f, 0.5f, 0.25f}},
		{3, {-0.5f, 0.0f, 0.5f}},
	};

	class CpuPyramid {

	public:
		int64_t frameIndex = -1;
		std::vector<Matf> mY;

		CpuPyramid(MainData& data);
	};

	//frame input buffer, number of frames = frameBufferCount
	std::vector<ImageYuv> mYUV;

	//holds image pyramids
	std::vector<CpuPyramid> mPyr;

	//buffers the last output frame, 3 mats, to be used to blend background of next frame
	std::vector<Matf> mPrevOut;

	//buffer for generating output from input yuv and transformation
	std::vector<Matf> mBuffer;
	Matf mYuvPlane, mFilterBuffer, mFilterResult;

	//final output
	ImageYuv mOutput;
};
