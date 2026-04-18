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

class CpuFrame : public FrameExecutor {

	using Matf4 = Mat<FloatVuyx>;

public:
	CpuFrame(MainData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool);

	void init() override;
	Image8& inputDestination(int64_t frameIndex) override;
	void inputData(int64_t frameIndex) override;
	int64_t createPyramid(int64_t frameIndex, AffineDataFloat trf = {}, bool warp = false) override;
	void adjustPyramid(int64_t frameIndex, float gamma) override;
	void computeStart(int64_t frameIndex, std::span<PointResult> results) override;
	void computeTerminate(int64_t frameIndex, std::span<PointResult> results) override;
	void outputData(int64_t frameIndex, AffineDataFloat trf) override;
	void getOutput(int64_t frameIndex, Image8& image) const override;
	bool getOutput(int64_t frameIndex, Image8& image, int cudaNv12stride, unsigned char* cudaNv12ptr) const override;
	Matf getTransformedOutput() const override;
	Matf getPyramid(int64_t frameIndex) const override;
	void getInput(int64_t frameIndex, Image8& image) const override;

private:
	std::vector<float> filterKernelY = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };

	std::vector<FloatVuyx> filterKernel4 = {
		{ 0.0f,  0.0f,  0.0625f, 0.0f },
		{ 0.25f, 0.25f, 0.25f,   0.0f },
		{ 0.5f,  0.5f,  0.375f,  0.0f },
		{ 0.25f, 0.25f, 0.25f,   0.0f },
		{ 0.0f,  0.0f,  0.0625f, 0.0f }
	};

	class CpuPyramid {

	public:
		int64_t frameIndex = -1;
		std::vector<Matf> mY;

		CpuPyramid(CoreData& data);
		Matf getCompletePyramid(int64_t index, size_t h, size_t w) const;
	};

	//frame input buffer
	ImageYuv mReadBuffer;
	std::vector<ImageYuv> mInput;

	//holds image pyramids
	std::vector<CpuPyramid> mPyr;

	//buffers the last output frame, to be used to blend background of next frame
	Matf4 mPrevOut;

	//buffer for generating output from input yuv and transformation
	Matf mFilterBuffer1, mFilterBuffer2;
	Matf4 mBuffer4, mFilterBuffer4, mFilterResult4;

	//final output
	ImageVuyxFloat mOutput;

	void createPyramidLevels(CpuPyramid& pyr);
};
