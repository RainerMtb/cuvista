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

//---------------------------------------------------------------------
//---------- CPU FRAME ------------------------------------------------
//---------------------------------------------------------------------

struct FilterKernel {
	static const int maxSize = 8;
	int siz;
	float k[maxSize];
};

class CpuFrame : public MovieFrame {

public:
	CpuFrame(MainData& data, MovieReader& reader, MovieWriter& writer);

	void inputData() override;
	void createPyramid(int64_t frameIndex) override;
	void computeStart(int64_t frameIndex) override;
	void computeTerminate(int64_t frameIndex) override;
	void outputData(const AffineTransform& trf, OutputContext outCtx) override;
	Matf getTransformedOutput() const override;
	Matf getPyramid(size_t idx) const override;
	ImageYuv getInput(int64_t index) const override;
	void getInputFrame(int64_t frameIndex, ImagePPM& image) override;
	void getTransformedOutput(int64_t frameIndex, ImagePPM& image) override;
	std::string getClassName() const override;
	std::string getClassId() const override;

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
};
