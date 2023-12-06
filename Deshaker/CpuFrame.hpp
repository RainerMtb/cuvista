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

class CpuFrame : public MovieFrame {

public:
	CpuFrame(MainData& data);

	void inputData(ImageYuv& frame) override;
	void createPyramid() override;
	void computeStart() override;
	void computeTerminate() override;
	void outputData(const AffineTransform& trf, OutputContext outCtx) override;
	Mat<float> getTransformedOutput() const override;
	Mat<float> getPyramid(size_t idx) const override;
	ImageYuv getInput(int64_t index) const override;
	void getCurrentInputFrame(ImagePPM& image) override;
	void getCurrentOutputFrame(ImagePPM& image) override;
	std::string name() const override { return "Cpu Only"; }

protected:

	class CpuFrameItem {

	public:
		int64_t frameIndex = -1;
		std::vector<Mat<float>> mY;

		CpuFrameItem(MainData& data);
	};

	//frame input buffer, number of frames = frameBufferCount
	std::vector<ImageYuv> mYUV;

	//holds image pyramids
	std::vector<CpuFrameItem> mPyr;

	//buffers the last output frame, 3 mats, to be used to blend background of next frame
	std::vector<Mat<float>> mPrevOut;

	//buffer for generating output from input yuv and transformation
	std::vector<Mat<float>> mBuffer;
	Mat<float> mYuv, mFilterBuffer, mFilterResult;
};
