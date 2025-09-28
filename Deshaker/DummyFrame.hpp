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

class DummyFrame : public FrameExecutor {

private:
	std::vector<ImageYuv> mFrames;

public:
	DummyFrame(MainData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool);

	void inputData(int64_t frameIndex, const ImageYuv& inputFrame) override;
	void createPyramid(int64_t frameIndex, AffineDataFloat trf, bool warp) override {};
	void computeStart(int64_t frameIndex, std::vector<PointResult>& results) override {};
	void computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) override {};
	void outputData(int64_t frameIndex, AffineDataFloat trf) override;
	void getOutputYuv(int64_t frameIndex, ImageYuv& image) const override;
	void getOutputImage(int64_t frameIndex, ImageBaseRgb& image) const override;
	bool getOutputNvenc(int64_t frameIndex, ImageNV12& image, unsigned char* cudaNv12ptr) const override;
	Mat<float> getTransformedOutput() const override;
	Mat<float> getPyramid(int64_t frameIndex) const override;
	void getInput(int64_t frameIndex, ImageYuv& image) const override;
	void getInput(int64_t frameIndex, ImageRGBA& image) const override;
	void getWarped(int64_t frameIndex, ImageRGBA& image) override;
};
