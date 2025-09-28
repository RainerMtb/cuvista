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

#include "clHeaders.hpp"
#include "clFunctions.hpp"
#include "CoreData.hpp"
#include "DeviceInfoBase.hpp"
#include "Mat.hpp"
#include "DeviceInfo.hpp"
#include "Image2.hpp"
#include "FrameExecutor.hpp"


class OpenClExecutor : public FrameExecutor {

private:
	cl::Data clData;

public:
	OpenClExecutor(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool);

	void init();
	void inputData(int64_t frameIndex, const ImageYuv& inputFrame) override;
	void createPyramid(int64_t frameIndex, AffineDataFloat trf, bool warp) override;
	void computeStart(int64_t frameIndex, std::vector<PointResult>& results) override;
	void computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) override;
	void outputData(int64_t frameIndex, AffineDataFloat trf) override;
	void getOutputYuv(int64_t frameIndex, ImageYuv& image) const override;
	void getOutputImage(int64_t frameIndex, ImageBaseRgb& image) const override;
	bool getOutputNvenc(int64_t frameIndex, ImageNV12& image, unsigned char* cudaNv12ptr) const override;
	Matf getTransformedOutput() const override;
	Matf getPyramid(int64_t frameIndex) const override;
	void getInput(int64_t frameIndex, ImageYuv& image) const override;
	void getInput(int64_t frameIndex, ImageRGBA& image) const override;
	void getWarped(int64_t frameIndex, ImageRGBA& image) override;

private:
	void compute(int64_t frameIndex, const CoreData& core, int rowStart, int rowEnd);
	void readImage(cl::Image src, size_t destPitch, void* dest, cl::CommandQueue queue) const;
	void readImage(cl::Image2DArray src, int idx, size_t destPitch, void* dest, cl::CommandQueue queue) const;
};

namespace cl {

	void probeRuntime(OpenClInfo& clinfo); //called on startup
}
