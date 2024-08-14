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
#include "CoreData.hpp"
#include "DeviceInfoBase.hpp"
#include "Mat.hpp"
#include "DeviceInfo.hpp"

namespace cl {
	void probeRuntime(OpenClInfo& clinfo); //called on startup
	void init(CoreData& core, ImageYuv& inputFrame, const DeviceInfoBase* device); //called from constructor of MovieFrame
	void shutdown(const CoreData& core); //called from destructor of MovieFrame

	void inputData(int64_t frameIdx, const CoreData& core, const ImageYuv& inputFrame);
	void createPyramid(int64_t frameIdx, const CoreData& core);
	void computeStart(int64_t frameIdx, const CoreData& core);
	void computeTerminate(int64_t frameIdx, const CoreData& core, std::vector<PointResult>& results);
	void outputData(int64_t frameIdx, const CoreData& core, std::array<double, 6> trf);
	void outputDataCpu(int64_t frameIndex, const CoreData& core, ImageYuv& image);
	void outputDataCpu(int64_t frameIndex, const CoreData& core, ImageRGBA& image);

	void getInput(int64_t idx, ImageYuv& image, const CoreData& core);
	Matf getTransformedOutput(const CoreData& core);
	void getPyramid(float* pyramid, int64_t index, const CoreData& core);
	void getCurrentInputFrame(ImageRGBA& image, int64_t idx);
	void getTransformedOutput(ImageRGBA& image);

	void readImage(Image src, size_t destPitch, void* dest, CommandQueue queue);
	void compute(int64_t frameIdx, const CoreData& core, int rowStart, int rowEnd);
}
