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
#include "DeviceInfo.hpp"
#include "Mat.h"


class DeviceInfoCl : public DeviceInfo {
public:
	cl::Device device;

	DeviceInfoCl(DeviceType type, size_t targetIndex, int64_t maxPixel, cl::Device device)
		: DeviceInfo(type, targetIndex, maxPixel)
		, device { device } 
	{}

	std::string getName() const override;
};

struct OpenClInfo {
	std::vector<DeviceInfoCl> devices;
	std::string version;
};

namespace cl {
	OpenClInfo probeRuntime(); //called on startup
	void init(CoreData& core, ImageYuv& inputFrame, OpenClInfo clinfo, std::size_t devIdx); //called from constructor of MovieFrame
	void shutdown(const CoreData& core); //called from destructor of MovieFrame

	void inputData(int64_t frameIdx, const CoreData& core, const ImageYuv& inputFrame);
	void createPyramid(int64_t frameIdx, const CoreData& core);
	void computePartOne();
	void computePartTwo();
	void computeTerminate(int64_t frameIdx, const CoreData& core, std::vector<PointResult>& results);
	void outputData(int64_t frameIdx, const CoreData& core, OutputContext outCtx, std::array<double, 6> trf);

	ImageYuv getInput(int64_t idx, const CoreData& core);
	Matf getTransformedOutput(const CoreData& core);
	void getPyramid(float* pyramid, size_t idx, const CoreData& core);
	void getCurrentInputFrame(ImagePPM& image, int64_t idx);
	void getCurrentOutputFrame(ImagePPM& image);
}
