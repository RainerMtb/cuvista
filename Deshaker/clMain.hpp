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

#include "AffineTransform.hpp"
#include "CoreData.cuh"
#include "Image.hpp"
#include "DeviceInfo.hpp"


struct OpenClInfo {
	std::vector<DeviceInfoCl> devices;
	std::string version;
};

namespace cl {
	OpenClInfo probeRuntime(); //called on startup
	void init(CoreData& core, ImageYuv& inputFrame, std::size_t devIdx); //called from constructor of MovieFrame
	void shutdown();

	void inputData(int64_t frameIdx, const CoreData& core, const ImageYuv& inputFrame);
	void createPyramid(int64_t frameIdx, const CoreData& core);
	void computePartOne();
	void computePartTwo();
	void computeTerminate();
	void outputData(int64_t frameIdx, const CoreData& core, OutputContext outCtx, std::array<double, 6> trf);

	ImageYuv getInput(int64_t idx);
	Matf getTransformedOutput();
	void getPyramid(float* pyramid, size_t idx, const CoreData& core);
	bool getCurrentInputFrame(ImagePPM& image);
	bool getCurrentOutputFrame(ImagePPM& image);
}
