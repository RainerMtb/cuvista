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

#include <map>
#include <memory>

#include "OutputOption.hpp"

enum class DeviceType {
	CPU,
	CUDA,
	OPEN_CL,
	AVX,
	UNKNOWN,
};

class MainData;
class FrameExecutor;
class MovieFrame;

class DeviceInfoBase {
public:
	std::vector<OutputOption> videoEncodingOptions;
	int64_t maxPixel;

	DeviceInfoBase(int64_t maxPixel)
		: maxPixel { maxPixel } 
	{}

	virtual DeviceType getType() const = 0;
	virtual std::string getName() const = 0;
	virtual std::string getNameShort() const = 0;
	virtual std::shared_ptr<FrameExecutor> create(MainData& data, MovieFrame& frame) = 0;
};
