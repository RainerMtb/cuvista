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

#include <vector>
#include <string>
#include <memory>

class MovieFrame;
class MainData;
class MovieReader;
class MovieWriter;

enum class Codec {
	AUTO,
	H264,
	H265,
	AV1,
};

enum class EncodingDevice {
	AUTO,
	NVENC,
	CPU,
};

struct EncodingOption {
	EncodingDevice device;
	Codec codec;
};

enum class DeviceType {
	CPU,
	CUDA,
	OPEN_CL,
	AVX,
};

class DeviceInfoBase {
public:
	DeviceType type;
	std::vector<EncodingOption> encodingOptions;
	int64_t maxPixel;

	DeviceInfoBase(DeviceType type, int64_t maxPixel)
		: type { type }, maxPixel { maxPixel } {}

	virtual std::string getName() const = 0;

	virtual std::string getNameShort() const = 0;

	virtual std::shared_ptr<MovieFrame> createClass(MainData& data, MovieReader& reader, MovieWriter& writer) = 0;
};
