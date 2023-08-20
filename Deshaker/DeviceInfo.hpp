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

#pragma once

#include "CudaInfo.hpp"
#include <vector>
#include <string>

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
};

class DeviceInfo {
public:
	DeviceType type;
	int targetIndex;
	std::vector<EncodingOption> encodingOptions;
	size_t maxPixel;

	DeviceInfo(DeviceType type, int targetIndex, size_t maxPixel) : type { type }, targetIndex { targetIndex }, maxPixel { maxPixel } {}

	virtual std::string getName() const = 0;
};

class DeviceInfoCpu : public DeviceInfo {
public:
	DeviceInfoCpu(DeviceType type, int targetIndex, size_t maxPixel) : DeviceInfo(type, targetIndex, maxPixel) {}

	DeviceInfoCpu() : DeviceInfoCpu(DeviceType::CPU, 0, 0) {}

	virtual std::string getName() const override;
};

class DeviceInfoCuda : public DeviceInfo {
public:
	cudaDeviceProp props;

	DeviceInfoCuda(DeviceType type, int targetIndex, size_t maxPixel, cudaDeviceProp props) : DeviceInfo(type, targetIndex, maxPixel), props { props } {}

	virtual std::string getName() const override;
};
