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

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "DeviceInfo.hpp"

class DeviceInfoCuda : public DeviceInfo {
public:
	cudaDeviceProp props;

	DeviceInfoCuda(DeviceType type, size_t targetIndex, int64_t maxPixel, cudaDeviceProp props)
		: DeviceInfo(type, targetIndex, maxPixel)
		, props { props } 
	{}

	std::string getName() const override;
};

//info about cuda devices
struct CudaInfo {
	std::vector<DeviceInfoCuda> devices;

	int nvidiaDriverVersion = 0;
	int cudaRuntimeVersion = 0;
	int cudaDriverVersion = 0;

	int nppMajor = 0;
	int nppMinor = 0;
	int nppBuild = 0;

	uint32_t nvencVersionApi;
	uint32_t nvencVersionDriver;

	std::string nvidiaDriverToString() const {
		return std::to_string(nvidiaDriverVersion / 100) + "." + std::to_string(nvidiaDriverVersion % 100);
	}

	std::string cudaRuntimeToString() const {
		return std::to_string(cudaRuntimeVersion / 1000) + "." + std::to_string(cudaRuntimeVersion % 1000 / 10);
	}

	std::string cudaDriverToString() const {
		return std::to_string(cudaDriverVersion / 1000) + "." + std::to_string(cudaDriverVersion % 1000 / 10);
	}
};