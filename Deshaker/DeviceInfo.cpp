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

#include "clHeaders.hpp"
#include "DeviceInfo.hpp"
#include <thread>
#include <format>

std::string DeviceInfoCpu::getName() const {
	return std::string("CPU: Software only, ") + std::to_string(std::thread::hardware_concurrency()) + " threads";
}

std::string DeviceInfoCuda::getName() const {
	return std::format("Cuda: {}, Compute {}.{}, {} Mb", props.name, props.major, props.minor, props.totalGlobalMem / 1024 / 1024);
}

std::string DeviceInfoCl::getName() const {
	std::string name = device->getInfo<CL_DEVICE_NAME>();
	std::string vendor = device->getInfo<CL_DEVICE_VENDOR>();
	cl_ulong memSize = device->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
	//cl_version version = device->getInfo<CL_DEVICE_NUMERIC_VERSION>();
	return std::format("OpenCL: {}, {}, {} Mb", name, vendor, memSize / 1024 / 1024);
}