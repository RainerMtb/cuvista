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

#include "DeviceInfo.hpp"
#include "CpuFrame.hpp"
#include "AvxFrame.hpp"
#include "clMain.hpp"
#include "CudaFrame.hpp"
#include "DummyFrame.hpp"
#include "MainData.hpp"
#include "MovieFrame.hpp"

#include <thread>
#include <format>

 //CPU Device Info
std::string DeviceInfoCpu::getName() const {
	return std::format("CPU, {}, {} threads", cpuInfo.brand_string, std::to_string(std::thread::hardware_concurrency()));
}

std::string DeviceInfoCpu::getNameShort() const {
	return "Cpu";
}

std::shared_ptr<FrameExecutor> DeviceInfoCpu::create(MainData& data, MovieFrame& frame) {
	return std::make_shared<CpuFrame>(data, *this, frame, frame.mPool);
}


//Avx Device Info
std::string DeviceInfoAvx::getName() const {
	return std::format("AVX 512, {}", cpuInfo.brand_string);
}

std::string DeviceInfoAvx::getNameShort() const {
	return "Avx 512";
}

std::shared_ptr<FrameExecutor> DeviceInfoAvx::create(MainData& data, MovieFrame& frame) {
	return std::make_shared<AvxFrame>(data, *this, frame, frame.mPool);
}


//OPENCL Info
std::string DeviceInfoOpenCl::getName() const {
	std::string name = device->getInfo<CL_DEVICE_NAME>();
	std::string vendor = device->getInfo<CL_DEVICE_VENDOR>();
	return std::format("OpenCL, {}, {}", name, vendor);
}

std::string DeviceInfoOpenCl::getNameShort() const {
	return "OpenCL";
}

std::shared_ptr<FrameExecutor> DeviceInfoOpenCl::create(MainData& data, MovieFrame& frame) {
	return std::make_shared<OpenClFrame>(data, *this, frame, frame.mPool);
}

std::ostream& operator << (std::ostream& os, const DeviceInfoOpenCl& info) {
	auto w = info.device->getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
	auto h = info.device->getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();
	os << "Device Vendor:        " << info.device->getInfo<CL_DEVICE_VENDOR>() << std::endl;
	os << "Device Name:          " << info.device->getInfo<CL_DEVICE_NAME>() << std::endl;
	os << "Total Global Memory:  " << info.device->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1024 / 1024 << " Mb" << std::endl;
	os << "Driver Version        " << info.device->getInfo<CL_DRIVER_VERSION>() << std::endl;
	os << "OpenCL Version:       " << info.versionDevice / 1000 << "." << info.versionDevice % 1000 << std::endl;
	os << "OpenCL C Version:     " << info.versionC / 1000 << "." << info.versionC % 1000 << std::endl;
	os << "Compute Units:        " << info.device->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
	os << "Max Image Size:       " << w << " x " << h << std::endl;
	return os;
}


//CUDA Info
std::ostream& operator << (std::ostream& os, const DeviceInfoCuda& info) {
	os << "Device Name:          " << info.props->name << std::endl;
	os << "Compute Version:      " << info.props->major << "." << info.props->minor << std::endl;
	os << "Clock Rate:           " << info.props->clockRate / 1000 << " Mhz" << std::endl;
	os << "Total Global Memory:  " << info.props->totalGlobalMem / 1024 / 1024 << " Mb" << std::endl;
	os << "Multiprocessor count: " << info.props->multiProcessorCount << std::endl;
	os << "Max Texture Size:     " << info.props->maxTexture2D[0] << " x " << info.props->maxTexture2D[1] << std::endl;
	os << "Shared Mem per Block: " << info.props->sharedMemPerBlock / 1024 << " kb" << std::endl;
	return os;
}

std::string DeviceInfoCuda::getName() const {
	return std::format("Cuda, {}, Compute {}.{}", props->name, props->major, props->minor);
}

std::string DeviceInfoCuda::getNameShort() const {
	return "Cuda";
}

std::shared_ptr<FrameExecutor> DeviceInfoCuda::create(MainData& data, MovieFrame& frame) {
	return std::make_shared<CudaFrame>(data, *this, frame, frame.mPool);
}


std::string CudaInfo::runtimeToString() const {
	return std::to_string(cudaRuntimeVersion / 1000) + "." + std::to_string(cudaRuntimeVersion % 1000 / 10);
}

std::string CudaInfo::driverToString() const {
	return std::to_string(cudaDriverVersion / 1000) + "." + std::to_string(cudaDriverVersion % 1000 / 10);
}

std::string CudaInfo::nvencApiToString() const {
	return std::to_string(nvencVersionApi / 1000) + "." + std::to_string(nvencVersionApi % 1000 / 10);
}

std::string CudaInfo::nvencDriverToString() const {
	return std::to_string(nvencVersionDriver / 1000) + "." + std::to_string(nvencVersionDriver % 1000 / 10);
}


std::string DeviceInfoNull::getName() const {
	return "";
}

std::string DeviceInfoNull::getNameShort() const {
	return "";
}

std::shared_ptr<FrameExecutor> DeviceInfoNull::create(MainData& data, MovieFrame& frame) {
	return std::make_shared<DummyFrame>(data, *this, frame, frame.mPool);
}