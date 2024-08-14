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
#include "cpu_features/cpuinfo_x86.h"
#include "CpuFrame.hpp"
#include "AvxFrame.hpp"
#include "OpenClFrame.hpp"
#include "CudaFrame.hpp"

#include <thread>
#include <format>

using namespace cpu_features;

static const X86Info cpu = GetX86Info();

 //CPU Device Info
std::string DeviceInfoCpu::getName() const {
	X86Microarchitecture arch = GetX86Microarchitecture(&cpu);
	return std::format("CPU, {}, {} threads", cpu.brand_string, std::to_string(std::thread::hardware_concurrency()));
}

std::string DeviceInfoCpu::getNameShort() const {
	return "Cpu";
}

//full name of this cpu
std::string DeviceInfoCpu::getCpuName() const {
	return cpu.brand_string;
}

std::shared_ptr<MovieFrame> DeviceInfoCpu::createClass(MainData& data, MovieReader& reader, MovieWriter& writer) {
	return std::make_shared<CpuFrame>(data, reader, writer);
}

//CPU Device Info
std::string DeviceInfoAvx::getName() const {
	X86Microarchitecture arch = GetX86Microarchitecture(&cpu);
	return std::format("AVX 512, {}", cpu.brand_string);
}

std::string DeviceInfoAvx::getNameShort() const {
	return "Avx 512";
}

//check if avx512 is available
bool DeviceInfoAvx::hasAvx512() const {
	return cpu.features.avx512f & cpu.features.avx512vl & cpu.features.avx512bw;
}

std::shared_ptr<MovieFrame> DeviceInfoAvx::createClass(MainData& data, MovieReader& reader, MovieWriter& writer) {
	return std::make_shared<AvxFrame>(data, reader, writer);
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

std::shared_ptr<MovieFrame> DeviceInfoOpenCl::createClass(MainData& data, MovieReader& reader, MovieWriter& writer) {
	return std::make_shared<OpenClFrame>(data, reader, writer);
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

std::shared_ptr<MovieFrame> DeviceInfoCuda::createClass(MainData& data, MovieReader& reader, MovieWriter& writer) {
	return std::make_shared<CudaFrame>(data, reader, writer);
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