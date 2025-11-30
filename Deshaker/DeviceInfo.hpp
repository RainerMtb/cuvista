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

#include "DeviceInfoBase.hpp"
#include "FrameExecutor.hpp"
extern "C" { 
#include "cpuinfo_x86.h" 
}

//CpuFrame
class DeviceInfoCpu : public DeviceInfoBase {
public:
	DeviceInfoCpu();

	DeviceType getType() const override;
	std::string getName() const override;
	std::string getNameShort() const override;
	std::shared_ptr<FrameExecutor> create(MainData& data, MovieFrame& frame) override;
	cpu_features::X86Features getCpuFeatures() const;
};

//AvxFrame
class DeviceInfoAvx : public DeviceInfoBase {
public:
	DeviceInfoAvx();

	DeviceType getType() const override;
	std::string getName() const override;
	std::string getNameShort() const override;
	std::shared_ptr<FrameExecutor> create(MainData& data, MovieFrame& frame) override;
};

namespace cl { class Device; }

//OpenClFrame
class DeviceInfoOpenCl : public DeviceInfoBase {
public:
	std::shared_ptr<cl::Device> device;
	int versionDevice = 0;
	int versionC = 0;
	int pitch = 0;
	std::vector<std::string> extensions;
	std::string platformVersion;

	inline static std::string warning = "";

	DeviceInfoOpenCl(int64_t maxPixel);

	DeviceType getType() const override;
	std::string getName() const override;
	std::string getNameShort() const override;
	std::shared_ptr<FrameExecutor> create(MainData& data, MovieFrame& frame) override;

	friend std::ostream& operator << (std::ostream& os, const DeviceInfoOpenCl& info);
};

struct cudaDeviceProp;
class NvEncoder;

//Cuda
class DeviceInfoCuda : public DeviceInfoBase {

private:
	inline static int cudaRuntimeVersion = 0;
	inline static int cudaDriverVersion = 0;
	inline static uint32_t nvencVersionApi = 0;
	inline static uint32_t nvencVersionDriver = 0;

public:
	std::shared_ptr<cudaDeviceProp> props;
	std::shared_ptr<NvEncoder> nvenc;
	int cudaIndex = 0;

	inline static std::string nvidiaDriverVersion = "";
	inline static std::string warning = "";

	DeviceInfoCuda(int64_t maxPixel);

	static std::vector<DeviceInfoCuda> probeCuda();
	static std::string runtimeToString();
	static std::string driverToString();
	static std::string nvencApiToString();
	static std::string nvencDriverToString();

	DeviceType getType() const override;
	std::string getName() const override;
	std::string getNameShort() const override;
	std::shared_ptr<FrameExecutor> create(MainData& data, MovieFrame& frame) override;

	friend std::ostream& operator << (std::ostream& os, const DeviceInfoCuda& info);

	bool operator < (const DeviceInfoCuda& other) const;
};

//Null Device
class DeviceInfoNull : public DeviceInfoBase {
public:
	DeviceInfoNull();

	DeviceType getType() const override;
	std::string getName() const override;
	std::string getNameShort() const override;
	std::shared_ptr<FrameExecutor> create(MainData& data, MovieFrame& frame) override;
};