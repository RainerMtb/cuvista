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


//CpuFrame
class DeviceInfoCpu : public DeviceInfoBase {
public:
	std::vector<OutputOption> encoders;

	DeviceInfoCpu();

	DeviceType getType() const override;
	std::string getName() const override;
	std::string getNameShort() const override;
	std::shared_ptr<FrameExecutor> create(MainData& data, MovieFrame& frame) override;
};


//AvxFrame
class DeviceInfoAvx : public DeviceInfoBase {
public:
	DeviceInfoAvx();

	DeviceType getType() const override;
	std::string getName() const override;
	std::string getNameShort() const override;
	std::shared_ptr<FrameExecutor> create(MainData& data, MovieFrame& frame) override;

	bool hasAvx10() const;
	bool hasAvx512() const;
	bool hasAvx2() const;
};


namespace cl { class Device; }

//OpenClFrame
struct OpenClDevice : public DeviceInfoBase {
	std::shared_ptr<cl::Device> device;
	int versionDevice = 0;
	int versionC = 0;
	int pitch = 0;
	std::vector<std::string> extensions;
	std::string platformVersion;

	OpenClDevice(int64_t maxPixel);

	DeviceType getType() const override;
	std::string getName() const override;
	std::string getNameShort() const override;
	std::shared_ptr<FrameExecutor> create(MainData& data, MovieFrame& frame) override;

	friend std::ostream& operator << (std::ostream& os, const OpenClDevice& info);
};

class DeviceInfoOpenCl {

public:
	std::vector<OpenClDevice> devices;
	std::string driverWarning = "";
};


//Vulkan, only for video encoding
class DeviceInfoVulkanCollection : public DeviceInfoBase {
public:
	std::vector<OutputOption> encoders;

	DeviceInfoVulkanCollection();

	DeviceType getType() const override;
	std::string getName() const override;
	std::string getNameShort() const override;
	std::shared_ptr<FrameExecutor> create(MainData& data, MovieFrame& frame) override;

};


//Cuda
struct CudaDevice : public DeviceInfoCudaBase {
	std::vector<OutputOption> encoders;
	int cudaIndex;

	CudaDevice(int64_t maxPixel);

	DeviceType getType() const override;
	std::string getName() const override;
	std::string getNameShort() const override;
	std::shared_ptr<FrameExecutor> create(MainData& data, MovieFrame& frame) override;

	bool operator < (const CudaDevice& other) const;

	friend std::ostream& operator << (std::ostream& os, const CudaDevice& info);
};

class DeviceInfoCudaCollection {

private:
	int mCudaRuntimeVersion = 0;
	int mCudaDriverVersion = 0;

public:
	std::vector<CudaDevice> devices;
	std::vector<OutputOption> encoders;
	std::string nvidiaDriverVersion = "";
	std::string nvidiaDriverWarning = "";
	uint32_t nvencVersionApi = 0;
	uint32_t nvencVersionDriver = 0;
	int encodingDeviceIndex = 0;

	void probeCuda();
	std::string runtimeToString() const;
	std::string driverToString() const;
	std::string nvencApiToString() const;
	std::string nvencDriverToString() const;
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