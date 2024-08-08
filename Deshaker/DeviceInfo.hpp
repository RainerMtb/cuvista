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
#include "CpuFrame.hpp"
#include "AvxFrame.hpp"
#include "OpenClFrame.hpp"
#include "CudaFrame.hpp"

template <class T> class DeviceInfo {};

//CpuFrame
template <> class DeviceInfo<CpuFrame> : public DeviceInfoBase {
public:
	DeviceInfo()
		: DeviceInfoBase(DeviceType::CPU, 16384) {}

	std::string getName() const override;

	std::string getNameShort() const override;

	std::string getCpuName() const;

	std::shared_ptr<MovieFrame> createClass(MainData& data, MovieReader& reader, MovieWriter& writer) override;
};

//AvxFrame
template <> class DeviceInfo<AvxFrame> : public DeviceInfoBase {
public:
	DeviceInfo()
		: DeviceInfoBase(DeviceType::AVX, 16384) {}

	std::string getName() const override;

	std::string getNameShort() const override;

	std::shared_ptr<MovieFrame> createClass(MainData& data, MovieReader& reader, MovieWriter& writer) override;

	bool hasAvx512() const;
};

//OpenClFrame
template <> class DeviceInfo<OpenClFrame> : public DeviceInfoBase {
public:
	cl::Device device;
	int versionDevice = 0;
	int versionC = 0;
	int pitch = 0;
	std::vector<std::string> extensions;

	DeviceInfo(DeviceType type, int64_t maxPixel)
		: DeviceInfoBase(type, maxPixel) {}

	std::string getName() const override;

	std::string getNameShort() const override;

	std::shared_ptr<MovieFrame> createClass(MainData& data, MovieReader& reader, MovieWriter& writer) override;

	friend std::ostream& operator << (std::ostream& os, const DeviceInfo<OpenClFrame>& info);
};

struct OpenClInfo {
	std::vector<DeviceInfo<OpenClFrame>> devices;
	std::string version;
};

//Cuda
template <> class DeviceInfo<CudaFrame> : public DeviceInfoBase {
public:
	cudaDeviceProp props = {};
	int cudaIndex = 0;

	DeviceInfo(DeviceType type, int64_t maxPixel)
		: DeviceInfoBase(type, maxPixel) {}

	std::string getName() const override;

	std::string getNameShort() const override;

	std::shared_ptr<MovieFrame> createClass(MainData& data, MovieReader& reader, MovieWriter& writer) override;

	friend std::ostream& operator << (std::ostream& os, const DeviceInfo<CudaFrame>& info);
};

//info about cuda devices
struct CudaInfo {
	std::vector<DeviceInfo<CudaFrame>> devices;

	std::string nvidiaDriverVersion = "";
	int cudaRuntimeVersion = 0;
	int cudaDriverVersion = 0;

	int nppMajor = 0;
	int nppMinor = 0;
	int nppBuild = 0;

	uint32_t nvencVersionApi;
	uint32_t nvencVersionDriver;

	std::string runtimeToString() const;
	std::string driverToString() const;
	std::string nvencApiToString() const;
	std::string nvencDriverToString() const;
};