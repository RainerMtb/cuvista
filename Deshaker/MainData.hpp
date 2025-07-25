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

#include "Mat.hpp"
#include "AVException.hpp"
#include "FFmpegUtil.hpp"
#include "RandomSource.hpp"
#include "SelfTest.hpp"
#include "Util.hpp"
#include "Version.hpp"
#include "CoreData.hpp"
#include "DeviceInfo.hpp"

class MovieReader;

enum class DeshakerPass {
	NONE,
	COMBINED,
	CONSECUTIVE,
	//FIRST_PASS,
	//SECOND_PASS,
};

enum class OutputType {
	NONE,
	PIPE,
	VIDEO_FILE,
	RAW_YUV_FILE,
	SEQUENCE_BMP,
	SEQUENCE_JPG,
};

enum class ProgressType {
	NONE,
	MULTILINE,
	REWRITE_LINE,
	NEW_LINE,
	GRAPH,
};

enum class DecideYNA {
	YES,
	NO,
	ASK,
};

class MainData : public CoreData {

private:
	std::map<std::string, Color> colorMap = {
		{"red",     Color::RED},
		{"green",   Color::GREEN},
		{"blue",    Color::BLUE},
		{"black",   Color::BLACK},
		{"white",   Color::WHITE},
		{"magenta", Color::MAGENTA},
		{"yellow",  Color::YELLOW},
		{"cyan",    Color::CYAN},
	};

	//utililty class holding command line parameters, extension of vector
	class Parameters : public std::vector<std::string> {

	private:
		size_t index = 0;

	public:
		Parameters(std::vector<std::string> args) : std::vector<std::string>(args) {}

		//for debugging
		Parameters(std::initializer_list<const char*> args) : std::vector<std::string>(args.begin(), args.end()) {}

		//string at current index
		std::string& str();

		//has more strings in the vector
		bool hasArgs();

		//check if next argument is provided string
		bool nextArg(std::string&& param);

		//check if next argument is provided string, and if so return matching parameter
		bool nextArg(std::string&& param, std::string& nextParam);
	};

	//convert string to upper case
	std::string str_toupper(const std::string& s) const;

	bool checkFileForWriting(const std::string& file, DecideYNA permission) const;

public:
	std::map<std::string, Codec> mapStringToCodec = {
		{"AUTO", Codec::AUTO},
		{"H264", Codec::H264},
		{"H265", Codec::H265},
		{"AV1", Codec::AV1},
	};
	std::map<Codec, std::string> mapCodecToString = {
		{Codec::AUTO, "AUTO"},
		{Codec::H264, "H264"},
		{Codec::H265, "H265"},
		{Codec::AV1, "AV1"},
	};
	std::map<std::string, EncodingDevice> mapStringToDevice = {
		{"AUTO", EncodingDevice::AUTO},
		{"NVENC", EncodingDevice::NVENC},
		{"CPU", EncodingDevice::CPU},
	};
	std::map<EncodingDevice, std::string> mapDeviceToString = {
		{EncodingDevice::AUTO, "AUTO"},
		{EncodingDevice::NVENC, "NVENC"},
		{EncodingDevice::CPU, "CPU"},
	};

	struct ValueLimits {
		double radsecMin = 0.1, radsecMax = 10.0;
		double imZoomMin = 0.1, imZoomMax = 10.0;
		int radiusMin = 1, radiusMax = 500;
		int wMin = 100, hMin = 100;
		int levelsMin = 1, levelsMax = 6;
		int irMin = 0, irMax = 3;
	} limits;

	
	std::vector<DeviceInfoBase*> deviceList;
	DeviceInfoCpu deviceInfoCpu;
	DeviceInfoAvx deviceInfoAvx;
	OpenClInfo clinfo;
	CudaInfo cudaInfo;
	bool deviceRequested = false;
	size_t deviceSelected = 0;
	DeviceInfoNull deviceInfoNull;
	std::optional<int> cpuThreadsRequired = std::nullopt;

	std::shared_ptr<SamplerBase<PointContext>> sampler = std::make_shared<Sampler<PointContext, PseudoRandomSource>>();

	DeshakerPass pass = DeshakerPass::COMBINED;

	//output related
	util::NullOutstream nullStream;
	EncodingOption requestedEncoding = { EncodingDevice::AUTO, Codec::AUTO };
	EncodingOption selectedEncoding = { EncodingDevice::AUTO, Codec::AUTO };
	std::ostream* console = &std::cout;
	OutputType videoOutputType = OutputType::NONE;
	DecideYNA overwriteOutput = DecideYNA::ASK;
	std::optional<uint8_t> crf = std::nullopt;
	std::optional<double> stackPosition = std::nullopt;

	bool printHeader = true;
	bool printSummary = true;

	std::string fileIn;					//input file path
	std::string fileOut;				//output file path
	std::string trajectoryFile;			//file to read or write trajectory data
	std::string resultsFile;			//output file path
	std::string resultImageFile;        //file to write result images, grayscale background and transform vectors
	std::string flowFile;               //file to write video showing optical flow

	bool dummyFrame = false;
	ProgressType progressType = ProgressType::MULTILINE;

	//parameters for computation of trajectory, at least 0.33, greater -> more stable camera
	double cSigmaParam = 1.25;

	int64_t maxFrames = std::numeric_limits<int32_t>::max();
	Color backgroundColor = Color::rgb(0, 150, 0);

	std::chrono::steady_clock::time_point timePoint;

	//------------------------------------
	// METHODS
	//------------------------------------

	void probeInput(std::vector<std::string> args);

	std::vector<DeviceInfoCuda> probeCuda();

	std::vector<DeviceInfoOpenCl> probeOpenCl();

	void collectDeviceInfo();

	void validate(const MovieReader& reader);

	void showDeviceInfo() const;

	std::ostream& showDeviceInfo(std::ostream& os) const;

	void showBasicInfo() const;

	void showVersionInfo() const;

	void showHeader() const;

	void showIntro(const std::string& deviceName, const MovieReader& reader) const;

	size_t deviceCountCuda() const;

	size_t deviceCountOpenCl() const;

	void timeStart();

	double timeElapsedSeconds() const;

	std::string getCpuName() const;

	bool hasAvx512() const;

	bool hasAvx2() const;
};
