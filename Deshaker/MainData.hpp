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

#include "Mat.h"
#include "AVException.hpp"
#include "FFmpegUtil.hpp"
#include "Stats.hpp"
#include "NullStream.hpp"
#include "RandomSource.hpp"
#include "cuDeshaker.cuh"
#include "clMain.hpp"
#include "DeviceInfo.hpp"

inline std::string CUVISTA_VERSION = "0.9.7";

enum class DeshakerPass {
	NONE, 
	COMBINED, 
	FIRST_PASS, 
	SECOND_PASS, 
	CONSECUTIVE
};

enum class OutputType {
	NONE,
	PIPE,
	TCP,
	VIDEO_FILE,
	BMP,
	JPG,
};

enum class ProgressType {
	NONE,
	REWRITE_LINE,
	GRAPH,
	DETAILED,
};

enum class DecideYNA {
	YES,
	NO,
	ASK,
};

class MainData : public CoreData {

private:

	std::map<std::string, std::vector<unsigned char>> colorMap = {
		{"red",     {255,   0,   0}},
		{"green",   {  0, 255,   0}},
		{"blue",    {  0,   0, 255}},
		{"black",   {  0,   0,   0}},
		{"white",   {255, 255, 255}},
		{"magenta", {255,   0, 255}},
		{"yellow",  {255, 255,   0}},
		{"cyan",    {  0, 255, 255}},
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
		int levelsMin = 1, levelsMax = 8;
		int irMin = 0, irMax = 7;
	} limits;

	Stats status;
	DeviceInfoCpu deviceInfoCpu;
	std::vector<DeviceInfo*> deviceList;
	std::vector<DeviceInfoCuda> deviceListCuda;
	CudaInfo cudaInfo;
	OpenClInfo clinfo;
	bool deviceRequested = false;
	size_t deviceSelected = 0;

	InputContext inputCtx;
	DeshakerPass pass = DeshakerPass::COMBINED;

	//output related
	EncodingOption requestedEncoding = { EncodingDevice::AUTO, Codec::AUTO };
	EncodingOption selectedEncoding = { EncodingDevice::AUTO, Codec::AUTO };
	std::ostream* console = &std::cout;
	OutputType videoOutputType = OutputType::NONE;
	DecideYNA overwriteOutput = DecideYNA::ASK;

	bool showHeader = true;

	std::string fileIn;					//input file path
	std::string fileOut;				//output file path
	std::string trajectoryFile;			//file to read or write trajectory data
	std::string resultsFile;			//output file path
	std::string resultImageFile;        //file to write result images, gray, transform lines

	std::string tcp_address;
	unsigned short tcp_port = 0;

	bool dummyFrame = false;
	ProgressType progressType = ProgressType::REWRITE_LINE;

	//parameters for computation of trajectory, at least 0.33, greater -> more stable camera
	double cSigmaParam = 1.25;

	//cpu threads to use in cpu-compute and computing transform parameters, leave room for other things
	size_t cpuThreads = std::max(1u, std::thread::hardware_concurrency() * 3 / 4);

	int64_t maxFrames = std::numeric_limits<int32_t>::max();

	std::unique_ptr<RNGbase> rng = std::make_unique<RNG<RandomSource>>();

	ColorRgb bgcol_rgb { 0, 50, 0 };

	//------------------------------------
	// METHODS
	//------------------------------------

	void probeInput(std::vector<std::string> args);

	void probeCuda();

	void probeOpenCl();

	void collectDeviceInfo();

	void validate(InputContext& input);

	void validate();

	void showIntro() const;

	void showBasicInfo() const;

	void showDeviceInfo();

	size_t deviceCountCuda() const;

	size_t deviceCountOpenCl() const;
};
