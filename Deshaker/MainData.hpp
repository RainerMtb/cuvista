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

inline std::string CUVISTA_VERSION = "0.9";

enum class DeshakerPass {
	NONE, 
	COMBINED, 
	FIRST_PASS, 
	SECOND_PASS, 
	CONSECUTIVE
};

enum class EncodingDevice {
	AUTO,
	GPU,
	CPU,
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

struct DeviceProps {
	cudaDeviceProp cudaProps;
	std::vector<OutputCodec> codecs;
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
	std::string str_toupper(const std::string& s);

public:

	std::map<std::string, OutputCodec> codecMap = {
		{"AUTO", OutputCodec::AUTO},
		{"H264", OutputCodec::H264},
		{"H265", OutputCodec::H265},
	};
	std::map<std::string, EncodingDevice> encodingDeviceMap = {
		{"AUTO", EncodingDevice::AUTO},
		{"GPU", EncodingDevice::GPU},
		{"CPU", EncodingDevice::CPU},
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
	CudaInfo cudaInfo;
	std::vector<DeviceProps> deviceProps;
	bool deviceRequested = false;

	InputContext inputCtx;
	DeshakerPass pass = DeshakerPass::COMBINED;

	//output related
	std::ostream* console = &std::cout;
	EncodingDevice encodingDevice = EncodingDevice::AUTO;
	OutputCodec videoCodec = OutputCodec::AUTO;
	OutputType videoOutputType = OutputType::NONE;
	bool overwriteOutput = false;
	std::string fileIn;					//input file path
	std::string fileOut;				//output file path
	std::string trajectoryFile;			//file to read or write trajectory data
	std::string resultsFile;			//output file path
	std::string resultImageFile;        //file to write result images, gray, transform lines

	std::string tcp_address;
	unsigned short tcp_port = 0;

	bool dummyFrame = false;
	ProgressType progressType = ProgressType::REWRITE_LINE;

	//parameters for computation of trajectory
	double cSigmaParam = 1.0;			//at least 0.33, greater -> more stable camera

	//parameters for computation of consensus
	//size_t cMaxIterCount = 120;			//max number of iterations for consensus set
	ptrdiff_t cMinConsensPoints = 8;	    //min numbers of points for consensus set
	int cConsLoopCount = 8;			        //max number of loops when searching for consensus set
	int cConsLoopPercent = 95;		        //percentage of points for next loop 0..100
	double cConsensDistance = 1.5;          //max distance for a point to be in the consensus set

	//cpu threads to use in cpu-compute and computing transform parameters
	size_t cpuThreads = std::max(1u, std::thread::hardware_concurrency() * 3 / 4); //leave room for other things

	//filter for unsharp masking output
	const std::vector<std::vector<float>> kernelFilter = {
		{ 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f },
		{ 0.25, 0.5, 0.25 },
		{ 0.25, 0.5, 0.25 },
	};

	//filter for differences in x and y when reading
	const std::vector<float> filterKernel = { -0.5f, 0.0f, 0.5f };

	int64_t maxFrames = std::numeric_limits<int64_t>::max();

	std::unique_ptr<RNGbase> rng = std::make_unique<RNG<RandomSource>>();

	ColorRgb bgcol_rgb { 0, 50, 0 };

	void probeCudaDevices();

	bool canDeviceEncode();

	void probeInput(std::vector<std::string> args);

	void validate(InputContext& input);

	void validate();

	void showIntro() const ;

	void showBasicInfo() const ;

	void showDeviceInfo() const;
};
