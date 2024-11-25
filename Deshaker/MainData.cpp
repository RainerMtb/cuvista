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

#include "MainData.hpp"
#include "DeviceInfo.hpp"
#include "DeshakerHelpText.hpp"
#include "KeyboardInput.hpp"
#include "MovieReader.hpp"
#include "MovieWriter.hpp"
#include "CudaFrame.hpp"
#include "NvidiaDriver.hpp"
#include "OpenClFrame.hpp"
#include "SelfTest.hpp"

#include <algorithm>
#include <regex>
#include <vector>
#include <map>
#include <filesystem>


//test or ask for file writing
bool MainData::checkFileForWriting(const std::string& file, DecideYNA permission) const {
	if (file.empty()) {
		throw AVException("output file missing");
	}
	if (permission == DecideYNA::NO) {
		throw CancelException("output file exists, aborting...");
	}
	if (permission == DecideYNA::ASK && static_cast<bool>(std::ifstream(file))) {
		*console << "file '" << file << "' exists, overwrite [y/n] " << std::flush;
		while (true) {
			std::optional<char> och = getKeyboardInput();
			if (och && *och == 'y') {
				*console << "y" << std::endl;
				break;
			}
			if (och && *och == 'n') {
				*console << "n" << std::endl;
				throw CancelException("cancelled writing to file, aborting...");
			}
		}
	}
	return true;
}

//start overall timing
void MainData::timeStart() {
	timePoint = std::chrono::steady_clock::now();
}

//elapsed time in seconds
double MainData::timeElapsedSeconds() const {
	auto timeNow = std::chrono::steady_clock::now();
	std::chrono::duration<double> secs = timeNow - timePoint;
	return secs.count();
}

void MainData::probeInput(std::vector<std::string> argsInput) {
	//vector args contains command line parameters
	Parameters args(argsInput);

	std::string next;
	while (args.hasArgs()) {
		if (args.nextArg("i", next)) {
			fileIn = next;

		} else if (args.nextArg("o", next)) {
			console = &std::cout;
			if (str_toupper(next) == "NULL") {
				//do not do any output
				videoOutputType = OutputType::NONE;

			} else if (str_toupper(next) == "PIPE:0") {
				//activate pipe output
				console = &nullStream;
				printHeader = false;
				videoOutputType = OutputType::PIPE;

			} else if (str_toupper(next).ends_with(".BMP")) {
				videoOutputType = OutputType::SEQUENCE_BMP;
				fileOut = next;

			} else if (str_toupper(next).ends_with(".JPG")) {
				videoOutputType = OutputType::SEQUENCE_JPG;
				fileOut = next;

			} else {
				//default file output
				videoOutputType = OutputType::VIDEO_FILE;
				fileOut = next;
			}

		} else if (args.nextArg("trf", next)) {
			//trajectory file
			trajectoryFile = next;

		} else if (args.nextArg("res", next)) {
			//file to store detailed results
			resultsFile = next;

		} else if (args.nextArg("resim", next)) {
			//produce and write transformation images
			resultImageFile = next;

		} else if (args.nextArg("flow", next)) {
			//optical flow video
			flowFile = next;

		} else if (args.nextArg("pass", next)) {
			int p = std::stoi(next);
			if (p == 0) { 
				pass = DeshakerPass::COMBINED; 
			} else if (p == 1) { 
				pass = DeshakerPass::FIRST_PASS; 
			} else if (p == 2) { 
				pass = DeshakerPass::SECOND_PASS; 
			} else if (p == 12) { 
				pass = DeshakerPass::CONSECUTIVE; 
			} else throw AVException("invalid value for pass: " + next);

		} else if (args.nextArg("radius", next)) {
			//temporal radius in seconds before and after current frame
			radsec = std::stod(next);

		} else if (args.nextArg("zoom", next)) {
			//zoom to apply after transformation
			std::regex pattern1("([-]*\\d+)");
			std::regex pattern2("(\\d+):(\\d+)");
			std::smatch matcher;
			//check for dynamic zoom
			if (std::regex_match(next, matcher, pattern2)) {
				zoomMin = 1.0 + std::stoi(matcher[1]) / 100.0;
				zoomMax = 1.0 + std::stoi(matcher[2]) / 100.0;
			} else if (std::regex_match(next, matcher, pattern1)) {
				zoomMin = 1.0 + std::stoi(matcher[1]) / 100.0;
				zoomMax = 1.0 + std::stoi(matcher[1]) / 100.0;
			} else throw AVException("invalid zoom: " + next);

		} else if (args.nextArg("device", next)) {
			deviceRequested = true;
			deviceSelected = next == "cpu" ? 0 : std::stoi(next);

		} else if (args.nextArg("bgmode", next)) {
			//background either blend previous frames or use defined color
			if (next == "blend") {
				bgmode = BackgroundMode::BLEND;

			} else if (next == "color") {
				bgmode = BackgroundMode::COLOR;

			} else {
				throw AVException("invalid background mode: " + next);
			}

		} else if (args.nextArg("cputhreads", next)) {
			//number of threads to use on cpu
			cpuThreadsRequired = { std::stoul(next) };

		} else if (args.nextArg("bgcolor", next)) {
			//background color
			auto it = colorMap.find(next); //some color literals
			if (it == colorMap.end()) {
				std::regex pattern("(\\d+):(\\d+):(\\d+)"); //rgb triplet
				std::smatch matcher;
				if (std::regex_match(next, matcher, pattern)) {
					for (int i = 0; i < 3; i++) {
						bgcol_rgb.colors[i] = (unsigned char) std::stoi(matcher[i].str());
					}

				} else {
					throw AVException("invalid color definition: " + next);
				}

			} else {
				bgcol_rgb = { it->second[0], it->second[1],it->second[2] };
			}

			bgmode = BackgroundMode::COLOR;

		} else if (args.nextArg("crf", next)) {
			*crf = std::stoi(next); //output encoder crf value

		} else if (args.nextArg("y")) {
			overwriteOutput = DecideYNA::YES; //always overwrite

		} else if (args.nextArg("n")) {
			overwriteOutput = DecideYNA::NO; //never overwrite

		} else if (args.nextArg("info")) {
			showHeader();
			*console << std::endl;
			showDeviceInfo(*console);
			MessagePrinterConsole mpc(console);
			runSelfTest(mpc, deviceList);
			throw SilentQuitException();

		} else if (args.nextArg("version")) {
			showVersionInfo();

		} else if (args.nextArg("h") || args.nextArg("help") || args.nextArg("?")) {
			*console << helpString;
			throw SilentQuitException();

		} else if (args.nextArg("progress", next)) {
			int i = std::stoi(next);
			if (i == 0) progressType = ProgressType::NONE;
			else if (i == 1) progressType = ProgressType::REWRITE_LINE;
			else if (i == 2) progressType = ProgressType::NEW_LINE;
			else if (i == 3) progressType = ProgressType::GRAPH;
			else if (i == 4) progressType = ProgressType::DETAILED;
			else throw AVException("invalid progress type: " + next);

		} else if (args.nextArg("noheader")) {
			printHeader = false;

		} else if (args.nextArg("quiet")) {
			printHeader = false;
			progressType = ProgressType::NONE;

		} else if (args.nextArg("levels", next)) {
			zCount = std::stoi(next);

		} else if (args.nextArg("ir", next)) {
			//integration radius
			ir = std::stoi(next);
			iw = ir * 2 + 1;

		} else if (args.nextArg("codec", next)) {
			auto item = mapStringToCodec.find(str_toupper(next));
			if (item == mapStringToCodec.end())
				throw AVException("invalid codec: " + next);
			else
				requestedEncoding.codec = item->second;

		} else if (args.nextArg("encoder", next)) {
			auto item = mapStringToDevice.find(str_toupper(next));
			if (item == mapStringToDevice.end())
				throw AVException("invalid encoding device: " + next);
			else
				requestedEncoding.device = item->second;

		} else if (args.nextArg("frames", next)) {
			maxFrames = std::stoll(next);

		} else if (args.nextArg("copyframes")) {
			dummyFrame = true;

		// } else if (args.nextArg("rand", next)) {
		// 	if (std::stoi(next) == 1) rng = std::make_unique<RNG<PseudoRandomSource>>();
		// 	else if (std::stoi(next) == 2) rng = std::make_unique<RNG<std::random_device>>();
		// 	else if (std::stoi(next) == 3) rng = std::make_unique<RNG<std::default_random_engine>>();
		// 	else throw AVException("invalid random generator selection: " + next);

		} else if (args.nextArg("stack", next)) {
			double val = std::stod(next);
			if (val >= -1.0 && val <= 1.0) {
				stackPosition = val;

			} else {
				throw AVException("invalid value for combining frames: " + next);
			}

		} else {
			throw AVException("invalid parameter '" + args.str() + "'");
		}
	}

	//show title
	if (printHeader) {
		showHeader();
	}

	if (args.empty()) {
		showBasicInfo();
		throw CancelException();
	}

	//check output file presence and permissions
	if (videoOutputType == OutputType::VIDEO_FILE) {
		if (pass == DeshakerPass::COMBINED || pass == DeshakerPass::CONSECUTIVE) {
			checkFileForWriting(fileOut, overwriteOutput);
		}
		if (pass == DeshakerPass::FIRST_PASS) {
			checkFileForWriting(trajectoryFile, overwriteOutput);
		}
		if (pass == DeshakerPass::SECOND_PASS) {
			if (trajectoryFile.empty()) throw AVException("trajectory file missing");
			checkFileForWriting(fileOut, overwriteOutput);
		}
		auto path1 = std::filesystem::path(fileIn);
		auto path2 = std::filesystem::path(fileOut);
		std::error_code ec;
		if (std::filesystem::equivalent(path1, path2, ec)) {
			throw AVException("cannot use the same file for input and output");
		}
	}
}

void MainData::collectDeviceInfo() {
	//sort cuda devices by compute
	std::sort(cudaInfo.devices.begin(), cudaInfo.devices.end());

	//cpu encoders
	std::vector<EncodingOption> cpuEncoders = {
		{EncodingDevice::CPU, Codec::H264},
		{EncodingDevice::CPU, Codec::H265},
		{EncodingDevice::CPU, Codec::AV1},
	};

	//CPU device
	deviceInfoCpu.encodingOptions = cpuEncoders;
	if (cudaInfo.devices.size() > 0) {
		DeviceInfoCuda& dic = cudaInfo.devices.front();
		std::copy(dic.encodingOptions.begin(), dic.encodingOptions.end(), std::back_inserter(deviceInfoCpu.encodingOptions));
	}
	deviceList.push_back(&deviceInfoCpu);

	//check for Avx512
	if (hasAvx512()) {
		deviceInfoAvx.encodingOptions = deviceInfoCpu.encodingOptions;
		deviceList.push_back(&deviceInfoAvx);
	}
	
	///OpenCL devices
	for (DeviceInfoOpenCl& dev : clinfo.devices) {
		std::copy(cpuEncoders.begin(), cpuEncoders.end(), std::back_inserter(dev.encodingOptions));
		deviceList.push_back(&dev);
	}

	//Cuda devices
	for (DeviceInfoCuda& dev : cudaInfo.devices) {
		std::copy(cpuEncoders.begin(), cpuEncoders.end(), std::back_inserter(dev.encodingOptions));
		deviceList.push_back(&dev);
	}
}

void MainData::validate(const MovieReader& reader) {
	if (this->cpuThreadsRequired.has_value()) this->cpuThreads = this->cpuThreadsRequired.value();
	else this->cpuThreads = std::max(1u, std::thread::hardware_concurrency() * 3 / 4);

	//main metrics
	this->w = reader.w;
	this->h = reader.h;

	//no output to console if frame output through pipe
	if (videoOutputType == OutputType::PIPE) progressType = ProgressType::NONE;

	//pyramid levels to compute
	this->zMax = this->zCount - 1;
	int div0 = 1 << zMax;
	int points = std::max(w / div0, h / div0);
	while (points > MAX_POINTS_COUNT) {
		points /= 2;
		zMax++;
	}
	this->zMin = zMax - zCount + 1;
	this->pyramidLevels = zMax + 1;

	//number of rows for one complete pyramid
	int hh = h;
	int rowCount = hh;
	for (int i = 0; i < zMax; i++) {
		hh /= 2;
		rowCount += hh;
	}
	this->pyramidRowCount = rowCount;

	//number of point to compute, leave one pixel around the border for computing delta values
	int div = 1 << zMax;
	this->ixCount = w / div - 2 * ir - 3;
	this->iyCount = h / div - 2 * ir - 3;
	this->resultCount = ixCount * iyCount;

	//check background color value
	for (int i = 0; i < 3; i++) {
		int col = bgcol_rgb.colors[i];
		if (col < 0 || col > 255) throw AVException("invalid background color value: " + std::to_string(col));
	}
	//set background yuv
	bgcol_yuv = bgcol_rgb.toNormalized();

	int pitchBase = 256;
	cpupitch = (w + pitchBase - 1) / pitchBase * pitchBase;

	//number of frames to buffer
	this->radius = (int) std::round(radsec * reader.fps());
	this->bufferCount = radius + 2;

	//zoom fallback rate
	this->zoomFallback = 1.0 - zoomFallbackTotal / radius;

	if (pass == DeshakerPass::FIRST_PASS) {
		this->bufferCount = 2;
	}
	if (pass == DeshakerPass::SECOND_PASS) {
		this->pyramidCount = 0; //no need for pyramid on pass 2
		this->bufferCount = 2;
	}
	if (pass == DeshakerPass::CONSECUTIVE) {
		this->bufferCount = 2;
	}

	//choose device and encoding
	if (deviceSelected >= deviceList.size()) {
		throw AVException("invalid device number: " + std::to_string(deviceRequested));
	}
	if (deviceRequested == false) {
		deviceSelected = deviceList.size() - 1;
	}
	if (requestedEncoding.device == EncodingDevice::AUTO) {
		if (deviceList[deviceSelected]->type == DeviceType::CUDA) selectedEncoding.device = EncodingDevice::NVENC;
		else selectedEncoding.device = EncodingDevice::CPU;
	} else {
		selectedEncoding.device = requestedEncoding.device;
	}

	//check certain values ranges for sanity
	if (radsec < limits.radsecMin || radsec > limits.radsecMax) throw AVException("invalid temporal radius: " + std::to_string(radsec));
	if (radius < limits.radiusMin || radius > limits.radiusMax) throw AVException("invalid image radius: " + std::to_string(radius));
	if (w < limits.wMin) throw AVException("invalid input video width: " + std::to_string(w));
	if (h < limits.hMin) throw AVException("invalid input video height: " + std::to_string(h));
	size_t mp = deviceList[deviceSelected]->maxPixel;
	if (w > mp) throw AVException("frame width exceeds maximum of " + std::to_string(mp) + " px");
	if (h > mp) throw AVException("frame height exceeds maximum of " + std::to_string(mp) + " px");
	if (w % 2 != 0 || h % 2 != 0) throw AVException("width and height must be factors of two");
	if (zoomMin > zoomMax) throw AVException("invalid zoom values, max zoom must be greater min zoom");
	if (zoomMin < limits.imZoomMin || zoomMin > limits.imZoomMax) throw AVException("invalid zoom value");
	if (zoomMax < limits.imZoomMin || zoomMax > limits.imZoomMax) throw AVException("invalid zoom value");

	if (zCount < limits.levelsMin || zCount > limits.levelsMax) throw AVException("invalid pyramid levels:" + std::to_string(pyramidLevels));
	if (ir < limits.irMin || ir > limits.irMax) throw AVException("invalid integration radius:" + std::to_string(ir));

	//check ffmpeg versions
	//if (ffmpeg_check_versions() == false) {
	//	throw AVException("different version of ffmpeg was used at buildtime");
	//}
}

//show info about input and output
void MainData::showIntro(const std::string& deviceName, const MovieReader& reader) const {
	//input source
	*console << "FILE IN: " << fileIn << std::endl;

	//streams in input
	for (size_t i = 0; i < reader.inputStreams.size(); i++) {
		const StreamContext& sc = reader.inputStreams[i];
		StreamInfo info = sc.inputStreamInfo();
		*console << "  Stream " << i
			<< ": type: " << info.streamType
			<< ", codec: " << info.codec
			<< ", duration: " << info.durationString
			<< " --> " << streamHandlerMap.at(sc.handling)
			<< std::endl;
	}

	//video info
	*console << "VIDEO w=" << w << ", h=" << h
		<< ", frames total=" << (reader.frameCount == 0 ? "unknown" : std::to_string(reader.frameCount))
		<< ", fps=" << reader.fps() << " (" << reader.fpsNum << ":" << reader.fpsDen << ")"
		<< ", radius=" << radius
		<< std::endl;

	//output to be written
	if (videoOutputType == OutputType::VIDEO_FILE && pass != DeshakerPass::FIRST_PASS) *console << "FILE OUT: " << fileOut << std::endl;
	if (trajectoryFile.empty() == false) *console << "TRAJECTORY FILE: " << trajectoryFile << std::endl;
	if (resultsFile.empty() == false) *console << "CALCULATION DETAIL OUTPUT: " << resultsFile << std::endl;
	if (videoOutputType == OutputType::SEQUENCE_BMP) *console << "IMAGE SEQUENCE: " << ImageWriter::makeFilenameSamples(fileOut, "bmp") << std::endl;
	if (videoOutputType == OutputType::SEQUENCE_JPG) *console << "IMAGE SEQUENCE: " << ImageWriter::makeFilenameSamples(fileOut, "jpg") << std::endl;
	if (resultImageFile.empty() == false) *console << "CALCULATION DETAIL IMAGES: " << ImageWriter::makeFilenameSamples(resultImageFile, "bmp") << std::endl;
	if (flowFile.empty() == false) *console << "OPTICAL FLOW VIDEO: " << flowFile << std::endl;

	//device info
	*console << "  USING: " << deviceName << std::endl;
}

//default output when no arguments are given
void MainData::showBasicInfo() const {
	*console << "usage: cuvista [-i inputfile -o outputfile] [options...]" << std::endl;
	*console << "use -h to get full help" << std::endl;
}

//show info about system
void MainData::showDeviceInfo() const {
	showDeviceInfo(*console);
	throw SilentQuitException();
}

//output info about system to stream
std::ostream& MainData::showDeviceInfo(std::ostream& os) const {
	//display all devices
	os << "Devices found on this system:" << std::endl;
	for (int i = 0; i < deviceList.size(); i++) {
		os << " #" << i << ": " << deviceList[i]->getName() << std::endl;
	}

	//ffmpeg
	os << std::endl << "FFMPEG:" << std::endl;
	os << "libavutil identifier:  " << LIBAVUTIL_IDENT << std::endl;
	os << "libavcodec identifier:  " << LIBAVCODEC_IDENT << std::endl;
	os << "libavformat identifier: " << LIBAVFORMAT_IDENT << std::endl;
	os << "libswscale identifier: " << LIBSWSCALE_IDENT << std::endl;
	os << "libswresample identifier: " << LIBSWRESAMPLE_IDENT << std::endl;
	if (ffmpeg_check_versions() == false) os << "warning: different version of ffmpeg was used at buildtime" << std::endl;

	//display nvidia info
	os << std::endl;
	os << "Nvidia/Cuda System Details:" << std::endl;
	if (cudaInfo.nvidiaDriverVersion.size() > 0) {
		os << "Nvidia Driver: " << cudaInfo.nvidiaDriverVersion << std::endl;
	} else {
		os << "Nvidia driver not found" << std::endl;
	}
	if (cudaInfo.warning.empty() == false) {
		os << "warning: " << cudaInfo.warning << std::endl;
	}

	//display cuda info
	if (deviceCountCuda() > 0) {
		os << "Cuda Runtime:  " << cudaInfo.runtimeToString() << std::endl;
		os << "Cuda Driver:   " << cudaInfo.driverToString() << std::endl;
		os << "Nvenc Api:     " << cudaInfo.nvencApiToString() << std::endl;
		os << "Nvenc Driver:  " << cudaInfo.nvencDriverToString() << std::endl;
	}
	os << std::endl;

	for (auto& info : cudaInfo.devices) {
		os << "Cuda Device:" << std::endl;
		os << info << std::endl;
	}

	//display OpenCL info
	if (clinfo.devices.size() == 0) {
		os << "OpenCL devices not found" << std::endl;
	}
	if (clinfo.warning.empty() == false) {
		os << "warning: " << clinfo.warning << std::endl;
	}

	for (auto& info : clinfo.devices) {
		os << "OpenCL Device:" << std::endl;
		os << info << std::endl;
	}
	return os;
}

void MainData::showVersionInfo() const {
	*console << "cuvista " << CUVISTA_VERSION << std::endl;
	throw SilentQuitException();
}

void MainData::showHeader() const {
	*console << "CUVISTA - Cuda Video Stabilizer, Version " << CUVISTA_VERSION << std::endl;
	*console << "Copyright (c) 2024 Rainer Bitschi, Email: cuvista@a1.net" << std::endl;
}

std::string MainData::str_toupper(const std::string& s) const {
	std::string out = s;
	std::transform(out.begin(), out.end(), out.begin(), [] (unsigned char c) { return std::toupper(c); });
	return out;
}

//-------------------------------
//   INPUT PARAMETERS
//-------------------------------

std::string& MainData::Parameters::str() {
	return *(data() + index);
}

bool MainData::Parameters::hasArgs() {
	return index < size();
}

bool MainData::Parameters::nextArg(std::string&& param) {
	std::string arg = "-" + param;
	bool ok = index < size() && *(data() + index) == arg;
	if (ok) index++;
	return ok;
}

bool MainData::Parameters::nextArg(std::string&& param, std::string& nextParam) {
	std::string arg = "-" + param;
	bool ok = index < size() - 1 && *(data() + index) == arg;
	if (ok) {
		nextParam = *(data() + index + 1);
		index += 2;
	}
	return ok;
}

void MainData::probeCuda() {
	cudaInfo.probeCuda();
}

size_t MainData::deviceCountCuda() const {
	return cudaInfo.devices.size();
}

void MainData::probeOpenCl() {
	cl::probeRuntime(clinfo);
}

size_t MainData::deviceCountOpenCl() const {
	return clinfo.devices.size();
}

std::string MainData::getCpuName() const {
	return deviceInfoCpu.getName();
}

bool MainData::hasAvx512() const {
	auto features = deviceInfoCpu.getCpuFeatures();
	return features.avx512f & features.avx512vl & features.avx512bw & features.avx512dq & features.sse3 & features.avx2;
}

bool MainData::hasAvx2() const {
	auto features = deviceInfoCpu.getCpuFeatures();
	return features.sse3 & features.avx2;
}
