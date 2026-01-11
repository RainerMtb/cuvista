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
#include "SystemStuff.hpp"
#include "MovieReader.hpp"
#include "MovieWriter.hpp"
#include "CudaFrame.hpp"
#include "NvidiaDriver.hpp"
#include "clMain.hpp"
#include "SelfTest.hpp"

#include <algorithm>
#include <regex>
#include <vector>
#include <filesystem>


//test or ask for file writing
bool MainData::checkFileForWriting(const std::string& file, DecideYNA permission) const {
	if (file.empty()) {
		throw AVException("output file missing");
	}
	if (permission == DecideYNA::NO) {
		throw CancelException("output file exists, aborting...");
	}
	if (permission == DecideYNA::ASK && std::filesystem::exists(std::filesystem::path(file))) {
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
			if (str_toupper(next) == "NULL") {
				//do not do any output
				outputOption = OutputOption::OPTION_NONE;
				fileOut = "";

			} else if (str_toupper(next).ends_with(".BMP")) {
				outputOption = OutputOption::IMAGE_BMP;
				fileOut = next;

			} else if (str_toupper(next).ends_with(".JPG")) {
				outputOption = OutputOption::IMAGE_JPG;
				fileOut = next;

			} else if (str_toupper(next).ends_with(".YUV")) {
				outputOption = OutputOption::RAW_YUV444;
				fileOut = next;

			} else if (str_toupper(next).ends_with(".NV12")) {
				outputOption = OutputOption::RAW_NV12;
				fileOut = next;

			} else {
				fileOut = next;
				if (outputOption.group == OutputGroup::VIDEO_NVENC || outputOption.group == OutputGroup::VIDEO_FFMPEG) {
					//keep selected option

				} else {
					//default video file output
					outputOption = OutputOption::OPTION_AUTO;
				}
			}

		} else if (args.nextArg("enc", next)) {
			outputOption = OutputOption::find(str_toupper(next));
			if (outputOption == OutputOption::PIPE_RAW || outputOption == OutputOption::PIPE_ASF) {
				console = &nullStream;
				printHeader = false;
				progressType = ProgressType::NONE;

			} else if (outputOption.group == OutputGroup::INVALID)
				throw AVException("invalid encoding option: " + next);

		} else if (args.nextArg("trf", next)) {
			//trajectory file
			trajectoryFile = next;

		} else if (args.nextArg("res", next)) {
			//file to store detailed results
			resultsFile = next;

		} else if (args.nextArg("resim")) {
			//produce and write transformation images
			outputOption = OutputOption::IMAGE_RESULTS;

		} else if (args.nextArg("resvid")) {
			//produce and write transformation video
			outputOption = OutputOption::VIDEO_RESULTS;

		} else if (args.nextArg("flow")) {
			//optical flow video
			outputOption = OutputOption::VIDEO_FLOW;

		} else if (args.nextArg("mode", next)) {
			int i = std::stoi(next);
			if (i >= 0 && i <= defaultParam.modeMax) mode = i;
			else throw AVException("invalid value for mode: " + next);

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

		} else if (args.nextArg("bgcolor", next)) {
			//background color
			auto it = colorMap.find(next); //some color literals
			if (it == colorMap.end()) {
				std::smatch matcher;
				if (std::regex_match(next, matcher, std::regex("(\\d{1,3}):(\\d{1,3}):(\\d{1,3})$"))) {
					//decimal numbers seperated by colon
					int r = std::stoi(matcher[1].str());
					int g = std::stoi(matcher[2].str());
					int b = std::stoi(matcher[3].str());
					backgroundColor = Color::rgb(r, g, b);

				} else if (std::regex_match(next, matcher, std::regex("#([[:xdigit:]]{2})([[:xdigit:]]{2})([[:xdigit:]]{2})$"))) {
					//webcolor
					int r = std::stoi(matcher[1].str(), nullptr, 16);
					int g = std::stoi(matcher[2].str(), nullptr, 16);
					int b = std::stoi(matcher[3].str(), nullptr, 16);
					backgroundColor = Color::rgb(r, g, b);

				} else {
					throw AVException("invalid color definition: " + next);
				}

			} else {
				//color literal
				backgroundColor = it->second;
			}

			//also set background mode
			bgmode = BackgroundMode::COLOR;

		} else if (args.nextArg("cputhreads", next)) {
			//number of threads to use on cpu
			cpuThreadsRequired = { std::stoul(next) };

		} else if (args.nextArg("cudathreads", next)) {
			cudaThreads = std::stoi(next);

		} else if (args.nextArg("crf", next)) {
			requestedCrf = std::stoi(next); //output encoder crf value

		} else if (args.nextArg("y")) {
			overwriteOutput = DecideYNA::YES; //always overwrite

		} else if (args.nextArg("n")) {
			overwriteOutput = DecideYNA::NO; //never overwrite

		} else if (args.nextArg("info")) {
			showHeader();
			*console << std::endl;
			showDeviceInfo(*console);
			showEncodingInfo(*console);
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
			else if (i == 1) progressType = ProgressType::MULTILINE;
			else if (i == 2) progressType = ProgressType::REWRITE_LINE;
			else if (i == 3) progressType = ProgressType::NEW_LINE;
			else if (i == 4) progressType = ProgressType::GRAPH;
			else throw AVException("invalid progress type: " + next);

		} else if (args.nextArg("noheader")) {
			printHeader = false;

		} else if (args.nextArg("showheader")) {
			printHeader = true;

		} else if (args.nextArg("nosummary")) {
			printSummary = false;

		} else if (args.nextArg("showsummary")) {
			printSummary = true;

		} else if (args.nextArg("quiet")) {
			printHeader = false;
			printSummary = false;
			progressType = ProgressType::NONE;

		} else if (args.nextArg("levels", next)) {
			//pyramid levels to compute
			pyramidLevelsRequested = std::stoi(next);

		} else if (args.nextArg("ir", next)) {
			//integration radius
			ir = std::stoi(next);
			iw = ir * 2 + 1;

		} else if (args.nextArg("roicrop", next)) {
			//define region of interest for computing
			std::smatch matcher;
			if (std::regex_match(next, matcher, std::regex("(\\d+):(\\d+)$")))
				roiCrop = { std::stoi(matcher[1].str()), std::stoi(matcher[2].str()) };
			else
				throw AVException("invalid roicrop definition: " + next);

		} else if (args.nextArg("codec", next)) {
			throw AVException("invalid option, use -enc instead");

		} else if (args.nextArg("encoder", next)) {
			throw AVException("invalid option, use -enc instead");

		} else if (args.nextArg("frames", next)) {
			//limit output frames
			maxFrames = std::stoll(next);

		} else if (args.nextArg("copyframes")) {
			dummyFrame = true;

		 } else if (args.nextArg("rng", next)) {
		 	if (std::stoi(next) == 1) sampler = std::make_shared<UrbgSampler<PointContext, PseudoRandomSource>>();
		 	else if (std::stoi(next) == 2) sampler = std::make_shared<UrbgSampler<PointContext, std::random_device>>();
		 	else if (std::stoi(next) == 3) sampler = std::make_shared<UrbgSampler<PointContext, std::default_random_engine>>();
		 	else throw AVException("invalid random generator selection: " + next);

		} else if (args.nextArg("stack", next)) {
			std::smatch matcher;
			if (std::regex_match(next, matcher, std::regex("(\\d+):(\\d+)$"))) {
				stackCrop = { std::stoi(matcher[1].str()), std::stoi(matcher[2].str()) };
				outputOption = OutputOption::VIDEO_STACK;

			} else {
				throw AVException("invalid parameter for stacking: " + next);
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
		throw SilentQuitException();
	}

	//check output file presence and permissions
	std::string fileOutCheck = fileOut;
	if (outputOption.isVideoFile()) {
		fileOutCheck = fileOut;

	} else if (outputOption == OutputOption::IMAGE_BMP) {
		fileOutCheck = ImageWriter::makeFilename(fileOut, 0, "bmp");

	} else if (outputOption == OutputOption::IMAGE_JPG) {
		fileOutCheck = ImageWriter::makeFilename(fileOut, 0, "jpg");
	}

	if (fileOutCheck.size() > 0) {
		std::filesystem::path path1 = fileIn;
		std::filesystem::path path2 = fileOutCheck;
		std::error_code ec;
		//use check with error code parameter as non existing output file will throw an exception otherwise
		bool filesEqual = std::filesystem::equivalent(path1, path2, ec);
		std::string ecstr = ec.message();
		if (filesEqual) {
			throw AVException("cannot use the same file for input and output");
		}
		checkFileForWriting(fileOutCheck, overwriteOutput);
	}
}

void MainData::collectDeviceInfo() {
	//sort cuda devices by compute
	std::sort(deviceInfoCuda.begin(), deviceInfoCuda.end());

	//ffmpeg available encoders
	std::vector<OutputOption> cpuEncoders = {
		OutputOption::FFMPEG_H264,
		OutputOption::FFMPEG_HEVC,
		OutputOption::FFMPEG_AV1,
		OutputOption::FFMPEG_FFV1
	};

	//nvenc available encoders taken from first cuda device
	std::vector<OutputOption> nvencEncoders = {};
	if (deviceInfoCuda.size() > 0) {
		nvencEncoders = deviceInfoCuda.front().videoEncodingOptions;
	}

	//CPU device
	deviceInfoCpu.videoEncodingOptions = cpuEncoders;
	std::copy(nvencEncoders.begin(), nvencEncoders.end(), std::back_inserter(deviceInfoCpu.videoEncodingOptions));
	deviceList.push_back(&deviceInfoCpu);

	//check for Avx512
	if (deviceInfoAvx.hasAvx512()) {
		deviceInfoAvx.videoEncodingOptions = deviceInfoCpu.videoEncodingOptions;
		deviceList.push_back(&deviceInfoAvx);
	}
	
	///OpenCL devices
	for (DeviceInfoOpenCl& dev : deviceInfoOpenCl) {
		dev.videoEncodingOptions = deviceInfoCpu.videoEncodingOptions;
		deviceList.push_back(&dev);
	}

	//Cuda devices
	for (DeviceInfoCuda& dev : deviceInfoCuda) {
		std::copy(cpuEncoders.begin(), cpuEncoders.end(), std::back_inserter(dev.videoEncodingOptions));
		deviceList.push_back(&dev);
	}
}

void MainData::validate(const MovieReader& reader) {
	if (this->cpuThreadsRequired.has_value()) this->cpuThreads = this->cpuThreadsRequired.value();
	else this->cpuThreads = std::max(2u, std::thread::hardware_concurrency() * 3 / 4);

	//main metrics
	this->w = reader.w;
	this->h = reader.h;

	//pyramid levels to compute
	this->zMax = pyramidLevelsRequested - 1;

	int div0 = 1 << zMax;
	int numPoints = (w / div0) * (h / div0);
	while (numPoints > maxResultCount) {
		numPoints /= 4;
		zMax++;
	}
	this->zMin = zMax - pyramidLevelsRequested + 1;
	this->pyramidLevels = zMax + 1;

	//number of rows for one complete pyramid
	int hh = h;
	int rowCount = hh;
	for (int i = 0; i < zMax; i++) {
		hh /= 2;
		rowCount += hh;
	}
	this->pyramidRowCount = rowCount;

	//number of points to compute, leaving one pixel around the border so delta values can be computed
	int div = 1 << zMax;
	this->ixCount = w / div - 2 * ir - 3;
	this->iyCount = h / div - 2 * ir - 3;
	this->resultCount = ixCount * iyCount;
	//std::cout << "zMin " << zMin << ", zMax " << zMax << ", ixCount " << ixCount << ", iyCount " << iyCount << std::endl;

	//check background color value
	for (int i = 0; i < 3; i++) {
		int col = backgroundColor.getChannel(i);
		if (col < 0 || col > 255) throw AVException("invalid background color value: " + std::to_string(col));
	}
	//set background yuv color vector
	backgroundColor.toYUVfloat(&bgcolorYuv.y, &bgcolorYuv.u, &bgcolorYuv.v);

	int pitchBase = 256;
	cpupitch = (w + pitchBase - 1) / pitchBase * pitchBase;

	//number of frames to buffer
	this->radius = (int) std::round(radsec * reader.fps());

	//dynamic zoom fallback rate per frame
	this->zoomFallback = 1.0 - zoomFallbackTotal / radius;

	//choose device and encoding
	if (deviceSelected >= deviceList.size()) {
		throw AVException("invalid device number: " + std::to_string(deviceRequested));
	}
	if (deviceRequested == false) {
		deviceSelected = deviceList.size() - 1;
	}
	if (outputOption.group == OutputGroup::AUTO) {
		DeviceInfoBase* dev = deviceList[deviceSelected];
		outputOption = dev->videoEncodingOptions.front();
	}
	if (requestedCrf == -1) {
		//set default crf from option list
		selectedCrf = outputOption.defaultCrf();

	} else {
		selectedCrf = (uint8_t) requestedCrf;
	}

	//check certain values ranges for sanity
	if (radsec < defaultParam.radsecMin || radsec > defaultParam.radsecMax) throw AVException("invalid temporal radius: " + std::to_string(radsec));
	if (radius < defaultParam.radiusMin || radius > defaultParam.radiusMax) throw AVException("invalid image radius: " + std::to_string(radius));
	if (w < defaultParam.wMin) throw AVException("invalid input video width: " + std::to_string(w));
	if (h < defaultParam.hMin) throw AVException("invalid input video height: " + std::to_string(h));
	size_t mp = deviceList[deviceSelected]->maxPixel;
	if (w > mp) throw AVException("frame width exceeds maximum of " + std::to_string(mp) + " px");
	if (h > mp) throw AVException("frame height exceeds maximum of " + std::to_string(mp) + " px");
	if (w % 2 != 0 || h % 2 != 0) throw AVException("width and height must be factors of two");
	if (zoomMin > zoomMax) throw AVException("invalid zoom values, max zoom must be greater min zoom");
	if (zoomMin < defaultParam.imZoomMin || zoomMin > defaultParam.imZoomMax) throw AVException("invalid zoom value");
	if (zoomMax < defaultParam.imZoomMin || zoomMax > defaultParam.imZoomMax) throw AVException("invalid zoom value");
	if (ixCount < 8 || iyCount < 8) throw AVException("invalid settings");

	if (pyramidLevelsRequested < defaultParam.levelsMin || pyramidLevelsRequested > defaultParam.levelsMax) throw AVException("invalid pyramid levels: " + std::to_string(pyramidLevels));
	if (ir < defaultParam.irMin || ir > defaultParam.irMax) throw AVException("invalid integration radius: " + std::to_string(ir));

	if (cudaThreads > 32) throw AVException("invalid cuda threads parameter " + std::to_string(cudaThreads));

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
	for (size_t i = 0; i < reader.mInputStreams.size(); i++) {
		const StreamContext& sc = reader.mInputStreams[i];
		StreamInfo info = sc.inputStreamInfo();
		*console << "  Stream " << i
			<< ": type: " << info.streamType
			<< ", codec: " << info.codec
			<< ", duration: " << info.durationString
			;

		*console << " --> ";
		for (int idx = 0; idx < sc.outputStreams.size(); idx++) {
			StreamHandling handling = sc.outputStreams[idx]->handling;
			*console << idx << ":" << streamHandlerMap.at(handling);
		}
		*console << std::endl;
	}

	//video info
	*console << "VIDEO w=" << w << ", h=" << h
		<< ", frames=" << (reader.frameCount < 1 ? "unknown" : std::to_string(reader.frameCount))
		<< ", fps=" << reader.fps() << " (" << reader.fpsNum << ":" << reader.fpsDen << ")"
		<< ", radius=" << radius
		<< std::endl;

	//output to be written
	if (outputOption.isVideoFile()) *console << "FILE OUT: " << outputOption.fullName() << ", " << fileOut << std::endl;
	if (trajectoryFile.empty() == false) *console << "TRAJECTORY FILE: " << trajectoryFile << std::endl;
	if (resultsFile.empty() == false) *console << "CALCULATION DETAIL OUTPUT: " << resultsFile << std::endl;
	if (outputOption == OutputOption::IMAGE_BMP) *console << "IMAGE SEQUENCE: " << ImageWriter::makeFilenameSamples(fileOut, "bmp") << std::endl;
	if (outputOption == OutputOption::IMAGE_JPG) *console << "IMAGE SEQUENCE: " << ImageWriter::makeFilenameSamples(fileOut, "jpg") << std::endl;
	if (outputOption == OutputOption::IMAGE_RESULTS) *console << "RESULTS IMAGES: " << ImageWriter::makeFilenameSamples(fileOut, "bmp") << std::endl;

	//device info
	*console << "\x1B[1;36m" << "  USING: " << deviceName << "\x1B[0m" << std::endl;
}

//default output when no arguments are given
void MainData::showBasicInfo() const {
	*console << "usage: cuvista [-i inputfile -o outputfile] [options...]" << std::endl;
	*console << "use -h to get full help" << std::endl;
}

//show info about system
void MainData::showDeviceInfo() const {
	showDeviceInfo(*console);
	showEncodingInfo(*console);
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
	os << std::endl << "FFMPEG Versions:" << std::endl;
	os << "libavutil identifier:     " << LIBAVUTIL_IDENT << std::endl;
	os << "libavcodec identifier:    " << LIBAVCODEC_IDENT << std::endl;
	os << "libavformat identifier:   " << LIBAVFORMAT_IDENT << std::endl;
	os << "libswscale identifier:    " << LIBSWSCALE_IDENT << std::endl;
	os << "libswresample identifier: " << LIBSWRESAMPLE_IDENT << std::endl;
	if (ffmpeg_check_versions() == false) {
		os << "warning: different versions of ffmpeg are used at compiletime and runtime" << std::endl;
	}

	//display nvidia info
	os << std::endl;
	os << "Nvidia/Cuda System Details:" << std::endl;
	if (DeviceInfoCuda::nvidiaDriverVersion.size() > 0) {
		os << "Nvidia Driver: " << DeviceInfoCuda::nvidiaDriverVersion << std::endl;
	} else {
		os << "Nvidia driver not found" << std::endl;
	}
	if (DeviceInfoCuda::warning.empty() == false) {
		os << "warning: " << DeviceInfoCuda::warning << std::endl;
	}

	//display cuda info
	if (deviceInfoCuda.size() > 0) {
		os << "Cuda Runtime:  " << DeviceInfoCuda::runtimeToString() << std::endl;
		os << "Cuda Driver:   " << DeviceInfoCuda::driverToString() << std::endl;
		os << "Nvenc Api:     " << DeviceInfoCuda::nvencApiToString() << std::endl;
		os << "Nvenc Driver:  " << DeviceInfoCuda::nvencDriverToString() << std::endl;
	}
	os << std::endl;

	for (auto& info : deviceInfoCuda) {
		os << "Cuda Device:" << std::endl;
		os << info << std::endl;
	}

	//display OpenCL info
	os << "OpenCL devices found: " << deviceInfoOpenCl.size();
	if (DeviceInfoOpenCl::warning.empty() == false) {
		os << ", " << DeviceInfoOpenCl::warning;
	}
	os << std::endl << std::endl;

	for (const DeviceInfoOpenCl& info : deviceInfoOpenCl) {
		os << "OpenCL Device:" << std::endl;
		os << info << std::endl;
	}

	return os;
}

//show video encoding options
std::ostream& MainData::showEncodingInfo(std::ostream& os) const {
	//available encoding options
	os << "Available Video Encoding Options:" << std::endl;
	std::vector<OutputOption> videoOptions = deviceInfoCpu.videoEncodingOptions;
	std::sort(videoOptions.begin(), videoOptions.end(), [] (const OutputOption& o1, const OutputOption o2) { return o1.fullName() < o2.fullName(); });
	for (const OutputOption& o : videoOptions) {
		os << o.fullName() << std::endl;
	}

	return os;
}

void MainData::showVersionInfo() const {
	*console << "cuvista " << CUVISTA_VERSION << std::endl;
	throw SilentQuitException();
}

void MainData::showHeader() const {
	*console << "CUVISTA - Cuda Video Stabilizer, Version " << CUVISTA_VERSION << std::endl;
	*console << "Copyright (c) 2026 Rainer Bitschi, cuvista@a1.net, https://github.com/RainerMtb/cuvista" << std::endl;
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

std::vector<DeviceInfoCuda> MainData::probeCuda() {
	return DeviceInfoCuda::probeCuda();
}

std::vector<DeviceInfoOpenCl> MainData::probeOpenCl() {
	return cl::probeRuntime();
}

std::string MainData::getCpuName() const {
	return deviceInfoCpu.getName();
}

bool MainData::hasAvx10() const {
	return deviceInfoAvx.hasAvx10();
}

bool MainData::hasAvx512() const {
	return deviceInfoAvx.hasAvx512();
}

bool MainData::hasAvx2() const {
	return deviceInfoAvx.hasAvx2();
}
