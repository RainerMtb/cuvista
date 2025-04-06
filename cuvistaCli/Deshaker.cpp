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

#include "Deshaker.hpp"

std::ostream& printError(std::ostream& os, const std::string& msg1) {
	//print in red
	return os << "\x1B[1;31m" << msg1 << "\x1B[0m" << std::endl;
}

DeshakerResult deshake(std::vector<std::string> argsInput, std::ostream* console) {
	std::set_terminate([] {
		printError(std::cout, "error: std::terminate was called");
		std::exit(-20);
	});

	//set command line arguments for debugging
	//argsInput = { "-frames", "10", "-i", "d:/VideoTest/02.mp4", "-o", "c:/temp/im%04d.jpg", "-y"}; //debugging

	//main program start
	MainData data;
	data.console = console;

	std::unique_ptr<MovieReader> reader;
	std::unique_ptr<MovieWriter> writer;
	std::shared_ptr<ProgressDisplay> progress;
	std::shared_ptr<FrameExecutor> executor;
	std::shared_ptr<MovieFrame> frame;
	AuxWriters auxWriters;

	try {
		data.probeOpenCl();
		data.probeCuda();
		data.collectDeviceInfo();
		data.probeInput(argsInput);

		//create MovieReader
		reader = std::make_unique<FFmpegReader>();
		reader->open(data.fileIn);
		data.validate(*reader);

		//----------- create appropriate MovieWriter
		if (data.stackPosition)
			writer = std::make_unique<StackedWriter>(data, *reader, *data.stackPosition);
		else if (data.videoOutputType == OutputType::PIPE)
			writer = std::make_unique<PipeWriter>(data, *reader);
		else if (data.videoOutputType == OutputType::SEQUENCE_BMP)
			writer = std::make_unique<BmpImageWriter>(data, *reader);
		else if (data.videoOutputType == OutputType::SEQUENCE_JPG)
			writer = std::make_unique<JpegImageWriter>(data, *reader);
		else if (data.videoOutputType == OutputType::VIDEO_FILE && data.selectedEncoding.device == EncodingDevice::CPU)
			writer = std::make_unique<FFmpegWriter>(data, *reader);
		else if (data.videoOutputType == OutputType::VIDEO_FILE && data.selectedEncoding.device == EncodingDevice::NVENC)
			writer = std::make_unique<CudaFFmpegWriter>(data, *reader);
		else
			writer = std::make_unique<BaseWriter>(data, *reader);

		writer->open(data.requestedEncoding);

		//----------- create secondary Writers
		if (!data.trajectoryFile.empty()) {
			auxWriters.push_back(std::make_unique<AuxTransformsWriter>(data));
		}
		if (!data.resultsFile.empty()) {
			auxWriters.push_back(std::make_unique<ResultDetailsWriter>(data));
		}
		if (!data.resultImageFile.empty()) {
			auxWriters.push_back(std::make_unique<ResultImageWriter>(data));
		}
		if (!data.flowFile.empty()) {
			auxWriters.push_back(std::make_unique<OpticalFlowWriter>(data, *reader));
		}
		for (auto& aw : auxWriters) {
			aw->open();
		}

		//----------- create Frame Handler for selected loop
		if (data.pass == DeshakerPass::COMBINED) {
			frame = std::make_shared<MovieFrameCombined>(data, *reader, *writer);

		} else if (data.pass == DeshakerPass::CONSECUTIVE) {
			frame = std::make_shared<MovieFrameConsecutive>(data, *reader, *writer);
		}

		//----------- create Frame Executor Class
		if (data.dummyFrame) {
			DeviceInfoBase& dib = data.deviceInfoNull;
			executor = dib.create(data, *frame);

		} else {
			DeviceInfoBase* dib = data.deviceList[data.deviceSelected];
			executor = dib->create(data, *frame);
		}

	} catch (const SilentQuitException&) {
		return { 0 };

	} catch (const CancelException& e) {
		printError(*data.console, e.what());
		return { -1 };

	} catch (const AVException& e) {
		printError(*data.console, std::string("error: ") + e.what());
		if (errorLogger().hasError()) {
			printError(*data.console, std::string("error: ") + errorLogger().getErrorMessage());
		}
		return { -2 };

	} catch (const std::invalid_argument& e) {
		printError(*data.console, std::string("invalid value: ") + e.what());
		if (errorLogger().hasError()) {
			printError(*data.console, std::string("error: ") + errorLogger().getErrorMessage());
		}
		return { -3 };

	} catch (...) {
		printError(*data.console, "unknown error in cuvista");
		return { -4 };
	}

	//setup progress output
	if (data.progressType == ProgressType::MULTILINE) progress = std::make_shared<ProgressDisplayMultiLine>(*frame, data.console);
	else if (data.progressType == ProgressType::REWRITE_LINE) progress = std::make_shared<ProgressDisplayRewriteLine>(*frame, data.console);
	else if (data.progressType == ProgressType::NEW_LINE) progress = std::make_shared<ProgressDisplayNewLine>(*frame, data.console);
	else if (data.progressType == ProgressType::GRAPH) progress = std::make_shared<ProgressDisplayGraph>(*frame, data.console);
	else progress = std::make_shared<ProgressDisplayNone>(*frame);

	// --------------------------------------------------------------
	// --------------- main loop start ------------------------------
	// --------------------------------------------------------------
	data.timeStart();
	UserInputConsole inputHandler(*data.console);

	if (errorLogger().hasNoError()) {
		frame->runLoop(progress, inputHandler, auxWriters, executor);
	}

	// --------------------------------------------------------------
	// --------------- main loop end --------------------------------
	// --------------------------------------------------------------

	//close input and output, destruct frame object
	//copy statistics before destructing
	int64_t framesWritten = writer->frameIndex;

	//final console messages
	if (errorLogger().hasError()) {
		printError(*data.console, "ERROR STACK:");
		std::vector<ErrorEntry> errorList = errorLogger().getErrors();
		for (int i = 0; i < errorList.size(); i++) {
			printError(*data.console, std::format("[{}] {}", i, errorList[i].msg));
		}

	} else {
		progress->update(true);
		progress->terminate();
	}

	//collect debugging data
	DeshakerResult result;
	result.statusCode = errorLogger().hasError() ? -10 : 0;
	result.frameCount = reader->frameCount;
	result.framesRead = reader->frameIndex;
	result.framesWritten = writer->frameIndex;
	result.framesEncoded = writer->frameEncoded;
	result.bytesEncoded = writer->encodedBytesTotal;
	result.trajectory = frame->mTrajectory.getTrajectory();
	result.executorName = executor->mDeviceInfo.getName();

	//destruct writer before frame
	auxWriters.clear();
	writer.reset();
	reader.reset();
	executor.reset();
	frame.reset();

	//stopwatch
	double secs = data.timeElapsedSeconds();
	double fps = framesWritten / secs;
	if (framesWritten > 0 && data.printSummary) {
		std::string time = secs < 60.0 ? std::format("{:.1f} sec", secs) : std::format("{:.1f} min", secs / 60.0);
		std::string str = std::format("{} frames written in {} at {:.1f} fps", framesWritten, time, fps);
		*data.console << "\x1B[1;36m" << str << "\x1B[0m" << std::endl;
	}

	result.secs = secs;
	return result;
}
