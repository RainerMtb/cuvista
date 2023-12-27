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
#include "CudaWriter.hpp"
#include "MovieReader.hpp"
#include "MovieFrame.hpp"
#include "CpuFrame.hpp"
#include "CudaFrame.hpp"
#include "OpenClFrame.hpp"
#include "FrameResult.hpp"
#include "ProgressDisplayConsole.hpp"
#include "UserInputConsole.hpp"

int deshake(int argsCount, char** args) {
	//command line arguments for debugging
	//std::vector<std::string> argsInput = { "-device", "0", "-frames", "10", "-i", "d:/VideoTest/02.mp4", "-y", "-resim", "d:/VideoTest/im/im%04d.bmp"};
	std::vector<std::string> argsInput(args, args + argsCount);

	//main program start
	MainData data;
	std::unique_ptr<MovieReader> reader = std::make_unique<NullReader>();
	std::unique_ptr<MovieWriter> writer = std::make_unique<NullWriter>(data, *reader);
	std::unique_ptr<MovieFrame> frame = std::make_unique<DefaultFrame>(data, *reader, *writer);
	AuxWriters auxWriters;
	
	try {
		data.probeOpenCl();
		data.probeCuda();
		data.probeInput(argsInput);
		data.collectDeviceInfo();

		//create MovieReader
		reader = std::make_unique<FFmpegReader>();
		reader->open(data.fileIn);
		data.validate(*reader);

		//----------- create appropriate MovieWriter
		if (data.pass == DeshakerPass::FIRST_PASS)
			writer = std::make_unique<TransformsWriterMain>(data, *reader);
		else if (data.blendInput.enabled)
			writer = std::make_unique<StackedWriter>(data, *reader);
		else if (data.videoOutputType == OutputType::PIPE)
			writer = std::make_unique<PipeWriter>(data, *reader);
		else if (data.videoOutputType == OutputType::TCP)
			writer = std::make_unique<TCPWriter>(data, *reader);
		else if (data.videoOutputType == OutputType::BMP)
			writer = std::make_unique<BmpImageWriter>(data, *reader);
		else if (data.videoOutputType == OutputType::JPG)
			writer = std::make_unique<JpegImageWriter>(data, *reader);
		else if (data.videoOutputType == OutputType::VIDEO_FILE && data.selectedEncoding.device == EncodingDevice::CPU)
			writer = std::make_unique<FFmpegWriter>(data, *reader);
		else if (data.videoOutputType == OutputType::VIDEO_FILE && data.selectedEncoding.device == EncodingDevice::NVENC)
			writer = std::make_unique<CudaFFmpegWriter>(data, *reader);
		else 
			writer = std::make_unique<NullWriter>(data, *reader);

		writer->open(data.requestedEncoding);

		//----------- create Frame Handler Class
		if (data.dummyFrame) {
			//skip stabilizing stuff to test decoding and encoding
			frame = std::make_unique<DummyFrame>(data, *reader, *writer);

		} else if (data.deviceList[data.deviceSelected]->type == DeviceType::CPU) {
			//only on CPU
			frame = std::make_unique<CpuFrame>(data, *reader, *writer);

		} else if (data.deviceList[data.deviceSelected]->type == DeviceType::CUDA) {
			//use CUDA GPU
			frame = std::make_unique<CudaFrame>(data, *reader, *writer);

		} else if (data.deviceList[data.deviceSelected]->type == DeviceType::OPEN_CL) {
			//use OpenCL
			frame = std::make_unique<OpenClFrame>(data, *reader, *writer);
		}

		//----------- create secondary Writers
		if (data.pass != DeshakerPass::FIRST_PASS && data.pass != DeshakerPass::SECOND_PASS && data.trajectoryFile.empty() == false) {
			auxWriters.push_back(std::make_unique<AuxTransformsWriter>(data));
		}
		if (!data.resultsFile.empty()) {
			auxWriters.push_back(std::make_unique<ResultDetailsWriter>(data));
		}
		if (!data.resultImageFile.empty()) {
			auxWriters.push_back(std::make_unique<ResultImageWriter>(data));
		}
		for (auto& aw : auxWriters) {
			aw->open();
		}

	} catch (const CancelException& e) {
		*data.console << e.what() << std::endl;
		return -1;

	} catch (const AVException& e) {
		*data.console << "error: " << e.what() << std::endl;
		return -2;

	} catch (const std::invalid_argument& e) {
		*data.console << "invalid value: " << e.what() << std::endl;
		return -3;

	} catch (...) {
		*data.console << "error setting up the system" << std::endl;
		return -4;
	}

	//setup progress output
	std::unique_ptr<ProgressDisplay> progress;
	if (data.progressType == ProgressType::REWRITE_LINE) progress = std::make_unique<ProgressDisplayRewriteLine>(*frame, data.console);
	else if (data.progressType == ProgressType::NEW_LINE) progress = std::make_unique<ProgressDisplayNewLine>(*frame, data.console);
	else if (data.progressType == ProgressType::GRAPH) progress = std::make_unique<ProgressDisplayGraph>(*frame, data.console);
	else if (data.progressType == ProgressType::DETAILED) progress = std::make_unique<ProgressDisplayDetailed>(*frame, data.console);
	else progress = std::make_unique<ProgressDisplayNone>(*frame);

	// --------------------------------------------------------------
	// --------------- main loop start ------------------------------
	// --------------------------------------------------------------
	data.timeStart();
	UserInputConsole inputHandler(*data.console);

	if (errorLogger.hasNoError()) {
		frame->runLoop(data.pass, *progress, inputHandler, auxWriters);
	}

	// --------------------------------------------------------------
	// --------------- main loop end --------------------------------
	// --------------------------------------------------------------

	//close input and output, destruct frame object
	//copy statistics before destructing
	int framesEncoded = writer->frameEncoded;

	//final console messages
	if (errorLogger.hasError()) {
		*data.console << std::endl << "ERROR STACK:" << std::endl;
		std::vector<ErrorEntry> errorList = errorLogger.getErrors();
		for (int i = 0; i < errorList.size(); i++) {
			*data.console << "[" << i << "] " << errorList[i].msg << std::endl;
		}

	} else {
		progress->update(true);
		progress->terminate();
	}

	//destruct writer before frame
	auxWriters.clear();
	writer.reset();
	reader.reset();
	frame.reset();

	//stopwatch
	double secs = data.timeElapsedSeconds();
	double fps = framesEncoded / secs;
	if (framesEncoded > 0 && data.showHeader) {
		std::string time = secs < 60.0 ? std::format("{:.1f} sec", secs) : std::format("{:.1f} min", secs / 60.0);
		std::string str = std::format("{} frames written in {} at {:.1f} fps", framesEncoded, time, fps);
		*data.console << str << std::endl;
	}

	return errorLogger.hasError();
}