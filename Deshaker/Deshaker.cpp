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
#include "MovieFrame.hpp"
#include "FrameResult.hpp"
#include "ProgressDisplayConsole.hpp"
#include "UserInputConsole.hpp"

int deshake(int argsCount, char** args) {
	//command line arguments for debugging
	//std::vector<std::string> argsInput = { "-device", "0", "-frames", "10", "-i", "d:/VideoTest/02.mp4", "-y", "-resim", "d:/VideoTest/im/im%04d.bmp"};
	std::vector<std::string> argsInput(args, args + argsCount);

	//main program start
	MainData data;
	std::unique_ptr<MovieFrame> frame = std::make_unique<DefaultFrame>(data);
	std::unique_ptr<MovieReader> reader = std::make_unique<FFmpegReader>();
	std::unique_ptr<MovieWriter> writer = std::make_unique<NullWriter>(data);

	try {
		data.probeCudaDevices();
		data.probeInput(argsInput);
		//create MovieReader
		InputContext ctx = reader->open(data.fileIn);
		data.validate(ctx);

		//----------- create appropriate MovieWriter
		if (data.pass == DeshakerPass::FIRST_PASS)
			writer = std::make_unique<NullWriter>(data);
		else if (data.videoOutputType == OutputType::PIPE)
			writer = std::make_unique<PipeWriter>(data);
		else if (data.videoOutputType == OutputType::TCP)
			writer = std::make_unique<TCPWriter>(data);
		else if (data.videoOutputType == OutputType::BMP)
			writer = std::make_unique<BmpImageWriter>(data);
		else if (data.videoOutputType == OutputType::JPG)
			writer = std::make_unique<JpegImageWriter>(data);
		else if (data.videoOutputType == OutputType::VIDEO_FILE) {
			if (data.deviceNum == -1 && data.encodingDevice == EncodingDevice::AUTO)
				writer = std::make_unique<FFmpegWriter>(data);
			else if (data.encodingDevice == EncodingDevice::AUTO && data.canDeviceEncode())
				writer = std::make_unique<CudaFFmpegWriter>(data);
			else if (data.encodingDevice == EncodingDevice::GPU)
				writer = std::make_unique<CudaFFmpegWriter>(data);
			else
				writer = std::make_unique<FFmpegWriter>(data);
		} else writer = std::make_unique<NullWriter>(data);

		writer->open(data.videoCodec);

		//----------- create Frame Handler Class
		if (data.dummyFrame) {
			//skip stabilizing stuff to test decoding and encoding
			frame = std::make_unique<DummyFrame>(data);

		} else if (data.deviceNum == -1) {
			//only on CPU
			frame = std::make_unique<CpuFrame>(data);

		} else {
			//use CUDA GPU
			frame = std::make_unique<GpuFrame>(data);
		}

	} catch (const CancelException& ce) {
		*data.console << ce.what() << std::endl;
		return -1;

	} catch (const AVException& e) {
		errorLogger.logError("ERROR: " + std::string(e.what()));

	} catch (const std::invalid_argument& e) {
		errorLogger.logError("ERROR: invalid value: " + std::string(e.what()));

	} catch (...) {
		errorLogger.logError("ERROR: invalid parameter");
	}

	//setup progress output
	std::unique_ptr<ProgressDisplay> progress;
	if (data.progressType == ProgressType::REWRITE_LINE) progress = std::make_unique<ProgressDisplayRewriteLine>(data);
	else if (data.progressType == ProgressType::GRAPH) progress = std::make_unique<ProgressDisplayGraph>(data);
	else if (data.progressType == ProgressType::DETAILED) progress = std::make_unique<ProgressDisplayDetailed>(data);
	else progress = std::make_unique<ProgressDisplayNone>(data);

	//setup loop worker
	std::unique_ptr<MovieFrame::DeshakerLoop> loop;
	if (data.pass == DeshakerPass::COMBINED) loop = std::make_unique<MovieFrame::DeshakerLoopCombined>();
	else if (data.pass == DeshakerPass::FIRST_PASS) loop = std::make_unique<MovieFrame::DeshakerLoopFirst>();
	else if (data.pass == DeshakerPass::SECOND_PASS) loop = std::make_unique<MovieFrame::DeshakerLoopSecond>();
	else if (data.pass == DeshakerPass::CONSECUTIVE) loop = std::make_unique<MovieFrame::DeshakerLoopClassic>();
	else loop = std::make_unique<MovieFrame::DeshakerLoop>();

	// --------------------------------------------------------------
	// --------------- main loop start ------------------------------
	// --------------------------------------------------------------
	auto t1 = std::chrono::high_resolution_clock::now();
	UserInputConsole inputHandler(*data.console);

	if (errorLogger.hasNoError()) {
		loop->run(*frame, *progress, *reader, *writer, inputHandler);
	}

	// --------------------------------------------------------------
	// --------------- main loop end --------------------------------
	// --------------------------------------------------------------

	//close input and output, destruct frame object
	writer.reset(); //destruct writer before frame
	reader.reset();
	frame.reset();

	//stopwatch
	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration nanos = t2 - t1;
	double secs = nanos.count() / 1e9;

	//final console messages
	Stats& status = data.status;
	if (errorLogger.hasError()) {
		*data.console << std::endl << "ERROR STACK:" << std::endl;
		std::vector<ErrorEntry> errorList = errorLogger.getErrors();
		for (int i = 0; i < errorList.size(); i++) {
			*data.console << "[" << i << "] " << errorList[i].msg << std::endl;
		}

	} else {
		progress->update(true);
		progress->terminate();

		if (data.frameCount > 0 && data.frameCount != status.frameWriteIndex && inputHandler.doContinue() && data.showHeader) {
			*data.console << "warning: number of frames processed does not match expected frame count" << std::endl;
		}
	}

	if (status.frameWriteIndex > 0 && data.showHeader) {
		double fps = (status.frameInputIndex + status.frameWriteIndex) / 2.0 / secs;
		std::string time = secs < 60.0 ? std::format("{:.1f} sec", secs) : std::format("{:.1f} min", secs / 60.0);
		*data.console << status.frameEncodeIndex << " frames written in " << time << " at " << std::setprecision(1) << std::fixed << fps << " fps" << std::endl;
	}

	return errorLogger.hasError();
}