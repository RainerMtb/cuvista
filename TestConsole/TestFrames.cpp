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

#include "TestMain.hpp"
#include "clTest.hpp"
#include "Util.hpp"
#include "SelfTestData.hpp"

#include <fstream>
#include <iterator>

struct Result {
	Matf pyramid;
	Matf output;
	std::vector<PointResult> results;
	std::string name;
	ImageYuv image;
	ImageRGBA input;
	std::string error;
};

template <class T> Result runPyramid(MainData& data, int deviceIndex) {
	errorLogger.clearErrors();
	try {
		std::vector<unsigned char> bytes = util::base64_decode(movieTestData);
		MemoryFFmpegReader reader(bytes);
		reader.open("");
		data.collectDeviceInfo();
		data.validate(reader);
		BaseWriter writer(data, reader);

		MovieFrameCombined frame(data, reader, writer);
		std::unique_ptr<FrameExecutor> executor = std::make_unique<T>(data, *data.deviceList[deviceIndex], frame, frame.mPool);
		std::string name = executor->mDeviceInfo.getNameShort();

		std::cout << "running " << name << std::endl;
		reader.read(frame.mBufferFrame);
		executor->inputData(reader.frameIndex, frame.mBufferFrame);
		executor->createPyramid(reader.frameIndex);

		reader.read(frame.mBufferFrame);
		executor->inputData(reader.frameIndex, frame.mBufferFrame);
		executor->createPyramid(reader.frameIndex);

		executor->computeStart(reader.frameIndex, frame.mResultPoints);
		executor->computeTerminate(reader.frameIndex, frame.mResultPoints);

		ImageRGBA im(data.h, data.w);
		executor->getInput(0, im);

		AffineTransform trf;
		trf.addRotation(0.2).addTranslation(-40, 30);
		trf.frameIndex = 0;
		executor->outputData(0, trf);
		writer.prepareOutput(*executor);
		Result result = {
			executor->getPyramid(0),
			executor->getTransformedOutput(),
			frame.mResultPoints,
			name,
			writer.getOutputFrame(),
			im,
			(errorLogger.hasError() ? errorLogger.getErrorMessage() : "")
		};

		return result;

	} catch (AVException e) {
		std::cout << "exception " << e.what() << std::endl;

	} catch (...) {
		std::cout << "general exception" << std::endl;
	}

	return {};
}

void compareFramesPlatforms() {
	std::cout << "comparing platforms..." << std::endl;
	std::vector<Result> results(4);

	{
		//CPU
		MainData data;
		data.deviceRequested = true;
		results[0] = runPyramid<CpuFrame>(data, 0);
	}

	{
		//AVX
		MainData data;
		data.deviceRequested = true;
		data.cpuThreadsRequired = 1;
		results[1] = runPyramid<AvxFrame>(data, 1);
	}

	{
		//Cuda
		MainData data;
		data.deviceRequested = true;
		data.probeCuda();
		results[2] = runPyramid<CudaFrame>(data, 2);
	}

	{
		//OpenCL
		MainData data;
		data.deviceRequested = true;
		data.probeOpenCl();
		results[3] = runPyramid<OpenClFrame>(data, 2);
	}
	std::cout << std::endl;
	//results[0].input.saveAsColorBMP("f:/0.bmp");
	//results[1].input.saveAsColorBMP("f:/1.bmp");
	//results[2].output.saveAsBinary("f:/2.dat");
	//results[0].image.saveAsColorBMP("f:/0.bmp");
	//results[1].image.saveAsColorBMP("f:/1.bmp");

	for (int i = 0; i < results.size(); i++) {
		if (results[i].error.empty() == false) std::cout << "error " << i << " " << results[i].name << ": " << results[i].error << std::endl;
	}

	Result& r1 = results[0];
	for (int i = 1; i < results.size(); i++) {
		Result& r2 = results[i];

		//compare image data
		std::cout << "comparing: " << r1.name << " vs " << r2.name << std::endl;
		std::cout << (r1.pyramid.equalsExact(r2.pyramid) ? "pyramids EQUAL" : "pyramids DIFFER <<<<<<") << std::endl;
		std::cout << (r1.output.equalsExact(r2.output) ? "warped output EQUAL" : "warped output DIFFER <<<<<<") << std::endl;
		std::cout << (r1.image == r2.image ? "image EQUAL" : "image DIFFER <<<<<<") << std::endl;
		std::cout << (r1.input == r2.input ? "input EQUAL" : "input DIFFER <<<<<<<") << std::endl;

		//compare results
		int deltaCount = 0;
		for (size_t i = 0; i < r1.results.size() && i < r2.results.size(); i++) {
			if (r1.results[i] != r2.results[i]) {
				if (deltaCount == 0) std::cout << "results DIFFER: ix0=" << r1.results[i].ix0 << ", iy0=" << r2.results[i].iy0 << std::endl;
				deltaCount++;
			}
		}
		if (deltaCount == 0) std::cout << "results EQUAL" << std::endl;
		else std::cout << "results difference count " << deltaCount << " of " << r1.results.size() << std::endl;
		std::cout << std::endl;
	}

	std::cout << errorLogger.getErrorMessage() << std::endl;
}

void analyzeFrames() {
	std::string file = "d:/VideoTest/02.mp4";
	std::string dest = "f:/";
	int frameFrom = 100;
	int frameTo = 105;

	MainData data;
	data.deviceRequested = true;
	data.resultImageFile = dest;
	FFmpegReader reader;
	reader.open(file);
	data.collectDeviceInfo();
	data.validate(reader);
	ResultImageWriter writer(data);
	writer.open();
	MovieFrameCombined frame(data, reader, writer);
	std::unique_ptr<FrameExecutor> executor = std::make_unique<CpuFrame>(data, *data.deviceList[0], frame, frame.mPool);

	while (true) {
		reader.read(frame.mBufferFrame);

		if (reader.frameIndex == frameFrom - data.radius) {
			executor->inputData(reader.frameIndex, frame.mBufferFrame);
			executor->createPyramid(reader.frameIndex);
		} else if (reader.frameIndex > frameFrom - data.radius) {
			executor->inputData(reader.frameIndex, frame.mBufferFrame);
			executor->createPyramid(reader.frameIndex);
			executor->computeStart(reader.frameIndex, frame.mResultPoints);
			executor->computeTerminate(reader.frameIndex, frame.mResultPoints);
		}
		
		if (writer.frameIndex >= frameFrom) {
			frame.computeTransform(reader.frameIndex);
			const AffineTransform& currentTransform = frame.mFrameResult.getTransform();
			frame.mTrajectory.addTrajectoryTransform(currentTransform);
			const AffineTransform& finalTransform = frame.mTrajectory.computeSmoothTransform(data, writer.frameIndex);
			executor->outputData(writer.frameIndex, finalTransform);
			writer.prepareOutput(*executor);
			writer.write(*executor);
		} else {
			frame.mTrajectory.addTrajectoryTransform(AffineTransform());
			writer.frameIndex++;
		}

		if (reader.frameIndex == frameTo) {
			break;
		}
		if (errorLogger.hasError()) {
			std::cout << errorLogger.getErrorMessage() << std::endl;
			break;
		}
	}

	std::cout << "done" << std::endl;
}