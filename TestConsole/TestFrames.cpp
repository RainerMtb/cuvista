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
	errorLogger().clear();
	try {
		std::vector<unsigned char> bytes = util::base64_decode(movieTestData);
		MemoryFFmpegReader reader(bytes);
		reader.open("");
		data.collectDeviceInfo();
		data.validate(reader);
		OutputWriter writer(data, reader);

		MovieFrameCombined frame(data, reader, writer);
		std::unique_ptr<FrameExecutor> executor = std::make_unique<T>(data, *data.deviceList[deviceIndex], frame, frame.mPool);
		std::string name = executor->mDeviceInfo.getNameShort();
		executor->init();

		std::cout << "running " << name << std::endl;
		reader.read(frame.mBufferFrame);
		executor->inputData(reader.frameIndex, frame.mBufferFrame);
		executor->createPyramid(reader.frameIndex, {}, false);

		reader.read(frame.mBufferFrame);
		executor->inputData(reader.frameIndex, frame.mBufferFrame);
		executor->createPyramid(reader.frameIndex, {}, false);

		executor->computeStart(reader.frameIndex, frame.mResultPoints);
		executor->computeTerminate(reader.frameIndex, frame.mResultPoints);

		ImageRGBA im(data.h, data.w);
		executor->getInput(0, im);

		AffineTransform trf;
		trf.addRotation(0.2).addTranslation(-40, 30);
		trf.frameIndex = 0;
		executor->outputData(0, trf);
		Result result = {
			executor->getPyramid(0),
			executor->getTransformedOutput(),
			frame.mResultPoints,
			name,
			writer.getOutputFrame(),
			im,
			(errorLogger().hasError() ? errorLogger().getErrorMessage() : "")
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
		//data.cpuThreadsRequired = 1;
		results[1] = runPyramid<AvxFrame>(data, 1);
	}

	{
		//Cuda
		MainData data;
		data.deviceRequested = true;
		data.deviceInfoCuda = data.probeCuda();
		results[2] = runPyramid<CudaFrame>(data, 2);
	}

	{
		//OpenCL
		MainData data;
		data.deviceRequested = true;
		data.deviceInfoOpenCl = data.probeOpenCl();
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

	std::cout << errorLogger().getErrorMessage() << std::endl;
}

void testVideo1() {
	std::string inFile = "f:/pic/input.mp4";
	std::string outputFile = "f:/pic/videoOut.yuv";
	std::string inputFile = "f:/pic/videoInput.yuv";
	std::string resultsFile = "f:/pic/results.dat";

	MainData data;
	data.fileIn = inFile;
	data.deviceRequested = true;
	data.mode = 1;
	std::vector<DeviceInfoCuda> cudaDevices = data.probeCuda();
	data.collectDeviceInfo();
	FFmpegReader reader;
	reader.open(inFile);
	data.validate(reader);

	int maxFrames = 200;
	RawMemoryStoreWriter writer(maxFrames);
	std::shared_ptr<MovieFrame> frame = std::make_shared<MovieFrameConsecutive>(data, reader, writer);
	std::shared_ptr<FrameExecutor> executor = std::make_shared<CudaFrame>(data, cudaDevices[0], *frame, frame->mPool);
	executor->init();
	frame->runLoop(executor);

	//output raw yuv files
	writer.writeYuvFiles(outputFile, inputFile);

	//write results binary
	std::ofstream resfile(resultsFile, std::ios::binary);
	auto writefcn = [&] (auto value) { resfile.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
	writefcn(data.w);
	writefcn(data.h);

	int idx = 0;
	for (std::span<PointResult> spr : writer.results) {
		for (const PointResult& pr : spr) {
			writefcn(idx);
			writefcn(pr.idx);
			writefcn(pr.x);
			writefcn(pr.y);
			writefcn(pr.u);
			writefcn(pr.v);
			writefcn((char) pr.isValid());
			writefcn((char) pr.isConsens);
		}
		idx++;
	}

	std::cout << std::endl << "results " << resultsFile << std::endl;
	std::cout << "input frames " << writer.inputFrames.size() << " " << inputFile << std::endl;
	std::cout << "output frames " << writer.outputFramesYuv.size() << " " << outputFile << std::endl;
}

// read and transform distinct images
void createTransformImages() {
	SimpleYuvWriter yuvFile("f:/pic/video.yuv");

	for (int i = 0; i < 10; i++) {
		std::string inFile1 = std::format("f:/pic/{:04}.bmp", i);
		std::string inFile2 = std::format("f:/pic/{:04}.bmp", i + 1);
		std::string outFile = std::format("f:/pic/out{:02}.bmp", i + 1);
		std::cout << "writing " << outFile << std::endl;

		//std::cout << "reading images" << std::endl;
		ImageYuv im1 = ImageBGR::readFromBMP(inFile1).toYUV();
		ImageYuv im2 = ImageBGR::readFromBMP(inFile2).toYUV();

		MainData data;
		data.collectDeviceInfo();
		ImageReader reader;
		reader.h = im1.h;
		reader.w = im1.w;
		reader.frameCount = 2;
		data.validate(reader);

		OutputWriter writer(data, reader);
		MovieFrameConsecutive frame(data, reader, writer);
		CpuFrame cpuframe(data, data.deviceInfoCpu, frame, frame.mPool);

		reader.readImage(frame.mBufferFrame, im1);
		cpuframe.inputData(reader.frameIndex, frame.mBufferFrame);
		cpuframe.createPyramid(reader.frameIndex, {}, false);

		reader.readImage(frame.mBufferFrame, im2);
		cpuframe.inputData(reader.frameIndex, frame.mBufferFrame);
		cpuframe.createPyramid(reader.frameIndex, {}, false);

		//std::cout << "computing transform" << std::endl;
		cpuframe.computeStart(reader.frameIndex, frame.mResultPoints);
		cpuframe.computeTerminate(reader.frameIndex, frame.mResultPoints);
		frame.mFrameResult.computeTransform(frame.mResultPoints, i + 1LL);
		const AffineTransform& trf = frame.getTransform();

		//save pairwise output images
		im1.writeText("input " + std::to_string(i), 0, 1080);
		yuvFile.write(im1);
		cpuframe.outputData(0, trf);
		ImageYuv imOut = writer.getOutputFrame();
		imOut.writeText("output " + std::to_string(i + 1), 0, 1080);
		yuvFile.write(imOut);

		//std::cout << "writing result" << std::endl;
		ResultImageWriter riw(data);
		riw.write(trf, frame.mResultPoints, i, im2, outFile);

		std::cout << "done" << std::endl;
	}
}