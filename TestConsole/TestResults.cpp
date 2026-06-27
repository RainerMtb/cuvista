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

void testVideo1() {
	std::cout << "test video" << std::endl;
	//std::string inFile = "d:/VideoTest/02short.mp4";
	std::string inFile = "f:/pic/input.mp4";
	std::string resultsFile = "f:/pic/results.dat";
	std::string videoInput = "f:/pic/videoInput.yuv";

	MainData data;
	data.fileIn = inFile;
	data.deviceRequested = true;
	data.mode = 1;
	std::vector<DeviceInfoCuda> cudaDevices = data.probeCuda();
	data.collectDeviceInfo();
	ff::loadFFmpegLibrary();
	auto reader = ff::createReader(ReaderType::FFMPEG);
	reader->open(inFile);
	data.validate(*reader);

	int maxFrames = 200;
	RawMemoryStoreWriter writer(maxFrames);
	std::shared_ptr<MovieFrame> frame = std::make_shared<MovieFrameConsecutive>(data, *reader, writer);
	std::shared_ptr<FrameExecutor> executor = std::make_shared<CudaFrame>(data, cudaDevices[0], *frame, frame->mPool);
	executor->init();
	frame->runLoop(executor);

	//write results binary
	std::cout << "write results " << resultsFile << std::endl;
	std::ofstream resfile(resultsFile, std::ios::binary);
	auto writefcn = [&] (auto value) { resfile.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
	writefcn(data.w);
	writefcn(data.h);
	writefcn(data.ixCount);
	writefcn(data.iyCount);

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

	//write video
	std::cout << "write video " << videoInput << std::endl;
	writer.writeInputFile(videoInput, maxFrames, executor->mPool);
	std::cout << "done" << std::endl;
	ff::freeFFmpegLibrary();
}

void testLuma2() {
	std::cout << "analyze image pair" << std::endl;
	MainData data;
	data.deviceRequested = true;
	data.mode = 1;
	std::vector<DeviceInfoCuda> cudaDevices = data.probeCuda();
	data.collectDeviceInfo();
	NullReader reader;
	reader.w = 1920;
	reader.h = 1080;
	reader.frameCount = 2;
	data.validate(reader);
	NullWriter writer(data, reader);
	MovieFrameConsecutive frame(data, reader, writer);
	CudaFrame executor(data, cudaDevices[0], frame, frame.mPool);

	executor.init();
	ImageYuv src = ImageYuv::readBmpFile("D:/VideoTest/06b.30.bmp");
	Image8& input = executor.inputDestination(0);
	src.copyTo(input);
	executor.inputData(0);
	std::vector<int> histOld(256);
	executor.createPyramid(0, histOld, {}, false);

	ImageYuv concat(1080, 1920 * 2);
	src.copyTo(concat, 0, 1920, 255);

	ImageYuv im = ImageYuv::readBmpFile("D:/VideoTest/06b.29.bmp");
	for (int i = 0; i < 5; i++) {
		im.copyTo(src);
		float gamma = 1.0f + 0.05f * i;
		src.adjustGamma(gamma);

		std::string str = std::format("{:.3f}", gamma);
		src.writeText(str, 860, 0, im::TextAlign::TOP_CENTER);
		src.copyTo(concat, 0, 0, 255);
		concat.saveBmpColor(std::format("f:/image{}a.bmp", i));

		Image8& input = executor.inputDestination(1);
		src.copyTo(input);
		executor.inputData(1);
		std::vector<int> hist(256);
		executor.createPyramid(1, hist, {}, false);
		frame.checkPyramidGamma(1, hist, histOld, executor);
		executor.computeStart(1, frame.mResultPoints);
		executor.computeTerminate(1, frame.mResultPoints);
		executor.computeTransform(frame.mResultPoints, 1);

		ImageRGBA dest(1080, 1920);
		FrameResultData resultData = executor.mFrame.getResultData();
		ResultImageWriter::writeImage(resultData, frame.mResultPoints, 1, dest, frame.mPool, false);
		dest.writeText(str, 860, 20, im::TextAlign::TOP_CENTER);
		dest.saveBmpColor(std::format("f:/image{}b.bmp", i));

		auto sizes = resultData.getClusterSizes();
		std::cout << "#" << i << " gamma=" << gamma << " [" << sizes.size() << "] " << util::collectionToString(sizes, 10) << std::endl;
	}
}