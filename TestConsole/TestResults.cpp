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
	std::string inFile = "d:/VideoTest/02short.mp4"; //std::string inFile = "f:/pic/input.mp4";
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
}

void testVideo2() {
	std::cout << "analyze image pair" << std::endl;
	FrameResult::storeDebugData = true;
	
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
	ImageYuv src = ImageBGR::readFromBMP("D:/VideoTest/beach.86.bmp").toYUV();
	std::cout << "rms " << src.lumaRms() << std::endl;
	executor.inputData(0, src);
	executor.createPyramid(0);

	ImageYuv im = ImageBGR::readFromBMP("D:/VideoTest/beach.87.bmp").toYUV();
	for (int i = 0; i < 5; i++) {
		src = im.copy();
		float gamma = 1.0f - 0.05f * i;
		std::string str = std::format("{:.3f}", gamma);
		src.gamma(gamma);
		std::cout << "gamma " << gamma << " lumaRms " << src.lumaRms() << std::endl;
		src.writeText(str, 860, 0, 2, 2, im::TextAlign::TOP_CENTER);
		src.saveAsColorBMP(std::format("f:/image{}a.bmp", i));

		executor.inputData(1, src);
		executor.createPyramid(1);
		executor.computeStart(1, frame.mResultPoints);
		executor.computeTerminate(1, frame.mResultPoints);
		AffineTransform trf = executor.computeTransform(frame.mResultPoints, 1);

		ImageRGBA dest(1080, 1920);
		ResultImageWriter::writeImage(trf, frame.mResultPoints, 1, dest, frame.mPool, false);
		dest.writeText(str, 860, 0, 2, 2, im::TextAlign::TOP_CENTER);
		dest.saveAsColorBMP(std::format("f:/image{}b.bmp", i));

		FrameResult::DebugData data = FrameResult::debugData;
		std::cout << "[" << data.clusterSizes.size() << "] " << util::collectionToString(data.clusterSizes, 15) << std::endl;
	}
}