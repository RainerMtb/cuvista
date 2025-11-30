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

ThreadPool pool(4);

static void filter() {
	int w = 2000;
	int h = 1000;
	Matd m = Matd::generate(h, w, [&] (size_t r, size_t c) { return (r + 1.0) / h * (c + 1.0) / w; });
	Matd x = Matd::allocate(h, w);
	std::vector<double> k = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };

	auto t1 = std::chrono::high_resolution_clock::now();
	m.filter1D(k.data(), k.size(), Matd::Direction::HORIZONTAL, x, pool);
	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> sec = t2 - t1;
	std::cout << "elapsed " << sec * 1000 << std::endl;
}

static void runInit(MainData& data, std::unique_ptr<FrameExecutor>& ex, AffineTransform& trf) {
	MovieReader& reader = ex->mFrame.mReader;
	reader.read(ex->mFrame.mBufferFrame);
	ex->inputData(reader.frameIndex, ex->mFrame.mBufferFrame);
	ex->createPyramid(reader.frameIndex, {}, false);

	reader.read(ex->mFrame.mBufferFrame);
	ex->inputData(reader.frameIndex, ex->mFrame.mBufferFrame);
	ex->createPyramid(reader.frameIndex, {}, false);

	ex->computeStart(reader.frameIndex, ex->mFrame.mResultPoints);
	ex->computeTerminate(reader.frameIndex, ex->mFrame.mResultPoints);
	ex->outputData(0, trf);
	ex->mFrame.mWriter.writeOutput(*ex);
}

void filterCompare() {
	std::cout << "compare filtering on cpu and gpu" << std::endl;
	std::string file = "d:/VideoTest/02.mp4";
	MainData dataGpu, dataCpu;

	std::unique_ptr<FrameExecutor> gpu, cpu;
	AffineTransform trf;
	trf.addRotation(0.2).addTranslation(-40, 30);
	trf.frameIndex = 0;
	Matd::precision(16);

	{
		//GPU
		dataGpu.deviceInfoCuda = dataGpu.probeCuda();
		dataGpu.collectDeviceInfo();
		dataGpu.fileIn = file;
		FFmpegReader reader;
		reader.open(file);
		dataGpu.validate(reader);
		OutputWriter writer(dataGpu, reader);
		MovieFrame frame(dataGpu, reader, writer);
		gpu = std::make_unique<CudaFrame>(dataGpu, *dataGpu.deviceList[2], frame, frame.mPool);
		runInit(dataGpu, gpu, trf);
	}
	{
		//CPU
		dataCpu.collectDeviceInfo();
		dataCpu.deviceRequested = true;
		dataCpu.deviceSelected = 0;
		dataCpu.fileIn = file;
		FFmpegReader reader;
		reader.open(file);
		dataCpu.validate(reader);
		OutputWriter writer(dataCpu, reader);
		MovieFrame frame(dataGpu, reader, writer);
		cpu = std::make_unique<CpuFrame>(dataCpu, *dataCpu.deviceList[0], frame, frame.mPool);
		runInit(dataCpu, cpu, trf);
	}

	std::vector pc = cpu->mFrame.mResultPoints;
	std::vector pg = gpu->mFrame.mResultPoints;
	bool isEqual = true;
	for (int i = 0; i < pc.size(); i++) {
		if (pc[i] != pg[i]) {
			std::cout << "result ix0=" << pc[i].ix0 << " iy0=" << pc[i].iy0 << " mismatch" << std::endl;
			isEqual = false;
		}
	}
	Matf c = cpu->getTransformedOutput();
	Matf g = gpu->getTransformedOutput();
	bool matEqual = c.equalsExact(g);
	std::cout << "equal: " << std::boolalpha << (isEqual && matEqual) << std::endl;

	//c.saveAsBinary("D:/VideoTest/out/cpu.dat");
	//g.saveAsBinary("D:/VideoTest/out/gpu.dat");
	std::cout << errorLogger().getErrorMessage() << std::endl;
}