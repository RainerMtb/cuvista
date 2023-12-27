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

void filter() {
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

void runInit(MainData& data, std::unique_ptr<MovieFrame>& frame, AffineTransform& trf) {
	frame->mReader.read(frame->mBufferFrame);
	frame->inputData();
	frame->createPyramid(frame->mReader.frameIndex);

	frame->mReader.read(frame->mBufferFrame);
	frame->inputData();
	frame->createPyramid(frame->mReader.frameIndex);

	frame->computeStart(frame->mReader.frameIndex);
	frame->computeTerminate(frame->mReader.frameIndex);
	frame->outputData(trf, frame->mWriter.getOutputContext());
}

void filterCompare() {
	std::cout << "compare filtering on cpu and gpu" << std::endl;
	std::string file = "d:/VideoTest/02.mp4";
	MainData dataGpu, dataCpu;
	std::unique_ptr<MovieFrame> gpu, cpu;
	AffineTransform trf;
	trf.addRotation(0.2).addTranslation(-40, 30);
	trf.frameIndex = 0;
	Matd::precision(16);

	{
		//GPU
		dataGpu.probeCuda();
		dataGpu.collectDeviceInfo();
		dataGpu.fileIn = file;
		FFmpegReader reader;
		reader.open(file);
		dataGpu.validate(reader);
		NullWriter writer(dataGpu, reader);
		gpu = std::make_unique<CudaFrame>(dataGpu, reader, writer);
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
		NullWriter writer(dataCpu, reader);
		cpu = std::make_unique<CpuFrame>(dataCpu, reader, writer);
		runInit(dataCpu, cpu, trf);
	}

	std::vector pc = cpu->mResultPoints;
	std::vector pg = gpu->mResultPoints;
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
}