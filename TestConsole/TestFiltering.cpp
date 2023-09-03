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
	Mat m = Mat<double>::generate(h, w, [&] (size_t r, size_t c) { return (r + 1.0) / h * (c + 1.0) / w; });
	Mat x = Mat<double>::allocate(h, w);
	std::vector<double> k = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };

	auto t1 = std::chrono::high_resolution_clock::now();
	m.filter1D_h(k.data(), k.size(), x, pool);
	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> sec = t2 - t1;
	std::cout << "elapsed " << sec * 1000 << std::endl;
}

void runInit(MainData& data, std::unique_ptr<MovieFrame>& frame, AffineTransform& trf, MovieReader* reader, MovieWriter* writer) {
	Stats& status = data.status;
	status.reset();
	reader->read(frame->bufferFrame, status);
	status.frameReadIndex++;
	frame->inputData(frame->bufferFrame);
	frame->createPyramid();
	status.frameInputIndex++;

	reader->read(frame->bufferFrame, status);
	frame->inputData(frame->bufferFrame);
	frame->createPyramid();

	frame->computePartOne();
	frame->computePartTwo();
	frame->computeTerminate();
	frame->outputData(trf, writer->getOutputData());
}

void filterCompare() {
	std::cout << "compare filtering on cpu and gpu" << std::endl;
	std::string file = "d:/VideoTest/04.ts";
	MainData dataGpu, dataCpu;
	std::unique_ptr<MovieFrame> gpu, cpu;
	AffineTransform trf;
	trf.addRotation(0.2).addTranslation(-40, 30);
	Matd::precision(16);

	{
		//GPU
		dataGpu.probeCuda();
		dataGpu.fileIn = file;
		FFmpegReader reader;
		InputContext ctx = reader.open(file);
		dataGpu.validate(ctx);
		NullWriter writer(dataGpu);
		gpu = std::make_unique<CudaFrame>(dataGpu);
		runInit(dataGpu, gpu, trf, &reader, &writer);
	}
	{
		//CPU
		dataCpu.deviceRequested = true;
		dataCpu.deviceSelected = 0;
		dataCpu.fileIn = file;
		FFmpegReader reader;
		InputContext ctx = reader.open(file);
		dataCpu.validate(ctx);
		NullWriter writer(dataCpu);
		cpu = std::make_unique<CpuFrame>(dataCpu);
		runInit(dataCpu, cpu, trf, &reader, &writer);
	}

	std::vector pc = cpu->resultPoints;
	std::vector pg = gpu->resultPoints;
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

	c.saveAsBinary("D:/VideoTest/out/cpu.dat");
	g.saveAsBinary("D:/VideoTest/out/gpu.dat");
}