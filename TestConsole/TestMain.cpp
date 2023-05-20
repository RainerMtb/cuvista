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
#include <chrono>
#include <algorithm>

void compare() {
	{
		MainData data;
		data.probeCudaDevices();
		InputContext ctx = { 1080, 1920, 2, 1 };
		data.validate(ctx);
		NullReader reader;
		NullWriter writer(data);
		GpuFrame frame(data);

		frame.inputFrame.readFromPGM("d:/VideoTest/v00.pgm");
		frame.inputData(frame.inputFrame);
		frame.createPyramid();
		data.status.frameInputIndex++;

		frame.inputFrame.readFromPGM("D:/VideoTest/v01.pgm");
		frame.inputData(frame.inputFrame);
		frame.createPyramid();
		frame.computeStart();
		frame.computeTerminate();
		//frame.getTransformedOutput().saveAsBinary("f:/test.dat");
		std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	}

	{
		MainData data;
		InputContext ctx = { 1080, 1920, 2, 1 };
		data.validate(ctx);
		NullReader reader;
		NullWriter writer(data);
		CpuFrame frame(data);

		frame.inputFrame.readFromPGM("d:/VideoTest/v00.pgm");
		frame.inputData(frame.inputFrame);
		frame.createPyramid();
		data.status.frameInputIndex++;

		frame.inputFrame.readFromPGM("D:/VideoTest/v01.pgm");
		frame.inputData(frame.inputFrame);
		frame.createPyramid();
		frame.computeStart();
		frame.computeTerminate();
	}
}

void check() {
	int frameSkip = 0;
	int frameOut = 20;

	MainData data;
	FFmpegReader reader;
	//InputContext ctx = reader.open("d:/videotest/07.mp4");
	InputContext ctx = reader.open("//READYNAS/Data/Documents/x.orig/bikini.1.1.avi");

	std::cout << "using file " << ctx.source << std::endl;
	data.validate(ctx);
	NullWriter writer(data);
	CpuFrame frame(data);
	OutputContext oc = { true, false, &writer.outputFrame, nullptr, 0 };
	ResultImage resim(data, {});

	for (int i = 0; i < frameSkip; i++) {
		reader.read(frame.inputFrame, data.status);
		if (i % 25 == 0)
			std::cout << "skipping " << i << std::endl;
	}

	frame.inputData(frame.inputFrame);
	frame.createPyramid();
	data.status.frameInputIndex++;

	for (int i = 0; i < frameOut; i++) {
		std::cout << "reading " << data.status.frameInputIndex << std::endl;
		reader.read(frame.inputFrame, data.status);
		frame.inputData(frame.inputFrame);
		frame.createPyramid();
		frame.computeStart();
		frame.computeTerminate();
		const AffineTransform& trf = frame.computeTransform(frame.resultPoints);
		std::string fname = std::format("c:/temp/im{:03d}.bmp", i);
		resim.write(frame.mFrameResult, i, frame.getInput(i), fname);
		data.status.frameInputIndex++;
	}

}

int main() {
	std::cout << "----------------------------" << std::endl << "MatrixTestMain:" << std::endl;
	check();
	//qrdec();
	//for (size_t i = 1; i <= 32; i++) cudaInvTest(i);
	//text();
	//filterCompare();
	//matPerf();
	//matTest();
	//subMat();
	//iteratorTest();
	//similarTransformPerformance();
	//cudaInvSimple();
	//cudaInvPerformanceTest();
	//cudaInvEqualityTest();
	//cudaFMAD();
	//cudaInvParallel();
	//readAndWriteOneFrame();
	//checkVersions();
	//transform();
}