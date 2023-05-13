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

void test() {
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

int main() {
	std::cout << "----------------------------" << std::endl << "MatrixTestMain:" << std::endl;
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