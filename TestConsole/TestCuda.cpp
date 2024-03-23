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

void imageOutput() {
	int frameOut = 10;

	MainData data;
	FFmpegReader reader;
	data.probeCuda();
	data.collectDeviceInfo();
	reader.open("d:/VideoTest/02.mp4");
	data.validate(reader);
	NullWriter writer(data, reader);
	AvxFrame frame(data, reader, writer);
	OutputContext oc = { false, false, nullptr, nullptr };
	ResultImageWriter resim(data);
	data.resultImageFile = "f:/im%03d.bmp";

	reader.read(frame.mBufferFrame);
	frame.inputData();
	frame.createPyramid(frame.mReader.frameIndex);

	for (int i = 0; i < frameOut; i++) {
		std::cout << "reading " << frame.mReader.frameIndex << std::endl;
		reader.read(frame.mBufferFrame);
		frame.inputData();
		frame.createPyramid(frame.mReader.frameIndex);
		frame.computeStart(frame.mReader.frameIndex);
		frame.computeTerminate(frame.mReader.frameIndex);
		frame.computeTransform(frame.mReader.frameIndex);
		resim.write(frame);
	}

}

void cudaInvTest(size_t s1, size_t s2) {
	for (size_t s = s1; s <= s2; s++) {
		Matd a = Matd::rand(s, s, -20, 50, 1000);
		Matd ainv = Matd::allocate(s, s);
		bool isOK = cutest::cudaInv(a.data(), ainv.data(), s);
		Matd b = ainv.times(a);

		std::cout << "Cuda inv test, dim=" << s << "; ";
		if (!b.equalsIdentity()) {
			Matd cpuinv = a.inv().value();
			Matd cpub = cpuinv.times(a);
			std::cout << "FAIL IDENTITIY TEST" << std::endl;
			if (s > 10) printf("MAX absolute value %f\n", b.abs().max());
			else b.toConsole("I");
			//a.saveAsCSV("c:/video/fail.csv");

		} else {
			std::cout << "OK" << std::endl;
		}
	}
}

void cudaInvSimple() {
	std::cout << "----------------------------" << std::endl << "Simple Cuda Inv Test:" << std::endl;
	int s = 3;
	Matd a = Matd::fromRowData(s, s, { 2, 8, 1, 4, 4, -1, -1, 2, 12 });
	//Matd a = Matd::hilb(s);
	Matd ainv = Matd::allocate(s, s);
	bool isOK = cutest::cudaInv(a.data(), ainv.data(), s);
	a.toConsole("A");
	ainv.toConsole("Ainv");
	ainv.times(a).toConsole("check Ainv * A = I");
}

void cudaInvPerformanceTest() {
	double pi = std::numbers::pi;
	std::cout << "----------------------------" << std::endl << "Cuda Inv Performance Test:" << std::endl;
	for (size_t i = 0; i < 20; i++) {
		std::chrono::nanoseconds time(0);
		size_t smax = 32;
		bool loopok = true;
		for (int s = 1; s <= smax; s++) {
			Matd a = Matd::generate(s, s, [&] (size_t r, size_t c) { return sqrt(2.0 / s) * std::cos((r + 0.5) * (c + 0.5) * pi / s); });
			Matd ainv = Matd::allocate(s, s);

			auto t1 = std::chrono::high_resolution_clock::now();
			bool isok = cutest::cudaInv(a.data(), ainv.data(), s);
			auto t2 = std::chrono::high_resolution_clock::now();
			time += t2 - t1;
			Matd check = ainv.times(a);
			loopok &= isok && check.equalsIdentity();
		}
		std::cout << "loop " << i << " result: " << (loopok ? "ok" : "fail") << ", runtime " << time / 1'000'000.0 << " ms" << std::endl;
	}
}

void cudaInvEqualityTest() {
	std::cout << "----------------------------" << std::endl << "Cuda Inv Equal Test:" << std::endl;
	size_t matdim = 3;
	Matd a = Matd::fromRowData(matdim, matdim, { 2, 8, 1, 4, 16, -1, -1, 2, 12 });
	//Matd a = Matd::fromRows(matdim, matdim, { 1, 0, 0, 4, 5, 6, -3, -2, -1 });
	//Matd a = Matd::hilb(matdim);
	a.toConsole("input");
	//inverse on cpu
	Matd invCpu = a.inv().value();
	//inverse on gpu
	Matd invGpu = Matd::allocate(matdim, matdim);
	cutest::cudaInv(a.data(), invGpu.data(), matdim);
	//compare
	Matd::precision(10);
	invCpu.toConsole("cpu");
	invGpu.toConsole("gpu");
	std::cout << "AinvCpu == AinvGpu: " << std::boolalpha << invCpu.equalsExact(invGpu) << std::endl;
}

void cudaInvParallel() {
	std::cout << "----------------------------" << std::endl << "Cuda Inv Parallel Test:" << std::endl;
	size_t s = 6;
	const int count = 16000;
	double* input = new double[s * s * count];
	double* output = new double[s * s * count];
	std::chrono::microseconds time(0);
	bool isok = true;
	double pi = std::numbers::pi;

	for (int i = 0; i < 4; i++) {
		for (int idx = 0; idx < count; idx++) {
			Matd m = Matd::fromArray(s, s, input + idx * s * s, false);
			m.setArea([&] (size_t r, size_t c) { return sqrt(2.0 / s) * std::cos((r + 0.5) * (c + 0.5) * pi / s); });
		}

		auto t1 = std::chrono::high_resolution_clock::now();
		cutest::invParallel_6(input, output, count);
		auto t2 = std::chrono::high_resolution_clock::now();
		auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
		std::cout << "loop " << i << ", cuda time " << us.count() / 1000.0 << " ms" << std::endl;
		time += us;

		for (int idx = 0; idx < count; idx++) {
			Matd invgpu = Matd::fromArray(s, s, output + idx * s * s, false);
			Matd m = Matd::fromArray(s, s, input + idx * s * s, false);
			Matd mm = m;
			Matd invcpu = LUDecompositor(mm).inv().value();
			isok &= (invcpu == invgpu);
		}
	}

	delete[] output, input;
	std::cout << "parallel result: " << (isok ? "ok" : "fail") << ", runtime " << time.count() / 1000.0 << " ms" << std::endl;
}

void compare() {
	{
		MainData data;
		data.probeCuda();
		data.collectDeviceInfo();
		NullReader reader;
		reader.h = 1080;
		reader.w = 1920;
		data.validate(reader);
		NullWriter writer(data, reader);
		CudaFrame frame(data, reader, writer);

		frame.mBufferFrame.readFromPGM("d:/VideoTest/v00.pgm");
		frame.inputData();
		frame.createPyramid(frame.mReader.frameIndex);

		frame.mBufferFrame.readFromPGM("D:/VideoTest/v01.pgm");
		frame.inputData();
		frame.createPyramid(frame.mReader.frameIndex);
		frame.computeStart(frame.mReader.frameIndex);
		frame.computeTerminate(frame.mReader.frameIndex);
		//frame.getTransformedOutput().saveAsBinary("f:/test.dat");
		std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	}

	{
		MainData data;
		NullReader reader;
		reader.h = 1080;
		reader.w = 1920;
		data.validate(reader);
		NullWriter writer(data, reader);
		CpuFrame frame(data, reader, writer);

		frame.mBufferFrame.readFromPGM("d:/VideoTest/v00.pgm");
		frame.inputData();
		frame.createPyramid(frame.mReader.frameIndex);

		frame.mBufferFrame.readFromPGM("D:/VideoTest/v01.pgm");
		frame.inputData();
		frame.createPyramid(frame.mReader.frameIndex);
		frame.computeStart(frame.mReader.frameIndex);
		frame.computeTerminate(frame.mReader.frameIndex);
	}
}