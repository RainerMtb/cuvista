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

void cudaTextureRead() {
	ImageYuv im1, im2;
	ImageYuv input(1080, 1920);
	for (int r = 0; r < input.h; r++) {
		for (int c = 0; c < input.w; c++) {
			input.at(0, r, c) = (unsigned char) (r);
		}
	}

	{
		MainData data;
		NullReader reader;
		reader.w = 1920;
		reader.h = 1080;
		reader.frameCount = 1;
		data.collectDeviceInfo();
		data.validate(reader);

		OutputWriter writer(data, reader);
		MovieFrameConsecutive frame(data, reader, writer);
		CpuFrame frameExecutor(data, data.deviceInfoCpu, frame, frame.mPool);

		frameExecutor.inputData(0, input);

		AffineTransform trf;
		trf.setParam(0.952379970131, 0.001367827033, 33.316623121580, 26.105044749792);
		frameExecutor.outputData(0, trf);
		im1 = writer.getOutputFrame();
	}

	{
		MainData data;
		NullReader reader;
		reader.w = 1920;
		reader.h = 1080;
		reader.frameCount = 1;
		data.probeCuda();
		data.collectDeviceInfo();
		data.validate(reader);

		OutputWriter writer(data, reader);
		MovieFrameConsecutive frame(data, reader, writer);
		CudaFrame frameExecutor(data, data.cudaInfo.devices[0], frame, frame.mPool);

		frameExecutor.inputData(0, input);

		AffineTransform trf;
		trf.setParam(0.952379970131, 0.001367827033, 33.316623121580, 26.105044749792);
		frameExecutor.outputData(0, trf);
		im2 = writer.getOutputFrame();
	}

	for (int z = 0; z < 3; z++) {
		for (int r = 0; r < 1080; r++) {
			for (int c = 0; c < 1920; c++) {
				if (im1.at(z, r, c) != im2.at(z, r, c)) {
					std::cout << z << "/" << r << "/" << c << " " << (int)im1.at(z, r, c) << " : " << (int)im2.at(z, r, c) << std::endl;
				}
			}
		}
	}
	std::cout << std::boolalpha << "images equal: " << (im1 == im2) << std::endl;

}
