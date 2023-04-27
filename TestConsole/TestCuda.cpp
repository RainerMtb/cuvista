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

void cudaInvTest(size_t s) {
	Matd a = Matd::rand(s, s, -20, 50);

	Matd ainv = Matd::allocate(s, s);
	bool isOK = cutest::cudaInv(a.data(), ainv.data(), s);
	Matd b = ainv.times(a);

	std::cout << "inv test, dim=" << s << "; ";
	if (!b.equalsIdentity()) {
		Matd cpuinv = a.inv().value();
		Matd cpub = cpuinv.times(a);
		if (!cpub.equalsIdentity()) {
			std::cout << "FAIL IDENTITIY TEST also on CPU" << std::endl;

		} else {
			std::cout << "FAIL IDENTITIY TEST" << std::endl;
			if (s > 10) printf("MAX absolute value %f\n", b.abs().max());
			else b.toConsole("ID");
			//a.saveAsCSV("c:/video/fail.csv");
		}

	} else {
		std::cout << "OK" << std::endl;
	}
}

void cudaInvSimple() {
	std::cout << "----------------------------" << std::endl << "Simple Cuda Inv Test:" << std::endl;
	int s = 3;
	Matd a = Matd::fromRows(s, s, { 2, 8, 1, 4, 4, -1, -1, 2, 12 });
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
		std::chrono::microseconds time(0);
		size_t smax = 32;
		bool loopok = true;
		for (int s = 1; s <= smax; s++) {
			Matd a = Matd::generate(s, s, [&] (size_t r, size_t c) { return sqrt(2.0 / s) * std::cos((r + 0.5) * (c + 0.5) * pi / s); });
			Matd ainv = Matd::allocate(s, s);

			auto t1 = std::chrono::high_resolution_clock::now();
			bool isok = cutest::cudaInv(a.data(), ainv.data(), s);
			auto t2 = std::chrono::high_resolution_clock::now();
			time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
			Matd check = ainv.times(a);
			loopok &= isok && check.equalsIdentity();
		}
		std::cout << "loop result: " << (loopok ? "ok" : "fail") << ", runtime " << time.count() / 1000.0 << " ms" << std::endl;
	}
}

void cudaInvEqualityTest() {
	std::cout << "----------------------------" << std::endl << "Cuda Inv Equal Test:" << std::endl;
	size_t matdim = 3;
	Matd a = Matd::fromRows(matdim, matdim, { 2, 8, 1, 4, 16, -1, -1, 2, 12 });
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
	std::cout << "AinvCpu == AinvGpu: " << std::boolalpha << invCpu.equals(invGpu, 0.0) << std::endl;
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

void cudaFMAD() {
	std::cout << "----------------------------" << std::endl << "FMAD Test:" << std::endl;
	int dim = 6;
	Matd src = Matd::fromRows(dim, dim, {
			0.5190191908, 0.554734511666, -0.279658085787, 1.33774798228, 0.631098670842, 0.136308250122,
			-0.198174314278, 0.560896368225, 0.512723252871, 0.968780516617, 0.178123438854, 1.36943674621,
			0.290164022862, -0.376174565165, 0.646963172062, -0.282636814666, -0.164510095229, -0.318205238958,
			0.296907252501, 0.433538232122, 1.27454239024, 0.827192179753, -0.359835208543, 0.870983602104,
			1.11645514311, 0.553288641361, 0.83733511859, 1.39369706849, -0.402843149795, -0.286130267234,
			0.548914236748, 1.04112317837, 0.744972004257, -0.0729558143778, -0.107012297344, -0.00469514097982
		}
	);
	//Matd::printPrecision(17);
	Matd a = src;
	LUDecompositor<double> lu(a);
	Matd ai = lu.inv().value();
	ai.toConsole("inv cpu");

	Matd ai2 = Matd::allocate(dim, dim);
	cutest::cudaInv(src.data(), ai2.data(), dim);
	ai2.toConsole("inv cuda");

	std::cout << "expect different result from cuda in release mode when -fmad=true" << std::endl;
	bool isEqual = ai.equals(ai2, 0.0);
	std::cout << "inv cpu EQUALS inv gpu: " << (isEqual ? "YES" : "NO") << std::endl;
	if (!isEqual) {
		Matd delta = ai.minus(ai2).toConsole("delta", 5);
	}
	//ai.times(a).minus(Matd::eye(dim)).toConsole("zero check");
}

