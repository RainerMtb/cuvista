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
#include "clTest.hpp"

void openClInvTest(size_t s1, size_t s2) {
	LoadResult res = cltest::loadKernels({ luinvFunction, luinvTestKernel });

	for (size_t s = s1; s <= s2; s++) {
		Matd a = Matd::rand(s, s, -20, 50);
		Matd ainv = Matd::allocate(s, s);
		bool isOK = cltest::cl_inv(res, a.data(), ainv.data(), s);
		Matd b = ainv.times(a);

		std::cout << "OpenCL inv test, dim=" << s << "; ";
		if (!b.equalsIdentity()) {
			Matd cpuinv = a.inv().value();
			Matd cpub = cpuinv.times(a);
			if (!cpub.equalsIdentity()) {
				std::cout << "FAIL IDENTITIY TEST also on CPU" << std::endl;

			} else {
				std::cout << "FAIL IDENTITIY TEST" << std::endl;
				if (s > 10) printf("MAX absolute value %f\n", b.abs().max());
				else b.toConsole("I");
				//a.saveAsCSV("c:/video/fail.csv");
			}

		} else {
			std::cout << "OK" << std::endl;
		}
	}
}

void pyramid() {
	MainData data;
	AffineTransform trf;
	trf.addRotation(0.2).addTranslation(-40, 30);

	data.deviceRequested = true;
	data.deviceRequested = 1;
	data.probeOpenCl();
	data.fileIn = "d:/VideoTest/04.ts";
	FFmpegReader r;
	InputContext ctx = r.open(data.fileIn);
	data.validate(ctx);
	NullWriter w(data);
	OpenClFrame f(data);
	
	MovieFrame* frame = &f;
	MovieReader* reader = &r;
	MovieWriter* writer = &w;
	Stats& status = data.status;
	status.reset();
	reader->read(frame->bufferFrame, status);
	status.frameReadIndex++;
	frame->inputData(frame->bufferFrame);
	frame->createPyramid();
	status.frameInputIndex++;

	reader->read(frame->bufferFrame, status);
	std::cout << "input" << std::endl;
	frame->inputData(frame->bufferFrame);
	std::cout << "pyramid" << std::endl;
	frame->createPyramid();

	frame->computePartOne();
	frame->computePartTwo();
	frame->computeTerminate();
	frame->outputData(trf, writer->getOutputData());

	frame->getPyramid(0).saveAsBinary("f:/pyr_g0.dat");

	if (errorLogger.hasError()) {
		std::cout << errorLogger.getErrorMessage() << std::endl;
	}
}