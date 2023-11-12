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
	LoadResult res = cltest::loadKernels({ luinvFunction, testKernels }, "luinvTest");
	if (res.status != CL_SUCCESS) return;

	for (size_t s = s1; s <= s2; s++) {
		Matd a = Matd::rand(s, s, -20, 50, 1000);
		Matd ainvCPU = a.inv().value();
		Matd ainvOCL = Matd::allocate(s, s);
		bool isOK = cltest::cl_inv(res, a.data(), ainvOCL.data(), s);

		std::cout << "OpenCL inv test, dim=" << s << "; ";
		if (!ainvCPU.equalsExact(ainvOCL)) {
			Matd delta = ainvCPU.minus(ainvOCL);
			std::cout << "FAIL " << std::endl;
			if (s > 6) printf("MAX absolute delta %g\n", delta.abs().max());
			else delta.toConsole();

		} else {
			std::cout << "OK" << std::endl;
		}
	}
}

void openClInvGroupTest(int w1, int w2) {
	LoadResult res = cltest::loadKernels({ luinvFunction, testKernels }, "luinvGroupTest");
	if (res.status != CL_SUCCESS) return;

	size_t s = 6;
	for (int groupWidth = w1; groupWidth <= w2; groupWidth++) {
		Matd a = Matd::rand(s * groupWidth, s, -10, 20, 1000);
		Matd ainv = Matd::allocate(s * groupWidth, s);
		for (int i = 0; i < groupWidth; i++) {
			ainv.setArea(i * s, 0, a.subMat(i * s, 0, s, s).inv().value());
		}

		Matd clmat = Matd::allocate(s * groupWidth, s);
		bool isOK = cltest::cl_inv_group(res, a.data(), clmat.data(), groupWidth, s);
		double deltaMax = ainv.minus(clmat).abs().max();
		std::cout << "group test " << groupWidth << " ";
		if (isOK && deltaMax == 0.0) {
			std::cout << "exactly equal" << std::endl;
		} else if (isOK) {
			std::cout << "max delta " << deltaMax << std::endl;
		} else {
			std::cout << "FAIL" << std::endl;
		}

		//std::cout << "cpu" << std::endl << ainv << std::endl;
		//std::cout << "ocl" << std::endl << clmat << std::endl;
		//std::cout << "delta" << std::endl << (ainv - clmat) << std::endl;
	}
}

void openClnorm1Test() {
	LoadResult res = cltest::loadKernels({ norm1Function, testKernels }, "norm1Test");
	if (res.status != CL_SUCCESS) return;

	int s = 6;
	for (int i = 0; i < 10; i++) {
		Matd a = Matd::rand(s, s, -10, 20);
		double norm1cpu = a.norm1();
		double norm1ocl = cltest::cl_norm1(res, a.data(), s);
		if (norm1cpu == norm1ocl) {
			std::cout << "norm1 test OK" << std::endl;
		} else {
			std::cout << "norm1 test FAIL" << std::endl;
		}
	}
}

template <class T> std::pair<Matf, Matf> runPyramid(MainData& data) {
	FFmpegReader reader;
	data.inputCtx = reader.open(data.fileIn);
	data.collectDeviceInfo();
	data.validate();
	NullWriter writer(data);
	std::unique_ptr<MovieFrame> frame = std::make_unique<T>(data);
	Stats& status = data.status;
	status.reset();
	reader.read(frame->bufferFrame, status);
	status.frameReadIndex++;
	frame->inputData(frame->bufferFrame);
	frame->createPyramid();
	status.frameInputIndex++;

	reader.read(frame->bufferFrame, status);
	frame->inputData(frame->bufferFrame);
	frame->createPyramid();

	frame->computePartOne();
	frame->computePartTwo();
	frame->computeTerminate();

	AffineTransform trf;
	trf.addRotation(0.2).addTranslation(-40, 30);
	frame->outputData(trf, writer.getOutputData());
	return { frame->getPyramid(0), frame->getTransformedOutput()};
}

void pyramid() {
	std::cout << "compare pyramids" << std::endl;

	AffineTransform trf;
	trf.addRotation(0.2).addTranslation(-40, 30);
	Matf pyrCpu, pyrOcl, pyrCuda;
	Matf outCpu, outOcl, outCuda;

	{
		//CPU
		MainData data;
		data.deviceRequested = true;
		data.deviceRequested = 0;
		data.fileIn = "d:/VideoTest/04.ts";
		auto ret = runPyramid<CpuFrame>(data);
		pyrCpu = ret.first;
		outCpu = ret.second;
		//ret.second.saveAsColorBMP("f:/testCpu.bmp");
	}

	{
		//OpenCL
		MainData data;
		data.deviceRequested = true;
		data.deviceRequested = 1;
		data.probeOpenCl();
		data.fileIn = "d:/VideoTest/04.ts";
		auto ret = runPyramid<OpenClFrame>(data);
		pyrOcl = ret.first;
		outOcl = ret.second;
		//ret.second.saveAsColorBMP("f:/testOcl.bmp");
	}

	{
		//Cuda
		MainData data;
		data.deviceRequested = true;
		data.deviceRequested = 1;
		data.probeCuda();
		data.fileIn = "d:/VideoTest/04.ts";
		auto ret = runPyramid<CudaFrame>(data);
		pyrCuda = ret.first;
		outCuda = ret.second;
		//ret.second.saveAsColorBMP("f:/testCuda.bmp");
	}

	//outCpu.saveAsBinary("f:/outCpu.dat");
	//outOcl.saveAsBinary("f:/outOcl.dat");
	std::cout << "comparing CPU and CUDA: ";
	std::cout << (pyrCpu.equalsExact(pyrCuda) ? "pyramids equal" : "pyramids DIFFER");
	std::cout << ", ";
	std::cout << (outCpu.equalsExact(outCuda) ? "warped output equal" : "warped output DIFFER") << std::endl;

	std::cout << "comparing CPU and OPENCL: ";
	std::cout << (pyrCpu.equalsExact(pyrOcl) ? "pyramids equal" : "pyramids DIFFER");
	std::cout << ", ";
	std::cout << (outCpu.equalsExact(outOcl) ? "warped output equal" : "warped output DIFFER") << std::endl;

	if (errorLogger.hasError()) {
		std::cout << errorLogger.getErrorMessage() << std::endl;
	}
}