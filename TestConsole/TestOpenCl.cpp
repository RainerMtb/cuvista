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
	LoadResult res = cltest::loadKernels({ norm1Function, luinvFunction, testKernels }, "luinvTest");
	if (res.status != CL_SUCCESS) return;

	for (size_t s = s1; s <= s2; s++) {
		Matd a = Matd::rand(s, s, -20, 50, 1000);
		Matd ainvCPU = a.inv().value();
		Matd ainvOCL = Matd::allocate(s, s);
		bool isOK = cltest::cl_inv(res, a.data(), ainvOCL.data(), s);

		std::cout << "OpenCL inv test, dim=" << s << " ";
		Matd delta = ainvCPU.minus(ainvOCL);
		double deltaMax = delta.abs().max();
		if (deltaMax == 0.0) {
			std::cout << "EXACT" << std::endl;

		} else if (deltaMax < 1e-12) {
			std::cout << "delta " << deltaMax << std::endl;

		} else {
			delta.toConsole();
		}
	}
}

void openClInvGroupTest(int w1, int w2) {
	LoadResult res = cltest::loadKernels({ norm1Function, luinvFunction, testKernels }, "luinvGroupTest");
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
			std::cout << "EXACT" << std::endl;
		} else if (isOK) {
			std::cout << "max delta " << deltaMax << std::endl;
		} else {
			std::cout << "FAIL" << std::endl;
		}
	}
}

void openClnorm1Test() {
	LoadResult res = cltest::loadKernels({ norm1Function, luinvFunction, testKernels }, "norm1Test");
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

struct Result {
	Matf pyramid;
	Matf output;
	std::vector<PointResult> results;
};

template <class T> Result runPyramid(MainData& data) {
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
	return { frame->getPyramid(0), frame->getTransformedOutput(), frame->resultPoints };
}

void pyramid() {
	std::cout << "comparing platforms..." << std::endl;

	AffineTransform trf;
	trf.addRotation(0.2).addTranslation(-40, 30);
	Result cpu, ocl, cuda;

	{
		//CPU
		MainData data;
		data.deviceRequested = true;
		data.deviceSelected = 0;
		data.fileIn = "d:/VideoTest/04.ts";
		cpu = runPyramid<CpuFrame>(data);
		//ret.second.saveAsColorBMP("f:/testCpu.bmp");
	}

	{
		//Cuda
		MainData data;
		data.deviceRequested = true;
		data.deviceSelected = 1;
		data.probeCuda();
		data.fileIn = "d:/VideoTest/04.ts";
		cuda = runPyramid<CudaFrame>(data);
		//ret.second.saveAsColorBMP("f:/testCuda.bmp");
	}

	{
		//OpenCL
		MainData data;
		data.deviceRequested = true;
		data.deviceSelected = 1;
		data.probeOpenCl();
		data.fileIn = "d:/VideoTest/04.ts";
		ocl = runPyramid<OpenClFrame>(data);
		//ret.second.saveAsColorBMP("f:/testOcl.bmp");
	}

	//outCpu.saveAsBinary("f:/outCpu.dat");
	//outCuda.saveAsBinary("f:/outCuda.dat");
	std::cout << "comparing CPU and CUDA: ";
	std::cout << (cpu.pyramid.equalsExact(cuda.pyramid) ? "pyramids equal" : "pyramids DIFFER");
	std::cout << ", ";
	std::cout << (cpu.output.equalsExact(cuda.output) ? "warped output equal" : "warped output DIFFER");
	std::cout << ", ";
	std::cout << (cpu.results == cuda.results ? "results equal" : "results DIFFER");
	std::cout << std::endl;

	std::cout << "comparing CPU and OPEN CL: ";
	std::cout << (cpu.pyramid.equalsExact(ocl.pyramid) ? "pyramids equal" : "pyramids DIFFER");
	std::cout << ", ";
	std::cout << (cpu.output.equalsExact(ocl.output) ? "warped output equal" : "warped output DIFFER");
	std::cout << ", ";
	std::cout << (cpu.results == ocl.results ? "results equal" : "results DIFFER");
	std::cout << std::endl;

	if (errorLogger.hasError()) {
		std::cout << errorLogger.getErrorMessage() << std::endl;
	}
}