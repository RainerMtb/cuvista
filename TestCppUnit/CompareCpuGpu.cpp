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

#include "pch.h"
#include "CppUnitTest.h"
#include "Utils.hpp"
#include "MovieFrame.hpp"

std::string file = "d:/VideoTest/04.ts";

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CudaTest {

	TEST_CLASS(CompareCpuGpu) {

private:

	inline static MainData dataGpu, dataCpu;
	inline static std::unique_ptr<MovieFrame> gpu, cpu;

	double sqr(double d) { return d * d; }

	static void runInit(MainData& data, std::unique_ptr<MovieFrame>& frame, AffineTransform& trf, MovieReader* reader, MovieWriter* writer) {
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

		frame->computeStart();
		frame->computeTerminate();
		frame->outputData(trf, writer->getOutputData());
	}
	
public:
	TEST_CLASS_INITIALIZE(init) {
		AffineTransform trf;
		trf.addRotation(0.2).addTranslation(-40, 30);

		{
			//GPU
			dataGpu.probeCudaDevices();
			dataGpu.fileIn = file;
			FFmpegReader reader;
			InputContext ctx = reader.open(file);
			dataGpu.validate(ctx);
			NullWriter writer(dataGpu);
			gpu = std::make_unique<GpuFrame>(dataGpu);
			runInit(dataGpu, gpu, trf, &reader, &writer);
		}
		{
			//CPU
			dataCpu.deviceRequested = true;
			dataCpu.deviceNum = -1;
			dataCpu.fileIn = file;
			FFmpegReader reader;
			InputContext ctx = reader.open(file);
			dataCpu.validate(ctx);
			NullWriter writer(dataCpu);
			cpu = std::make_unique<CpuFrame>(dataCpu);
			runInit(dataCpu, cpu, trf, &reader, &writer);
		}
	}

	TEST_CLASS_CLEANUP(shutdown) {
		gpu.reset();
		cpu.reset();
	}

	TEST_METHOD(status) {
		Assert::IsTrue(errorLogger.hasNoError());
	}


	TEST_METHOD(pyramid) {
		//gpu->getPyramid(0).saveAsBinary("f:/pyr_g0.dat");
		//cpu->getPyramid(0).saveAsBinary("f:/pyr_c0.dat");
		Mat cpu0 = cpu->getPyramid(0);
		Assert::AreNotEqual(0.0f, cpu0.sum());
		Assert::IsTrue(cpu0.equals(gpu->getPyramid(0), 0), L"pyramid 0 mismatch");

		Mat cpu1 = cpu->getPyramid(1);
		Assert::AreNotEqual(0.0f, cpu1.sum());
		Assert::IsTrue(cpu1.equals(gpu->getPyramid(1), 0), L"pyramid 1 mismatch");
	}

	TEST_METHOD(equalPointResultSize) {
		Assert::AreEqual(cpu->resultPoints.size(), gpu->resultPoints.size());
	}

	TEST_METHOD(equalPointResults) {
		std::vector pc = cpu->resultPoints;
		std::vector pg = gpu->resultPoints;
		for (int i = 0; i < pc.size(); i++) {
			//only check when both results are numerically stable
			Assert::AreEqual(pc[i], pg[i], L"results not equal");
		}
	}

	TEST_METHOD(transform) {
		Mat cpuMat = cpu->getTransformedOutput();
		Mat gpuMat = gpu->getTransformedOutput();
		//cpuMat.saveAsBinary("f:/matCpu.dat");
		//gpuMat.saveAsBinary("f:/matGpu.dat");

		float deltaMax = cpuMat.minus(gpuMat).abs().max();
		std::wstring msg = L"delta=" + std::to_wstring(deltaMax);
		Assert::AreEqual(0.0f, deltaMax, msg.c_str());

		Assert::IsTrue(cpuMat.equalsExact(gpuMat));
	}

	TEST_METHOD(transformConsistency) {
		ThreadPool pool(4);
		size_t siz = 20;
		std::vector<AffineTransform> trfs(siz);

		//compute transformation matrix multiple times
		for (int i = 0; i < siz; i++) {
			FrameResult frameResult(dataCpu);
			std::unique_ptr<RNGbase> rng = std::make_unique<RNG<RandomSource>>();
			frameResult.computeTransform(cpu->resultPointsOld, dataCpu, pool, rng.get());
			trfs[i] = frameResult.mTransform;
		}

		//check
		const AffineTransform& first = trfs[0];
		for (int i = 1; i < siz; i++) {
			Assert::AreEqual(first, trfs[i]);
		}
	}

	TEST_METHOD(transformRandomConsistency) {
		ThreadPool pool(4);
		size_t siz = 20;
		std::vector<AffineTransform> trfs(siz);

		//compute transformation matrix multiple times
		std::unique_ptr<RNGbase> rng = std::make_unique<RNG<std::default_random_engine>>();
		for (int i = 0; i < siz; i++) {
			FrameResult frameResult(dataCpu);
			frameResult.computeTransform(cpu->resultPointsOld, dataCpu, pool, rng.get());
			trfs[i] = frameResult.mTransform;
		}

		Matd avg = Matd::zeros(1, 4);
		Matd dev = Matd::zeros(1, 4);

		//average
		for (int i = 0; i < siz; i++) {
			Affine2D& a = trfs[i];
			avg += Matd::fromRow({ a.scale(), a.rot(), a.dX(), a.dY() });
		}
		avg /= (double) siz;

		//deviation sum of squares
		for (int i = 0; i < siz; i++) {
			Affine2D& a = trfs[i];
			Matd m = Matd::fromRow({ a.scale(), a.rot(), a.dX(), a.dY() });
			Matd d = (avg - m);
			dev += d.timesEach(d);
		}
		dev /= (double) siz;

		//check
		for (int k = 0; k < dev.numel(); k++) {
			Assert::IsTrue(dev[0][k] < 0.2);
		}
	}

	};
}