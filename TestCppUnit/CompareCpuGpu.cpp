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
#include "CpuFrame.hpp"
#include "OpenClFrame.hpp"
#include "CudaFrame.hpp"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CudaTest {

	TEST_CLASS(CompareCpuGpu) {

private:

	inline static MainData dataCuda, dataCpu;
	inline static std::unique_ptr<MovieFrame> frameCuda, frameCpu;

	double sqr(double d) { return d * d; }

	template <class T> static std::unique_ptr<MovieFrame> runInit(MainData& data) {
		AffineTransform trf;
		trf.addRotation(0.2).addTranslation(-40, 30);
		trf.frameIndex = 0;
		FFmpegReader reader;
		reader.open("d:/VideoTest/02.mp4");
		data.validate(reader);
		TestWriter writer(data, reader);
		std::unique_ptr<MovieFrame> frame = std::make_unique<T>(data, reader, writer);

		reader.read(frame->mBufferFrame);
		frame->inputData();
		frame->createPyramid(reader.frameIndex);

		reader.read(frame->mBufferFrame);
		frame->inputData();
		frame->createPyramid(reader.frameIndex);
		frame->computeStart(reader.frameIndex);
		frame->computeTerminate(reader.frameIndex);
		frame->outputData(trf, writer.getOutputContext());
		return frame;
	}
	
public:
	TEST_CLASS_INITIALIZE(init) {
		AffineTransform trf;
		trf.addRotation(0.2).addTranslation(-40, 30);

		//Cuda
		dataCuda.probeCuda();
		dataCuda.probeOpenCl();
		dataCuda.collectDeviceInfo();
		dataCuda.deviceSelected = 2;
		frameCuda = runInit<CudaFrame>(dataCuda);

		//CPU
		dataCpu.deviceSelected = 0;
		dataCpu.collectDeviceInfo();
		frameCpu = runInit<CpuFrame>(dataCpu);
	}

	TEST_CLASS_CLEANUP(shutdown) {
		frameCuda.reset();
		frameCpu.reset();
	}

	TEST_METHOD(status) {
		Assert::IsTrue(errorLogger.hasNoError(), toWString(errorLogger.getErrorMessage()).c_str());
	}

	TEST_METHOD(pyramid) {
		//frameCuda->getPyramid(0).saveAsBinary("f:/pyr_g0.dat");
		//frameCpu->getPyramid(0).saveAsBinary("f:/pyr_c0.dat");
		Mat cpu0 = frameCpu->getPyramid(0);
		Assert::AreNotEqual(0.0f, cpu0.sum());
		Assert::IsTrue(cpu0.equals(frameCuda->getPyramid(0), 0), L"pyramid 0 mismatch");

		Mat cpu1 = frameCpu->getPyramid(1);
		Assert::AreNotEqual(0.0f, cpu1.sum());
		Assert::IsTrue(cpu1.equals(frameCuda->getPyramid(1), 0), L"pyramid 1 mismatch");
	}

	TEST_METHOD(equalPointResultSize) {
		Assert::AreEqual(frameCpu->mResultPoints.size(), frameCuda->mResultPoints.size());
	}

	TEST_METHOD(equalPointResults) {
		std::vector pc = frameCpu->mResultPoints;
		std::vector pg = frameCuda->mResultPoints;
		for (int i = 0; i < pc.size(); i++) {
			//only check when both results are numerically stable
			Assert::AreEqual(pc[i], pg[i], L"results not equal");
		}
	}

	TEST_METHOD(transform) {
		Mat cpuMat = frameCpu->getTransformedOutput();
		Mat gpuMat = frameCuda->getTransformedOutput();
		Assert::AreEqual(cpuMat.rows(), gpuMat.rows(), L"row dimension does not agree");
		Assert::AreEqual(cpuMat.cols(), gpuMat.cols(), L"col dimension does not agree");
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
			FrameResult frameResult(dataCpu, pool);
			frameResult.computeTransform(frameCpu->mResultPoints, pool, -1);
			trfs[i] = frameResult.getTransform();
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
			FrameResult frameResult(dataCpu, pool);
			frameResult.computeTransform(frameCpu->mResultPoints, pool, -1);
			trfs[i] = frameResult.getTransform();
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