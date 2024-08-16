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
#include "MovieFrame.hpp"
#include "CudaFrame.hpp"
#include "clMain.hpp"
#include "CpuFrame.hpp"
#include "AvxFrame.hpp"

#include "MainData.hpp"
#include "Utils.hpp"
#include "ProgressDisplayConsole.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CudaTest {

	TEST_CLASS(FrameTest) {

private:

	std::vector<PointResult> run(FrameExecutor& ex, MovieFrame& frame, MainData& data) {
		//read first frame
		frame.mReader.read(frame.mBufferFrame);
		ex.inputData(frame.mReader.frameIndex, frame.mBufferFrame);
		ex.createPyramid(frame.mReader.frameIndex);

		//read second frame
		frame.mReader.read(frame.mBufferFrame);
		ex.inputData(frame.mReader.frameIndex, frame.mBufferFrame);
		ex.createPyramid(frame.mReader.frameIndex);

		//compute
		ex.computeStart(frame.mReader.frameIndex, frame.mResultPoints);
		ex.computeTerminate(frame.mReader.frameIndex, frame.mResultPoints);
		Assert::IsTrue(errorLogger.hasNoError(), toWString(errorLogger.getErrorMessage()).c_str());
		return frame.mResultPoints;
	}

public:
	TEST_METHOD(compareFrames1) {
		UserInputDefault input;
		std::vector<ImageYuv> cpuImages;
		{
			//cpu test run
			MainData data;
			TestReader reader;
			reader.open("");
			data.collectDeviceInfo();
			data.validate(reader);
			TestWriter writer(data, reader);
			MovieFrame frame(data, reader, writer);
			auto ex = std::make_shared<CpuFrame>(data, *data.deviceList[0], frame, frame.mPool);
			ex->init();
			auto progress = std::make_shared<ProgressDisplayNone>(frame);
			AuxWriters writers;
			frame.runLoop(progress, input, writers, ex);
			cpuImages = writer.outputFrames;
			Assert::IsTrue(errorLogger.hasNoError());
		}

		std::vector<ImageYuv> gpuImages;
		{
			//gpu test run
			MainData data;
			data.probeCuda();
			TestReader reader;
			reader.open("");
			data.collectDeviceInfo();
			data.validate(reader);
			TestWriter writer(data, reader);
			MovieFrame frame(data, reader, writer);
			auto ex = std::make_shared<CudaFrame>(data, *data.deviceList[2], frame, frame.mPool);
			ex->init();
			auto progress = std::make_shared<ProgressDisplayNone>(frame);
			AuxWriters writers;
			frame.runLoop(progress, input, writers, ex);
			gpuImages = writer.outputFrames;
			Assert::IsTrue(errorLogger.hasNoError());
		}

		Assert::AreEqual(cpuImages.size(), gpuImages.size());

		for (size_t i = 0; i < cpuImages.size(); i++) {
			ImageYuv gpu = gpuImages[i];
			ImageYuv cpu = cpuImages[i];
			//cpu.saveAsBMP("f:/cpu.bmp");
			//gpu.saveAsBMP("f:/gpu.bmp");
			Assert::IsTrue(gpu == cpu, L"cpu and gpu frames not identical");
		}
	}

	TEST_METHOD(compareFrameResults) {
		std::vector<PointResult> resCpu, resGpu;
		std::vector<Mat<float>> pyramids;
		std::string file = "d:/VideoTest/02.mp4";;

		{
			MainData data;
			data.fileIn = file;
			FFmpegReader reader;
			reader.open(file);
			data.collectDeviceInfo();
			data.validate(reader);
			BaseWriter writer(data, reader);
			MovieFrame frame(data, reader, writer);
			CpuFrame ex(data, *data.deviceList[0], frame, frame.mPool);
			ex.init();

			resCpu = run(ex, frame, data);
			pyramids.push_back(ex.getPyramid(0));
			pyramids.push_back(ex.getPyramid(1));
		}

		{
			MainData data;
			data.probeCuda();
			FFmpegReader reader;
			reader.open(file);
			data.collectDeviceInfo();
			data.validate(reader);
			BaseWriter writer(data, reader);
			MovieFrame frame(data, reader, writer);
			CudaFrame ex(data, *data.deviceList[2], frame, frame.mPool);
			ex.init();

			resGpu = run(ex, frame, data);
			pyramids.push_back(ex.getPyramid(0));
			pyramids.push_back(ex.getPyramid(1));
		}

		//pyramids[0].saveAsBinary("f:/p0.dat");
		//pyramids[2].saveAsBinary("f:/p1.dat");
		Assert::IsTrue(pyramids[0].equals(pyramids[2], 0), L"pyramids mismatch"); //first frame each
		Assert::IsTrue(pyramids[1].equals(pyramids[3], 0), L"pyramids mismatch"); //second frame each

		Assert::IsTrue(resCpu.size() == resGpu.size());
		for (int i = 0; i < resCpu.size(); i++) {
			PointResult& pc = resCpu[i];
			PointResult& pg = resGpu[i];

			//only check when both results are numerically stable
			if (pc.result == pg.result && pc.result > PointResultType::RUNNING) {
				//message in case of failure
				std::wstring msg = L"idx=" + std::to_wstring(pc.ix0) + L"/" + std::to_wstring(pc.iy0) +
					L", ug=" + std::to_wstring(pg.u) + L", uc=" + std::to_wstring(pc.u) +
					L", vg=" + std::to_wstring(pg.v) + L", vc=" + std::to_wstring(pc.v) + L", gpu res=" + std::to_wstring(pg.resultValue());

				//compare
				Assert::IsTrue(pc == pg, msg.c_str());
				//Assert::IsTrue(std::isnan(pc.v) && std::isnan(pg.v) || std::abs(pcpc.v == pg.v, msg.c_str());
			}
		}
	}

private:
	struct Result {
		std::vector<PointResult> res;
		Matf pyr, out;
		ImageYuv im;
		std::string id;
	};

	template <class T> Result compareFrame2func(MainData& data, int deviceIndex) {
		Result res;
		AffineTransform trf(0, 0.95, 0.3, 2, 3);
		data.collectDeviceInfo();
		NullReader reader;
		reader.w = 1920;
		reader.h = 1080;
		data.validate(reader);
		TestWriter writer(data, reader);
		MovieFrame frame(data, reader, writer);
		std::unique_ptr<FrameExecutor> ex = std::make_unique<T>(data, *data.deviceList[deviceIndex], frame, frame.mPool);
		ex->init();

		frame.mBufferFrame.readFromPGM("d:/VideoTest/v00.pgm");
		frame.mBufferFrame.index = 0;
		reader.frameIndex = 0;
		ex->inputData(reader.frameIndex, frame.mBufferFrame);
		ex->createPyramid(reader.frameIndex);

		frame.mBufferFrame.readFromPGM("D:/VideoTest/v01.pgm");
		frame.mBufferFrame.index = 1;
		reader.frameIndex = 1;
		ex->inputData(reader.frameIndex, frame.mBufferFrame);
		ex->createPyramid(reader.frameIndex);
		ex->computeStart(reader.frameIndex, frame.mResultPoints);
		ex->computeTerminate(reader.frameIndex, frame.mResultPoints);
		ex->outputData(0, trf);
		writer.prepareOutput(*ex);
		writer.write(*ex);
		Assert::IsTrue(errorLogger.hasNoError(), toWString(errorLogger.getErrorMessage()).c_str());

		return { frame.mResultPoints, ex->getPyramid(0), ex->getTransformedOutput(), writer.outputFrames[0], ex->mDeviceInfo.getNameShort() };
	}

public:

	TEST_METHOD(compareFrames2) {
		//FILE* ptr;
		//freopen_s(&ptr, "f:/redir.txt", "w", stdout);
		MainData data[4];
		Result res[4];
		res[0] = compareFrame2func<CpuFrame>(data[0], 0);
		res[1] = compareFrame2func<AvxFrame>(data[1], 1);
		
		data[2].probeCuda();
		res[2] = compareFrame2func<CudaFrame>(data[2], 2);
		
		data[3].probeOpenCl();
		res[3] = compareFrame2func<OpenClFrame>(data[3], 2);

		//check pyramids against cpu
		for (int i = 1; i < 4; i++) {
			std::wstring err = toWString("pyramids not equal: " + res[i].id);
			Assert::IsTrue(res[0].pyr.equalsExact(res[i].pyr), err.c_str());
		}

		//check output mats
		for (int i = 1; i < 4; i++) {
			std::wstring err = toWString("output mats not equal: " + res[i].id);
			Assert::IsTrue(res[0].out.equalsExact(res[i].out), err.c_str());
		}

		//check output images
		//imCpu.saveAsBMP("f:/imcpu.bmp");
		//imGpu.saveAsBMP("f:/imgpu.bmp");
		for (int i = 1; i < 4; i++) {
			std::wstring err = toWString("images not equal: " + res[i].id);
			Assert::IsTrue(res[0].im == res[i].im, err.c_str());
		}

		//check results
		for (int i = 1; i < 4; i++) {
			std::wstring err = toWString("restults not equal: " + res[i].id);
			for (int k = 0; k < res[k].res.size(); k++) {
				const PointResult& pr1 = res[0].res[k];
				const PointResult& pr2 = res[i].res[k];
			}
		}
	}

	};
}