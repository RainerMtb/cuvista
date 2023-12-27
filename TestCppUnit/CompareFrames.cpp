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
#include "OpenClFrame.hpp"
#include "CpuFrame.hpp"
#include "MainData.hpp"
#include "Utils.hpp"
#include "ProgressDisplayConsole.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CudaTest {

	//make up frame data in memory for testing
	class TestReader : public MovieReader {

	private:
		int64_t testFrameCount = 20;

	public:
		void open(std::string_view source) override {
			frameCount = testFrameCount;
			h = 100;
			w = 200;
		}

		void read(ImageYuv& frame) override {
			frameIndex++;
			for (int64_t z = 0; z < 3; z++) {
				int64_t base = this->frameIndex * 2 + z * 5 + 30;
				unsigned char* plane = frame.plane(z);
				for (int64_t r = 0; r < frame.h; r++) {
					for (int64_t c = 0; c < frame.w; c++) {
						int64_t pix = std::clamp(base + r / 10, 0LL, 255LL);
						plane[r * frame.stride + c] = (unsigned char) (pix);
					}
				}
			}
			frame.index = this->frameIndex;
			endOfInput = this->frameIndex == testFrameCount;
		}
	};

	//store resulting images in vector
	class TestWriter : public MovieWriter {

	public:
		std::vector<ImageYuv> outputFrames;

		TestWriter(MainData& data, MovieReader& reader) : 
			MovieWriter(data, reader) {}

		void write(const MovieFrame& frame) override {
			this->frameIndex++;
			outputFrames.push_back(outputFrame);
		}
	};


	TEST_CLASS(FrameTest) {

private:

	std::vector<PointResult> run(MovieFrame& frame, MainData& data) {
		//read first frame
		frame.mReader.read(frame.mBufferFrame);
		frame.inputData();
		frame.createPyramid(frame.mReader.frameIndex);

		//read second frame
		frame.mReader.read(frame.mBufferFrame);
		frame.inputData();
		frame.createPyramid(frame.mReader.frameIndex);

		//compute
		frame.computeStart(frame.mReader.frameIndex);
		frame.computeTerminate(frame.mReader.frameIndex);
		Assert::IsTrue(errorLogger.hasNoError());
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
			CpuFrame frame(data, reader, writer);
			ProgressDisplayNone progress(frame);
			AuxWriters writers;
			frame.runLoop(DeshakerPass::COMBINED, progress, input, writers);
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
			CudaFrame frame(data, reader, writer);
			ProgressDisplayNone progress(frame);
			AuxWriters writers;
			frame.runLoop(DeshakerPass::COMBINED, progress, input, writers);
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
		std::string file = "d:/VideoTest/04.ts";

		{
			MainData data;
			data.fileIn = file;
			FFmpegReader reader;
			reader.open(file);
			data.collectDeviceInfo();
			data.validate(reader);
			NullWriter writer(data, reader);
			CpuFrame frame(data, reader, writer);

			resCpu = run(frame, data);
			pyramids.push_back(frame.getPyramid(0));
			pyramids.push_back(frame.getPyramid(1));
		}

		{
			MainData data;
			data.probeCuda();
			data.fileIn = file;
			FFmpegReader reader;
			reader.open(file);
			data.collectDeviceInfo();
			data.validate(reader);
			NullWriter writer(data, reader);
			CudaFrame frame(data, reader, writer);

			resGpu = run(frame, data);
			pyramids.push_back(frame.getPyramid(0));
			pyramids.push_back(frame.getPyramid(1));
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
	};

	template <class T> Result compareFrame2func(MainData& data) {
		Result res;

		AffineTransform trf(0, 0.95, 0.3, 2, 3);
		data.collectDeviceInfo();
		NullReader reader;
		reader.w = 1920;
		reader.h = 1080;
		data.validate(reader);
		NullWriter writer(data, reader);
		std::unique_ptr<MovieFrame> frame = std::make_unique<T>(data, reader, writer);

		frame->mBufferFrame.readFromPGM("d:/VideoTest/v00.pgm");
		frame->mBufferFrame.index = 0;
		reader.frameIndex = 0;
		frame->inputData();
		frame->createPyramid(frame->mReader.frameIndex);

		frame->mBufferFrame.readFromPGM("D:/VideoTest/v01.pgm");
		frame->mBufferFrame.index = 1;
		reader.frameIndex = 1;
		frame->inputData();
		frame->createPyramid(frame->mReader.frameIndex);
		res.pyr = frame->getPyramid(0);
		frame->computeStart(frame->mReader.frameIndex);
		frame->computeTerminate(frame->mReader.frameIndex);
		frame->outputData(trf, writer.getOutputContext());
		res.out = frame->getTransformedOutput();
		res.im = writer.outputFrame;

		res.res = frame->mResultPoints;
		Assert::IsTrue(errorLogger.hasNoError());

		return res;
	}

public:

	TEST_METHOD(compareFrames2) {
		MainData dataCpu;
		Result resCpu = compareFrame2func<CpuFrame>(dataCpu);
		MainData dataCuda;
		dataCuda.probeCuda();
		Result resGpu = compareFrame2func<CudaFrame>(dataCuda);
		MainData dataOcl;
		dataOcl.probeOpenCl();
		Result resOcl = compareFrame2func<OpenClFrame>(dataOcl);

		//check pyramid
		Assert::IsTrue(resCpu.pyr.equalsExact(resGpu.pyr), L"pyramids Cuda are not equal");
		Assert::IsTrue(resCpu.pyr.equalsExact(resOcl.pyr), L"pyramids OpenCl are not equal");

		//check output mats
		Assert::IsTrue(resCpu.out.equalsExact(resGpu.out), L"output mats Cuda are not equal");
		Assert::IsTrue(resCpu.out.equalsExact(resOcl.out), L"output mats OpenCl are not equal");

		//check output images
		//imCpu.saveAsBMP("f:/imcpu.bmp");
		//imGpu.saveAsBMP("f:/imgpu.bmp");
		Assert::IsTrue(resCpu.im == resGpu.im, L"images Cuda are not equal");
		Assert::IsTrue(resCpu.im == resOcl.im, L"images OpenCl are not equal");

		//check results Cuda
		for (int i = 0; i < resCpu.res.size(); i++) {
			const PointResult& cpu = resCpu.res[i];
			const PointResult& gpu = resGpu.res[i];
			Assert::AreEqual(cpu, gpu);
		}

		//check results OpenCl
		for (int i = 0; i < resCpu.res.size(); i++) {
			const PointResult& cpu = resCpu.res[i];
			const PointResult& gpu = resOcl.res[i];
			Assert::AreEqual(cpu, gpu);
		}
	}

	};
}