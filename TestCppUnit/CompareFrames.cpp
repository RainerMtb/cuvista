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
		int w = 200;
		int h = 100;
		int64_t frameCount = 20;
		int64_t readIndex = 0;

	public:
		InputContext open(std::string_view source) override {
			return { h, w, 10, 1, -1, -1, frameCount };
		}

		void read(ImageYuv& frame, Stats& status) override {
			for (int64_t z = 0; z < 3; z++) {
				int64_t base = readIndex * 2 + z * 5 + 30;
				unsigned char* plane = frame.plane(z);
				for (int64_t r = 0; r < frame.h; r++) {
					for (int64_t c = 0; c < frame.w; c++) {
						int64_t pix = std::clamp(base + r / 10, 0LL, 255LL);
						plane[r * frame.stride + c] = (unsigned char) (pix);
					}
				}
			}
			frame.frameIdx = readIndex;
			status.endOfInput = readIndex == frameCount;
			readIndex++;
		}
	};

	//store resulting images in vector
	class TestWriter : public MovieWriter {

	public:
		std::vector<ImageYuv> outputFrames;

		TestWriter(MainData& data) : MovieWriter(data) {}

		void write() override {
			outputFrames.push_back(outputFrame);
		}
	};


	TEST_CLASS(FrameTest) {

private:

	std::vector<PointResult> run(MovieFrame& frame, MainData& data, MovieReader& reader) {
		Stats& status = data.status;
		status.reset();

		//read first frame
		reader.read(frame.bufferFrame, status);
		status.frameReadIndex++;
		frame.inputData(frame.bufferFrame);
		frame.createPyramid();
		status.frameInputIndex++;

		//read second frame
		reader.read(frame.bufferFrame, status);
		frame.inputData(frame.bufferFrame);
		frame.createPyramid();

		//compute
		frame.computeStart();
		frame.computeTerminate();
		Assert::IsTrue(errorLogger.hasNoError());
		return frame.resultPoints;
	}

public:
	TEST_METHOD(compareFrames1) {
		UserInputDefault input;
		std::vector<ImageYuv> cpuImages;
		{
			//cpu test run
			MainData data;
			ProgressDisplayNone progress(data);
			TestReader reader;
			data.inputCtx = reader.open("");
			data.collectDeviceInfo();
			data.validate();
			TestWriter writer(data);
			CpuFrame frame(data);
			reader.read(frame.bufferFrame, data.status);
			MovieFrame::DeshakerLoopCombined loop;
			loop.run(frame, progress, reader, writer, input);
			cpuImages = writer.outputFrames;
			Assert::IsTrue(errorLogger.hasNoError());
		}

		std::vector<ImageYuv> gpuImages;
		{
			//gpu test run
			MainData data;
			ProgressDisplayNone progress(data);
			data.probeCuda();
			TestReader reader;
			data.inputCtx = reader.open("");
			data.collectDeviceInfo();
			data.validate();
			TestWriter writer(data);
			CudaFrame frame(data);
			reader.read(frame.bufferFrame, data.status);
			MovieFrame::DeshakerLoopCombined loop;
			loop.run(frame, progress, reader, writer, input);
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
			data.inputCtx = reader.open(file);
			data.collectDeviceInfo();
			data.validate();
			CpuFrame frame(data);

			resCpu = run(frame, data, reader);
			pyramids.push_back(frame.getPyramid(0));
			pyramids.push_back(frame.getPyramid(1));
		}

		{
			MainData data;
			data.probeCuda();
			data.fileIn = file;
			FFmpegReader reader;
			data.inputCtx = reader.open(file);
			data.collectDeviceInfo();
			data.validate();
			CudaFrame frame(data);

			resGpu = run(frame, data, reader);
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

	TEST_METHOD(compareFrames2) {
		AffineTransform trf(0.95, 0.3, 2, 3);

		//gpu
		std::vector<PointResult> resGpu;
		Matf pyrGpu, outGpu;
		ImageYuv imGpu;
		{
			MainData data;
			data.probeCuda();
			data.inputCtx = { 1080, 1920, 2, 1 };
			data.collectDeviceInfo();
			data.validate();
			NullReader reader;
			NullWriter writer(data);
			CudaFrame frame(data);

			frame.inputFrame.readFromPGM("d:/VideoTest/v00.pgm");
			frame.inputData(frame.inputFrame);
			frame.createPyramid();
			data.status.frameInputIndex++;

			frame.inputFrame.readFromPGM("D:/VideoTest/v01.pgm");
			frame.inputData(frame.inputFrame);
			frame.createPyramid();
			pyrGpu = frame.getPyramid(0);
			frame.computeStart();
			frame.computeTerminate();
			frame.outputData(trf, writer.getOutputContext());
			outGpu = frame.getTransformedOutput();
			imGpu = writer.outputFrame;

			resGpu = frame.resultPoints;
			Assert::IsTrue(errorLogger.hasNoError());
		}

		//cpu
		std::vector<PointResult> resCpu;
		Matf pyrCpu, outCpu;
		ImageYuv imCpu;
		{
			MainData data;
			data.inputCtx = { 1080, 1920, 2, 1 };
			data.collectDeviceInfo();
			data.validate();
			NullReader reader;
			NullWriter writer(data);
			CpuFrame frame(data);

			frame.inputFrame.readFromPGM("d:/VideoTest/v00.pgm");
			frame.inputData(frame.inputFrame);
			frame.createPyramid();
			data.status.frameInputIndex++;

			frame.inputFrame.readFromPGM("D:/VideoTest/v01.pgm");
			frame.inputData(frame.inputFrame);
			frame.createPyramid();
			pyrCpu = frame.getPyramid(0);
			frame.computeStart();
			frame.computeTerminate();
			frame.outputData(trf, writer.getOutputContext());
			outCpu = frame.getTransformedOutput();
			imCpu = writer.outputFrame;

			resCpu = frame.resultPoints;
			Assert::IsTrue(errorLogger.hasNoError());
		}

		//check pyramid
		Assert::IsTrue(pyrCpu.equalsExact(pyrGpu), L"pyramids are not equal");

		//check output mats
		Assert::IsTrue(outCpu.equalsExact(outGpu), L"output mats are not equal");

		//check output images
		Assert::IsTrue(imCpu == imGpu, L"images are not equal");

		//check results
		for (int i = 0; i < resGpu.size(); i++) {
			const PointResult& cpu = resCpu[i];
			const PointResult& gpu = resGpu[i];
			Assert::AreEqual(cpu, gpu);
		}
	}

	};
}