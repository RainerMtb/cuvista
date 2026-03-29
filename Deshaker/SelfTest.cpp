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

#include "SelfTest.hpp"
#include "SelfTestData.hpp"
#include "AVException.hpp"
#include "MainData.hpp"
#include "MovieWriter.hpp"
#include "MovieReader.hpp"
#include "MovieFrame.hpp"

void MessagePrinterConsole::print(const std::string& str) {
	*out << str << std::flush;
}

void MessagePrinterConsole::printNewLine() {
	*out << std::endl << std::flush;
}

void runSelfTest(util::MessagePrinter& out, std::vector<DeviceInfoBase*> deviceList) {
	out.printNewLine();
	out.print("Testing available devices...");
	out.printNewLine();

	std::vector<unsigned char> movieData = util::base64_decode(movieTestData);

	uint64_t crcInputFirst =  0x2bf32ea3d882dce0;
	uint64_t crcInputSecond = 0xfd1547af06114037;
	uint64_t crcPyramid =     0x430664d35d1bddfa;
	uint64_t crcLuma =        0x3586033f4901bd51;
	uint64_t crcResult =      0x9a14e1549f60775f;
	uint64_t crcTransformed = 0x1ab110a9e388b23c;
	uint64_t crcOutput =      0x93abc29a172ba683;
	uint64_t crcNv12 =        0x15ab051f2d2045cb;

	//util::debugLogger = std::make_shared<util::DebugLoggerTcp>("10.0.0.1", 5555);
	for (size_t i = 0; i < deviceList.size(); i++) {
		errorLogger().clear();
		out.print(" #");
		out.print(std::to_string(i));
		out.print(": ");
		std::string name = deviceList[i]->getNameShort();
		std::string str = name;
		str.resize(10, ' ');
		out.print(str);
		out.print(": testing... ");

		MainData data;
		data.backgroundColor = Color::rgb(0, 50, 0);
		data.deviceList.push_back(deviceList[i]);
		//data.cpuThreadsRequired = { 1 };
		MemoryFFmpegReader reader(movieData);
		reader.open();
		data.validate(reader);
		OutputWriter writer(data, reader);
		
		bool check = true;
		try {
			//frame executor
			MovieFrameCombined frame(data, reader, writer);
			std::shared_ptr<FrameExecutor> executor = deviceList[i]->create(data, frame);
			executor->init();
			if (errorLogger().hasError()) throw AVException(errorLogger().getErrorMessage());

			//first frame
			//std::cout << "reading" << std::endl;
			reader.read(*executor);
			executor->inputData(reader.frameIndex);
			int64_t luma1 = executor->createPyramid(reader.frameIndex, {}, false);
			//second frame
			reader.read(*executor);
			executor->inputData(reader.frameIndex);
			int64_t luma2 = executor->createPyramid(reader.frameIndex, {}, false);
			//compute
			//std::cout << "computing" << std::endl;
			executor->computeStart(reader.frameIndex, frame.mResultPoints);
			executor->computeTerminate(reader.frameIndex, frame.mResultPoints);
			//input
			ImageRGBA inputFirst(data.h, data.w);
			executor->getInput(0, inputFirst);

			//input
			ImageVuyx inputSecond(data.h, data.w);
			executor->getInput(1, inputSecond);
			//output
			AffineTransform trf;
			trf.addRotation(0.2).addTranslation(-40, 30);
			executor->outputData(0, trf);
			writer.writeOutput(*executor);

			//-------------- executing checks -----------------------
			
			//inputFirst.saveBmpPlanes(std::format("f:/in1_{}.bmp", name));
			if (uint64_t crc = inputFirst.crc(); crc != crcInputFirst) {
				util::debugLogger->format("{} fail input1 {:x}", name, crc);
				out.print("FAIL input1 ");
				check = false;
			}
			//inputSecond.saveBmpPlanes(std::format("f:/in2_{}.bmp", name));
			if (uint64_t crc = inputSecond.crc(); crc != crcInputSecond) {
				util::debugLogger->format("{} fail input2 {:x}", name, crc);
				out.print("FAIL input2 ");
				check = false;
			}

			Matf pyramid = executor->getPyramid(0);
			//pyramid.saveAsBMP(std::format("f:/pyr_{}.bmp", name), 1.0f);
			if (uint64_t crc = pyramid.crc(); crc != crcPyramid) {
				util::debugLogger->format("{} fail pyramid {:x}", name, crc);
				out.print("FAIL pyramid ");
				check = false;
			}

			if (uint64_t crc = util::CRC64().addDirect(luma1).addDirect(luma2).result(); crc != crcLuma) {
				util::debugLogger->format("{} fail luma {:x}", name, crc);
				out.print("FAIL luma ");
				check = false;
			}

			{
				//check result data
				util::CRC64 crc64;
				for (const PointResult& pr : frame.mResultPoints) {
					crc64.addDirect(pr.result);
					crc64.addDirect(pr.idx);
					crc64.addDirect(pr.ix0);
					crc64.addDirect(pr.iy0);
					crc64.addDirect(pr.x);
					crc64.addDirect(pr.y);
					crc64.addDirect(pr.u);
					crc64.addDirect(pr.v);
				}
				if (uint64_t crc = crc64.result(); crc != crcResult) {
					util::debugLogger->format("{} fail result {:x}", name, crc);
					out.print("FAIL result ");
					check = false;
				}
			}

			//transformed float data
			Matf transformed = executor->getTransformedOutput();
			//transformed.saveAsBinary(std::format("f:/trf_{}.mat", name));
			//transformed.saveAsBMP(std::format("f:/trf_{}.bmp", name), 1.0f);
			if (uint64_t crc = transformed.crc(); crc != crcTransformed) {
				util::debugLogger->format("{} fail transformed {:x}", name, crc);
				out.print("FAIL transformed ");
				check = false;
			}

			//output yuv
			const ImageVuyx& output = writer.getOutputFrame();
			//output.saveBmpColor(std::format("f:/out_{}.bmp", name));
			if (uint64_t crc = output.crc(); crc != crcOutput) {
				util::debugLogger->format("{} fail output {:x}", name, crc);
				out.print("FAIL output ");
				check = false;
			}

			//output nv12
			ImageNV12 nv12(data.h, data.w, data.stride);
			executor->getOutput(0, nv12, 0, nullptr);
			//nv12.saveBmpColor(std::format("f:/nv12_{}.bmp", name));
			if (uint64_t crc = nv12.crc(); crc != crcNv12) {
				util::debugLogger->format("{} fail nv12 {:x}", name, crc);
				out.print("FAIL nv12 ");
				check = false;
			}

		} catch (AVException e) {
			out.print("ERROR: ");
			out.print(e.what());
			check = false;
		}

		if (errorLogger().hasError()) {
			out.print(errorLogger().getErrorMessage());
			check = false;
		}
		if (check) {
			out.print("OK");
		}

		out.printNewLine();
	}
	out.print("Testing complete");
	out.printNewLine();
}
