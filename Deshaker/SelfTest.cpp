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

	uint64_t crcInputFirst =  0xe72e7c43c685dc4c;
	uint64_t crcInputSecond = 0x8975f438e501390a;
	uint64_t crcPyramid =     0x430664d35d1bddfa;
	uint64_t crcLuma =        0x3586033f4901bd51;
	uint64_t crcResult =      0x9a14e1549f60775f;
	uint64_t crcTransformed = 0x7904805c67659a0f;
	uint64_t crcOutput =      0xfc36c9a2927c627a;

	//for (size_t i = 3; i < 4; i++) {
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
			ImageAyuv inputSecond(data.h, data.w);
			executor->getInput(1, inputSecond);
			//output
			AffineTransform trf;
			trf.addRotation(0.2).addTranslation(-40, 30);
			executor->outputData(0, trf);
			writer.writeOutput(*executor);

			//executing checks
			//inputFirst.saveBmpPlanes(std::format("f:/in1_{}.bmp", name));
			if (uint64_t crc = inputFirst.crc(); crc != crcInputFirst) {
				debugLogger->format("{} fail input 1 {:x}", name, crc);
				out.print("FAIL input 1 ");
				check = false;
			}
			//inputSecond.saveBmpPlanes(std::format("f:/in2_{}.bmp", name));
			if (uint64_t crc = inputSecond.crc(); crc != crcInputSecond) {
				debugLogger->format("{} fail input 2 {:x}", name, crc);
				out.print("FAIL input 2 ");
				check = false;
			}

			Matf pyramid = executor->getPyramid(0);
			//pyramid.saveAsBMP(std::format("f:/pyr_{}.bmp", name), 1.0f);
			if (uint64_t crc = pyramid.crc(); crc != crcPyramid) {
				debugLogger->format("{} fail pyramid {:x}", name, crc);
				out.print("FAIL pyramid ");
				check = false;
			}

			if (uint64_t crc = util::CRC64().addDirect(luma1).addDirect(luma2).result(); crc != crcLuma) {
				debugLogger->format("{} fail luma {:x}", name, crc);
				out.print("FAIL luma ");
				check = false;
			}

			{
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
					debugLogger->format("{} fail result {:x}", name, crc);
					out.print("FAIL result ");
					check = false;
				}
			}

			Matf transformed = executor->getTransformedOutput();
			//transformed.saveAsBinary(std::format("f:/trf_{}.mat", name));
			transformed.saveAsBMP(std::format("f:/trf_{}.bmp", name), 1.0f);
			if (uint64_t crc = transformed.crc(); crc != crcTransformed) {
				debugLogger->format("{} fail transformed {:x}", name, crc);
				out.print("FAIL transformed ");
				check = false;
			}

			const ImageAyuv& output = writer.getOutputFrame();
			//output.saveBmpPlanes(std::format("f:/out_{}.bmp", name));
			if (uint64_t crc = output.crc(); crc != crcOutput) {
				debugLogger->format("{} fail output {:x}", name, crc);
				out.print("FAIL output ");
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
