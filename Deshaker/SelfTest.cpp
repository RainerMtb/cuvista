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
	std::vector<unsigned char> movieData = util::base64_decode(movieTestData);
	out.printNewLine();
	out.print("Testing available devices...");
	out.printNewLine();

	for (size_t i = 0; i < deviceList.size(); i++) {
		errorLogger().clear();
		out.print(" #");
		out.print(std::to_string(i));
		out.print(": ");
		std::string name = deviceList[i]->getNameShort();
		name.resize(10, ' ');
		out.print(name);
		out.print(": testing... ");

		MainData data;
		data.backgroundColor = Color::rgb(0, 50, 0);
		data.deviceList.push_back(deviceList[i]);
		MemoryFFmpegReader reader(movieData);
		reader.open("");
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
			reader.read(frame.mBufferFrame);
			executor->inputData(reader.frameIndex, frame.mBufferFrame);
			executor->createPyramid(reader.frameIndex, {}, false);
			//second frame
			reader.read(frame.mBufferFrame);
			executor->inputData(reader.frameIndex, frame.mBufferFrame);
			executor->createPyramid(reader.frameIndex, {}, false);
			//compute
			//std::cout << "computing" << std::endl;
			executor->computeStart(reader.frameIndex, frame.mResultPoints);
			executor->computeTerminate(reader.frameIndex, frame.mResultPoints);
			//input
			ImageRGBA input(data.h, data.w);
			executor->getInput(0, input);
			//input.saveAsColorBMP(std::string("f:/input" + std::to_string(i) + ".bmp"));
			//output
			AffineTransform trf;
			trf.addRotation(0.2).addTranslation(-40, 30);
			executor->outputData(0, trf);
			writer.writeOutput(*executor);

			//checks
			//std::cout << "running checks" << std::endl;
			if (uint64_t crc = input.crc(); crc != crcInput) {
				//std::cout << std::hex << crc << " ";
				out.print("FAIL input ");
				check = false;
			}
			if (uint64_t crc = executor->getPyramid(0).crc(); crc != crcPyramid) {
				//std::cout << std::hex << crc << " ";
				out.print("FAIL pyramid ");
				check = false;
			}
			if (uint64_t crc = executor->getTransformedOutput().crc(); crc != crcTransformed) {
				//std::cout << std::hex << crc << " ";
				out.print("FAIL transformed ");
				check = false;
			}
			if (uint64_t crc = writer.getOutputFrame().crc(); crc != crcOutput) {
				//std::cout << std::hex << crc << " ";
				out.print("FAIL output ");
				check = false;
			}
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
				//std::cout << std::hex << crc << " ";
				out.print("FAIL result ");
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
