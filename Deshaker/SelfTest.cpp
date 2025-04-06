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
		errorLogger().clearErrors();
		out.print(" #");
		out.print(std::to_string(i));
		out.print(": ");
		std::string name = deviceList[i]->getNameShort();
		name.resize(10, ' ');
		out.print(name);
		out.print(": testing... ");

		MainData data;
		data.bgcol_rgb = { 0, 50, 0 };
		data.deviceList.push_back(deviceList[i]);
		MemoryFFmpegReader reader(movieData);
		reader.open("");
		data.validate(reader);
		BaseWriter writer(data, reader);
		
		//executor and frame
		MovieFrameCombined frame(data, reader, writer);
		auto executor = deviceList[i]->create(data, frame);
		
		//first frame
		//std::cout << "reading" << std::endl;
		reader.read(frame.mBufferFrame);
		executor->inputData(reader.frameIndex, frame.mBufferFrame);
		executor->createPyramid(reader.frameIndex);
		//second frame
		reader.read(frame.mBufferFrame);
		executor->inputData(reader.frameIndex, frame.mBufferFrame);
		executor->createPyramid(reader.frameIndex);
		//compute
		//std::cout << "computing" << std::endl;
		executor->computeStart(reader.frameIndex, frame.mResultPoints);
		executor->computeTerminate(reader.frameIndex, frame.mResultPoints);
		//input
		ImageRGBA input(data.h, data.w);
		executor->getInput(0, input);
		//output
		AffineTransform trf;
		trf.addRotation(0.2).addTranslation(-40, 30);
		executor->outputData(0, trf);
		writer.prepareOutput(*executor);

		//checks
		//std::cout << "running checks" << std::endl;
		bool check = true;
		if (uint64_t crc = input.crc(); crc != crcInput) {
			out.print("FAIL input ");
			check = false;
		}
		if (uint64_t crc = executor->getPyramid(0).crc(); crc != crcPyramid) {
			out.print("FAIL pyramid ");
			check = false;
		}
		if (uint64_t crc = executor->getTransformedOutput().crc(); crc != crcTransformed) {
			out.print("FAIL transformed ");
			check = false;
		}
		if (uint64_t crc = writer.getOutputFrame().crc(); crc != crcOutput) {
			//writer.getOutputFrame().saveAsColorBMP("f:/test.bmp");
			out.print("FAIL output ");
			check = false;
		}
		util::CRC64 crc64;
		for (const PointResult& pr : frame.mResultPoints) {
			crc64.add(pr.result); 
			crc64.add(pr.idx);
			crc64.add(pr.ix0);
			crc64.add(pr.iy0);
			crc64.add(pr.x);
			crc64.add(pr.y);
			crc64.add(pr.u);
			crc64.add(pr.v);
		}
		if (uint64_t crc = crc64.result(); crc != crcResult) {
			out.print("FAIL result ");
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
