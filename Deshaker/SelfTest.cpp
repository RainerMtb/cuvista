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
		errorLogger.clearErrors();
		out.print(" #");
		out.print(std::to_string(i));
		out.print(": ");
		std::string name = deviceList[i]->getNameShort();
		name.resize(10, ' ');
		out.print(name);
		out.print(": testing... ");

		MainData data;
		data.deviceList.push_back(deviceList[i]);
		MemoryFFmpegReader reader(movieData);
		reader.open("");
		data.validate(reader);
		BaseWriter writer(data, reader);
		auto frame = deviceList[i]->createClass(data, reader, writer);

		//first frame
		reader.read(frame->mBufferFrame);
		frame->inputData();
		frame->createPyramid(frame->mReader.frameIndex);
		//second frame
		reader.read(frame->mBufferFrame);
		frame->inputData();
		frame->createPyramid(frame->mReader.frameIndex);
		//compute
		frame->computeStart(frame->mReader.frameIndex);
		frame->computeTerminate(frame->mReader.frameIndex);
		//input
		ImageRGBA input(data.h, data.w);
		frame->getInput(0, input);
		//output
		AffineTransform trf;
		trf.addRotation(0.2).addTranslation(-40, 30);
		trf.frameIndex = 0;
		frame->outputData(trf);
		writer.prepareOutput(*frame);

		//checks
		bool check = true;
		if (input.crc() != crcInput) {
			out.print("FAIL input ");
			check = false;
		}
		if (frame->getPyramid(0).crc() != crcPyramid) {
			out.print("FAIL pyramid ");
			check = false;
		}
		if (frame->getTransformedOutput().crc() != crcTransformed) {
			out.print("FAIL transformed ");
			check = false;
		}
		if (writer.getOutputFrame().crc() != crcOutput) {
			out.print("FAIL output ");
			check = false;
		}
		util::CRC64 crc;
		for (const PointResult& pr : frame->mResultPoints) {
			crc.add(pr.result); 
			crc.add(pr.idx);
			crc.add(pr.ix0);
			crc.add(pr.iy0);
			crc.add(pr.px);
			crc.add(pr.py);
			crc.add(pr.x);
			crc.add(pr.y);
		}
		if (crc.result() != crcResult) {
			out.print("FAIL result ");
			check = false;
		}
		if (errorLogger.hasError()) {
			out.print(errorLogger.getErrorMessage());
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