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

#include "TestMain.hpp"
//#include "clTest.hpp"
//#include "Util.hpp"
//#include "SelfTestData.hpp"

#include <fstream>
#include <iterator>

struct Result {
	Matf pyramid;
	Matf output;
	std::vector<PointResult> results;
	std::string name;
	ImageYuv image;
	ImageRGBA input;
	std::string error;
};

template <class T> Result runPyramid(MainData& data) {
	try {
		std::cout << "1" << std::endl;
		//std::vector<unsigned char> bytes = util::base64_decode(movieTestData);
		std::cout << "2" << std::endl;
		//MemoryFFmpegReader reader(bytes);
		FFmpegReader reader;
		std::cout << "3" << std::endl;
		//reader.open("");
		reader.open("//READYNAS/Data/VideoTest/02.mp4");
		std::cout << "4" << std::endl;
		data.collectDeviceInfo();
		std::cout << "5" << std::endl;
		data.validate(reader);
		std::cout << "6" << std::endl;
		BaseWriter writer(data, reader);
		std::cout << "7" << std::endl;
		std::unique_ptr<MovieFrame> frame = std::make_unique<T>(data, reader, writer);
		std::cout << "running " << frame->getId().nameShort << std::endl;
		reader.read(frame->mBufferFrame);
		frame->inputData();
		frame->createPyramid(frame->mReader.frameIndex);
		std::cout << "8" << std::endl;

		reader.read(frame->mBufferFrame);
		std::cout << "9" << std::endl;
		frame->inputData();
		std::cout << "10" << std::endl;
		frame->createPyramid(frame->mReader.frameIndex);
		std::cout << "11" << std::endl;

		frame->computeStart(frame->mReader.frameIndex);
		std::cout << "12" << std::endl;
		frame->computeTerminate(frame->mReader.frameIndex);
		std::cout << "13" << std::endl;

		ImageRGBA im(data.h, data.w);
		frame->getInput(0, im);

		AffineTransform trf;
		trf.addRotation(0.2).addTranslation(-40, 30);
		trf.frameIndex = 0;
		frame->outputData(trf);
		writer.prepareOutput(*frame);
		Result result = {
			frame->getPyramid(0),
			frame->getTransformedOutput(),
			frame->mResultPoints,
			frame->getId().nameShort,
			writer.getOutputFrame(),
			im,
			(errorLogger.hasError() ? errorLogger.getErrorMessage() : "")
		};

		return result;

	} catch (AVException e) {
		std::cout << "exception " << e.what() << std::endl;

	} catch (...) {
		std::cout << "general exception" << std::endl;
	}

	return {};
}

void compareFramesPlatforms() {
	std::cout << "comparing platforms..." << std::endl;
	std::vector<Result> results(4);

	{
		//CPU
		MainData data;
		results[0] = runPyramid<CpuFrame>(data);
	}

	{
		//AVX
		MainData data;
		results[1] = runPyramid<AvxFrame>(data);
	}

	{
		//Cuda
		MainData data;
		data.deviceRequested = true;
		data.deviceSelected = 2;
		data.probeCuda();
		results[2] = runPyramid<CudaFrame>(data);
	}

	{
		//OpenCL
		MainData data;
		data.deviceRequested = true;
		data.deviceSelected = 2;
		data.probeOpenCl();
		results[3] = runPyramid<OpenClFrame>(data);
	}
	std::cout << std::endl;
	//results[0].input.saveAsColorBMP("f:/0.bmp");
	//results[1].input.saveAsColorBMP("f:/1.bmp");
	//results[2].output.saveAsBinary("f:/2.dat");
	//results[0].image.saveAsColorBMP("f:/0.bmp");
	//results[1].image.saveAsColorBMP("f:/1.bmp");

	for (int i = 0; i < results.size(); i++) {
		if (results[i].error.empty() == false) std::cout << "error " << i << " " << results[i].name << ": " << results[i].error << std::endl;
	}

	Result& r1 = results[0];
	for (int i = 1; i < results.size(); i++) {
		Result& r2 = results[i];

		//compare image data
		std::cout << "comparing: " << r1.name << " vs " << r2.name << std::endl;
		std::cout << (r1.pyramid.equalsExact(r2.pyramid) ? "pyramids EQUAL" : "pyramids DIFFER <<<<<<") << std::endl;
		std::cout << (r1.output.equalsExact(r2.output) ? "warped output EQUAL" : "warped output DIFFER <<<<<<") << std::endl;
		std::cout << (r1.image == r2.image ? "image EQUAL" : "image DIFFER <<<<<<") << std::endl;
		std::cout << (r1.input == r2.input ? "input EQUAL" : "input DIFFER <<<<<<<") << std::endl;

		//compare results
		int deltaCount = 0;
		for (size_t i = 0; i < r1.results.size() && i < r2.results.size(); i++) {
			if (r1.results[i] != r2.results[i]) {
				if (deltaCount == 0) std::cout << "results DIFFER: ix0=" << r1.results[i].ix0 << ", iy0=" << r2.results[i].iy0 << std::endl;
				deltaCount++;
			}
		}
		if (deltaCount == 0) std::cout << "results EQUAL" << std::endl;
		else std::cout << "results difference count " << deltaCount << std::endl;
		std::cout << std::endl;
	}

	if (errorLogger.hasError()) {
		std::cout << errorLogger.getErrorMessage() << std::endl;
	}
}