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
#include "clTest.hpp"
struct Result {
	Matf pyramid;
	Matf output;
	std::vector<PointResult> results;
	std::string name;
	ImageYuv image;
};

template <class T> Result runPyramid(MainData& data) {
	FFmpegReader reader;
	reader.open(data.fileIn);
	data.collectDeviceInfo();
	data.validate(reader);
	NullWriter writer(data, reader);
	std::unique_ptr<MovieFrame> frame = std::make_unique<T>(data, reader, writer);
	std::cout << "running " << frame->getClassId() << std::endl;
	reader.read(frame->mBufferFrame);
	frame->inputData();
	frame->createPyramid(frame->mReader.frameIndex);

	reader.read(frame->mBufferFrame);
	frame->inputData();
	frame->createPyramid(frame->mReader.frameIndex);

	frame->computeStart(frame->mReader.frameIndex);
	frame->computeTerminate(frame->mReader.frameIndex);

	//ImagePPM im(1080, 1920);
	//frame->getInput(0, im);
	//im.saveAsPGM("f:/im.pgm");

	Result result;
	if (errorLogger.hasError()) {
		std::cout << errorLogger.getErrorMessage() << std::endl;

	} else {
		AffineTransform trf;
		trf.addRotation(0.2).addTranslation(-40, 30);
		trf.frameIndex = 0;
		frame->outputData(trf, writer.getOutputContext());
		result = { frame->getPyramid(0), frame->getTransformedOutput(), frame->mResultPoints, frame->getClassId(), writer.getOutputFrame()};
	}

	return result;
}

void compareAllFrames() {
	std::cout << "comparing platforms..." << std::endl;
	std::vector<Result> results(4);
	std::string fileName = "d:/VideoTest/02.mp4";

	{
		//CPU
		MainData data;
		data.fileIn = fileName;
		results[0] = runPyramid<CpuFrame>(data);
	}

	{
		//AVX
		MainData data;
		data.fileIn = fileName;
		results[1] = runPyramid<AvxFrame>(data);
	}

	{
		//Cuda
		MainData data;
		data.deviceRequested = true;
		data.deviceSelected = 2;
		data.probeCuda();
		data.fileIn = fileName;
		results[2] = runPyramid<CudaFrame>(data);
	}

	{
		//OpenCL
		MainData data;
		data.deviceRequested = true;
		data.deviceSelected = 2;
		data.probeOpenCl();
		data.fileIn = fileName;
		results[3] = runPyramid<OpenClFrame>(data);
	}
	std::cout << std::endl;
	//results[0].pyramid.saveAsBinary("f:/0.dat");
	//results[1].pyramid.saveAsBinary("f:/1.dat");
	//results[0].output.saveAsBinary("f:/0.dat");
	//results[1].output.saveAsBinary("f:/1.dat");
	//results[0].image.saveAsColorBMP("f:/0.bmp");
	//results[2].image.saveAsColorBMP("f:/1.bmp");

	//compare pyramid and output
	Result& r1 = results[0];
	for (int i = 1; i < results.size(); i++) {
		Result& r2 = results[i];

		std::cout << "comparing:" << std::endl << r1.name << " // " << r2.name << std::endl;
		std::cout << (r1.pyramid.equalsExact(r2.pyramid) ? "pyramids EQUAL" : "pyramids DIFFER <<<<<<") << std::endl;
		std::cout << (r1.output.equalsExact(r2.output) ? "warped output EQUAL" : "warped output DIFFER <<<<<<") << std::endl;
		std::cout << (r1.image == r2.image ? "image EQUAL" : "image DIFFER <<<<<<") << std::endl;
	}
	return;

	//compare results
	for (int i = 1; i < results.size(); i++) {
		Result& r2 = results[i];
		bool isEqual = true;
		for (size_t i = 0; i < r1.results.size() && i < r2.results.size(); i++) {
			if (r1.results[i] != r2.results[i]) {
				isEqual = false;
				std::cout << "result DIFFER: ix0=" << r1.results[i].ix0 << ", iy0=" << r2.results[i].iy0 << std::endl;
			}
		}
		if (isEqual) {
			std::cout << "results equal" << std::endl;
		}
		std::cout << std::endl;
	}

	if (errorLogger.hasError()) {
		std::cout << errorLogger.getErrorMessage() << std::endl;
	}
}