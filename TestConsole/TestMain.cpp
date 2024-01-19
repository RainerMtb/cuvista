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
#include <chrono>
#include <algorithm>


int main() {
	std::cout << "----------------------------" << std::endl << "TestMain:" << std::endl;
	//imageOutput();
	//qrdec();
	//text();
	//filterCompare();
	//matPerf();
	//matTest();
	//subMat();
	//iteratorTest();
	//similarTransformPerformance();
	//cudaInvSimple();
	//cudaInvPerformanceTest();
	//cudaInvEqualityTest();
	//cudaFMAD();
	//cudaInvParallel();
	//readAndWriteOneFrame();
	//checkVersions();
	//transform();
	//cudaInvTest(1, 32);

	pyramid();
	//openClInvTest(1, 32);
	//openClInvGroupTest(1, 9);
	//openClnorm1Test();
	//flow();
}

class OpticalFlowImageWriter : public OpticalFlowWriter {
public:
	OpticalFlowImageWriter(MainData& data, MovieReader& reader) :
		OpticalFlowWriter(data, reader) {
		codec_ctx = avcodec_alloc_context3(NULL);
	}

	void writeFlow(const MovieFrame& frame, const std::string& fileName) {
		OpticalFlowWriter::writeFlow(frame);
		imageInterpolated.saveAsBMP("f:/flow.bmp");
	}
};

void flow() {
	std::cout << "-------- Test Flow" << std::endl;
	std::string file = "d:/VideoTest/02.mp4";
	MainData data;
	data.collectDeviceInfo();
	data.fileIn = file;
	FFmpegReader reader;
	reader.open(file);
	data.validate(reader);
	NullWriter writer(data, reader);
	CpuFrame frame(data, reader, writer);
	reader.read(frame.mBufferFrame);
	frame.inputData();
	frame.createPyramid(reader.frameIndex);

	reader.read(frame.mBufferFrame);
	frame.inputData();
	frame.createPyramid(reader.frameIndex);

	frame.computeStart(reader.frameIndex);
	frame.computeTerminate(reader.frameIndex);

	OpticalFlowImageWriter fw(data, reader);
	fw.writeFlow(frame, "f:/flow.bmp");

	std::vector<PointResult>& pr = frame.mResultPoints;
	for (PointResultType type : { 
		PointResultType::FAIL_SINGULAR, PointResultType::FAIL_ITERATIONS, PointResultType::FAIL_ETA_NAN,
		PointResultType::RUNNING, PointResultType::SUCCESS_ABSOLUTE_ERR, PointResultType::SUCCESS_STABLE_ITER 
	}) {
		auto num = std::count_if(pr.cbegin(), pr.cend(), [&] (const PointResult& pr) { return pr.result == type; });
		std::cout << "type " << int(type) << ", count " << num << " = " << 100.0 * num / pr.size() << "%" << std::endl;
	}

	frame.computeTransform(reader.frameIndex);
	AffineTransform trf = frame.getTransform();
	trf.toConsole("transform: ", 2);
}