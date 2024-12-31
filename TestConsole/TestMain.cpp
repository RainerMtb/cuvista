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

void analyseFrames(const std::string& in1, const std::string& in2, const std::string& out) {
	//std::cout << "reading images" << std::endl;
	ImageYuv im1 = ImageBGR::readFromBMP(in1).toYUV();
	ImageYuv im2 = ImageBGR::readFromBMP(in2).toYUV();

	MainData data;
	data.collectDeviceInfo();
	ImageReader reader;
	reader.h = im1.h;
	reader.w = im1.w;
	data.validate(reader);

	NullWriter writer(data, reader);
	MovieFrameFirst frame(data, reader, writer);
	CpuFrame cpuframe(data, data.deviceInfoCpu, frame, frame.mPool);
	
	reader.readImage(frame.mBufferFrame, im1);
	cpuframe.inputData(reader.frameIndex, frame.mBufferFrame);
	cpuframe.createPyramid(reader.frameIndex);

	reader.readImage(frame.mBufferFrame, im2);
	cpuframe.inputData(reader.frameIndex, frame.mBufferFrame);
	cpuframe.createPyramid(reader.frameIndex);

	//std::cout << "computing transform" << std::endl;
	cpuframe.computeStart(reader.frameIndex, frame.mResultPoints);
	cpuframe.computeTerminate(reader.frameIndex, frame.mResultPoints);
	frame.computeTransform(reader.frameIndex);
	const AffineTransform& trf = frame.getTransform();

	//std::cout << "writing result" << std::endl;
	ResultImageWriter riw(data);
	riw.writeImage(trf, frame.mResultPoints, 1, im2, out);
	
	std::cout << "done" << std::endl;
}

int main() {
	std::cout << "----------------------------" << std::endl << "TestMain:" << std::endl;
	//imageOutput();
	//qrdec();
	//draw();
	//filterCompare();
	//matPerf();
	//matTest();
	//subMat();
	//iteratorTest();
	//cudaInvSimple();
	//cudaInvPerformanceTest();
	//cudaInvEqualityTest();
	//cudaFMAD();
	//cudaInvParallel();
	//readAndWriteOneFrame();
	//checkVersions();
	//transform();
	//cudaInvTest(1, 32);

	//openClInvTest(1, 32);
	//openClInvGroupTest(1, 9);
	//openClnorm1Test();
	//flow();
	//pinvTest();
	//compareInv();
	//similarTransform();

	//testSampler();
	//compareFramesPlatforms();
	//avxCompute();

	//testZoom();
	//analyzeFrames();

	for (int i = 0; i < 10; i++) {
		std::string in1 = std::format("f:/pic/{:04}.bmp", i);
		std::string in2 = std::format("f:/pic/{:04}.bmp", i + 1);
		std::string out = std::format("f:/pic/out{:02}.bmp", i + 1);
		std::cout << "writing " << out << std::endl;
		analyseFrames(in1, in2, out);
	}
}