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

void x() {
	MainData data;
	data.deviceRequested = true;
	data.deviceSelected = 0;
	FFmpegReader reader;
	reader.open("d:/VideoTest/02.mp4");
	data.collectDeviceInfo();
	data.validate(reader);
	NullWriter writer(data, reader);
	std::unique_ptr<MovieFrame> frame = std::make_unique<AvxFrame>(data, reader, writer);
	std::cout << "running " << frame->getClassName() << std::endl;
	reader.read(frame->mBufferFrame);
	frame->inputData();
	frame->createPyramid(frame->mReader.frameIndex);
	frame->createPyramid(frame->mReader.frameIndex);
	frame->createPyramid(frame->mReader.frameIndex);
	frame->createPyramid(frame->mReader.frameIndex);
	frame->createPyramid(frame->mReader.frameIndex);
	frame->createPyramid(frame->mReader.frameIndex);
	frame->createPyramid(frame->mReader.frameIndex);
	frame->createPyramid(frame->mReader.frameIndex);
	frame->createPyramid(frame->mReader.frameIndex);
	frame->createPyramid(frame->mReader.frameIndex);
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

	//openClInvTest(1, 32);
	//openClInvGroupTest(1, 9);
	//openClnorm1Test();
	//flow();
	//pinvTest();
	pyramid();
	//x();
}
