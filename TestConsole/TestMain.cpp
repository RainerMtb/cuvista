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

void func() {
	ImageYuv im1, im2;

	{
		MainData data;
		data.collectDeviceInfo();
		FFmpegReader reader;
		reader.open("d:/VideoTest/02.mp4");
		data.validate(reader);

		BaseWriter writer(data, reader);
		MovieFrameConsecutive frame(data, reader, writer);
		CpuFrame frameExecutor(data, data.deviceInfoCpu, frame, frame.mPool);

		for (int i = 0; i < 6; i++)	reader.read(frame.mBufferFrame);
		frameExecutor.inputData(0, frame.mBufferFrame);

		AffineTransform trf;
		trf.setParam(0.952379970131, 0.001367827033, 33.316623121580, 26.105044749792);
		frameExecutor.outputData(0, trf);
		writer.prepareOutput(frameExecutor);
		im1 = writer.getOutputFrame();
	}

	{
		MainData data;
		data.probeCuda();
		data.collectDeviceInfo();
		FFmpegReader reader;
		reader.open("d:/VideoTest/02.mp4");
		data.validate(reader);

		BaseWriter writer(data, reader);
		MovieFrameConsecutive frame(data, reader, writer);
		CudaFrame frameExecutor(data, data.cudaInfo.devices[0], frame, frame.mPool);

		for (int i = 0; i < 6; i++)	reader.read(frame.mBufferFrame);
		frameExecutor.inputData(0, frame.mBufferFrame);

		AffineTransform trf;
		trf.setParam(0.952379970131, 0.001367827033, 33.316623121580, 26.105044749792);
		frameExecutor.outputData(0, trf);
		writer.prepareOutput(frameExecutor);
		im2 = writer.getOutputFrame();
	}

	for (int z = 0; z < 3; z++) {
		for (int r = 0; r < 1080; r++) {
			for (int c = 0; c < 1920; c++) {
				if (im1.at(z, r, c) != im2.at(z, r, c)) {
					std::cout << z << "/" << r << "/" << c << " " << (int) im1.at(z, r, c) << " : " << (int) im2.at(z, r, c) << std::endl;
				}
			}
		}
	}
	std::cout << std::boolalpha << "images equal: " << (im1 == im2) << std::endl;

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

	//createTransformImages();
	func();
}