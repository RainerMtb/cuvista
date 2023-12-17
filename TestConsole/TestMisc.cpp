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

void similarTransformPerformance() {
	//load PointResults
	Matd::precision(10);
	std::chrono::microseconds time;
	Matd mat = Matd::fromBinaryFile("D:/VideoTest/points.dat");
	size_t rows = mat.rows();
	std::vector<PointResult> points(rows);
	for (size_t i = 0; i < rows; i++) {
		points[i] = {
			.x = (int) mat[i][0],
			.y = (int) mat[i][1],
			.u = mat[i][2],
			.v = mat[i][3]
		}; //since c++ 20 designated initializers
	}
	std::cout << rows << " points" << std::endl;

	//slow method
	AffineTransform trans;
	trans.computeSimilarLoop(points.begin(), rows);
	Matd x1 = Matd::fromRow(trans.scale(), trans.rot(), trans.dX(), trans.dY()).trans();
	x1.toConsole("V1");
	for (int i = 0; i < 15; i++) {
		auto t1 = std::chrono::high_resolution_clock::now();
		trans.computeSimilarLoop(points.begin(), rows);
		auto t2 = std::chrono::high_resolution_clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
		std::cout << "time : " << time.count() / 1000.0 << " ms" << std::endl;
	}

	//direct method
	ThreadPool thr(4);
	trans.computeSimilarDirect(points.begin(), rows, thr);
	Matd x2 = Matd::fromRow(trans.scale(), trans.rot(), trans.dX(), trans.dY()).trans();
	x2.toConsole("V2");
	for (int i = 0; i < 15; i++) {
		auto t1 = std::chrono::high_resolution_clock::now();
		trans.computeSimilarDirect(points.begin(), rows, thr);
		auto t2 = std::chrono::high_resolution_clock::now();
		time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
		std::cout << "time : " << time.count() / 1000.0 << " ms" << std::endl;
	}

	std::cout << std::endl << "results equal: " << (x1 == x2 ? "YES" : "NO") << std::endl;
}

void readAndWriteOneFrame() {
	MainData data;
	data.probeCuda();
	data.collectDeviceInfo();
	{
		NullReader reader;
		reader.w = 1920;
		reader.h = 1080;
		data.validate(reader);
		NullWriter writer(data, reader);
		CudaFrame frame(data, reader, writer);

		frame.bufferFrame.readFromPGM("d:/VideoTest/v00.pgm");
		frame.bufferFrame.index = 0;
		frame.mReader.frameIndex = 0;
		frame.inputData();
		frame.createPyramid(frame.mReader.frameIndex);

		frame.bufferFrame.readFromPGM("D:/VideoTest/v01.pgm");
		frame.bufferFrame.index = 1;
		frame.mReader.frameIndex = 1;
		frame.inputData();
		frame.createPyramid(frame.mReader.frameIndex);
		frame.computeStart(frame.mReader.frameIndex);
		frame.computeTerminate(frame.mReader.frameIndex);

		AffineTransform trf;
		trf.addRotation(0.3).addTranslation(-200, 100);
		trf.frameIndex = 0;
		OutputContext oc = writer.getOutputContext();
		frame.outputData(trf, oc);
		std::string fileOut = "f:/test.bmp";
		std::cout << "writing file " << fileOut << std::endl;
		oc.outputFrame->saveAsBMP(fileOut);
	}
	std::cout << errorLogger.getErrorMessage() << std::endl;
}

void checkVersions() {
	std::cout << "check cuda devices" << std::endl;
	try {
		MainData data;
		data.probeCuda();
		data.probeOpenCl();
		data.showDeviceInfo();
	} catch (CancelException ignore) {}
}

void transform() {
	std::vector<PointResult> points = {
		{ 0, 0, 0, 0, 0, 3, 6, 0.5, 0.4 },
		{ 1, 0, 0, 0, 0, 5, 5, 0.5, 0.4 },
	};

	AffineTransform trf1;
	trf1.computeSimilarLoop(points.begin(), points.size());
	ThreadPool pool(2);
	AffineTransform trf2;
	trf2.computeSimilarDirect(points.begin(), points.size(), pool);

	trf1.toConsole("result classic loop ");
	trf2.toConsole("result direct method");
	bool isEqual = trf1.equals(trf2, 1e-14);
	std::cout << "results equal: " << std::boolalpha << isEqual << std::endl;
}

void text() {
	ImageYuv im(200, 500);
	im.writeText("abcdefghijklmnopqrstuvwxyz", 10, 10, 2, 3, ColorYuv::WHITE, ColorYuv::BLACK);
	im.writeText("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10, 50, 2, 3, ColorYuv::WHITE, ColorYuv::GRAY);
	im.writeText("äöüß %,.()/-=", 10, 90, 2, 3, ColorYuv::WHITE);
	std::string file = "D:/VideoTest/out/text1.bmp";
	std::cout << "save to " << file << std::endl;
	im.saveAsColorBMP(file);

	ImageBGR bgr(200, 500);
	bgr.writeText("A quick brown fox", 10, 10, 2, 3, { 255, 255, 0 });
	bgr.writeText("jumps over a lazy dog", 10, 50, 2, 3, { 255, 255, 0, 0.5 });
	bgr.writeText("A quick brown fox", 10, 90, 2, 3, { 255, 255, 0 }, { 0, 255, 255, 0.25 });
	bgr.writeText("jumps over a lazy dog", 10, 130, 2, 3, { 255, 255, 0, 0.5 }, { 0, 255, 255, 0.5 });
	file = "D:/VideoTest/out/text2.bmp";
	std::cout << "save to " << file << std::endl;
	bgr.saveAsBMP(file);
}