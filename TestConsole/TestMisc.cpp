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

Matd similarTransformFunc(std::vector<PointResult>& points, AffineSolver& solver) {
	solver.computeSimilar(points);
	Matd x = Matd::fromRow(solver.scale(), solver.rot(), solver.dX(), solver.dY()).trans();
	x.toConsole("X");
	for (int i = 0; i < 8; i++) {
		auto t1 = std::chrono::high_resolution_clock::now();
		solver.computeSimilar(points);
		auto t2 = std::chrono::high_resolution_clock::now();
		auto time = std::chrono::duration<double, std::milli>(t2 - t1);
		std::cout << "time : " << time.count() << " ms" << std::endl;
	}
	return x;
}

void similarTransform() {
	//load PointResults
	Matd::precision(10);

	int w = 100;
	int h = 75;
	int n = w * h;
	std::vector<PointResult> points(n);
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			PointResult& pr = points[1ll * y * w + x];
			pr.x = x;
			pr.y = y;
			pr.u = 4 + std::sin(x);
			pr.v = 3 + std::sin(y);
		}
	}
	std::cout << n << " points" << std::endl;

	//slow method
	AffineSolverSimple s1(points.size());
	Matd x1 = similarTransformFunc(points, s1);

	//direct method
	ThreadPool thr(4);
	AffineSolverFast s2(thr, points.size());
	Matd x2 = similarTransformFunc(points, s2);

	//avx solver
	AffineSolverAvx s3(points.size());
	Matd x3 = similarTransformFunc(points, s3);

	std::cout << std::endl;
	std::cout << "results equal 1-2: " << (x1 == x2 ? "YES" : "NO") << std::endl;
	std::cout << "results equal 1-3: " << (x1 == x3 ? "YES" : "NO") << std::endl;
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
		OutputWriter writer(data, reader);
		MovieFrame frame(data, reader, writer);
		CudaFrame ex(data, *data.deviceList[2], frame, frame.mPool);

		frame.mBufferFrame.readFromPGM("d:/VideoTest/v00.pgm");
		frame.mBufferFrame.index = 0;
		reader.frameIndex = 0;
		ex.inputData(reader.frameIndex, frame.mBufferFrame);
		ex.createPyramid(frame.mReader.frameIndex, {}, false);

		frame.mBufferFrame.readFromPGM("D:/VideoTest/v01.pgm");
		frame.mBufferFrame.index = 1;
		reader.frameIndex = 1;
		ex.inputData(reader.frameIndex, frame.mBufferFrame);
		ex.createPyramid(frame.mReader.frameIndex, {}, false);
		ex.computeStart(frame.mReader.frameIndex, frame.mResultPoints);
		ex.computeTerminate(frame.mReader.frameIndex, frame.mResultPoints);

		AffineTransform trf;
		trf.addRotation(0.3).addTranslation(-200, 100);
		trf.frameIndex = 0;
		ex.outputData(0, trf);
		writer.prepareOutput(ex);
		std::string fileOut = "f:/test.bmp";
		std::cout << "writing file " << fileOut << std::endl;
		writer.getOutputFrame().saveAsColorBMP(fileOut);
	}
	std::cout << errorLogger().getErrorMessage() << std::endl;
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

void draw(const std::string& filename) {
	using namespace im;

	//draw into bgr image
	ImageBGR bgr(600, 800);
	bgr.setColor(Color::GRAY);
	bgr.writeText("abcdefghijklmnopqrstuvwxyz", 10, 10, 2, 3, TextAlign::TOP_LEFT);
	bgr.writeText("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10, 50, 2, 3, TextAlign::TOP_LEFT);
	bgr.writeText("A quick brown fox", 10, 90, 2, 3, TextAlign::TOP_LEFT, Color::YELLOW, Color::BLACK);
	bgr.writeText("jumps over a lazy dog", 10, 120, 2, 3, TextAlign::TOP_LEFT, Color::rgba(255, 255, 255, 0.75), Color::GREEN);
	bgr.writeText("A quick brown fox", 10, 150, 2, 3, TextAlign::TOP_LEFT, Color::YELLOW, Color::rgba(0, 255, 255, 0.25));
	bgr.writeText("jumps over a lazy dog", 10, 180, 2, 3, TextAlign::TOP_LEFT, Color::rgba(255, 0, 255, 0.5), Color::rgba(0, 255, 255, 0.5));

	//charachter map
	int x0 = 500;
	int y0 = 20;
	for (int i = 0; i < 16; i++) {
		for (int k = 0; k < 16; k++) {
			char ch = i * 16 + k;
			std::string str({ ch });
			bgr.writeText(str, x0 + i * 16, y0 + k * 22, 2, 2, TextAlign::TOP_LEFT);
		}
	}

	//lines and dots
	double len = 100.0;
	double cx = 250.0;
	double cy = 350.0;
	double r = 1.75;
	for (double angle = 0.0; angle < 360.0; angle += 15.0) {
		double x1 = cx + len * std::cos(angle * std::numbers::pi / 180.0);
		double y1 = cy + len * std::sin(angle * std::numbers::pi / 180.0);
		bgr.drawLine(cx, cy, x1, y1, Color::BLUE);
		bgr.drawMarker(x1, y1, Color::RED, r);
	}

	//dots at fractional pixel values
	for (int i = 0; i < 5; i++) {
		for (int k = 0; k < 5; k++) {
			double x = 20 + 10.2 * i + 0.2 * k;
			double y = 250 + 10.2 * k + 0.2 * i;
			bgr.drawMarker(x, y, Color::GREEN, 1.5);
		}
	}

	//polygon
	bgr.drawLine(400, 300, 425, 320, Color::GREEN);
	bgr.drawLine(425, 320, 410, 375, Color::GREEN);
	bgr.drawLine(410, 375, 380, 355, Color::GREEN);
	bgr.drawLine(380, 355, 400, 300, Color::GREEN);

	//save to file
	std::cout << "save to " << filename << std::endl;
	bgr.saveAsColorBMP(filename);
}

class OpticalFlowImageWriter : public OpticalFlowWriter {
public:
	OpticalFlowImageWriter(MainData& data, MovieReader& reader) :
		OpticalFlowWriter(data, reader) {
		codec_ctx = avcodec_alloc_context3(NULL);
	}

	void writeFlow(const MovieFrame& frame, const std::string& fileName) {
		OpticalFlowWriter::writeFlow(frame);
		imageInterpolated.saveAsColorBMP("f:/flow.bmp");
	}
};

void flow() {
	std::chrono::time_point t1 = std::chrono::system_clock::now();
	std::cout << "-------- Test Flow" << std::endl;
	std::string file = "d:/VideoTest/02.mp4";
	MainData data;
	data.collectDeviceInfo();
	data.fileIn = file;
	FFmpegReader reader;
	reader.open(file);
	data.validate(reader);
	OutputWriter writer(data, reader);
	MovieFrame frame(data, reader, writer);
	CpuFrame ex(data, *data.deviceList[0], frame, frame.mPool);
	reader.read(frame.mBufferFrame);
	ex.inputData(reader.frameIndex, frame.mBufferFrame);
	ex.createPyramid(reader.frameIndex, {}, false);

	reader.read(frame.mBufferFrame);
	ex.inputData(reader.frameIndex, frame.mBufferFrame);
	ex.createPyramid(reader.frameIndex, {}, false);

	ex.computeStart(reader.frameIndex, frame.mResultPoints);
	ex.computeTerminate(reader.frameIndex, frame.mResultPoints);

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

	frame.mFrameResult.computeTransform(frame.mResultPoints, reader.frameIndex);
	AffineTransform trf = frame.getTransform();
	trf.toConsole("transform: ", 2);

	std::chrono::time_point t2 = std::chrono::system_clock::now();
	std::cout << "time [ms]: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << std::endl;
	std::cout << errorLogger().getErrorMessage() << std::endl;
}

void testZoom() {
	std::cout << "testing zoom calculation..." << std::endl;
	MainData data;
	Trajectory t;

	double x = 0.0;
	double y = 0.0;
	double deg = 15.0;
	double rad = deg * std::numbers::pi / 180.0;
	double z = t.calcRequiredZoom(x, y, rad, 100, 50);
	std::cout << z << std::endl;
}

void testSampler() {
	std::vector<int> data(10'000);
	std::iota(data.begin(), data.end(), 0);
	std::vector<int> samples(4);

	//std::shared_ptr<SamplerBase<int>> sampler = std::make_shared<Sampler<int, std::random_device>>();
	std::shared_ptr<SamplerBase<int>> sampler = std::make_shared<Sampler<int, PseudoRandomSource>>();

	for (int i = 0; i < 3; i++) {
		sampler->sample(data, samples);
		for (int i = 0; i < samples.size(); i++) std::cout << samples[i] << " ";
		std::cout << std::endl;
	}
}