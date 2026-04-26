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
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 10.0);
	Matd::precision(8);

	//load PointResults
	int w = 100;
	int h = 75;
	int n = w * h;
	std::vector<PointResult> points(n);
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			PointResult& pr = points[1ll * y * w + x];
			pr.x = x;
			pr.y = y;
			pr.u = 4 + std::sin(x) * dis(gen);
			pr.v = 3 + std::sin(y) * dis(gen);
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

	for (int i = 0; i < 15; i++) {
		std::ranges::shuffle(points, gen);

		AffineSolverSimple s1(points.size());
		AffineSolver& solver1 = s1;
		Matd t1 = solver1.computeSimilar(points).toParamsMat();

		ThreadPool thr(8);
		AffineSolverFast s2(thr, points.size());
		AffineSolver& solver2 = s2;
		Matd t2 = solver2.computeSimilar(points).toParamsMat();

		AffineSolverAvx s3(points.size());
		AffineSolver& solver3 = s3;
		Matd t3 = solver3.computeSimilar(points).toParamsMat();

		std::cout << "results equal 1-2: " << t1.minus(t2).timesTransposed().scalar() << ", ";
		std::cout << "results equal 1-3: " << t1.minus(t3).timesTransposed().scalar() << std::endl;
	}
}

void readAndWriteOneFrame() {
	MainData data;
	data.deviceInfoCuda = data.probeCuda();
	data.collectDeviceInfo();
	{
		NullReader reader;
		reader.w = 1920;
		reader.h = 1080;
		data.validate(reader);
		OutputWriter writer(data, reader);
		MovieFrame frame(data, reader, writer);
		CudaFrame ex(data, *data.deviceList[2], frame, frame.mPool);

		ImageVuyx im0 = ImageVuyx::readPgmFile("d:/VideoTest/v00.pgm");
		Image8& input0 = ex.inputDestination(0);
		im0.copyTo(input0);
		input0.index = 0;
		reader.frameIndex = 0;
		ex.inputData(reader.frameIndex);
		std::vector<int> hist1(256);
		ex.createPyramid(frame.mReader.frameIndex, hist1, {}, false);

		ImageVuyx im1 = ImageVuyx::readPgmFile("D:/VideoTest/v01.pgm");
		Image8& input1 = ex.inputDestination(1);
		im1.copyTo(input1);
		input1.index = 1;
		reader.frameIndex = 1;
		ex.inputData(reader.frameIndex);
		std::vector<int> hist2(256);
		ex.createPyramid(frame.mReader.frameIndex, hist2, {}, false);
		ex.computeStart(frame.mReader.frameIndex, frame.mResultPoints);
		ex.computeTerminate(frame.mReader.frameIndex, frame.mResultPoints);

		AffineTransform trf;
		trf.addRotation(0.3).addTranslation(-200, 100);
		trf.frameIndex = 0;
		ex.outputData(0, trf);
		writer.writeOutput(ex);
		std::string fileOut = "f:/test.bmp";
		std::cout << "writing file " << fileOut << std::endl;
		writer.getOutputFrame().saveBmpColor(fileOut);
	}
	std::cout << errorLogger().getErrorMessage() << std::endl;
}

void checkVersions() {
	std::cout << "check cuda devices" << std::endl;
	try {
		MainData data;
		data.deviceInfoCuda = data.probeCuda();
		data.deviceInfoOpenCl = data.probeOpenCl();
		data.showDeviceInfo();
	} catch (CancelException ignore) {}
}

void draw(const std::string& filename) {
	using namespace im;

	//draw into bgr image
	ImageBgr bgr(600, 800);
	bgr.setColor(Color::GRAY);
	bgr.writeText("abcdefghijklmnopqrstuvwxyz", 10, 10, TextAlign::TOP_LEFT, 2, 3);
	bgr.writeText("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10, 50, TextAlign::TOP_LEFT, 2, 3);
	bgr.writeText("A quick brown fox      (yellow)", 10, 90, TextAlign::TOP_LEFT, 2, 3, Color::YELLOW, Color::BLACK);
	bgr.writeText("jumps over a lazy dog   (white)", 10, 120, TextAlign::TOP_LEFT, 2, 3, Color::rgba(255, 255, 255, 0.75), Color::GREEN);
	bgr.writeText("A quick brown fox      (yellow)", 10, 150, TextAlign::TOP_LEFT, 2, 3, Color::YELLOW, Color::rgba(0, 255, 255, 0.25));
	bgr.writeText("jumps over a lazy dog (magenta)", 10, 180, TextAlign::TOP_LEFT, 2, 3, Color::rgba(255, 0, 255, 0.5), Color::rgba(0, 255, 255, 0.5));

	//charachter map
	int x0 = 500;
	int y0 = 20;
	for (int i = 0; i < 16; i++) {
		for (int k = 0; k < 16; k++) {
			char ch = i * 16 + k;
			std::string str({ ch });
			bgr.writeText(str, x0 + i * 16, y0 + k * 22, TextAlign::TOP_LEFT, 2, 2);
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
		bgr.drawLine(cx, cy, x1, y1, Color::BLUE, 1.0 - angle / 360.0);
		bgr.drawMarker(x1, y1, Color::RED, r);
	}

	bgr.drawLine(150, 550, 250, 550, Color::BLUE, 1.0);

	//lines with alpha
	Color col = Color::web("#F09B59");
	for (int i = 0; i < 15; i++) {
		double x = 400.0 + i * 20.0;
		bgr.drawLine(x, 400.0, x + 40.0, 500.0, col, i / 15.0);
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
	bgr.saveBmpColor(filename);
}

class OpticalFlowImageWriter : public OpticalFlowWriter {
public:
	OpticalFlowImageWriter(MainData& data, MovieReader& reader) :
		OpticalFlowWriter(data, reader) {
		codec_ctx = avcodec_alloc_context3(NULL);
	}

	void writeFlow(const MovieFrame& frame, const std::string& fileName) {
		OpticalFlowWriter::writeFlow(frame);
		imageInterpolated.saveBmpColor("f:/flow.bmp");
	}
};

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
	std::shared_ptr<SamplerBase<int>> sampler = std::make_shared<UrbgSampler<int, PseudoRandomSource>>();

	for (int i = 0; i < 3; i++) {
		sampler->sample(data, samples);
		for (int i = 0; i < samples.size(); i++) std::cout << samples[i] << " ";
		std::cout << std::endl;
	}
}