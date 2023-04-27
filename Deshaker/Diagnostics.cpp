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

#include "Diagnostics.hpp"
#include "MovieWriter.hpp"
#include <format>

ResultDetailsFile::ResultDetailsFile(const std::string& filename, const char delimiter) : std::ofstream(filename), delimiter { delimiter } {
	if (is_open()) {
		*this << "frameIdx" << delimiter << "ix0" << delimiter << "iy0" 
			<< delimiter << "px" << delimiter << "py" << delimiter << "u" << delimiter << "v" 
			<< delimiter << "isValid" << delimiter << "isConsens" << std::endl;

	} else {
		throw AVException("cannot open file '" + filename + "'");
	}
}

void ResultDetailsFile::write(const std::vector<PointResult>& results, int64_t frameIndex) {
	//for better performace first write into buffer string
	std::stringstream ss;
	for (auto& item : results) {
		ss << frameIndex << delimiter << item.ix0 << delimiter << item.iy0 << delimiter << item.px << delimiter << item.py << delimiter
			<< item.u << delimiter << item.v << delimiter << item.resultValue() << std::endl;
	}
	//write buffer to file
	*this << ss.str();
}

void ResultDetailsFile::run(const FrameResult& fr, int64_t frameIndex) {
	write(fr.mFiniteResults, frameIndex);
}

TransformsFile::TransformsFile(const std::string& filename, std::ios::openmode mode) : std::fstream(filename, mode) {
	if (is_open() && (mode & std::ios::out)) {
		//write signature
		*this << id;

	}  else if (is_open() && (mode & std::ios::in)) {
		//read and check signature
		std::string str = "    ";
		get(str.data(), 5);
		if (str != id) {
			errorLogger.logError("transforms file '" + filename + "' is not valid");
			close();
		}

	} else {
		throw AVException("error opening file '" + filename + "'");
	}
}

void TransformsFile::writeTransform(const Affine2D& transform, int64_t frameIndex) {
	writeValue(frameIndex);
	writeValue(transform.scale());
	writeValue(transform.dX());
	writeValue(transform.dY());
	writeValue(transform.rotMilliDegrees());
}

std::map<int64_t, TransformValues> TransformsFile::readTransformMap() {
	std::map<int64_t, TransformValues> transformsMap;
	if (is_open()) {
		while (!eof()) {
			int64_t frameIdx = 0;
			double s = 0, dx = 0, dy = 0, da = 0;
			readValue(frameIdx);
			readValue(s);
			readValue(dx);
			readValue(dy);
			readValue(da);

			transformsMap[frameIdx] = { s, dx, dy, da / 3600.0 * std::numbers::pi / 180.0 };
		}
	}
	return transformsMap;
}

void TransformsFile::run(const FrameResult& fr, int64_t frameIndex) {
	writeTransform(fr.mTransform, frameIndex);
}

void ResultImage::write(const FrameResult& fr, int64_t idx, const ImageYuv& yuv, const std::string& fname) {
	const AffineTransform& trf = fr.mTransform;

	//copy and scale Y plane to first color plane of bgr
	yuv.scaleTo(0, bgr, 0);
	//copy planes in bgr image making it grayscale bgr
	for (int z = 1; z < 3; z++) {
		for (int r = 0; r < bgr.h; r++) {
			for (int c = 0; c < bgr.w; c++) {
				bgr.at(z, r, c) = bgr.at(0, r, c);
			}
		}
	}
		
	//draw lines
	int numValid = (int) fr.mCountFinite;
	int numConsens = 0;
	for (int i = 0; i < numValid; i++) {
		const PointResult& pr = fr.mFiniteResults[i];
		double x2 = pr.px + pr.u;
		double y2 = pr.py + pr.v;
		ImageColor col = ColorBgr::RED;
		if (pr.distance < data.cConsensDistance) {
			col = ColorBgr::GREEN;
			numConsens++;
		}
		//blue line to computed transformation
		auto [tx, ty] = trf.transform(pr.x, pr.y);
		bgr.drawLine(pr.px, pr.py, tx + bgr.w / 2.0, ty + bgr.h / 2.0, ColorBgr::BLUE);

		//red or green if point is consens
		bgr.drawLine(pr.px, pr.py, x2, y2, col);
		bgr.drawDot(x2, y2, 1.25, 1.25, col);
	}

	//write text info
	int textScale = bgr.h / 540;
	double frac = numValid == 0 ? 0.0 : 100.0 * numConsens / numValid;
	std::string s1 = std::format("index {}, consensus {}/{} ({:.0f}%)", idx, numConsens, numValid, frac);
	bgr.writeText(s1, 0, bgr.h - textScale * 20, textScale, textScale, ColorBgr::WHITE, ColorBgr::BLACK);
	std::string s2 = std::format("transform dx={:.1f}, dy={:.1f}, scale={:.5f}, rot={:.1f}", trf.dX(), trf.dY(), trf.scale(), trf.rotMilliDegrees());
	bgr.writeText(s2, 0, bgr.h - textScale * 10, textScale, textScale, ColorBgr::WHITE, ColorBgr::BLACK);

	//save image to file
	bool result = bgr.saveAsBMP(fname);
	if (result == false) {
		errorLogger.logError("cannot write file '" + fname + "'");
	}
}

void ResultImage::run(const FrameResult& fr, int64_t frameIndex) {
	//get input image from buffers
	ImageYuv yuv = imageGetter(frameIndex);
	std::string fname = ImageWriter::makeFilename(data.resultImageFile, frameIndex);
	write(fr, frameIndex, yuv, fname);
}
