#include "AuxiliaryWriter.hpp"
#include "MovieFrame.hpp"


//-----------------------------------------------------------------------------------
// Computed Results per Point
//-----------------------------------------------------------------------------------

void  ResultDetailsWriter::open() {
	file = std::ofstream(mAuxData.resultsFile);
	if (file.is_open()) {
		file << "frameIdx" << delimiter << "ix0" << delimiter << "iy0"
			<< delimiter << "px" << delimiter << "py" << delimiter << "u" << delimiter << "v"
			<< delimiter << "isValid" << delimiter << "isConsens" << std::endl;

	} else {
		throw AVException("cannot open file '" + mAuxData.resultsFile + "'");
	}
}

void ResultDetailsWriter::write(const std::vector<PointResult>& results, int64_t frameIndex) {
	//for better performace first write into buffer string
	std::stringstream ss;
	for (auto& item : results) {
		ss << frameIndex << delimiter << item.ix0 << delimiter << item.iy0 << delimiter << item.px << delimiter << item.py << delimiter
			<< item.u << delimiter << item.v << delimiter << item.resultValue() << std::endl;
	}
	//write buffer to file
	file << ss.str();
}

void ResultDetailsWriter::write(const MovieFrame& frame) {
	write(frame.mFrameResult.mFiniteResults, frameIndex);
	this->frameIndex++;
}


//-----------------------------------------------------------------------------------
// Result Images
//-----------------------------------------------------------------------------------

void ResultImageWriter::write(const FrameResult& fr, int64_t idx, const ImageYuv& yuv, const std::string& fname) {
	const AffineTransform& trf = fr.transform();

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
	//green line -> consensus point
	//red line -> out of consens
	//blue line -> computed transform
	int numValid = (int) fr.mCountFinite;
	int numConsens = (int) fr.mCountConsens;
	for (int i = 0; i < numValid; i++) {
		const PointResult& pr = fr.mFiniteResults[i];
		double x2 = pr.px + pr.u;
		double y2 = pr.py + pr.v;

		//red or green if point is consens
		ImageColor col = i < numConsens ? ColorBgr::GREEN : ColorBgr::RED;
		bgr.drawLine(pr.px, pr.py, x2, y2, col);
		bgr.drawDot(x2, y2, 1.25, 1.25, col);

		//blue line to computed transformation
		auto [tx, ty] = trf.transform(pr.x, pr.y);
		bgr.drawLine(pr.px, pr.py, tx + bgr.w / 2.0, ty + bgr.h / 2.0, ColorBgr::BLUE);
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

void ResultImageWriter::write(const MovieFrame& frame) {
	//get input image from buffers
	ImageYuv yuv = frame.getInput(frameIndex);
	std::string fname = ImageWriter::makeFilename(mAuxData.resultImageFile, frameIndex);
	write(frame.mFrameResult, frameIndex, yuv, fname);
	this->frameIndex++;
}
