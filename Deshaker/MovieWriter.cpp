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

#include "MovieWriter.hpp"
#include "MovieFrame.hpp"


void AuxWriters::writeAll(const MovieFrame& frame) {
	for (auto it = begin(); it != end(); it++) {
		it->get()->write(frame);
	}
}

OutputContext MovieWriter::getOutputContext() {
	return { false, false, nullptr, nullptr };
}

std::future<void> MovieWriter::writeAsync(const MovieFrame& frame) {
	return std::async(std::launch::async, [&] { write(frame); });
}

OutputContext StandardMovieWriter::getOutputContext() {
	return { true, false, &outputFrame, nullptr };
}

std::string ImageWriter::makeFilename(const std::string& pattern, int64_t index) {
	const int siz = 256;
	char fname[siz];
	std::snprintf(fname, siz, pattern.c_str(), index);
	return fname;
}

std::string ImageWriter::makeFilename() const {
	return makeFilename(mData.fileOut, this->frameIndex);
}

//-----------------------------------------------------------------------------------
// BMP Images
//-----------------------------------------------------------------------------------

void BmpImageWriter::write(const MovieFrame& frame) {
	outputFrame.toBGR(image).saveAsBMP(makeFilename());
	outputBytesWritten += image.dataSizeInBytes();
	this->frameIndex++;
}

//-----------------------------------------------------------------------------------
// JPG Images through ffmpeg
//-----------------------------------------------------------------------------------

void JpegImageWriter::open(EncodingOption videoCodec) {
	const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
	ctx = avcodec_alloc_context3(codec);
	ctx->width = mData.w;
	ctx->height = mData.h;
	ctx->time_base = { 1, 1 };
	ctx->framerate = { 1, 1 };
	ctx->codec_type = AVMEDIA_TYPE_VIDEO;
	ctx->pix_fmt = AV_PIX_FMT_YUV444P;
	ctx->color_range = AVCOL_RANGE_JPEG;
	int ret = avcodec_open2(ctx, codec, NULL);
	if (ret < 0)
		throw AVException(av_make_error(ret, "cannot open mjpeg codec"));

	av_frame = av_frame_alloc();
	av_frame->format = ctx->pix_fmt;
	av_frame->width = mData.w;
	av_frame->height = mData.h;

	av_frame->linesize[0] = outputFrame.stride;
	av_frame->linesize[1] = outputFrame.stride;
	av_frame->linesize[2] = outputFrame.stride;
	av_frame->data[0] = outputFrame.plane(0);
	av_frame->data[1] = outputFrame.plane(1);
	av_frame->data[2] = outputFrame.plane(2);

	packet = av_packet_alloc();
}

void JpegImageWriter::write(const MovieFrame& frame) {
	av_frame->pts = this->frameIndex;
	int result = avcodec_send_frame(ctx, av_frame);
	if (result < 0)
		errorLogger.logError(av_make_error(result, "error sending frame"));

	result = avcodec_receive_packet(ctx, packet);
	if (result < 0)
		errorLogger.logError(av_make_error(result, "error receiving packet"));

	std::string fname = makeFilename();
	std::ofstream file(fname, std::ios::binary);
	if (file)
		file.write(reinterpret_cast<char*>(packet->data), packet->size);
	else
		errorLogger.logError("error opening file '" + fname + "'");

	outputBytesWritten += packet->size;
	av_packet_unref(packet);
	this->frameIndex++;
}

JpegImageWriter::~JpegImageWriter() {
	av_packet_free(&packet);
	avcodec_free_context(&ctx);
}

//-----------------------------------------------------------------------------------
// YUV444 packed without striding pixels
//-----------------------------------------------------------------------------------

void RawWriter::packYuv() {
	unsigned char* yuv = outputFrame.data();
	char* dest = yuvPacked.data();

	for (int i = 0; i < 3ull * outputFrame.h; i++) {
		std::copy(yuv, yuv + outputFrame.w, dest);
		yuv += outputFrame.stride;
		dest += outputFrame.w;
	}
}


//-----------------------------------------------------------------------------------
// Computed Transformation Values
//-----------------------------------------------------------------------------------

std::map<int64_t, TransformValues> TransformsFile::readTransformMap(const std::string& trajectoryFile) {
	std::map<int64_t, TransformValues> transformsMap;
	std::ifstream file(trajectoryFile, std::ios::binary);
	if (file.is_open()) {
		//read and check signature
		std::string str = "    ";
		file.get(str.data(), 5);
		if (str != id) {
			errorLogger.logError("transforms file '" + trajectoryFile + "' is not valid");

		} else {
			while (!file.eof()) {
				int64_t frameIdx = 0;
				double s = 0, dx = 0, dy = 0, da = 0;
				file.read(reinterpret_cast<char*>(&frameIdx), sizeof(frameIdx));
				file.read(reinterpret_cast<char*>(&s), sizeof(s));
				file.read(reinterpret_cast<char*>(&dx), sizeof(dx));
				file.read(reinterpret_cast<char*>(&dy), sizeof(dy));
				file.read(reinterpret_cast<char*>(&da), sizeof(da));

				transformsMap[frameIdx] = { s, dx, dy, da / 3600.0 * std::numbers::pi / 180.0 };
			}
		}

	} else {
		errorLogger.logError("cannot open transforms file '" + trajectoryFile + "'");
	}
	return transformsMap;
}

void TransformsWriterMain::open(EncodingOption videoCodec) {
	TransformsFile::open();
}

void TransformsWriterMain::write(const MovieFrame& frame) {
	writeTransform(frame.mFrameResult.transform(), frameIndex);
	this->frameIndex++;
	this->outputBytesWritten = file.tellp();
}

void TransformsFile::open() {
	file = std::ofstream(mData.trajectoryFile, std::ios::binary);
	if (file.is_open()) {
		//write signature
		file << id;

	} else {
		throw AVException("error opening file '" + mData.trajectoryFile + "'");
	}
}

void TransformsFile::writeTransform(const Affine2D& transform, int64_t frameIndex) {
	writeValue(frameIndex);
	writeValue(transform.scale());
	writeValue(transform.dX());
	writeValue(transform.dY());
	writeValue(transform.rotMilliDegrees());
}

void AuxTransformsWriter::open() {
	TransformsFile::open();
}

void AuxTransformsWriter::write(const MovieFrame& frame) {
	writeTransform(frame.mFrameResult.transform(), frameIndex);
	this->frameIndex++;
}


//-----------------------------------------------------------------------------------
// Computed Results per Point
//-----------------------------------------------------------------------------------

void  ResultDetailsWriter::open() {
	file = std::ofstream(mData.resultsFile);
	if (file.is_open()) {
		file << "frameIdx" << delimiter << "ix0" << delimiter << "iy0"
			<< delimiter << "px" << delimiter << "py" << delimiter << "u" << delimiter << "v"
			<< delimiter << "isValid" << delimiter << "isConsens" << std::endl;

	} else {
		throw AVException("cannot open file '" + mData.resultsFile + "'");
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
	std::string fname = ImageWriter::makeFilename(mData.resultImageFile, frameIndex);
	write(frame.mFrameResult, frameIndex, yuv, fname);
	this->frameIndex++;
}
