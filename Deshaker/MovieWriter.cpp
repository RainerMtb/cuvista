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

#include <filesystem>
#include "MovieWriter.hpp"
#include "MovieFrame.hpp"


void AuxWriters::writeAll(const FrameExecutor& executor) {
	for (auto it = begin(); it != end(); it++) {
		it->get()->write(executor);
	}
}

void BaseWriter::prepareOutput(FrameExecutor& executor) {
	executor.getOutput(frameIndex, outputFrame);
}

void BaseWriter::write(const FrameExecutor& executor) {
	this->frameIndex++;
}

std::string ImageWriter::makeFilename(const std::string& pattern, int64_t index, const std::string& extension) {
	namespace fs = std::filesystem;
	fs::path out;

	if (pattern.empty() == false && fs::is_directory(pattern)) {
		std::string file = std::format("im{:04d}.{}", index, extension);
		out = fs::path(pattern) / fs::path(file);

	} else {
		const int siz = 512;
		char fname[siz];
		std::snprintf(fname, siz, pattern.c_str(), index);
		out = fs::path(fname);
	}
	return out.make_preferred().string();
}

std::string ImageWriter::makeFilenameSamples(const std::string& pattern, const std::string& extension) {
	std::string str = makeFilename(pattern, 0, extension) + ", " + makeFilename(pattern, 1, extension) + ", " + makeFilename(pattern, 2, extension);
	return str.substr(0, 100) + ", ...";
}

std::string ImageWriter::makeFilename(const std::string& extension) const {
	return makeFilename(mData.fileOut, this->frameIndex, extension);
}

//-----------------------------------------------------------------------------------
// BMP Images
//-----------------------------------------------------------------------------------

void BmpImageWriter::prepareOutput(FrameExecutor& executor) {
	worker.join();
	executor.getOutput(frameIndex, image);
}

void BmpImageWriter::write(const FrameExecutor& executor) {
	std::string fname = makeFilename("bmp");
	worker = std::jthread([&, fname] { image.saveAsColorBMP(fname); });
	outputBytesWritten += image.bytes();
	this->frameIndex++;
}

BmpImageWriter::~BmpImageWriter() {
	worker.join();
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

void JpegImageWriter::write(const FrameExecutor& executor) {
	av_frame->pts = this->frameIndex;
	int result = avcodec_send_frame(ctx, av_frame);
	if (result < 0)
		errorLogger.logError(av_make_error(result, "error sending frame"));

	result = avcodec_receive_packet(ctx, packet);
	if (result < 0)
		errorLogger.logError(av_make_error(result, "error receiving packet"));

	std::string fname = makeFilename("jpg");
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

				transformsMap[frameIdx] = { s, dx, dy, da / 60.0 * std::numbers::pi / 180.0 };
			}
		}

	} else {
		errorLogger.logError("cannot open transforms file '" + trajectoryFile + "'");
	}
	return transformsMap;
}

void TransformsFile::open(const std::string& trajectoryFile) {
	file = std::ofstream(trajectoryFile, std::ios::binary);
	if (file.is_open()) {
		//write signature
		file << id;

	} else {
		throw AVException("error opening file '" + trajectoryFile + "'");
	}
}

void TransformsFile::writeTransform(const Affine2D& transform, int64_t frameIndex) {
	writeValue(frameIndex);
	writeValue(transform.scale());
	writeValue(transform.dX());
	writeValue(transform.dY());
	writeValue(transform.rotMinutes());
}

void TransformsWriterMain::open(EncodingOption videoCodec) {
	TransformsFile::open(mData.trajectoryFile);
}

void TransformsWriterMain::write(const FrameExecutor& executor) {
	writeTransform(movieFrame->mFrameResult.getTransform(), frameIndex);
	this->frameIndex++;
	this->outputBytesWritten = file.tellp();
}

void AuxTransformsWriter::open() {
	TransformsFile::open(mData.trajectoryFile);
}

void AuxTransformsWriter::write(const FrameExecutor& executor) {
	writeTransform(executor.mFrame.mFrameResult.getTransform(), frameIndex);
	this->frameIndex++;
}


//-----------------------------------------------------------------------------------
// Optical Flow Video
//-----------------------------------------------------------------------------------

void OpticalFlowWriter::open() {
	open(mData.flowFile);
}

void OpticalFlowWriter::vectorToColor(double dx, double dy, unsigned char* r, unsigned char* g, unsigned char* b) {
	const double f = 20.0;
	double hue = std::atan2(dy, dx) / std::numbers::pi * 180.0 + 180.0;
	double val = std::clamp(std::sqrt(dx * dx + dy * dy) / f, 0.0, 1.0);
	im::hsv_to_rgb(hue, 1.0, val, r, g, b);
}

void OpticalFlowWriter::open(const std::string& sourceName) {
	//simplified ffmpeg video setup
	av_log_set_callback(ffmpeg_log);

	const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
	if (!codec)
		throw AVException("Could not find encoder");

	codec_ctx = avcodec_alloc_context3(codec);
	if (!codec_ctx)
		throw AVException("Could not allocate encoder context");

	//setup output file
	int result = avformat_alloc_output_context2(&fmt_ctx, NULL, NULL, sourceName.c_str());
	if (result < 0)
		throw AVException(av_make_error(result));

	videoStream = avformat_new_stream(fmt_ctx, NULL);
	if (!videoStream)
		throw AVException("could not create stream");

	codec_ctx->width = mData.w;
	codec_ctx->height = mData.h;
	codec_ctx->pix_fmt = pixfmt;
	codec_ctx->time_base = timeBase;
	codec_ctx->gop_size = GOP_SIZE;

	FFmpegWriter::openEncoder(codec, sourceName);

	//setup scaler to accept RGB
	sws_scaler_ctx = sws_getContext(mData.w, mData.h, AV_PIX_FMT_RGBA, mData.w, mData.h, pixfmt, SWS_BILINEAR, NULL, NULL, NULL);
	if (!sws_scaler_ctx)
		throw AVException("Could not get scaler context");

	//setup legend image showing flow colors
	double r = legendSize / 2.0;
	for (int y = 0; y < legendSize; y++) {
		double dy = r - y;
		for (int x = 0; x < legendSize; x++) {
			double dx = r - x;
			double dist = std::sqrt(dx * dx + dy * dy);
			if (dist < r) {
				legend.at(3, y, x) = 0xFF;
				vectorToColor(dx, -dy, legend.addr(0, y, x), legend.addr(1, y, x), legend.addr(2, y, x));

			} else {
				legend.at(3, y, x) = 0;
			}
		}
	}
}

void OpticalFlowWriter::writeFlow(const MovieFrame& frame) {
	std::vector<unsigned char> colorGray = { 128, 128, 128 };

	//convert vectors to hsv color values and then to rgb planar image
	std::vector<PointResult> results = frame.mResultPoints;
	for (const PointResult& pr : results) {
		if (pr.isValid()) {
			vectorToColor(-pr.u, pr.v, imageResults.addr(0, pr.iy0, pr.ix0), imageResults.addr(1, pr.iy0, pr.ix0), imageResults.addr(2, pr.iy0, pr.ix0));

		} else {
			//invalid result is gray pixel
			imageResults.setPixel(pr.iy0, pr.ix0, colorGray);
		}
	}

	//interpolate image
	auto fcn = [&] (size_t iy) {
		int div = 1 << mData.zMax;
		double y = 1.0 * iy / div - 1.0 - mData.ir;
		for (size_t ix = 0; ix < mData.w; ix++) {
			double x = 1.0 * ix / div - 1.0 - mData.ir;
			for (int i = 0; i < 3; i++) {
				unsigned char col = colorGray.at(i);
				if (x >= -0.5 && x < mData.ixCount + 0.5 && y >= -0.5 && y < mData.iyCount + 0.5) col = imageResults.sample(i, x, y);
				imageInterpolated.at(i, iy, ix) = col;
			}
		}
	};
	frame.mPool.addAndWait(fcn, 0, mData.h);
}

void OpticalFlowWriter::writeAVFrame(AVFrame* av_frame) {
	int result = avcodec_send_frame(codec_ctx, av_frame);
	if (result < 0)
		ffmpeg_log_error(result, "error encoding flow #1");

	while (result >= 0) {
		result = avcodec_receive_packet(codec_ctx, videoPacket);
		if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
			//do not report error here, need more frame data or end of file

		} else if (result < 0) {
			//report error, something wrong
			ffmpeg_log_error(result, "error encoding flow #2");

		} else {
			av_packet_rescale_ts(videoPacket, timeBase, videoStream->time_base);
			result = av_interleaved_write_frame(fmt_ctx, videoPacket);
			if (result < 0) {
				errorLogger.logError(av_make_error(result, "error writing flow packet"));
			}
		}
	}
}

void OpticalFlowWriter::write(const FrameExecutor& executor) {
	writeFlow(executor.mFrame);
	//imageInterpolated.saveAsBMP("f:/im.bmp");

	//stamp color legend onto image
	legend.copyTo(imageInterpolated, 0ull + mData.h - 10 - legendSize, 10);
	int tx = 10 + legendSize / 2;
	int ty = 0ull + mData.h - 10 - legendSize / 2;
	imageInterpolated.writeText(std::to_string(frameIndex), tx, ty, 1, 1, im::TextAlign::MIDDLE_CENTER, im::ColorRGBA::WHITE, im::ColorRGBA::BLACK);

	//encode rgba image
	uint8_t* src[] = { imageInterpolated.data(), nullptr, nullptr, nullptr};
	int strides[] = { imageInterpolated.stride, 0, 0, 0 };
	int sliceHeight = sws_scale(sws_scaler_ctx, src, strides, 0, imageInterpolated.h, av_frame->data, av_frame->linesize);

	av_frame->pts = frameIndex;
	writeAVFrame(av_frame);
	frameIndex++;
}

OpticalFlowWriter::~OpticalFlowWriter() {
	writeAVFrame(nullptr);
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

void ResultDetailsWriter::write(const FrameExecutor& executor) {
	write(executor.mFrame.mResultPoints, frameIndex);
	this->frameIndex++;
}


//-----------------------------------------------------------------------------------
// Result Images
//-----------------------------------------------------------------------------------

void ResultImageWriter::write(const AffineTransform& trf, const std::vector<PointResult>& res, int64_t idx, const ImageYuv& yuv, const std::string& fname) {
	using namespace im;

	//copy and scale Y plane to first color plane of bgr
	yuv.scaleByTwo(0, bgr, 0);
	//copy planes in bgr image making it grayscale bgr
	for (int z = 1; z < 3; z++) {
		for (int r = 0; r < bgr.h; r++) {
			for (int c = 0; c < bgr.w; c++) {
				bgr.at(z, r, c) = bgr.at(0, r, c);
			}
		}
	}

	//draw lines
	//draw blue lines first
	for (const PointResult& pr : res) {
		if (pr.isValid()) {
			double x2 = pr.px + pr.u;
			double y2 = pr.py + pr.v;

			//blue line to computed transformation
			auto [tx, ty] = trf.transform(pr.x, pr.y);
			bgr.drawLine(pr.px, pr.py, tx + bgr.w / 2.0, ty + bgr.h / 2.0, ColorBgr::BLUE, 0.5);
		}
	}

	//draw on top
	//green line if point is consens
	//red line if point is not consens
	int numValid = 0, numConsens = 0;
	ImageColor col;
	for (const PointResult& pr : res) {
		if (pr.isValid()) {
			numValid++;
			double x2 = pr.px + pr.u;
			double y2 = pr.py + pr.v;

			if (pr.isConsens) {
				col = ColorBgr::GREEN;
				numConsens++;

			} else {
				col = ColorBgr::RED;
			}
			bgr.drawLine(pr.px, pr.py, x2, y2, col);
			bgr.drawDot(x2, y2, 1.25, 1.25, col);
		}
	}

	//write text info
	int textScale = bgr.h / 540;
	double frac = numValid == 0 ? 0.0 : 100.0 * numConsens / numValid;
	std::string s1 = std::format("index {}, consensus {}/{} ({:.0f}%)", idx, numConsens, numValid, frac);
	bgr.writeText(s1, 0, bgr.h - textScale * 20, textScale, textScale, TextAlign::TOP_LEFT, ColorBgr::WHITE, ColorBgr::BLACK);
	std::string s2 = std::format("transform dx={:.1f}, dy={:.1f}, scale={:.5f}, rot={:.1f}", trf.dX(), trf.dY(), trf.scale(), trf.rotMinutes());
	bgr.writeText(s2, 0, bgr.h - textScale * 10, textScale, textScale, TextAlign::TOP_LEFT, ColorBgr::WHITE, ColorBgr::BLACK);

	//save image to file
	bool result = bgr.saveAsColorBMP(fname);
	if (result == false) {
		errorLogger.logError("cannot write file '" + fname + "'");
	}
}

void ResultImageWriter::write(const FrameExecutor& executor) {
	//get input image from buffers
	executor.getInput(frameIndex, yuv);
	std::string fname = ImageWriter::makeFilename(mData.resultImageFile, frameIndex, "bmp");
	write(executor.mFrame.getTransform(), executor.mFrame.mResultPoints, frameIndex, yuv, fname);
	this->frameIndex++;
}
