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

std::future<void> MovieWriter::writeAsync() {
	return std::async(std::launch::async, [&] { write(); });
}

OutputContext NullWriter::getOutputContext() {
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

void BmpImageWriter::write() {
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

void JpegImageWriter::write() {
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
	writeValue(transform.rotMilliDegrees());
}

void TransformsWriterMain::open(EncodingOption videoCodec) {
	TransformsFile::open(mData.trajectoryFile);
}

void TransformsWriterMain::write() {
	writeTransform(movieFrame->mFrameResult.getTransform(), frameIndex);
	this->frameIndex++;
	this->outputBytesWritten = file.tellp();
}

void AuxTransformsWriter::open() {
	TransformsFile::open(mAuxData.trajectoryFile);
}

void AuxTransformsWriter::write(const MovieFrame& frame) {
	writeTransform(frame.mFrameResult.getTransform(), frameIndex);
	this->frameIndex++;
}


//-----------------------------------------------------------------------------------
// Optical Flow Video
//-----------------------------------------------------------------------------------

void OpticalFlowWriter::open() {
	open(mData.flowFile);
}

void OpticalFlowWriter::vectorToColor(double dx, double dy, unsigned char* r, unsigned char* g, unsigned char* b) {
	double hue = std::atan2(dy, dx) / std::numbers::pi * 180.0 + 180.0;
	double val = std::clamp(std::sqrt(sqr(dx) + sqr(dy)) / 15.0, 0.0, 1.0);
	hsv_to_rgb(hue, 1.0, val, r, g, b);
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
	sws_scaler_ctx = sws_getContext(mData.w, mData.h, AV_PIX_FMT_BGR24, mData.w, mData.h, pixfmt, SWS_BICUBIC, NULL, NULL, NULL);
	if (!sws_scaler_ctx)
		throw AVException("Could not get scaler context");

	//setup legend image showing flow colors
	double r = legendSize / 2.0;
	for (int y = 0; y < legendSize; y++) {
		double dy = r - y;
		for (int x = 0; x < legendSize; x++) {
			double dx = r - x;
			double dist = std::sqrt(sqr(dx) + sqr(dy));
			if (dist < r) {
				legendMask.setPixel(y, x, { 1 });
				vectorToColor(dx, -dy, legend.addr(2, y, x), legend.addr(1, y, x), legend.addr(0, y, x));
			}
		}
	}
}

void OpticalFlowWriter::writeFlow(const MovieFrame& frame) {
	//convert vectors to hsv color values and then to rgb planar image
	std::vector<PointResult> results = frame.mResultPoints;
	for (const PointResult& pr : results) {
		if (pr.isValid()) {
			vectorToColor(-pr.u, pr.v, imageResults.addr(2, pr.iy0, pr.ix0), imageResults.addr(1, pr.iy0, pr.ix0), imageResults.addr(0, pr.iy0, pr.ix0));

		} else {
			//invalid result is gray pixel
			imageResults.setPixel(pr.iy0, pr.ix0, { 128, 128, 128 });
		}
	}

	//interpolate image
	auto fcn = [&] (size_t iy) {
		int div = 1 << mData.zMax;
		double y = 1.0 * iy / div - 1.0 - mData.ir;
		for (size_t ix = 0; ix < mData.w; ix++) {
			double x = 1.0 * ix / div - 1.0 - mData.ir;
			imageInterpolated.at(0, iy, ix) = imageResults.sample(0, x, y);
			imageInterpolated.at(1, iy, ix) = imageResults.sample(1, x, y);
			imageInterpolated.at(2, iy, ix) = imageResults.sample(2, x, y);
		}
	};
	frame.mPool.addAndWait(fcn, 0, mData.h);
}

void OpticalFlowWriter::write(const MovieFrame& frame) {
	writeFlow(frame);
	//imageInterpolated.saveAsBMP("f:/im.bmp");

	//stamp color legend onto image
	imageInterpolated.setArea(0ull + mData.h - 10 - legendSize, 10, legend, legendMask);

	//encode bgr image
	ImageBGR& fr = imageInterpolated;
	AVFrame* av_frame = av_frames[0];
	uint8_t* src[] = { fr.data(), nullptr, nullptr, nullptr};
	int strides[] = { fr.stride * 3, 0, 0, 0 };
	int sliceHeight = sws_scale(sws_scaler_ctx, src, strides, 0, fr.h, av_frame->data, av_frame->linesize);

	av_frame->pts = AuxiliaryWriter::frameIndex;
	writeAVFrame(av_frame);
	AuxiliaryWriter::frameIndex++;
	MovieWriter::frameIndex++;
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

OpticalFlowWriter::~OpticalFlowWriter() {
	writeAVFrame(nullptr);
}