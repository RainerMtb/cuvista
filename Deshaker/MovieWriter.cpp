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
#include "Stats.hpp"


OutputContext MovieWriter::getOutputContext() {
	return { true, false, &outputFrame, nullptr };
}

std::future<void> MovieWriter::writeAsync() {
	return std::async(std::launch::async, [&] () { write(); });
}

std::string ImageWriter::makeFilename(const std::string& pattern, int64_t index) {
	const int siz = 256;
	char fname[siz];
	std::snprintf(fname, siz, pattern.c_str(), index);
	return fname;
}

std::string ImageWriter::makeFilename() const {
	return makeFilename(mData.fileOut, mStatus.frameWriteIndex);
}

//-----------------------------------------------------------------------------------
// BMP Images
//-----------------------------------------------------------------------------------

void BmpImageWriter::write() {
	outputFrame.toBGR(image).saveAsBMP(makeFilename());
	mStatus.outputBytesWritten += image.dataSizeInBytes();
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

	frame = av_frame_alloc();
	frame->format = ctx->pix_fmt;
	frame->width = mData.w;
	frame->height = mData.h;

	frame->linesize[0] = outputFrame.stride;
	frame->linesize[1] = outputFrame.stride;
	frame->linesize[2] = outputFrame.stride;
	frame->data[0] = outputFrame.plane(0);
	frame->data[1] = outputFrame.plane(1);
	frame->data[2] = outputFrame.plane(2);

	packet = av_packet_alloc();
}

void JpegImageWriter::write() {
	frame->pts = mStatus.frameWriteIndex;
	int result = avcodec_send_frame(ctx, frame);
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

	mStatus.outputBytesWritten += packet->size;
	av_packet_unref(packet);
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
