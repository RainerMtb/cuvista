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

#include <numbers>
#include <filesystem>

#include "Writer.hpp"
#include "MovieFrame.hpp"


 //-----------------------------------------------------------------------------------
 // JPG Images via ffmpeg
 //-----------------------------------------------------------------------------------

JpegImageWriter::JpegImageWriter(MainData& data, MovieReader& reader) :
	ImageWriter(data, reader),
	output(data.h, data.w)
{}

void JpegImageWriter::open(OutputOption outputOption) {
	const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
	ctx = avcodec_alloc_context3(codec);
	ctx->width = mData.w;
	ctx->height = mData.h;
	ctx->time_base = { 1, 1 };
	ctx->framerate = { 1, 1 };
	ctx->codec_type = AVMEDIA_TYPE_VIDEO;
	ctx->pix_fmt = AV_PIX_FMT_YUVJ444P;
	ctx->color_range = AVCOL_RANGE_JPEG;
	ctx->flags |= AV_CODEC_FLAG_QSCALE;
	ctx->global_quality = FF_QP2LAMBDA * mData.selectedCrf; //values for crf from 31 (worst) to 1 (best)
	int retval;
	retval = avcodec_open2(ctx, codec, NULL);
	if (retval < 0)
		throw AVException(av_make_error(retval, "cannot open jpeg codec"));

	swsCtx = sws_getContext(mData.w, mData.h, AV_PIX_FMT_YUV444P, mData.w, mData.h, AV_PIX_FMT_YUVJ444P, 0, NULL, NULL, NULL);
	if (!swsCtx) {
		throw AVException("cannot get scaler context");
	}

	av_frame = av_frame_alloc();
	av_frame->format = ctx->pix_fmt;
	av_frame->width = mData.w;
	av_frame->height = mData.h;
	av_frame->quality = ctx->global_quality; //quality must be set both to AVContext and AVFrame
	av_frame_get_buffer(av_frame, 0);

	packet = av_packet_alloc();
}

void JpegImageWriter::writeOutput(const FrameExecutor& executor) {
	executor.getOutput(frameIndex, output);

	uint8_t* srcLines[] = { output.plane(0), output.plane(1), output.plane(2), nullptr, nullptr, nullptr, nullptr, nullptr };
	int srcLineSizes[] = { output.stride(), output.stride(), output.stride(), 0, 0, 0, 0, 0 };
	sws_scale(swsCtx, srcLines, srcLineSizes, 0, output.h(), av_frame->data, av_frame->linesize);

	av_frame->pts = this->frameIndex;
	int result = avcodec_send_frame(ctx, av_frame);
	if (result < 0)
		errorLogger().logError(av_make_error(result, "error sending frame"), ErrorSource::WRITER);

	result = avcodec_receive_packet(ctx, packet);
	if (result < 0)
		errorLogger().logError(av_make_error(result, "error receiving packet"), ErrorSource::WRITER);

	std::string fname = makeFilename("jpg");
	std::ofstream file(fname, std::ios::binary);
	if (file)
		file.write(reinterpret_cast<char*>(packet->data), packet->size);
	else
		errorLogger().logError("error opening output file '" + fname + "'", ErrorSource::WRITER);

	this->outputBytesWritten += packet->size;
	av_packet_unref(packet);

	this->encodedBytesTotal += std::filesystem::file_size(std::filesystem::path(fname));
	this->frameIndex++;
}

JpegImageWriter::~JpegImageWriter() {
	av_packet_free(&packet);
	sws_freeContext(swsCtx);
	av_frame_free(&av_frame);
	avcodec_free_context(&ctx);
}


//-----------------------------------------------------
// Write asf data to Pipe
//-----------------------------------------------------

AsfPipeWriter::AsfPipeWriter(MainData& data, MovieReader& reader) :
	FFmpegWriter(data, reader, 0),
	output(data.h, data.w)
{}

void AsfPipeWriter::open(OutputOption outputOption) {
	PipeWriter::openPipe();
	av_log_set_callback(ffmpeg_log);

	//setup output context
	const AVOutputFormat* ofmt = av_guess_format("asf", NULL, NULL);
	AVFormatContext* fmt = nullptr;
	int result = avformat_alloc_output_context2(&fmt, ofmt, NULL, NULL);
	if (result < 0)
		throw AVException(av_make_error(result, "cannot allocate output format"));

	//custom avio
	int bufsiz = 64 * 1024;
	mBuffer = (unsigned char*) av_malloc(bufsiz);
	av_avio = avio_alloc_context(mBuffer, bufsiz, 1, this, nullptr, &AsfPipeWriter::writeBuffer, nullptr);
	if (av_avio == NULL)
		throw AVException("cannot create io context");

	av_avio->seekable = 0; //no seek allowed
	fmt->pb = av_avio;
	fmt->flags |= AVFMT_FLAG_CUSTOM_IO;

	//open ffmpeg
	AVCodecID id = AV_CODEC_ID_FFVHUFF;
	FFmpegFormatWriter::openFormat(id, fmt, 1);
	FFmpegWriter::open({}, id, AV_PIX_FMT_YUV444P, mData.h, mData.w, mData.stride);
}

//for ffmpeg 7
int AsfPipeWriter::writeBuffer(void* opaque, unsigned char* buf, int siz) {
	return writeBuffer(opaque, (const unsigned char*) buf, siz); //forward to new constified function
}

//for ffmpeg 8
int AsfPipeWriter::writeBuffer(void* opaque, const unsigned char* buf, int siz) {
	//AsfPipeWriter* ptr = static_cast<AsfPipeWriter*>(opaque);
	size_t bytes = fwrite(buf, 1, siz, stdout);
	if (bytes != siz) {
		errorLogger().logError("error writing data to pipe", ErrorSource::WRITER);
	}
	return (int) bytes;
}

void AsfPipeWriter::writeOutput(const FrameExecutor& executor) {
	executor.getOutput(frameIndex, output);

	av_frame->data[0] = output.plane(0);
	av_frame->data[1] = output.plane(1);
	av_frame->data[2] = output.plane(2);
	av_frame->linesize[0] = output.stride();
	av_frame->linesize[1] = output.stride();
	av_frame->linesize[2] = output.stride();
	av_frame->pts = frameIndex;

	//generate and write packet through ffmpeg writer
	int result = sendFFmpegFrame(av_frame);
	while (result >= 0) {
		result = writeFFmpegPacket(av_frame);
	}

	this->frameIndex++;
}

void AsfPipeWriter::close() {
	FFmpegFormatWriter::close();
	fflush(stdout);
	PipeWriter::closePipe();
}

AsfPipeWriter::~AsfPipeWriter() {
	avio_context_free(&av_avio);
}


//-----------------------------------------------------------------------------------
// Optical Flow Video
//-----------------------------------------------------------------------------------

OpticalFlowWriter::OpticalFlowWriter(MainData& data, MovieReader& reader) :
	FFmpegWriter(data, reader, 1),
	imageInterpolated(data.h, data.w),
	imageResults(data.iyCount, data.ixCount)
{}

void OpticalFlowWriter::vectorToColor(double dx, double dy, unsigned char* r, unsigned char* g, unsigned char* b) {
	const double f = 20.0;
	double hue = std::atan2(dy, dx) / std::numbers::pi * 180.0 + 180.0;
	double val = std::clamp(std::sqrt(dx * dx + dy * dy) / f, 0.0, 1.0);
	im::hsv_to_rgb(hue, 1.0, val, r, g, b);
}

void OpticalFlowWriter::start() {
	start(mData.fileOut, AV_PIX_FMT_YUV420P);
}

void OpticalFlowWriter::start(const std::string& sourceName, AVPixelFormat pixfmt) {
	//simplified ffmpeg video setup
	av_log_set_callback(ffmpeg_log);

	//setup output context
	int result = avformat_alloc_output_context2(&fmt_ctx, NULL, NULL, sourceName.c_str());
	if (result < 0)
		throw AVException(av_make_error(result));

	result = avio_open(&fmt_ctx->pb, fmt_ctx->url, AVIO_FLAG_WRITE);
	if (result < 0)
		throw AVException("error opening output file '" + mData.fileOut + "'");

	for (size_t i = 0; i < mReader.inputStreamCount(); i++) {
		std::shared_ptr<StreamContext> sc = mReader.inputStream(i);
		auto osc = std::make_shared<OutputStreamContext>();
		osc->inputStream = sc->inputStream;

		if (sc->inputStream->index == mReader.videoStreamIndex) {
			osc->handling = StreamHandling::STREAM_STABILIZE;
			osc->outputStream = videoStream = createNewStream(fmt_ctx, sc->inputStream);

		} else {
			osc->handling = StreamHandling::STREAM_IGNORE;
		}

		outputStreams.push_back(osc);
		sc->outputStreams.push_back(osc);
	}

	FFmpegWriter::open({}, AV_CODEC_ID_H264, pixfmt, mData.h, mData.w, mData.stride);

	//setup scaler to accept RGB
	sws_scaler_ctx = sws_getContext(mData.w, mData.h, AV_PIX_FMT_RGBA, mData.w, mData.h, pixfmt, SWS_BILINEAR, NULL, NULL, NULL);
	if (!sws_scaler_ctx)
		throw AVException("Could not get scaler context");

	//setup legend image showing flow colors
	legendScale = std::max(1, std::min(mData.w, mData.h) / 512);
	legendSize = legendSizeBase * legendScale;
	legend = ImageRGBA(legendSize, legendSize);

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
	Color color = Color::GRAY;

	//convert vectors to hsv color values and then to rgb planar image
	std::vector<PointResult> results = frame.mResultPoints;
	for (const PointResult& pr : results) {
		if (pr.isValid()) {
			vectorToColor(-pr.u, pr.v, imageResults.addr(0, pr.iy0, pr.ix0), imageResults.addr(1, pr.iy0, pr.ix0), imageResults.addr(2, pr.iy0, pr.ix0));

		} else {
			//invalid result is gray pixel
			imageResults.setColor(pr.iy0, pr.ix0, color);
		}
	}

	//interpolate image
	auto fcn = [&] (size_t iy) {
		int div = 1 << mData.zMax;
		double y = 1.0 * iy / div - 1.0 - mData.ir;
		for (size_t ix = 0; ix < mData.w; ix++) {
			double x = 1.0 * ix / div - 1.0 - mData.ir;
			for (int i = 0; i < 3; i++) {
				unsigned char col = color.getChannel(i);
				if (x >= -0.5 && x < mData.ixCount + 0.5 && y >= -0.5 && y < mData.iyCount + 0.5) col = imageResults.sample(i, float(x), float(y));
				imageInterpolated.at(i, iy, ix) = col;
			}
		}
	};
	frame.mPool.addAndWait(fcn, 0, mData.h);
}

void OpticalFlowWriter::writeAVFrame(AVFrame* av_frame) {
	int result = avcodec_send_frame(codec_ctx, av_frame);
	if (result < 0)
		ffmpeg_log_error(result, "error encoding flow #1", ErrorSource::WRITER);

	while (result >= 0) {
		result = avcodec_receive_packet(codec_ctx, videoPacket);
		if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
			//do not report error here, need more frame data or end of file

		} else if (result < 0) {
			//report error, something wrong
			ffmpeg_log_error(result, "error encoding flow #2", ErrorSource::WRITER);

		} else {
			av_packet_rescale_ts(videoPacket, codec_ctx->time_base, videoStream->time_base);
			int siz = videoPacket->size;
			result = av_interleaved_write_frame(fmt_ctx, videoPacket);
			if (result < 0) {
				ffmpeg_log_error(result, "error writing flow packet", ErrorSource::WRITER);

			} else {
				encodedBytesTotal += siz;
			}

			outputBytesWritten = avio_tell(fmt_ctx->pb);
			frameEncoded++;
		}
	}
}

void OpticalFlowWriter::writeInput(const FrameExecutor& executor) {
	writeFlow(executor.mFrame);
	//imageInterpolated.saveAsBMP("f:/im.bmp");

	//stamp color legend onto image
	legend.copyTo(imageInterpolated, 0ull + mData.h - 10 - legendSize, 10, 192, executor.mPool);
	int tx = 10 + legendSize / 2;
	int ty = 0ull + mData.h - 10 - legendSize / 2;
	imageInterpolated.writeText(std::to_string(frameIndex), tx, ty, im::TextAlign::MIDDLE_CENTER, legendScale, legendScale);

	//encode rgba image
	uint8_t* src[] = { imageInterpolated.data(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
	int strides[] = { imageInterpolated.stride(), 0, 0, 0, 0, 0, 0, 0 };
	int sliceHeight = sws_scale(sws_scaler_ctx, src, strides, 0, imageInterpolated.h(), av_frame->data, av_frame->linesize);

	av_frame->pts = frameIndex;
	writeAVFrame(av_frame);
	frameIndex++;
}

void OpticalFlowWriter::writeOutput(const FrameExecutor& executor) {}

bool OpticalFlowWriter::flush() {
	writeAVFrame(nullptr);
	return false;
}
