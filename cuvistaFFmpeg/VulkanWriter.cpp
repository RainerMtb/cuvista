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

#include "Writer.hpp"
#include "Reader.hpp"
#include "MainData.hpp"

VulkanFFmpegWriter::VulkanFFmpegWriter(MainData& data, MovieReader& reader) :
	FFmpegWriter(data, reader, 0)
{}

bool VulkanFFmpegWriter::probe(OutputOption outputOption) {
	auto noopLogger = [] (void* avclass, int level, const char* fmt, va_list args) {};
	av_log_set_callback(noopLogger);

	return true;
}

void VulkanFFmpegWriter::openEncoder(OutputOption outputOption) {
	//find cpu encoder
	std::map<OutputOption, std::string> optionToCodecMap = {
		{ OutputOption::VULKAN_AV1, "av1_vulkan"},
		{ OutputOption::VULKAN_HEVC, "hevc_vulkan"},
		{ OutputOption::VULKAN_H264, "h264_vulkan"}
	};
	std::string codecName = optionToCodecMap.at(outputOption);
	const AVCodec* codec = avcodec_find_encoder_by_name(codecName.c_str());
	if (!codec) {
		throw AVException("Could not find encoder '" + codecName + "'");
	}

	//allocate cpu codec context
	codec_ctx = avcodec_alloc_context3(codec);
	if (!codec_ctx)
		throw AVException("Could not allocate encoder context");

	codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
	codec_ctx->width = mData.w;
	codec_ctx->height = mData.h;
	codec_ctx->pix_fmt = AV_PIX_FMT_VULKAN;
	codec_ctx->framerate = { mReader.fpsNum, mReader.fpsDen };
	codec_ctx->time_base = { mReader.fpsDen, mReader.fpsNum };
	codec_ctx->sample_aspect_ratio = { mReader.parNum, mReader.parDen };
	codec_ctx->gop_size = gopSize;
	codec_ctx->flags |= AV_CODEC_FLAG_QSCALE;
	av_opt_set(codec_ctx->priv_data, "qp", std::to_string(mData.selectedCrf).c_str(), 0);
	//codec_ctx->has_b_frames = 1;
	//codec_ctx->max_b_frames = 4;
	//codec_ctx->bit_rate = 5'000'000;

	if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
		codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

	//allocate hardware context
	AVHWDeviceType deviceType = av_hwdevice_find_type_by_name("vulkan");
	if (deviceType != AV_HWDEVICE_TYPE_VULKAN)
		throw AVException("Could not find device type VULKAN");

	int result = 0;
	result = av_hwdevice_ctx_create(&hw_ctx, deviceType, nullptr, nullptr, 0);
	if (result < 0)
		throw AVException(av_make_error(result, "Could not create vulkan device"));

	//check hardware capabilities
	AVHWFramesConstraints* constraints = nullptr;
	constraints = av_hwdevice_get_hwframe_constraints(hw_ctx, nullptr);
	if (constraints == nullptr)
		throw AVException("Could not get device constraints");

	const AVPixelFormat* ptr = constraints->valid_sw_formats;
	for (const AVPixelFormat* ptr = constraints->valid_sw_formats; *ptr != AV_PIX_FMT_NONE; ptr++) {
		vulkanSwFormats.push_back(*ptr);
	}
	av_hwframe_constraints_free(&constraints);

	//allocate hardwware frame context
	hwframes_ctx = av_hwframe_ctx_alloc(hw_ctx);
	avhw_frames_ctx = (AVHWFramesContext*) (hwframes_ctx->data);
	avhw_frames_ctx->format = AV_PIX_FMT_VULKAN;
	avhw_frames_ctx->sw_format = AV_PIX_FMT_NV12;
	avhw_frames_ctx->width = codec_ctx->width;
	avhw_frames_ctx->height = codec_ctx->height;

	result = av_hwframe_ctx_init(hwframes_ctx);
	if (result < 0)
		throw AVException(av_make_error(result, "Could not initialize hardware frames context"));

	codec_ctx->hw_frames_ctx = av_buffer_ref(hwframes_ctx);
	av_buffer_unref(&hwframes_ctx);

	//open encoder
	result = avcodec_open2(codec_ctx, codec, NULL);
	if (result < 0)
		throw AVException(av_make_error(result, "Error opening codec"));
}

void VulkanFFmpegWriter::open(OutputOption outputOption) {
	//open format
	AVCodecID codecID = optionToCodecIdMap[outputOption];
	FFmpegFormatWriter::openFormat(codecID, mData.fileOut, 1);

	//open vulkan encoder
	openEncoder(outputOption);

	//allocate nv12 image
	outputNV12 = ImageNV12(mData.h, mData.w);

	//allocate av_frame in NV12 format
	av_frame = av_frame_alloc();
	if (!av_frame)
		throw AVException("Could not allocate video frame");

	av_frame->format = AV_PIX_FMT_NV12;
	av_frame->width = codec_ctx->width;
	av_frame->height = codec_ctx->height;
	av_frame->data[0] = outputNV12.plane(0);
	av_frame->data[1] = outputNV12.plane(1);
	av_frame->linesize[0] = outputNV12.strideInBytes();
	av_frame->linesize[1] = outputNV12.strideInBytes();

	//allocate hardware frame but do not allocate buffer here
	hw_frame = av_frame_alloc();
	hw_frame->format = AV_PIX_FMT_VULKAN;
	hw_frame->width = codec_ctx->width;
	hw_frame->height = codec_ctx->height;

	//open format
	int result = 0;
	result = avcodec_parameters_from_context(videoStream->codecpar, codec_ctx);
	if (result < 0)
		throw AVException(av_make_error(result, "Error setting codec parameters"));

	result = avformat_init_output(fmt_ctx, NULL);
	if (result < 0)
		throw AVException(av_make_error(result, "Error initializing output"));

	result = avformat_write_header(fmt_ctx, NULL);
	if (result < 0)
		throw AVException(av_make_error(result, "Error writing file header"));
	else
		this->isHeaderWritten = true; //store info for proper closing

	videoPacket = av_packet_alloc();
	if (!videoPacket)
		throw AVException("Could not allocate encoder packet");
}

void VulkanFFmpegWriter::writeOutput(const FrameExecutor& executor) {
	executor.getOutput(frameIndex, outputNV12, outputNV12.stride(), nullptr);
	//outputNV12.writeText(std::to_string(frameIndex), 10, 10, im::TextAlign::TOP_LEFT);

	int result = 0;
	result = av_hwframe_get_buffer(codec_ctx->hw_frames_ctx, hw_frame, 0);
	if (result < 0)
		throw AVException("Could not get vulkan frame buffer");

	av_frame->pts = frameIndex;
	hw_frame->pts = frameIndex;
	result = av_hwframe_transfer_data(hw_frame, av_frame, 0);

	result = avcodec_send_frame(codec_ctx, hw_frame);
	av_frame_unref(hw_frame);

	while (true) {
		result = avcodec_receive_packet(codec_ctx, videoPacket);
		if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
			break;
			//do not report error here, need more frame data or end of file

		} else {
			//write packet to output, here packets arrive in dts order
			videoPacket->stream_index = videoStream->index;
			//std::printf("stream=%d pts=%zd dts=%zd\n", videoPacket->stream_index, videoPacket->pts, videoPacket->dts);
			writePacket(videoPacket, videoPacket->pts, videoPacket->dts, av_frame == nullptr);
		}
	}

	frameIndex++;
}

VulkanFFmpegWriter::~VulkanFFmpegWriter() {
	if (hwframes_ctx) {
		av_buffer_unref(&hwframes_ctx);
	}
	if (hw_ctx) {
		av_buffer_unref(&hw_ctx);
	}
	if (hw_frame) {
		av_frame_free(&hw_frame);
	}
}
