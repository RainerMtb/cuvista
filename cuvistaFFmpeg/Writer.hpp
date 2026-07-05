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

#pragma once

#include "MovieWriter.hpp"
#include "FFmpegMain.hpp"
#include "ThreadPool.hpp"
#include "ImageClasses.hpp"
#include "DeviceInfo.hpp"

#include <map>
#include <vector>

using namespace im;


//-----------------------------------------------------------------------------------
class JpegImageWriter : public ImageWriter {

private:
	ImageYuv output;
	AVCodecContext* ctx = nullptr;
	AVFrame* av_frame = nullptr;
	AVPacket* packet = nullptr;
	SwsContext* swsCtx = nullptr;

public:
	JpegImageWriter(MainData& data, MovieReader& reader);
	~JpegImageWriter() override;

	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class FFmpegFormatWriter : public NullWriter {

protected:
	std::map<OutputOption, AVCodecID> optionToCodecIdMap = {
		{ OutputOption::NVENC_H264, AV_CODEC_ID_H264 },
		{ OutputOption::NVENC_HEVC, AV_CODEC_ID_HEVC },
		{ OutputOption::NVENC_AV1, AV_CODEC_ID_AV1 },

		{ OutputOption::FFMPEG_H264, AV_CODEC_ID_H264 },
		{ OutputOption::FFMPEG_HEVC, AV_CODEC_ID_HEVC },
		{ OutputOption::FFMPEG_AV1, AV_CODEC_ID_AV1 },
		{ OutputOption::FFMPEG_FFV1, AV_CODEC_ID_FFV1 },

		{ OutputOption::VIDEO_STACK, AV_CODEC_ID_H264 },
		{ OutputOption::VIDEO_FLOW, AV_CODEC_ID_H264 },
	};

	std::vector<std::shared_ptr<OutputStreamContext>> outputStreams;

	uint32_t gopSize = 15; //interval of key frames
	ThreadPool encoderPool = ThreadPool(1);
	std::list<std::future<void>> encodingQueue; //queue for encoder thread

	AVFormatContext* fmt_ctx = nullptr;
	AVIOContext* av_avio = nullptr;
	AVStream* videoStream = nullptr;
	AVPacket* videoPacket = nullptr;
	bool isHeaderWritten = false;
	bool isFlushing = false;
	int64_t dtsWritten = INT64_MIN;

	FFmpegFormatWriter(MainData& data, MovieReader& reader);
	~FFmpegFormatWriter() override;

	void close() override;
	void openFormat(AVCodecID codecId);
	void openFormat(AVCodecID codecId, const std::string& sourceName, int queueSize);
	void openFormat(AVCodecID codecId, AVFormatContext* ctx, int queueSize);
	int writePacket(AVPacket* packet);
	void writePacket(AVPacket* pkt, int64_t ptsIdx, int64_t dtsIdx, bool terminate);
	void transcodeAudio(AVPacket* pkt, OutputStreamContext& osc, bool terminate);
	AVStream* createNewStream(AVFormatContext* fmt_ctx, AVStream* inStream);
};


//-----------------------------------------------------------------------------------
class FFmpegWriter : public FFmpegFormatWriter {

protected:
	int imageBufferSize;
	std::vector<ImageVuyx> imageBuffer;
	AVFrame* av_frame = nullptr;

	AVCodecContext* codec_ctx = nullptr;
	SwsContext* sws_scaler_ctx = nullptr;

	int sendFFmpegFrame(AVFrame* frame);
	int writeFFmpegPacket(AVFrame* av_frame);

	void open(std::span<std::string> codecNames, AVCodecID codecId, AVPixelFormat pixfmt, int h, int w, int stride);
	void open(const AVCodec* codec, AVPixelFormat pixfmt, int h, int w, int stride);
	void open(OutputOption outputOption, AVPixelFormat pixfmt, int h, int w, int stride, const std::string& sourceName);
	void write(int bufferIndex);

	FFmpegWriter(MainData& data, MovieReader& reader, int writeBufferSize) :
		FFmpegFormatWriter(data, reader),
		imageBufferSize { writeBufferSize } {}

public:
	FFmpegWriter(MainData& data, MovieReader& reader) :
		FFmpegWriter(data, reader, 4)
	{}

	~FFmpegWriter() override;
	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
	bool flush() override;
};


//-----------------------------------------------------------------------------------
class AsfPipeWriter : public FFmpegWriter, public PipeWriter {

private:
	unsigned char* mBuffer = nullptr;
	AVIOContext* av_avio = nullptr;
	ImageYuv output;

	static int writeBuffer(void* opaque, const unsigned char* buf, int bufsiz);
	static int writeBuffer(void* opaque, unsigned char* buf, int bufsiz);

public:
	AsfPipeWriter(MainData& data, MovieReader& reader);

	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
	void close() override;
	~AsfPipeWriter() override;
};


//-------------- writer to combine input and output side by side -------------------
class StackedWriter : public FFmpegWriter {

private:
	int mWidth;
	int mWidthTotal;
	ImageVuyx mInputFrame;
	ImageVuyx mOutputFrame;

public:
	StackedWriter(MainData& data, MovieReader& reader);

	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
};


//--------------- optical flow video ------------------------------------------------
class OpticalFlowWriter : public FFmpegWriter {

private:
	int legendSizeBase = 64;
	int legendScale = 1;

protected:
	int legendSize = 0;
	ImageRGBA legend;
	ImageRGBA imageInterpolated;
	ImageRGBA imageResults;

	void start(const std::string& sourceName, AVPixelFormat pixfmt);
	void writeFlow(const MovieFrame& frame);
	void writeAVFrame(AVFrame* av_frame);
	void vectorToColor(double dx, double dy, unsigned char* r, unsigned char* g, unsigned char* b);

public:
	OpticalFlowWriter(MainData& data, MovieReader& reader);

	void open(OutputOption outputOption) override {}
	void start() override;
	void writeInput(const FrameExecutor& executor) override;
	void writeOutput(const FrameExecutor& executor) override;
	bool flush() override;
};


//--------------- cuda writer -----------------------------------------------------

struct NvPacket;
class NvEncoder;

class CudaFFmpegWriter : public FFmpegFormatWriter {

protected:
	std::shared_ptr<NvEncoder> nvenc = nullptr;
	std::unique_ptr<std::list<NvPacket>> nvPackets; //encoded packets
	ImageNV12 outputNV12;
	int nv12stride = 0;

	void open(OutputOption outputOption, const DeviceInfoCuda* dic);
	void writePacketToFile(const NvPacket& nvpkt, bool terminate);
	void writePacketsToFile(std::list<NvPacket> nvpkts, bool terminate);
	void encodeFrame(int64_t frameIndex);

public:
	CudaFFmpegWriter(MainData& data, MovieReader& reader);
	~CudaFFmpegWriter() override;

	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
	bool flush() override;
};
