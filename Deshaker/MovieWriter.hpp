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

#include <fstream>
#include "MainData.hpp"
#include "FFmpegUtil.hpp"


//-----------------------------------------------------------------------------------
class MovieWriter {

public:
	ImageYuv outputFrame; //frame to write in YUV444 format

protected:
	std::map<OutputCodec, AVCodecID> codecMap = {
	{ OutputCodec::H264, AV_CODEC_ID_H264 },
	{ OutputCodec::H265, AV_CODEC_ID_HEVC },
	{ OutputCodec::JPEG, AV_CODEC_ID_MJPEG },
	{ OutputCodec::AUTO, AV_CODEC_ID_H264 },
	};

	Stats& mStatus;
	const MainData& mData; //central metadata structure

	MovieWriter(MainData& data) : 
		outputFrame(data.h, data.w, data.pitch), 
		mStatus { data.status }, 
		mData { data } 
	{}

public:
	virtual ~MovieWriter() = default;
	virtual OutputContext getOutputData();
	virtual void write() {}
	virtual void open(OutputCodec videoCodec) {}
	virtual bool terminate(bool init = false) { return false; }
};


//-----------------------------------------------------------------------------------
class NullWriter : public MovieWriter {

public:
	NullWriter(MainData& data) : MovieWriter(data) {}
};


//-----------------------------------------------------------------------------------
class ImageWriter : public MovieWriter {

protected:
	ImageWriter(MainData& data) : MovieWriter(data) {}
	std::string makeFilename() const;

public:
	static std::string makeFilename(const std::string& pattern, int64_t index);
};

//-----------------------------------------------------------------------------------
class BmpImageWriter : public ImageWriter {

private:
	ImageBGR image;

public:
	BmpImageWriter(MainData& data) : ImageWriter(data), image(data.h, data.w) {}
	void write() override;
};


//-----------------------------------------------------------------------------------
class JpegImageWriter : public ImageWriter {

private:
	AVCodecContext* ctx = nullptr;
	AVFrame* frame = nullptr;
	AVPacket* packet = nullptr;

public:
	JpegImageWriter(MainData& data) : ImageWriter(data) {}
	~JpegImageWriter() override;
	void open(OutputCodec videoCodec) override;
	void write() override;
};


//-----------------------------------------------------------------------------------
class RawWriter : public MovieWriter {

protected:
	std::vector<char> yuvPacked;

	//constructor
	RawWriter(MainData& data) : 
		MovieWriter(data), 
		yuvPacked(3ull * data.h * data.w) 
	{}

	//copy strided yuv data into a packed array [w * h * 3]
	void packYuv();
};


//-----------------------------------------------------------------------------------
class PipeWriter : public RawWriter {

public:
	PipeWriter(MainData& data) : RawWriter(data) {}
	~PipeWriter() override;
	void open(OutputCodec videoCodec) override;
	void write() override;
};


//-----------------------------------------------------------------------------------

//introduce custom structure in order to keep windows includes away
struct Sockets;

/*
send raw YUV444P data to tcp address
example to receive and encode via ffmpeg
ffmpeg -f rawvideo -pix_fmt yuv444p -video_size www:hhh -i tcp://127.0.0.1:4444 out.mp4
*/
class TCPWriter : public RawWriter {

protected:
	std::unique_ptr<Sockets> sockets;

public:
	TCPWriter(MainData& data);
	~TCPWriter() override;
	void open(OutputCodec videoCodec) override;
	void write() override;
};


//-----------------------------------------------------------------------------------
class FFmpegFormatWriter : public MovieWriter {

protected:
	uint32_t GOP_SIZE = 30; //interval of key frames

	AVFormatContext* fmt_ctx = nullptr;
	AVStream* videoStream = nullptr;
	AVPacket* videoPacket = nullptr;
	bool headerWritten = false;

	FFmpegFormatWriter(MainData& data) : MovieWriter(data) {}
	~FFmpegFormatWriter() override;
	void open(OutputCodec videoCodec) override;
	int writePacket(AVPacket* packet);
	void writePacket(AVPacket* pkt, int64_t ptsIdx, int64_t dtsIdx, bool terminate);
	void transcodeAudio(AVPacket* pkt, StreamContext& sc, bool terminate);
	//void encodeAudioFrame(StreamContext& sc, AVFrame* frame);

private:
	AVStream* newStream(AVFormatContext* fmt_ctx, AVStream* inStream);
	void rescaleAudioPacket(StreamContext& sc, AVPacket* pkt);
};


class FFmpegWriter : public FFmpegFormatWriter {

private:
	AVPixelFormat pixfmt = AV_PIX_FMT_YUV420P;
	AVCodecContext* codec_ctx = nullptr;
	AVFrame* frame = nullptr;
	SwsContext* sws_scaler_ctx = nullptr;

	int sendFFmpegFrame(AVFrame* frame);
	int writeFFmpegPacket();

public:
	FFmpegWriter(MainData& data) : FFmpegFormatWriter(data) {}
	~FFmpegWriter() override;
	void open(OutputCodec videoCodec) override;
	void write() override;
	bool terminate(bool init) override;
};

