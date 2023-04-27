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
	Stats& status;
	const MainData& data; //central metadata structure

	MovieWriter(MainData& data) : 
		outputFrame(data.h, data.w, data.pitch), 
		status { data.status }, 
		data { data } 
	{}

public:
	virtual ~MovieWriter() = default;
	virtual OutputContext getOutputData();
	virtual void write() {}
	virtual void open() {}
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
	virtual void write() override;
};


//-----------------------------------------------------------------------------------
class JpegImageWriter : public ImageWriter {

private:
	AVCodecContext* ctx = nullptr;
	AVFrame* frame = nullptr;
	AVPacket* packet = nullptr;

public:
	JpegImageWriter(MainData& data) : ImageWriter(data) {}
	virtual void open() override;
	virtual void write() override;
	virtual ~JpegImageWriter() override;
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
	virtual void open() override;
	virtual ~PipeWriter() override;
	virtual void write() override;
};


#include <ws2tcpip.h>
#include <WinSock2.h>

//undef conflicting macro
#undef max
#undef min

//-----------------------------------------------------------------------------------
class TCPWriter : public RawWriter {

protected:
	SOCKET mSock {};
	SOCKET mConn {};

public:
	TCPWriter(MainData& data) : RawWriter(data) {}
	virtual void open() override;
	virtual ~TCPWriter() override;
	virtual void write() override;
};


//-----------------------------------------------------------------------------------
class FFmpegFormatWriter : public MovieWriter {

protected:
	uint32_t GOP_SIZE = 30; //interval of key frames

	struct StreamsContext {
		int mapping = 0;
		AVStream* stream = nullptr;
	};
	std::vector<StreamsContext> outStreams;

	AVFormatContext* fmt_ctx = nullptr;
	AVStream* videoStream = nullptr;
	size_t videoIdx = 0;
	AVPacket* videoPacket = nullptr;
	bool headerWritten = false;

	FFmpegFormatWriter(MainData& data) : MovieWriter(data) {}
	virtual void open() override;
	virtual ~FFmpegFormatWriter() override;
	void writePacket(AVPacket* packet);
	void writePacket(AVPacket* pkt, int64_t ptsIdx, int64_t dtsIdx, bool terminate);
};


class FFmpegWriter : public FFmpegFormatWriter {

private:
	std::map<OutputCodec, AVCodecID> codecMap = {
		{ OutputCodec::H264, AV_CODEC_ID_H264 },
		{ OutputCodec::H265, AV_CODEC_ID_HEVC },
		{ OutputCodec::AUTO, AV_CODEC_ID_H264 },
	};

	AVPixelFormat pixfmt = AV_PIX_FMT_YUV420P;
	AVCodecContext* codec_ctx = nullptr;
	AVFrame* frame = nullptr;
	SwsContext* sws_scaler_ctx = nullptr;

	int sendFFmpegFrame(AVFrame* frame);
	int writeFFmpegPacket();

public:
	FFmpegWriter(MainData& data) : FFmpegFormatWriter(data) {}
	virtual void open() override;
	virtual ~FFmpegWriter() override;
	virtual void write() override;
	virtual bool terminate(bool init) override;
};


#include "NvEncoder.hpp"

//-----------------------------------------------------------------------------------
class CudaFFmpegWriter : public FFmpegFormatWriter {

private:
	std::map<OutputCodec, GUID> guidMap = {
		{ OutputCodec::H264, NV_ENC_CODEC_H264_GUID },
		{ OutputCodec::H265, NV_ENC_CODEC_HEVC_GUID },
		{ OutputCodec::AUTO, NV_ENC_CODEC_HEVC_GUID },
	};

	class FunctorLess {
	public:
		bool operator () (const GUID& g1, const GUID& g2) const;
	};
	std::map<GUID, AVCodecID, FunctorLess> guidToCodecMap = {
		{ NV_ENC_CODEC_H264_GUID, AV_CODEC_ID_H264 },
		{ NV_ENC_CODEC_HEVC_GUID, AV_CODEC_ID_HEVC },
	};

	NvEncoder nvenc;
	std::list<NvPacket> nvPackets; //encoded packets

	void writePacketToFile(const NvPacket& nvpkt, bool terminate);

public:
	CudaFFmpegWriter(MainData& data) : FFmpegFormatWriter(data), nvenc { NvEncoder(data.w, data.h) } {}
	virtual void open() override;
	virtual ~CudaFFmpegWriter() override;
	virtual OutputContext getOutputData();
	void write() override;
	virtual bool terminate(bool init) override;
};
