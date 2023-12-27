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
#include "Trajectory.hpp"
#include "Stats.hpp"

class MovieFrame;

//-----------------------------------------------------------------------------------
class MovieWriter : public WriterStats {

public:
	ImageYuv outputFrame; //frame to write in YUV444 format
	VideoPacketContext encodedFrame = {};

protected:
	const MainData& mData; //central metadata structure
	MovieReader& mReader;

	MovieWriter(MainData& data, MovieReader& reader) : 
		outputFrame(data.h, data.w, data.cpupitch), 
		mReader { reader },
		mData { data } {}

public:
	virtual ~MovieWriter() = default;
	virtual OutputContext getOutputContext();
	virtual void write(const MovieFrame& frame) { frameIndex++; }
	virtual std::future<void> writeAsync(const MovieFrame& frame);
	virtual void open(EncodingOption videoCodec) {}
	virtual bool startFlushing() { return false; }
	virtual bool flush() { return false; }
};


//-----------------------------------------------------------------------------------
class NullWriter : public MovieWriter {

public:
	NullWriter(MainData& data, MovieReader& reader) : 
		MovieWriter(data, reader) {}
};


//-----------------------------------------------------------------------------------
class ImageWriter : public MovieWriter {

protected:
	ImageWriter(MainData& data, MovieReader& reader) : 
		MovieWriter(data, reader) {}

	std::string makeFilename() const;

public:
	static std::string makeFilename(const std::string& pattern, int64_t index);
};

//-----------------------------------------------------------------------------------
class BmpImageWriter : public ImageWriter {

private:
	ImageBGR image;

public:
	BmpImageWriter(MainData& data, MovieReader& reader) : 
		ImageWriter(data, reader), 
		image(data.h, data.w) {}

	void write(const MovieFrame& frame) override;
};


//-----------------------------------------------------------------------------------
class JpegImageWriter : public ImageWriter {

private:
	AVCodecContext* ctx = nullptr;
	AVFrame* avframe = nullptr;
	AVPacket* packet = nullptr;

public:
	JpegImageWriter(MainData& data, MovieReader& reader) : 
		ImageWriter(data, reader) {}

	~JpegImageWriter() override;
	void open(EncodingOption videoCodec) override;
	void write(const MovieFrame& frame) override;
};


//-----------------------------------------------------------------------------------
class RawWriter : public MovieWriter {

protected:
	std::vector<char> yuvPacked;

	//constructor
	RawWriter(MainData& data, MovieReader& reader) :
		MovieWriter(data, reader), 
		yuvPacked(3ull * data.h * data.w) {}

	//copy strided yuv data into a packed array [w * h * 3]
	void packYuv();
};


//-----------------------------------------------------------------------------------
class PipeWriter : public RawWriter {

public:
	PipeWriter(MainData& data, MovieReader& reader) : 
		RawWriter(data, reader) {}

	~PipeWriter() override;
	void open(EncodingOption videoCodec) override;
	void write(const MovieFrame& frame) override;
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
	TCPWriter(MainData& data, MovieReader& reader);
	~TCPWriter() override;
	void open(EncodingOption videoCodec) override;
	void write(const MovieFrame& frame) override;
};


//-----------------------------------------------------------------------------------
class FFmpegFormatWriter : public MovieWriter {

protected:
	std::map<Codec, AVCodecID> codecMap = {
		{ Codec::H264, AV_CODEC_ID_H264 },
		{ Codec::H265, AV_CODEC_ID_HEVC },
		{ Codec::AV1, AV_CODEC_ID_AV1 },
		{ Codec::AUTO, AV_CODEC_ID_H264 },
	};

protected:
	uint32_t GOP_SIZE = 30; //interval of key frames

	AVFormatContext* fmt_ctx = nullptr;
	AVStream* videoStream = nullptr;
	AVPacket* videoPacket = nullptr;
	bool headerWritten = false;

	FFmpegFormatWriter(MainData& data, MovieReader& reader) : 
		MovieWriter(data, reader) {}
	~FFmpegFormatWriter() override;
	void open(EncodingOption videoCodec) override;
	int writePacket(AVPacket* packet);
	void writePacket(AVPacket* pkt, int64_t ptsIdx, int64_t dtsIdx, bool terminate);
	void transcodeAudio(AVPacket* pkt, StreamContext& sc, bool terminate);

private:
	AVStream* newStream(AVFormatContext* fmt_ctx, AVStream* inStream);
};


//-----------------------------------------------------------------------------------
class FFmpegWriter : public FFmpegFormatWriter {

private:
	AVPixelFormat pixfmt = AV_PIX_FMT_YUV420P;
	AVCodecContext* codec_ctx = nullptr;
	AVFrame* frame = nullptr;
	SwsContext* sws_scaler_ctx = nullptr;

	int sendFFmpegFrame(AVFrame* frame);
	int writeFFmpegPacket();

protected:
	void open(EncodingOption videoCodec, int w, int h);
	void write(ImageYuv& frame);

public:
	FFmpegWriter(MainData& data, MovieReader& reader) : 
		FFmpegFormatWriter(data, reader) {}
	~FFmpegWriter() override;
	void open(EncodingOption videoCodec) override;
	void write(const MovieFrame& frame) override;
	bool startFlushing() override;
	bool flush() override;
};


//-------------- writer to combine input and output side by side -------------------
class StackedWriter : public FFmpegWriter {

private:
	int widthTotal;
	ImageYuv combinedFrame; //frame that holds input and output side by side
	ImageYuv inputFrame;
	std::vector<unsigned char> bg;

public:
	StackedWriter(MainData& data, MovieReader& reader) :
		FFmpegWriter(data, reader), 
		widthTotal { data.w * 3 / 2 }, 
		combinedFrame(data.h, widthTotal, widthTotal),
		inputFrame(data.h, data.w, data.cpupitch) {}

	void open(EncodingOption videoCodec) override;
	OutputContext getOutputContext() override;
	void write(const MovieFrame& frame) override;
};


//------------- transformation file -------------------------------------------------
class TransformsFile {

protected:
	const MainData& mData;
	inline static std::string id = "CUVI";
	std::ofstream file;

	template <class T> void writeValue(T val) {
		file.write(reinterpret_cast<const char*>(&val), sizeof(val));
	}

public:
	TransformsFile(const MainData& data) :
		mData { data } {}

	void open();
	void writeTransform(const Affine2D& transform, int64_t frameIndex);
};


//------------- write binary transformation file ------------------------------------
class TransformsWriterMain : public MovieWriter, TransformsFile {

public:
	static std::map<int64_t, TransformValues> readTransformMap(const std::string& trajectoryFile);

	TransformsWriterMain(MainData& data, MovieReader& reader) :
		MovieWriter(data, reader),
		TransformsFile(data) {}

	virtual void open(EncodingOption videoCodec) override; //for use as a main writer
	virtual void write(const MovieFrame& frame) override;
};


//-------------- superclass for secondary writers -----------------------------------
class AuxiliaryWriter : public MovieWriter {

private:
	NullReader nullReader;

public:
	AuxiliaryWriter(MainData& data) :
		MovieWriter(data, nullReader) {}

	virtual void open() {}
};


//--------------- write transforms --------------------------------------------------
class AuxTransformsWriter : public AuxiliaryWriter, TransformsFile {

public:
	AuxTransformsWriter(MainData& data) :
		AuxiliaryWriter(data),
		TransformsFile(data) {}

	virtual void open() override;
	virtual void write(const MovieFrame& frame) override;
};

//--------------- write point results as large text file ----------------------------
class ResultDetailsWriter : public AuxiliaryWriter {

private:
	std::string delimiter = ";";
	std::ofstream file;

public:
	ResultDetailsWriter(MainData& data) :
		AuxiliaryWriter(data) {}

	virtual void open() override;
	virtual void write(const MovieFrame& frame) override;
	void write(const std::vector<PointResult>& results, int64_t frameIndex);
};


//--------------- write individual images to show point results ---------------------
class ResultImageWriter : public AuxiliaryWriter {

private:
	ImageBGR bgr;

public:
	ResultImageWriter(MainData& data) :
		AuxiliaryWriter(data), 
		bgr(data.h, data.w) {}

	virtual void open() override {}
	virtual void write(const MovieFrame& frame) override;
	void write(const FrameResult& fr, int64_t idx, const ImageYuv& yuv, const std::string& fname);
};


//--------------- collection of auxiliary writer instances
class AuxWriters : public std::vector<std::unique_ptr<AuxiliaryWriter>> {

public:
	void writeAll(const MovieFrame& frame);
};