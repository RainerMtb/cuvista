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
#include "AuxiliaryWriter.hpp"
#include "ThreadPool.hpp"

class MovieFrame;

//-----------------------------------------------------------------------------------
class MovieWriter : public WriterStats {

protected:
	const MainData& mData; //main metadata structure
	MovieReader& mReader;

	MovieWriter(MainData& data, MovieReader& reader) :
		mData { data },
		mReader { reader } {}

public:
	const MovieFrame* movieFrame = nullptr; //will be set in constructor of MovieFrame

	virtual ~MovieWriter() = default;
	virtual void open(EncodingOption videoCodec) {}
	virtual OutputContext getOutputContext();
	virtual void write() { frameIndex++; }
	virtual std::future<void> writeAsync();
	virtual bool startFlushing() { return false; }
	virtual bool flush() { return false; }
};


//-----------------------------------------------------------------------------------
class EmptyWriter : public MovieWriter {

public:
	EmptyWriter(MainData& data, MovieReader& reader) :
		MovieWriter(data, reader) {}
};


//-----------------------------------------------------------------------------------
class NullWriter : public MovieWriter {

protected:
	ImageYuv outputFrame; //frame to write in YUV444 format

public:
	NullWriter(MainData& data, MovieReader& reader) :
		MovieWriter(data, reader),
		outputFrame(data.h, data.w, data.cpupitch) {}

	OutputContext getOutputContext() override;
};


//-----------------------------------------------------------------------------------
class ImageWriter : public NullWriter {

protected:
	ImageWriter(MainData& data, MovieReader& reader) :
		NullWriter(data, reader) {}

	std::string makeFilename() const;

public:
	static std::string makeFilename(const std::string& pattern, int64_t index);
	static std::string makeFilenameSamples(const std::string& pattern);
};

//-----------------------------------------------------------------------------------
class BmpImageWriter : public ImageWriter {

private:
	ImageBGR image;

public:
	BmpImageWriter(MainData& data, MovieReader& reader) :
		ImageWriter(data, reader),
		image(data.h, data.w) {}

	void write() override;
};


//-----------------------------------------------------------------------------------
class JpegImageWriter : public ImageWriter {

private:
	AVCodecContext* ctx = nullptr;
	AVFrame* av_frame = nullptr;
	AVPacket* packet = nullptr;

public:
	JpegImageWriter(MainData& data, MovieReader& reader) :
		ImageWriter(data, reader) {}

	~JpegImageWriter() override;
	void open(EncodingOption videoCodec) override;
	void write() override;
};


//-----------------------------------------------------------------------------------
class RawWriter : public NullWriter {

protected:
	std::vector<char> yuvPacked;

	//constructor
	RawWriter(MainData& data, MovieReader& reader) :
		NullWriter(data, reader),
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
	TCPWriter(MainData& data, MovieReader& reader);
	~TCPWriter() override;
	void open(EncodingOption videoCodec) override;
	void write() override;
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

	ImageYuv outputFrame;

	uint32_t GOP_SIZE = 30; //interval of key frames
	VideoPacketContext encodedFrame = {};

	AVFormatContext* fmt_ctx = nullptr;
	AVStream* videoStream = nullptr;
	AVPacket* videoPacket = nullptr;
	bool headerWritten = false;

	FFmpegFormatWriter(MainData& data, MovieReader& reader) :
		MovieWriter(data, reader),
		outputFrame(data.h, data.w, data.cpupitch) {}

	~FFmpegFormatWriter() override;
	void open(EncodingOption videoCodec, const std::string& sourceName);
	int writePacket(AVPacket* packet);
	void writePacket(AVPacket* pkt, int64_t ptsIdx, int64_t dtsIdx, bool terminate);
	void transcodeAudio(AVPacket* pkt, StreamContext& sc, bool terminate);

private:
	AVStream* newStream(AVFormatContext* fmt_ctx, AVStream* inStream);
};


//-----------------------------------------------------------------------------------
class FFmpegWriter : public FFmpegFormatWriter {

protected:
	int writeBufferSize;
	std::vector<AVFrame*> av_frames;
	std::list<std::future<void>> futures;
	ThreadPool encoderPool = ThreadPool(1);

	AVPixelFormat pixfmt = AV_PIX_FMT_YUV420P;
	AVCodecContext* codec_ctx = nullptr;
	SwsContext* sws_scaler_ctx = nullptr;

	AVFrame* putAVFrame(ImageYuv& fr);
	int sendFFmpegFrame(AVFrame* frame);
	int writeFFmpegPacket(AVFrame* av_frame);

	void openEncoder(const AVCodec* codec, const std::string& sourceName);
	void open(EncodingOption videoCodec, int w, int h, const std::string& sourceName);
	void write(ImageYuv& frame);

	FFmpegWriter(MainData& data, MovieReader& reader, int writeBufferSize) :
		FFmpegFormatWriter(data, reader),
		writeBufferSize { writeBufferSize } {}

public:
	FFmpegWriter(MainData& data, MovieReader& reader) :
		FFmpegWriter(data, reader, 4) {}

	~FFmpegWriter() override;
	void open(EncodingOption videoCodec) override;
	OutputContext getOutputContext() override;
	void write() override;
	std::future<void> writeAsync() override;
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
	void write() override;
};


//------------- transformation file -------------------------------------------------
class TransformsFile {

protected:
	inline static std::string id = "CUVI";
	std::ofstream file;

	template <class T> void writeValue(T val) {
		file.write(reinterpret_cast<const char*>(&val), sizeof(val));
	}

public:
	TransformsFile(const MainData& data) {}

	static std::map<int64_t, TransformValues> readTransformMap(const std::string& trajectoryFile);

	void open(const std::string& trajectoryFile);
	void writeTransform(const Affine2D& transform, int64_t frameIndex);
};


//------------- write binary transformation file ------------------------------------
class TransformsWriterMain : public MovieWriter, public TransformsFile {

public:
	TransformsWriterMain(MainData& data, MovieReader& reader) :
		MovieWriter(data, reader),
		TransformsFile(data) {}

	void open(EncodingOption videoCodec) override; //for use as a main writer
	void write() override;
};


//--------------- write transforms --------------------------------------------------
class AuxTransformsWriter : public AuxiliaryWriter, public TransformsFile {

public:
	AuxTransformsWriter(MainData& data) :
		AuxiliaryWriter(data),
		TransformsFile(data) {}

	void open() override;
	void write(const MovieFrame& frame) override;
};


//optical flow video
class OpticalFlowWriter : public AuxiliaryWriter, public FFmpegWriter {

protected:
	ImageRGB imageResults;
	ImageBGR imageInterpolated;
	AVRational timeBase = { 1, 10 };

	int legendSize = 64;
	ImageBGR legend = ImageBGR(legendSize, legendSize);
	ImageGray legendMask = ImageGray(legendSize, legendSize);

	void open(const std::string& sourceName);
	void writeFlow(const MovieFrame& frame);
	void writeAVFrame(AVFrame* av_frame);
	void vectorToColor(double dx, double dy, unsigned char* r, unsigned char* g, unsigned char* b);

public:
	OpticalFlowWriter(MainData& data, MovieReader& reader) :
		AuxiliaryWriter(data),
		FFmpegWriter(data, reader, 1),
		imageResults(data.iyCount, data.ixCount),
		imageInterpolated(data.h, data.w) {}

	void open() override;
	void write(const MovieFrame& frame) override;
	~OpticalFlowWriter();
};