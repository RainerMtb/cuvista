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
#include "ThreadPool.hpp"

class MovieFrame;

//-----------------------------------------------------------------------------------
class MovieWriter : public WriterStats {

protected:
	const MainData& mData; //main metadata structure

	MovieWriter(MainData& data) :
		mData { data } {}

public:
	const MovieFrame* movieFrame = nullptr; //will be set in constructor of MovieFrame

	virtual ~MovieWriter() = default;
	virtual void open(EncodingOption videoCodec) {}
	virtual void open() {}
	virtual void prepareOutput(MovieFrame& frame) {}
	virtual void write(const MovieFrame& frame) { frameIndex++; }
	virtual bool startFlushing() { return false; }
	virtual bool flush() { return false; }
};


//-----------------------------------------------------------------------------------
class NullWriter : public MovieWriter {

protected:
	MovieReader& mReader;

public:
	NullWriter(MainData& data, MovieReader& reader) :
		MovieWriter(data),
		mReader { reader } {}
};


//-----------------------------------------------------------------------------------
class BaseWriter : public NullWriter {

protected:
	ImageYuv outputFrame; //frame to write in YUV444 format

public:
	BaseWriter(MainData& data, MovieReader& reader) :
		NullWriter(data, reader),
		outputFrame(data.h, data.w, data.cpupitch) {}

	const ImageYuv& getOutputFrame() { return outputFrame; }
	void prepareOutput(MovieFrame& frame) override;
};


//-----------------------------------------------------------------------------------
class ImageWriter : public BaseWriter {

protected:
	ImageWriter(MainData& data, MovieReader& reader) :
		BaseWriter(data, reader) {}

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

	void write(const MovieFrame& frame) override;
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
	void write(const MovieFrame& frame) override;
};


//-----------------------------------------------------------------------------------
class RawWriter : public BaseWriter {

protected:
	std::vector<char> yuvPacked;

	//constructor
	RawWriter(MainData& data, MovieReader& reader) :
		BaseWriter(data, reader),
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

// send raw YUV444P data to tcp address
// example to receive and encode via ffmpeg
// ffmpeg -f rawvideo -pix_fmt yuv444p -video_size www:hhh -i tcp://127.0.0.1:4444 out.mp4
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
class FFmpegFormatWriter : public NullWriter {

protected:
	std::map<Codec, AVCodecID> codecMap = {
		{ Codec::H264, AV_CODEC_ID_H264 },
		{ Codec::H265, AV_CODEC_ID_HEVC },
		{ Codec::AV1, AV_CODEC_ID_AV1 },
		{ Codec::AUTO, AV_CODEC_ID_H264 },
	};

	uint32_t GOP_SIZE = 30; //interval of key frames
	std::list<std::future<void>> encodingQueue;
	ThreadPool encoderPool = ThreadPool(1);

	AVFormatContext* fmt_ctx = nullptr;
	AVStream* videoStream = nullptr;
	AVPacket* videoPacket = nullptr;
	bool headerWritten = false;

	FFmpegFormatWriter(MainData& data, MovieReader& reader) :
		NullWriter(data, reader) {}

	~FFmpegFormatWriter() override;
	void open(EncodingOption videoCodec, const std::string& sourceName, int queueSize);
	int writePacket(AVPacket* packet);
	void writePacket(AVPacket* pkt, int64_t ptsIdx, int64_t dtsIdx, bool terminate);
	void transcodeAudio(AVPacket* pkt, StreamContext& sc, bool terminate);

private:
	VideoPacketContext encodedFrame = {};

	AVStream* newStream(AVFormatContext* fmt_ctx, AVStream* inStream);
};


//-----------------------------------------------------------------------------------
class FFmpegWriter : public FFmpegFormatWriter {

protected:
	int imageBufferSize;
	std::vector<ImageYuv> imageBuffer;
	AVFrame* av_frame = nullptr;

	AVPixelFormat pixfmt = AV_PIX_FMT_YUV420P;
	AVCodecContext* codec_ctx = nullptr;
	SwsContext* sws_scaler_ctx = nullptr;

	int sendFFmpegFrame(AVFrame* frame);
	int writeFFmpegPacket(AVFrame* av_frame);

	void openEncoder(const AVCodec* codec, const std::string& sourceName);
	void open(EncodingOption videoCodec, int h, int w, int stride, const std::string& sourceName);
	void write(int bufferIndex);

	FFmpegWriter(MainData& data, MovieReader& reader, int writeBufferSize) :
		FFmpegFormatWriter(data, reader),
		imageBufferSize { writeBufferSize } {}

public:
	FFmpegWriter(MainData& data, MovieReader& reader) :
		FFmpegWriter(data, reader, 4) {}

	~FFmpegWriter() override;
	void open(EncodingOption videoCodec) override;
	void prepareOutput(MovieFrame& frame) override;
	void write(const MovieFrame& frame) override;
	bool startFlushing() override;
	bool flush() override;
};


//-------------- writer to combine input and output side by side -------------------
class StackedWriter : public FFmpegWriter {

private:
	int widthTotal;
	ImageYuv inputFrame;
	ImageYuv outputFrame;
	std::vector<unsigned char> background;

public:
	StackedWriter(MainData& data, MovieReader& reader) :
		FFmpegWriter(data, reader, 1),
		widthTotal { data.w * 3 / 2 },
		inputFrame(data.h, data.w, data.cpupitch),
		outputFrame(data.h, data.w, data.cpupitch) {}

	void open(EncodingOption videoCodec) override;
	void prepareOutput(MovieFrame& frame) override;
	void write(const MovieFrame& frame) override;
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
	TransformsFile() {}

	static std::map<int64_t, TransformValues> readTransformMap(const std::string& trajectoryFile);

	void open(const std::string& trajectoryFile);
	void writeTransform(const Affine2D& transform, int64_t frameIndex);
};


//------------- write binary transformation file ------------------------------------
class TransformsWriterMain : public MovieWriter, public TransformsFile {

public:
	TransformsWriterMain(MainData& data, MovieReader& reader) :
		MovieWriter(data),
		TransformsFile() {}

	void open(EncodingOption videoCodec) override; //for use as a main writer
	void write(const MovieFrame& frame) override;
};


//signature class
class AuxiliaryWriter {

public:
	virtual void open() = 0;
	virtual void write(const MovieFrame& frame) = 0;
	virtual ~AuxiliaryWriter() {};
};


//--------------- write transforms --------------------------------------------------
class AuxTransformsWriter : public MovieWriter, public TransformsFile, public AuxiliaryWriter {

public:
	AuxTransformsWriter(MainData& data) :
		MovieWriter(data),
		TransformsFile() {}

	void open() override;
	void write(const MovieFrame& frame) override;
};


//optical flow video
class OpticalFlowWriter : public FFmpegWriter, public AuxiliaryWriter {

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
		FFmpegWriter(data, reader, 1),
		imageResults(data.iyCount, data.ixCount),
		imageInterpolated(data.h, data.w) {}

	void open() override;
	void write(const MovieFrame& frame) override;
	~OpticalFlowWriter();
};


//--------------- write point results as large text file ----------------------------
class ResultDetailsWriter : public MovieWriter, public AuxiliaryWriter {

private:
	std::string delimiter = ";";
	std::ofstream file;
	void write(const std::vector<PointResult>& results, int64_t frameIndex);

public:
	ResultDetailsWriter(MainData& data) :
		MovieWriter(data) {}

	virtual void open() override;
	virtual void write(const MovieFrame& frame) override;
};


//--------------- write individual images to show point results ---------------------
class ResultImageWriter : public MovieWriter, public AuxiliaryWriter {

private:
	ImageYuv yuv;
	ImageBGR bgr;
	void write(const AffineTransform& trf, const std::vector<PointResult>& res, int64_t idx, const ImageYuv& yuv, const std::string& fname);

public:
	ResultImageWriter(MainData& data) :
		MovieWriter(data),
		yuv(data.h, data.w),
		bgr(data.h, data.w) {}

	virtual void open() override {}
	virtual void write(const MovieFrame& frame) override;
};


//--------------- collection of auxiliary writer instances
class AuxWriters : public std::vector<std::unique_ptr<AuxiliaryWriter>> {

public:
	void writeAll(const MovieFrame& frame);
};