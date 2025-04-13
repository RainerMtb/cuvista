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

#include "MainData.hpp"
#include "Trajectory.hpp"
#include "Stats.hpp"
#include "ThreadPool.hpp"
#include "FrameExecutor.hpp"

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
	virtual void start() {}
	virtual void prepareOutput(FrameExecutor& executor) {}
	virtual void write(const FrameExecutor& executor) { frameIndex++; }
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
	void prepareOutput(FrameExecutor& executor) override;
	void write(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class ImageWriter : public BaseWriter {

protected:
	ImageWriter(MainData& data, MovieReader& reader) :
		BaseWriter(data, reader) {}

	std::string makeFilename(const std::string& extension) const;

public:
	static std::string makeFilename(const std::string& pattern, int64_t index, const std::string& extension);
	static std::string makeFilenameSamples(const std::string& pattern, const std::string& extension);
};

//-----------------------------------------------------------------------------------
class BmpImageWriter : public ImageWriter {

private:
	ImageRGBA image;
	std::jthread worker;

public:
	BmpImageWriter(MainData& data, MovieReader& reader) :
		ImageWriter(data, reader),
		worker { [] {} },
		image(data.h, data.w) {}

	~BmpImageWriter() override;
	void prepareOutput(FrameExecutor& executor) override;
	void write(const FrameExecutor& executor) override;
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
	void write(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class RawWriter : public BaseWriter {

private:
	std::ofstream file;

public:
	RawWriter(MainData& data, MovieReader& reader) :
		BaseWriter(data, reader) {}

	void open(EncodingOption videoCodec) override;
	void write(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class PipeWriter : public BaseWriter {

public:
	PipeWriter(MainData& data, MovieReader& reader) :
		BaseWriter(data, reader) {}

	~PipeWriter() override;
	void open(EncodingOption videoCodec) override;
	void write(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class FFmpegFormatWriter : public NullWriter {

protected:
	std::map<Codec, AVCodecID> codecToCodecIdMap = {
		{ Codec::H264, AV_CODEC_ID_H264 },
		{ Codec::H265, AV_CODEC_ID_HEVC },
		{ Codec::AV1, AV_CODEC_ID_AV1 },
		{ Codec::AUTO, AV_CODEC_ID_H264 },
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

	FFmpegFormatWriter(MainData& data, MovieReader& reader) :
		NullWriter(data, reader) {}

	~FFmpegFormatWriter() override;
	void open(AVCodecID codecId, const std::string& sourceName, int queueSize);
	void open(AVCodecID codecId);
	int writePacket(AVPacket* packet);
	void writePacket(AVPacket* pkt, int64_t ptsIdx, int64_t dtsIdx, bool terminate);
	void transcodeAudio(AVPacket* pkt, OutputStreamContext& osc, bool terminate);
	AVStream* newStream(AVFormatContext* fmt_ctx, AVStream* inStream);

private:
	VideoPacketContext encodedFrame = {};
};


//-----------------------------------------------------------------------------------
class FFmpegWriter : public FFmpegFormatWriter {

protected:
	std::map<AVCodecID, std::vector<std::string>> codecToNamesMap = {
		{AV_CODEC_ID_H264, {"libx264", "h264", "h264_qsv"}},
		{AV_CODEC_ID_HEVC, {"libx265", "hevc", "hevc_qsv"}},
		{AV_CODEC_ID_AV1, {"libsvtav1", "librav1e", "libaom-av1"}},
	};

	int imageBufferSize;
	std::vector<ImageYuv> imageBuffer;
	AVFrame* av_frame = nullptr;

	AVCodecContext* codec_ctx = nullptr;
	SwsContext* sws_scaler_ctx = nullptr;

	int sendFFmpegFrame(AVFrame* frame);
	int writeFFmpegPacket(AVFrame* av_frame);

	void open(AVCodecID codecId, AVPixelFormat pixfmt, int h, int w, int stride);
	void open(EncodingOption videoCodec, AVPixelFormat pixfmt, int h, int w, int stride, const std::string& sourceName);
	void write(int bufferIndex);

	FFmpegWriter(MainData& data, MovieReader& reader, int writeBufferSize) :
		FFmpegFormatWriter(data, reader),
		imageBufferSize { writeBufferSize } {}

public:
	FFmpegWriter(MainData& data, MovieReader& reader) :
		FFmpegWriter(data, reader, 4) {}

	~FFmpegWriter() override;
	void open(EncodingOption videoCodec) override;
	void prepareOutput(FrameExecutor& executor) override;
	void write(const FrameExecutor& executor) override;
	bool startFlushing() override;
	bool flush() override;
};


//-------------- writer to combine input and output side by side -------------------
class StackedWriter : public FFmpegWriter {

private:
	int mWidthTotal;
	double mStackPosition;
	ImageYuv mInputFrame;
	ImageYuv mInputFrameScaled;
	ImageYuv mOutputFrame;
	im::ColorYuv mBackgroundColor;

public:
	StackedWriter(MainData& data, MovieReader& reader, double stackPosition) :
		FFmpegWriter(data, reader, 1),
		mWidthTotal { data.w * 3 / 2 },
		mStackPosition { stackPosition },
		mInputFrame(data.h, data.w, data.cpupitch),
		mInputFrameScaled(data.h, data.w * 3 / 4),
		mOutputFrame(data.h, data.w, data.cpupitch) {}

	void open(EncodingOption videoCodec) override;
	void prepareOutput(FrameExecutor& executor) override;
	void write(const FrameExecutor& executor) override;
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
	void write(const FrameExecutor& executor) override;
};


//signature class
class AuxiliaryWriter {

public:
	virtual void open() = 0;
	virtual void write(const FrameExecutor& executor) = 0;
	virtual ~AuxiliaryWriter() {};
};


//--------------- write transforms --------------------------------------------------
class AuxTransformsWriter : public MovieWriter, public TransformsFile, public AuxiliaryWriter {

public:
	AuxTransformsWriter(MainData& data) :
		MovieWriter(data),
		TransformsFile() {}

	void open() override;
	void write(const FrameExecutor& executor) override;
};


//optical flow video
class OpticalFlowWriter : public FFmpegWriter, public AuxiliaryWriter {

private:
	int legendSizeBase = 64;
	int legendScale = 1;

protected:
	int legendSize = 0;
	ImageRGBA legend;
	ImageRGBA imageInterpolated;
	ImageRGBplanar imageResults;

	void open(const std::string& sourceName, AVPixelFormat pixfmt);
	void writeFlow(const MovieFrame& frame);
	void writeAVFrame(AVFrame* av_frame);
	void vectorToColor(double dx, double dy, unsigned char* r, unsigned char* g, unsigned char* b);

public:
	OpticalFlowWriter(MainData& data, MovieReader& reader) :
		FFmpegWriter(data, reader, 1),
		imageInterpolated(data.h, data.w),
		imageResults(data.iyCount, data.ixCount) {}

	void open() override;
	void write(const FrameExecutor& executor) override;
	~OpticalFlowWriter();
};


//--------------- write point results as large text file ----------------------------
class ResultDetailsWriter : public MovieWriter, public AuxiliaryWriter {

private:
	std::string mDelim = ";";
	std::ofstream mFile;

	void write(std::span<PointResult> results, int64_t frameIndex);

public:
	ResultDetailsWriter(MainData& data) :
		MovieWriter(data) {}

	virtual void open() override;
	virtual void write(const FrameExecutor& executor) override;

	static void write(std::span<PointResult> results, const std::string& filename);
};


//--------------- write individual images to show point results ---------------------
class ResultImageWriter : public MovieWriter, public AuxiliaryWriter {

private:
	ImageYuv yuv;
	ImageBGR bgr;

public:
	ResultImageWriter(MainData& data) :
		MovieWriter(data),
		yuv(data.h, data.w),
		bgr(data.h, data.w) {}

	virtual void open() override {}
	virtual void write(const FrameExecutor& executor) override;

	void writeImage(const AffineTransform& trf, std::span<PointResult> res, int64_t idx, const ImageYuv& yuv, const std::string& fname);
};


//--------------- collection of auxiliary writer instances
class AuxWriters : public std::vector<std::unique_ptr<AuxiliaryWriter>> {

public:
	void writeAll(const FrameExecutor& executor);
};