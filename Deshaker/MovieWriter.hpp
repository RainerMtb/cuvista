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


//-----------------------------------------------------------------------------------
//important: each writer must increment frame counter when beeing called
//-----------------------------------------------------------------------------------
class MovieWriter : public WriterStats {

public:
	virtual ~MovieWriter() = default;
	virtual void open(OutputOption outputOption) {}
	virtual void start() {}
	virtual void writeInput(const FrameExecutor& executor) {}
	virtual void writeOutput(const FrameExecutor& executor) {}
	virtual bool flush() { return false; }
	virtual void close() {}
};


//-----------------------------------------------------------------------------------
class MovieWriterBase : public MovieWriter {

protected:
	const MainData& mData;

public:
	MovieWriterBase(MainData& data) :
		mData(data) 
	{}
};


//-----------------------------------------------------------------------------------
class NullWriter : public MovieWriterBase {

protected:
	MovieReader& mReader;

public:
	NullWriter(MainData& data, MovieReader& reader) :
		MovieWriterBase(data),
		mReader { reader } 
	{}

	void writeOutput(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class MovieWriterCollection : public NullWriter {

private:
	std::vector<std::shared_ptr<MovieWriter>> writers;
	std::vector<int> hasFrames;

	void updateStats();

public:
	MovieWriterCollection(MainData& data, MovieReader& reader, std::vector<std::shared_ptr<MovieWriter>> writers);

	void open(OutputOption outputOption) override;
	void start() override;
	void writeInput(const FrameExecutor& executor) override;
	void writeOutput(const FrameExecutor& executor) override;
	bool flush() override;
	void close() override;
};


//-----------------------------------------------------------------------------------
class OutputWriter : public NullWriter {

protected:
	ImageVuyx outputFrame; //frame to get from main loop and to send to output

public:
	OutputWriter(MainData& data, MovieReader& reader, int outputStride) :
		NullWriter(data, reader),
		outputFrame(data.h, data.w, outputStride) 
	{}

	OutputWriter(MainData& data, MovieReader& reader) :
		OutputWriter(data, reader, data.stride4) 
	{}

	const ImageVuyx& getOutputFrame();
	void writeOutput(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class RawMemoryStoreWriter : public MovieWriter {

protected:
	size_t maxFrameCount;
	int inputFrameIndex = 0;
	bool doWriteInput;
	bool doWriteOutput;

public:
	std::list<ImageVuyx> outputFramesYuv;
	std::list<ImageRGBA> outputFramesRgba;
	std::list<ImageBGRA> outputFramesBgra;
	std::list<ImageVuyx> inputFrames;
	std::list<std::vector<PointResult>> results;

	RawMemoryStoreWriter(size_t maxFrameCount = 250, bool writeInput = true, bool writeOutput = true) :
		maxFrameCount { maxFrameCount },
		doWriteInput { writeInput },
		doWriteOutput { writeOutput} 
	{}

	void writeOutput(const FrameExecutor& executor) override;
	void writeInput(const FrameExecutor& executor) override;
	
	void writeYuvFiles(const std::string& inputFile, const std::string& outputFile, int maxFrames);
};


//-----------------------------------------------------------------------------------
class ImageWriter : public NullWriter {

protected:
	ImageWriter(MainData& data, MovieReader& reader) :
		NullWriter(data, reader) 
	{}

	std::string makeFilename(const std::string& extension) const;

public:
	static std::string makeFilename(const std::string& pattern, int64_t index, const std::string& extension);
	static std::string makeFilenameSamples(const std::string& pattern, const std::string& extension);
};


//-----------------------------------------------------------------------------------
class BmpImageWriter : public ImageWriter {

private:
	ImageBGRA imageBgra;
	std::jthread worker;

public:
	BmpImageWriter(MainData& data, MovieReader& reader) :
		ImageWriter(data, reader),
		worker { [] {} },
		imageBgra(data.h, data.w)
	{}

	void close() override;
	void writeOutput(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class JpegImageWriter : public ImageWriter {

private:
	ImageYuv output;
	AVCodecContext* ctx = nullptr;
	AVFrame* av_frame = nullptr;
	AVPacket* packet = nullptr;
	SwsContext* swsCtx = nullptr;

public:
	JpegImageWriter(MainData& data, MovieReader& reader) :
		ImageWriter(data, reader),
		output(data.h, data.w)
	{}

	~JpegImageWriter() override;
	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class RawNv12Writer : public NullWriter {

private:
	std::ofstream file;
	ImageNV12 nv12;

public:
	RawNv12Writer(MainData& data, MovieReader& reader) :
		NullWriter(data, reader),
		nv12(data.h, data.w, data.stride)
	{}

	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class RawYuvWriter : public NullWriter {

private:
	std::ofstream file;
	ImageYuv yuv;

public:
	RawYuvWriter(MainData& data, MovieReader& reader) :
		NullWriter(data, reader),
		yuv(data.h, data.w, data.stride)
	{}

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
class PipeWriter {

protected:
	virtual void openPipe();
	virtual void closePipe();
};


//-----------------------------------------------------------------------------------
class RawPipeWriter : public NullWriter, public PipeWriter {

private:
	ImageYuv output;

public:
	RawPipeWriter(MainData& data, MovieReader& reader) :
		NullWriter(data, reader),
		output(data.h, data.w)
	{}

	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
	void close() override;
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
	AsfPipeWriter(MainData& data, MovieReader& reader) :
		FFmpegWriter(data, reader, 0),
		output(data.h, data.w)
	{}

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
	StackedWriter(MainData& data, MovieReader& reader) :
		FFmpegWriter(data, reader, 1),
		mWidth { data.w - data.stackCrop.left - data.stackCrop.right },
		mWidthTotal { 2 * mWidth },
		mInputFrame(data.h, data.w, data.stride4),
		mOutputFrame(data.h, data.w, data.stride4) 
	{}

	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
};


//------------- transformation file -------------------------------------------------
class TransformsFile {

protected:
	inline static std::string id = "CUVI";
	std::ofstream mFile;

	template <class T> void writeValue(T val) {
		mFile.write(reinterpret_cast<const char*>(&val), sizeof(val));
	}

public:
	TransformsFile() {}

	static std::map<int64_t, TransformValues> readTransformMap(const std::string& trajectoryFile);

	void open(const std::string& trajectoryFile);
	void writeTransform(const Affine2D& transform, int64_t frameIndex);
};


//--------------- write transforms --------------------------------------------------
class TransformsWriter : public MovieWriterBase, public TransformsFile {

public:
	TransformsWriter(MainData& data) :
		MovieWriterBase(data),
		TransformsFile() 
	{}

	void start() override;
	void writeInput(const FrameExecutor& executor) override;
};


//optical flow video
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


//--------------- write point results as large text file ----------------------------
class ResultDetailsWriter : public MovieWriterBase {

private:
	std::string mDelim = ";";
	std::ofstream mFile;

	void write(std::span<PointResult> results, int64_t frameIndex);

public:
	ResultDetailsWriter(MainData& data) :
		MovieWriterBase(data) {}

	static void write(std::span<PointResult> results, const std::string& filename);

	void start() override;
	void writeInput(const FrameExecutor& executor) override;
};


//--------------- write individual images to show point results ---------------------
class ResultImageWriter : public MovieWriterBase {

private:
	ImageVuyx yuv;
	ImageBGRA bgra;

public:
	ResultImageWriter(MainData& data) :
		MovieWriterBase(data),
		yuv(data.h, data.w),
		bgra(data.h, data.w) 
	{}

	ResultImageWriter(MainData& data, MovieReader& reader) :
		ResultImageWriter(data)
	{}

	void start() override {}
	void writeInput(const FrameExecutor& executor) override;

	static void writeImage(const AffineTransform& trf, std::span<PointResult> res, int64_t idx, Image8& dest, 
		ThreadPoolBase& pool = defaultPool, bool drawTransformed = true);
};


//--------------- write individual images to show point results ---------------------
class ResultVideoWriter : public MovieWriterBase {

private:
	std::ofstream file;
	ImageNV12 nv12;
	ImageBGRA bgra;

public:
	ResultVideoWriter(MainData& data, MovieReader& reader) :
		MovieWriterBase(data)
	{}

	void open(OutputOption outputOption) override;
	void writeInput(const FrameExecutor& executor) override;
};


//--------------------------------------------------------------------------------------
//simple writer for debugging, writes YUV data to file, cannot be used in executor loop
class SimpleYuvWriter {

private:
	std::ofstream os;

public:
	SimpleYuvWriter(const std::string& file);

	void write(ImageVuyx& image);
};