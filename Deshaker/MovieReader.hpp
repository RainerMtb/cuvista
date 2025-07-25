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

#include "AVException.hpp"
#include "Stats.hpp"
#include "Image2.hpp"

#include <span>
#include <optional>


class MovieReader : public ReaderStats {

public:
	std::mutex mVideoPacketMutex;
	std::list<VideoPacketContext> mVideoPacketList;
	std::vector<StreamContext> mInputStreams;
	bool mStoreSidePackets = true;

	virtual ~MovieReader() = default;

	virtual void open(const std::string& source) = 0;
	virtual void start() {}
	virtual bool read(ImageYuv& inputFrame) = 0;
	virtual std::future<void> readAsync(ImageYuv& inputFrame);
	virtual void close() {}
	virtual void rewind() {}
	virtual bool seek(double fraction) { return true; }
	virtual int openAudioDecoder(OutputStreamContext& osc) { return -1; }

	std::optional<std::string> ptsForFrameAsString(int64_t frameIndex);
	std::optional<int64_t> ptsForFrameAsMillis(int64_t frameIndex);
	double ptsForFrame(int64_t frameIndex);

protected:
	int sideDataMaxSize = 20 * 1024 * 1024;
};


class NullReader : public MovieReader {

public:
	void open(const std::string& source) override {};
	bool read(ImageYuv& inputFrame) override;
};


class ImageReader : public MovieReader {

public:
	void open(const std::string& source) override {};
	bool read(ImageYuv& inputFrame) override { return false; };

	bool readImage(ImageYuv& inputFrame, const ImageYuv& sourceImage);
};


class FFmpegFormatReader : public MovieReader {

protected:
	bool isFormatOpen = false;
	bool isStoredPacket = false;
	AVFormatContext* av_format_ctx = nullptr;
	AVCodecContext* av_codec_ctx = nullptr;
	AVFrame* av_frame = nullptr;
	AVPacket* av_packet = nullptr;

	void close() override;
	~FFmpegFormatReader() override;

	void openInput(AVFormatContext* fmt, const std::string& source);
};


//main class to decode input
class FFmpegReader : public FFmpegFormatReader {

private:
	SwsContext* sws_scaler_ctx = nullptr;
	AVSampleFormat decodingSampleFormat = AV_SAMPLE_FMT_FLT;

public:
	~FFmpegReader() override;

	void open(const std::string& source) override;
	bool read(ImageYuv& inputFrame) override;
	void close() override;
	void rewind() override;
	int openAudioDecoder(OutputStreamContext& osc) override;

	bool seek(double fraction);
};


//provide data to ffmpeg from memory
class MemoryFFmpegReader : public FFmpegReader {

private:
	AVIOContext* av_avio = nullptr;
	unsigned char* mBuffer = nullptr;

	std::span<unsigned char> mData;
	int64_t mDataPos = 0;

	static int readBuffer(void* opaque, unsigned char* buf, int bufsiz);
	static int64_t seekBuffer(void* opaque, int64_t offset, int whence);
	
public:
	MemoryFFmpegReader(std::span<unsigned char> movieData);
	~MemoryFFmpegReader() override;

	void open(const std::string& source) override;
};
