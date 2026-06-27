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

#include "MovieReaderBase.hpp"
#include "FFmpegMain.hpp"

class FFmpegFormatReader : public MovieReader {

protected:
	bool isFormatOpen = false;
	bool isStoredPacket = false;
	AVFormatContext* av_format_ctx = nullptr;
	AVCodecContext* av_codec_ctx = nullptr;
	AVFrame* av_frame = nullptr;
	AVPacket* av_packet = nullptr;
	AVStream* av_stream = nullptr;
	std::vector<std::shared_ptr<StreamContext>> mInputStreams;

	size_t inputStreamCount() const override;
	std::shared_ptr<StreamContext> inputStream(size_t index) const override;
	std::shared_ptr<StreamContextBase> inputStreamBase(size_t index) const override;
	void close() override;
	~FFmpegFormatReader() override;

	void openInput(AVFormatContext* fmt, const std::string& source);
};


//main class to decode input
class FFmpegReader : public FFmpegFormatReader {

private:
	SwsContext* sws_scaler_ctx = nullptr;
	AVChannelLayout bufferChannelFormat = AV_CHANNEL_LAYOUT_STEREO;
	AVSampleFormat bufferSampleFormat = AV_SAMPLE_FMT_FLT;

public:
	~FFmpegReader() override;

	void open(const std::string& source) override;
	bool read(im::Image8& inputFrame) override;
	bool read(FrameExecutor& executor) override;
	void close() override;
	bool seek(double fraction);
	void rewind() override;
	int openAudioDecoder(std::shared_ptr<OutputStreamContextBase> sposc) override;
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
	~MemoryFFmpegReader() override;

	void open(std::span<unsigned char> movieData) override;
};
