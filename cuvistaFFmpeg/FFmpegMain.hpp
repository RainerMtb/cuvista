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

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libswscale/swscale.h"
#include "libswresample/swresample.h"
#include "libavutil/opt.h"
#include "libavutil/audio_fifo.h"
}

#include <string>
#include <vector>
#include <list>
#include <memory>
#include <mutex>

#include "FFmpegUtil.hpp"
#include "ErrorLogger.hpp"


class SidePacket {
public:
	int64_t frameIndex;
	AVPacket* packet;
	double pts;

	SidePacket(int64_t frameIndex, const AVPacket* packet);
	~SidePacket();
};

//structure per output stream
struct OutputStreamContext : public OutputStreamContextBase {
	AVStream* inputStream = nullptr;

	AVStream* outputStream = nullptr;
	std::list<SidePacket> sidePackets;
	std::list<DecodedAudioPacket> audioPackets;
	int64_t packetsWritten;

	AVCodecContext* audioInCtx = nullptr;
	const AVCodec* audioInCodec = nullptr;
	AVFrame* frameIn = nullptr;

	AVCodecContext* audioOutCtx = nullptr;
	const AVCodec* audioOutCodec = nullptr;
	AVPacket* outpkt = nullptr;
	AVFrame* frameOut = nullptr;
	int64_t lastPts = 0;
	int64_t ptsTranscoded = 0;
	int64_t ptsWritten = INT64_MIN;
	SwrContext* resampleCtx = nullptr;
	AVAudioFifo* fifo = nullptr;

	std::mutex mMutexSidePackets;

	std::list<DecodedAudioPacket> getAudioData(double ptsLimit) override;

	~OutputStreamContext();
};

//structure per stream in input file
struct StreamContext : public StreamContextBase {
	AVStream* inputStream = nullptr;
	int64_t durationMillis = -1;
	std::vector<std::shared_ptr<OutputStreamContext>> outputStreams;

	AVStream* getInputStream() override;
	StreamInfo inputStreamInfo() const override;
	int inputStreamIndex() const override;

	size_t outputStreamsCount() const override;
	std::shared_ptr<OutputStreamContextBase> addOutputStreamContext() override;
	std::shared_ptr<OutputStreamContextBase> getOutputStreamContext(size_t index) override;
};


//generate error string from ffmpeg return codes
std::string av_make_error(int errnum, const char* msg = "", const std::string& str = "");

//log error from ffmpeg return codes
void ffmpeg_log_error(int errnum, const char* msg, ErrorSource source);

//callback from ffmpeg to report errors
void ffmpeg_log(void* avclass, int level, const char* fmt, va_list args);


class MovieReader;
class MovieWriter;
class MainData;

#if defined(_WIN64)
#define LIBRARY_EXPORT extern "C" __declspec(dllexport)
#else
#define LIBRARY_EXPORT extern "C"
#endif

//library functions to get versions
LIBRARY_EXPORT const FFmpegVersions* versionsCompiled();
LIBRARY_EXPORT const FFmpegVersions* versionsRuntime();

//library functions to get classes
LIBRARY_EXPORT MovieReader* createReader(ReaderType readerType);
LIBRARY_EXPORT MovieWriter* createWriter(WriterType writerType, MainData& data, MovieReader& reader);
