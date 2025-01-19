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

#include <list>
#include <map>
#include "ErrorLogger.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/audio_fifo.h>
}

struct FFmpegVersions {
	unsigned int avutil, avcodec, avformat, swscale, swresample;

	auto operator <=> (const FFmpegVersions&) const = default;
};


//what to do with any input stream
enum class StreamHandling {
	STREAM_UNKNOWN,
	STREAM_IGNORE,
	STREAM_STABILIZE,
	STREAM_COPY,
	STREAM_TRANSCODE,
	STREAM_DECODE,
};

inline std::map<StreamHandling, std::string> streamHandlerMap = {
	{StreamHandling::STREAM_COPY, "copy"},
	{StreamHandling::STREAM_IGNORE, "ignore"},
	{StreamHandling::STREAM_STABILIZE, "stabilize"},
	{StreamHandling::STREAM_TRANSCODE, "transcode"},
	{StreamHandling::STREAM_DECODE, "decode"},
};

struct StreamInfo {
	std::string streamType;
	std::string codec;
	std::string durationString;
	AVMediaType mediaType;
};

class SidePacket {
public:
	int64_t frameIndex;
	AVPacket* packet;
	std::vector<uint8_t> audioData;
	double pts;

	SidePacket(int64_t frameIndex, double pts);
	SidePacket(int64_t frameIndex, const AVPacket* packet);
	~SidePacket();
};

//structure per stream in input file
struct StreamContext {
	AVStream* inputStream = nullptr;
	int64_t durationMillis = -1;
	AVStream* outputStream = nullptr;

	StreamHandling handling = StreamHandling::STREAM_IGNORE;
	std::list<SidePacket> packets;
	int64_t packetsWritten;

	AVCodecContext* audioInCtx = nullptr;
	AVCodecContext* audioOutCtx = nullptr;
	const AVCodec* audioInCodec = nullptr;
	const AVCodec* audioOutCodec = nullptr;
	AVPacket* outpkt = nullptr;
	AVFrame* frameIn = nullptr;
	AVFrame* frameOut = nullptr;
	int64_t lastPts = 0;
	int64_t pts = 0;
	SwrContext* resampleCtx = nullptr;
	AVAudioFifo* fifo = nullptr;

	StreamInfo inputStreamInfo() const;

	~StreamContext();
};

//timing values for input packets
struct VideoPacketContext {
	int64_t readIndex;
	int64_t pts;
	int64_t dts;
	int64_t duration;
	int64_t bestTimestamp;
};

//structure holding timing infos
struct Timings {
	int64_t pts, dts, duration;
	double ptsTime, dtsTime;

	friend std::ostream& operator << (std::ostream& ostream, const Timings& t);
};

//convert millis into readable string hh:mm:ss.fff
std::string timeString(int64_t millis);

//generate error string from ffmpeg return codes
std::string av_make_error(int errnum, const char* msg = "");

//log error from ffmpeg retunr codes
void ffmpeg_log_error(int errnum, const char* msg, ErrorSource source);

//callback from ffmpeg to report errors
void ffmpeg_log(void* avclass, int level, const char* fmt, va_list args);

//compare version strings at compiletime and runtime
bool ffmpeg_check_versions();