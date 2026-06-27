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

#include <unordered_map>
#include <string>
#include <memory>


enum class ReaderType {
	FFMPEG,
	MEMORY,
};

enum class WriterType {
	FFMPEG,
	CUDA,
	STACKED,
	FLOW,
	ASF_PIPE,
	JPEG_IMAGE,
};

struct FFmpegVersions {
	unsigned int avutil, avcodec, avformat, swscale, swresample;

	auto operator <=> (const FFmpegVersions& v) const = default;
	friend std::ostream& operator << (std::ostream& os, const FFmpegVersions& v);
	std::string toString(unsigned int val) const;
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

struct AVStream;
struct StreamContext;
struct OutputStreamContext;
class FrameExecutor;
namespace im { class Image8; }


enum class MediaType {
	AUDIO,
	VIDEO,
	OTHER,
};

//general info about stream
struct StreamInfo {
	std::string streamType;
	std::string codec;
	std::string durationString;
	MediaType mediaType;
	int index;

	std::string inputStreamSummary(const std::string& delimiter) const;
};

//packet of decodec audio for playing live
struct DecodedAudioPacket {
	int64_t frameIndex;
	double pts;
	std::vector<uint8_t> audioData;
};

//base class, implementation is in ffmpeg dll
struct OutputStreamContextBase {
	StreamHandling handling = StreamHandling::STREAM_IGNORE;

	virtual std::list<DecodedAudioPacket> getAudioData(double ptsLimit) = 0;
};

//base class, implementation is in ffmpeg dll
struct StreamContextBase {
	virtual AVStream* getInputStream() = 0;
	virtual StreamInfo inputStreamInfo() const = 0;
	virtual int inputStreamIndex() const = 0;

	virtual size_t outputStreamsCount() const = 0;
	virtual std::shared_ptr<OutputStreamContextBase> addOutputStreamContext() = 0;
	virtual std::shared_ptr<OutputStreamContextBase> getOutputStreamContext(size_t index) = 0;
};

//map enum to info string
inline std::unordered_map<StreamHandling, std::string> streamHandlerMap = {
	{StreamHandling::STREAM_COPY, "copy"},
	{StreamHandling::STREAM_IGNORE, "ignore"},
	{StreamHandling::STREAM_STABILIZE, "stabilize"},
	{StreamHandling::STREAM_TRANSCODE, "transcode"},
	{StreamHandling::STREAM_DECODE, "decode"},
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
std::string millisToTimeString(int64_t millis);
