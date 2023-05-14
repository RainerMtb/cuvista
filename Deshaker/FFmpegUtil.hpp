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

#include <vector>
#include <string>
#include <list>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}


//timing values for input packets
struct VideoPacketContext {
	int64_t readIndex;
	int64_t codedIndex;
	int64_t pts;
	int64_t dts;
	int64_t duration;
	int64_t pos;
};

//structure holding timing infos
struct Timings {
	int64_t pts, dts, duration;
	double ptsTime, dtsTime;

	friend std::ostream& operator << (std::ostream& ostream, const Timings& t);
};

struct StreamInfo {
	std::string streamType;
	std::string codec;
	std::string durationString;
};

//parameters describing input
class InputContext {

public:
	int h = 0, w = 0;
	int fpsNum = -1, fpsDen = -1;
	int64_t timeBaseNum = -1, timeBaseDen = -1;
	int64_t frameCount = 0;
	int64_t formatDuration = -1;

	std::vector<AVStream*> inputStreams;
	AVStream* videoStream = nullptr;

	double fps() const;
	StreamInfo streamInfo(AVStream* stream) const;
	StreamInfo videoStreamInfo() const;
};

std::string timeString(int64_t millis);

std::string av_make_error(int errnum, const char* msg = "");

void ffmpeg_log(void* avclass, int level, const char* fmt, va_list args);
