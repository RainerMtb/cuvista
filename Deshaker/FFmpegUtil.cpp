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

#include "FFmpegUtil.hpp"
#include <format>
#include <iostream>


std::string FFmpegVersions::toString(unsigned int val) const {
    return std::format("{}.{}.{}", (val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF);
}

std::ostream& operator << (std::ostream& os, const FFmpegVersions& v) {
    os << "FFMPEG Versions:" << std::endl;
    os << "libavutil:     " << v.toString(v.avutil) << std::endl;
    os << "libavcodec:    " << v.toString(v.avcodec) << std::endl;
    os << "libavformat:   " << v.toString(v.avformat) << std::endl;
    os << "libswscale:    " << v.toString(v.swscale) << std::endl;
    os << "libswresample: " << v.toString(v.swresample) << std::endl;
    return os;
}

std::ostream& operator << (std::ostream& ostream, const Timings& t) {
    ostream << "pts=" << t.pts << ", dts=" << t.dts << ", dur=" << t.duration;
    return ostream;
}

std::string millisToTimeString(int64_t millis) {
	int64_t sign = millis < 0 ? -1 : 1;
	millis = std::abs(millis);
	int64_t sec = millis / 1000;
	int64_t min = sec / 60;
	int64_t hrs = min / 60;

	millis %= 1000;
	sec %= 60;
	min %= 60;
	hrs %= 60;

	std::string timeString = "";
	if (hrs > 0) timeString = std::format("{}:{:02}:{:02}.{:03}", hrs * sign, min, sec, millis);
	else timeString = std::format("{:02}:{:02}.{:03}", min * sign, sec, millis);
	return timeString;
}

std::string StreamInfo::inputStreamSummary(const std::string& delimiter) const {
	return std::format("stream #{}{}type: {}, codec: {}, duration: {}", index, delimiter, streamType, codec, durationString);
}
