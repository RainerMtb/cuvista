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

#include <format>
#include "FFmpegUtil.hpp"
#include "ErrorLogger.hpp"


std::string av_make_error(int errnum, const char* msg) {
    char av_errbuf[AV_ERROR_MAX_STRING_SIZE];
    std::string info = msg;
    if (!info.empty()) info += ": ";
    info += av_make_error_string(av_errbuf, AV_ERROR_MAX_STRING_SIZE, errnum);
    return info;
}

void ffmpeg_log_error(int errnum, const char* msg) {
    errorLogger.logError(av_make_error(errnum, msg));
}

void ffmpeg_log(void* avclass, int level, const char* fmt, va_list args) {
    if (level <= AV_LOG_ERROR) {
        const size_t ffmpeg_bufsiz = 256;
        char ffmpeg_logbuf[ffmpeg_bufsiz];
        std::vsnprintf(ffmpeg_logbuf, ffmpeg_bufsiz, fmt, args);

        //trim trailing newline
        char* ptr = ffmpeg_logbuf;
        while (ptr != ffmpeg_logbuf + ffmpeg_bufsiz && *ptr != '\0') {
            ptr++;
        }
        ptr--;
        while (ptr >= ffmpeg_logbuf && *ptr == '\n') {
            *ptr = '\0';
            ptr--;
        }
        errorLogger.logError(ffmpeg_logbuf);
    }
};

std::ostream& operator << (std::ostream& ostream, const Timings& t) {
    ostream << "pts=" << t.pts << ", dts=" << t.dts << ", dur=" << t.duration;
    return ostream;
}

std::string timeString(int64_t millis) {
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

StreamContext::~StreamContext() {
    for (AVPacket* packet : packets) {
        av_packet_free(&packet);
    }
}