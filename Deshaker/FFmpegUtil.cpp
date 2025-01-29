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
#include "Util.hpp"
#include "ErrorLogger.hpp"
#include <format>
#include <numeric>


std::string av_make_error(int errnum, const char* msg) {
    char av_errbuf[AV_ERROR_MAX_STRING_SIZE];
    std::string info = msg;
    if (!info.empty()) info += ": ";
    info += av_make_error_string(av_errbuf, AV_ERROR_MAX_STRING_SIZE, errnum);
    return info;
}

void ffmpeg_log_error(int errnum, const char* msg, ErrorSource source) {
    errorLogger.logError(av_make_error(errnum, msg), source);
}

static constexpr FFmpegVersions ffmpeg_build_versions = { 
    LIBAVUTIL_VERSION_INT, 
    LIBAVCODEC_VERSION_INT, 
    LIBAVFORMAT_VERSION_INT, 
    LIBSWSCALE_VERSION_INT, 
    LIBSWRESAMPLE_VERSION_INT 
};

bool ffmpeg_check_versions() {
    FFmpegVersions ffmpeg_runtime_versions = {
        avutil_version(),
        avcodec_version(),
        avformat_version(),
        swscale_version(),
        swresample_version()
    };
    return ffmpeg_build_versions == ffmpeg_runtime_versions;
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
        errorLogger.logError(ffmpeg_logbuf, ErrorSource::FFMPEG);
    }
};

std::ostream& operator << (std::ostream& ostream, const Timings& t) {
    ostream << "pts=" << t.pts << ", dts=" << t.dts << ", dur=" << t.duration;
    return ostream;
}

SidePacket::SidePacket(int64_t frameIndex, double pts) {
    this->frameIndex = frameIndex;
    this->packet = nullptr;
    this->pts = pts;
}

SidePacket::SidePacket(int64_t frameIndex, const AVPacket* packet) {
    this->frameIndex = frameIndex;
    this->packet = av_packet_clone(packet);
    this->pts = std::numeric_limits<double>::quiet_NaN();
}

SidePacket::~SidePacket() {
    if (packet) {
        av_packet_free(&packet);
    }
}

StreamContext::~StreamContext() {
    packets.clear();

    if (audioInCtx) {
        avcodec_free_context(&audioInCtx);
    }
    if (audioOutCtx) {
        avcodec_free_context(&audioOutCtx);
    }
    if (outpkt) {
        av_packet_free(&outpkt);
    }
    if (frameIn) {
        av_frame_free(&frameIn);
    }
    if (frameOut) {
        av_frame_free(&frameOut);
    }
    if (resampleCtx) {
        swr_free(&resampleCtx);
    }
    if (fifo) {
        av_audio_fifo_free(fifo);
    }
}

StreamInfo StreamContext::inputStreamInfo() const {
    std::string tstr;
    if (inputStream->duration != AV_NOPTS_VALUE)
        tstr = util::millisToTimeString(inputStream->duration * inputStream->time_base.num * 1000 / inputStream->time_base.den);
    else if (durationMillis != -1)
        tstr = util::millisToTimeString(durationMillis);
    else
        tstr = "unknown";

    AVCodecParameters* param = inputStream->codecpar;
    return { av_get_media_type_string(param->codec_type), avcodec_get_name(param->codec_id), tstr, param->codec_type };
}