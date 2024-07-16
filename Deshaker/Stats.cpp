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

#include "Stats.hpp"
#include <format>

double ReaderStats::fps() const {
    return 1.0 * fpsNum / fpsDen;
}

StreamInfo ReaderStats::streamInfo(AVStream* stream) const {
    std::string tstr;
    if (stream->duration != AV_NOPTS_VALUE)
        tstr = timeString(stream->duration * stream->time_base.num * 1000 / stream->time_base.den);
    else if (avformatDuration != AV_NOPTS_VALUE)
        tstr = timeString(avformatDuration * stream->time_base.num / stream->time_base.den);
    else
        tstr = "unknown";

    AVCodecParameters* param = stream->codecpar;
    return { av_get_media_type_string(param->codec_type), avcodec_get_name(param->codec_id), tstr };
}

StreamInfo ReaderStats::videoStreamInfo() const {
    return streamInfo(videoStream);
}
