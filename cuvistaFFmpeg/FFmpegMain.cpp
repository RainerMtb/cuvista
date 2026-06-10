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

#include "FFmpegMain.hpp"

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libswscale/swscale.h"
#include "libswresample/swresample.h"
#include "libavutil/opt.h"
#include "libavutil/audio_fifo.h"
}

namespace ff {

    static constexpr FFmpegVersions ffmpeg_build_versions = {
        .avutil = LIBAVUTIL_VERSION_INT,
        .avcodec = LIBAVCODEC_VERSION_INT,
        .avformat = LIBAVFORMAT_VERSION_INT,
        .swscale = LIBSWSCALE_VERSION_INT,
        .swresample = LIBSWRESAMPLE_VERSION_INT
    };

    static FFmpegVersions ffmpeg_runtime_versions;

    const FFmpegVersions* versionsCompiled() {
        return &ffmpeg_build_versions;
    }

    const FFmpegVersions* versionsRuntime() {
        ffmpeg_runtime_versions = {
            .avutil = avutil_version(),
            .avcodec = avcodec_version(),
            .avformat = avformat_version(),
            .swscale = swscale_version(),
            .swresample = swresample_version()
        };
        return &ffmpeg_runtime_versions;
    }

    MovieReader* createReader(ReaderType readerType) {
        return nullptr;
    }

    MovieWriter* createWriter(WriterType writerType) {
        return nullptr;
    }
}
