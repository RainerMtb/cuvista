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

#include <set>
#include "MainData.hpp"
#include "FFmpegMain.hpp"
#include "Reader.hpp"
#include "Writer.hpp"
#include "CudaInterface.hpp"

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
    switch (readerType) {
    case ReaderType::FFMPEG:
        return new FFmpegReader();
        break;

    case ReaderType::MEMORY:
        return new MemoryFFmpegReader();
        break;
    }
    return nullptr;
}

//create and return a writer instance
MovieWriter* createWriter(OutputOption option, MainData& data, MovieReader& reader) {
    if (option == OutputOption::VIDEO_STACK)
        return new StackedWriter(data, reader);
    else if (option == OutputOption::VIDEO_FLOW)
        return new OpticalFlowWriter(data, reader);
    else if (option == OutputOption::PIPE_ASF)
        return new AsfPipeWriter(data, reader);
    else if (option == OutputOption::IMAGE_JPG)
        return new JpegImageWriter(data, reader);
    else if (option.group == OutputGroup::VIDEO_FFMPEG)
        return new FFmpegWriter(data, reader);
    else if (option.group == OutputGroup::VIDEO_NVENC)
        return new CudaFFmpegWriter(data, reader);
    else if (option.group == OutputGroup::VIDEO_VULKAN)
        return new VulkanFFmpegWriter(data, reader);
    else 
        return nullptr;
}

//check writer capability
void probeEncoders(MainData& data) {
    //check ffmpeg encoders
    {
        std::set<OutputOption> s;
        void* codecState = nullptr; //start with null
        const AVCodec* codec = av_codec_iterate(&codecState);
        while (codec) {
            if (codec->id == AV_CODEC_ID_H264) s.insert(OutputOption::FFMPEG_H264);
            if (codec->id == AV_CODEC_ID_HEVC) s.insert(OutputOption::FFMPEG_HEVC);
            if (codec->id == AV_CODEC_ID_AV1) s.insert(OutputOption::FFMPEG_AV1);
            if (codec->id == AV_CODEC_ID_FFV1) s.insert(OutputOption::FFMPEG_FFV1);
            codec = av_codec_iterate(&codecState);
        }
        data.deviceInfoCpu.encoders = std::vector<OutputOption>(s.crbegin(), s.crend());
    }

    //check cuda encoders
    {
        DeviceInfoCudaCollection& cuda = data.deviceInfoCuda;
        for (int i = 0; i < cuda.devices.size(); i++) {
            NvEncoder nvenc(i);
            nvenc.probeEncoding(&cuda.nvencVersionApi, &cuda.nvencVersionDriver);

            if (cuda.nvencVersionDriver >= cuda.nvencVersionApi) {
                //check supported codecs
                cuda.devices[i].encoders = nvenc.probeSupportedCodecs();
            }
        }
        if (cuda.devices.size() > 0) {
            cuda.encoders = cuda.devices[0].encoders;
        }
    }

    //check vulkan encoders
    {
        DeviceInfoVulkanCollection& vulkan = data.deviceInfoVulkan;
        NoOpReader reader;
        reader.fpsNum = 25;
        reader.fpsDen = 1;
        reader.parNum = 1;
        reader.parDen = 1;
        auto noopLogger = [] (void* avclass, int level, const char* fmt, va_list args) {};
        av_log_set_callback(noopLogger);
        std::vector<OutputOption> vulkanOptions = { OutputOption::VULKAN_AV1, OutputOption::VULKAN_HEVC, OutputOption::VULKAN_H264 };
        for (OutputOption op : vulkanOptions) {
            try {
                VulkanFFmpegWriter writer(data, reader);
                writer.openEncoder(op, false, 640, 480, 20);
                vulkan.encoders.push_back(op);

            } catch (AVException e) {

            } catch (...) {}
        }
    }
}

//set global symbols
void init(std::shared_ptr<ErrorLogger> errorLoggerInstance) {
    ::errorLoggerInstance = errorLoggerInstance;
}

//convert ffmpeg error number to message
std::string av_make_error(int errnum, const char* msg, const std::string& str) {
    std::string info = msg + str;
    if (info.size() > 0) info += ": ";
    char av_errbuf[AV_ERROR_MAX_STRING_SIZE];
    info += av_make_error_string(av_errbuf, AV_ERROR_MAX_STRING_SIZE, errnum);
    return info;
}

void ffmpeg_log_error(int errnum, const char* msg, ErrorSource source) {
    errorLogger().logError(av_make_error(errnum, msg), source);
}

void ffmpeg_log(void* avclass, int level, const char* fmt, va_list args) {
    if (level <= AV_LOG_INFO) {
        //collect ffmpeg log
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

        //set error message
        if (level <= AV_LOG_FATAL) {
            errorLogger().logError(ffmpeg_logbuf, ErrorSource::FFMPEG);

        } else {
            errorLogger().logFFmpeg(level, ffmpeg_logbuf);
        }
    }
};

SidePacket::SidePacket(int64_t frameIndex, const AVPacket* packet) :
    frameIndex { frameIndex },
    packet { av_packet_clone(packet) },
    pts { std::numeric_limits<double>::quiet_NaN() }
{}

SidePacket::~SidePacket() {
    if (packet) {
        av_packet_free(&packet);
    }
}

std::list<DecodedAudioPacket> OutputStreamContext::getAudioData(double ptsLimit) {
    std::unique_lock<std::mutex> lock(mMutexSidePackets);
    auto it = audioPackets.begin();
    while (it != audioPackets.end() && it->pts < ptsLimit) it++;
    std::list<DecodedAudioPacket> out;
    out.splice(out.begin(), audioPackets, audioPackets.begin(), it);
    return out;
}

OutputStreamContext::~OutputStreamContext() {
    sidePackets.clear();

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

AVStream* StreamContext::getInputStream() {
    return inputStream;
}

int StreamContext::inputStreamIndex() const {
    return inputStream->index;
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
    StreamInfo info;
    info.streamType = av_get_media_type_string(param->codec_type),
    info.codec = avcodec_get_name(param->codec_id),
    info.durationString = tstr,
    info.index = inputStream->index;

    info.mediaType = MediaType::OTHER;
    if (param->codec_type == AVMEDIA_TYPE_VIDEO) info.mediaType = MediaType::VIDEO;
    if (param->codec_type == AVMEDIA_TYPE_AUDIO) info.mediaType = MediaType::AUDIO;
    
    return info;
}

std::shared_ptr<OutputStreamContextBase> StreamContext::addOutputStreamContext() {
    auto osc = std::make_shared<OutputStreamContext>();
    osc->inputStream = inputStream;
    outputStreams.push_back(osc);
    return osc;
}

size_t StreamContext::outputStreamsCount() const {
    return outputStreams.size();
}

std::shared_ptr<OutputStreamContextBase> StreamContext::getOutputStreamContext(size_t index) {
    return outputStreams[index];
}
