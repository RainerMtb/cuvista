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

#include <filesystem>
#include "MovieReader.hpp"
#include "MovieWriter.hpp"
#include "MovieFrame.hpp"
#include "Util.hpp"


void FFmpegWriter::open(AVCodecID codecId, AVPixelFormat pixfmt, int h, int w, int stride) {
    int result = 0;

    //find and open codec
    const AVCodec* codec = nullptr;
    for (const std::string& codecName : codecToNamesMap[codecId]) {
        if (codec = avcodec_find_encoder_by_name(codecName.c_str())) break;
    }
    if (!codec)
        throw AVException("Could not find encoder");

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
        throw AVException("Could not allocate encoder context");

    codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_ctx->width = w;
    codec_ctx->height = h;
    codec_ctx->pix_fmt = pixfmt;
    codec_ctx->framerate = { mReader.fpsNum, mReader.fpsDen };
    codec_ctx->time_base = { mReader.fpsDen, mReader.fpsNum };
    codec_ctx->gop_size = gopSize;
    codec_ctx->max_b_frames = 4;
    //av_opt_set(codec_ctx->priv_data, "preset", "slow", 0);
    av_opt_set(codec_ctx->priv_data, "profile", "main", 0);
    av_opt_set(codec_ctx->priv_data, "x265-params", "log-level=error", 0);
    av_opt_set(codec_ctx->priv_data, "svtav1-params", "svt-log-level=1", 0);
    if (mData.crf) av_opt_set(codec_ctx->priv_data, "crf", std::to_string(mData.crf.value()).c_str(), 0);

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    //open encoder
    result = avcodec_open2(codec_ctx, codec, NULL);
    if (result < 0)
        throw AVException(av_make_error(result, "error opening codec"));

    result = avcodec_parameters_from_context(videoStream->codecpar, codec_ctx);
    if (result < 0)
        throw AVException(av_make_error(result, "error setting codec parameters"));

    result = avformat_init_output(fmt_ctx, NULL);
    if (result < 0)
        throw AVException(av_make_error(result, "error initializing output"));

    result = avformat_write_header(fmt_ctx, NULL);
    if (result < 0)
        throw AVException(av_make_error(result, "error writing file header"));
    else
        this->isHeaderWritten = true; //store info for proper closing

    //av_dump_format(fmt_ctx, 0, fmt_ctx->url, 1);

    videoPacket = av_packet_alloc();
    if (!videoPacket)
        throw AVException("Could not allocate encoder packet");

    //allocate one av_frame to be used on encoding
    av_frame = av_frame_alloc();
    if (!av_frame)
        throw AVException("Could not allocate video frame");

    av_frame->format = codec_ctx->pix_fmt;
    av_frame->width = codec_ctx->width;
    av_frame->height = codec_ctx->height;

    result = av_frame_get_buffer(av_frame, 0);
    if (result < 0)
        throw AVException("Could not get frame buffer");

    result = av_frame_make_writable(av_frame);
    if (result < 0)
        throw AVException("Could not make frame writable");
}


//set up ffmpeg encoder
void FFmpegWriter::open(EncodingOption videoCodec, AVPixelFormat pixfmt, int h, int w, int stride, const std::string& sourceName) {
    //open container format
    AVCodecID codecID = codecToCodecIdMap[videoCodec.codec];
    FFmpegFormatWriter::open(codecID, sourceName, imageBufferSize);

    open(codecID, pixfmt, h, w, stride);

    sws_scaler_ctx = sws_getContext(w, h, AV_PIX_FMT_YUV444P, w, h, pixfmt, SWS_BILINEAR, NULL, NULL, NULL);
    if (!sws_scaler_ctx) 
        throw AVException("Could not get scaler context");

    //allocate images to be cycled through
    for (int i = 0; i < imageBufferSize; i++) {
        imageBuffer.emplace_back(h, w, stride);
    }
}


//normal entry point for opening Writer
void FFmpegWriter::open(EncodingOption videoCodec) {
    open(videoCodec, AV_PIX_FMT_YUV420P, mData.h, mData.w, mData.cpupitch, mData.fileOut);
}


void FFmpegWriter::prepareOutput(FrameExecutor& executor) {
    int64_t idx = frameIndex % imageBufferSize;
    executor.getOutputYuv(frameIndex, imageBuffer[idx]);
}


int FFmpegWriter::sendFFmpegFrame(AVFrame* av_frame) {
    //util::ConsoleTimer timer("send ffmpeg");
    int result = avcodec_send_frame(codec_ctx, av_frame);
    if (result < 0)
        ffmpeg_log_error(result, "error encoding #1", ErrorSource::WRITER);
    return result;
}


int FFmpegWriter::writeFFmpegPacket(AVFrame* av_frame) {
    int result = avcodec_receive_packet(codec_ctx, videoPacket);
    if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) { 
        //do not report error here, need more frame data or end of file

    } else if (result < 0) { 
        //report error, something wrong
        ffmpeg_log_error(result, "error encoding #2", ErrorSource::WRITER);

    } else { 
        //write packet to output
        //packet pts starts at 0 and is incremented, but here packets arrive in dts order
        videoPacket->stream_index = videoStream->index;
        writePacket(videoPacket, videoPacket->pts, videoPacket->dts, av_frame == nullptr);
    }
    return result;
}


void FFmpegWriter::write(int bufferIndex) {
    assert(bufferIndex < imageBufferSize && frameIndex == imageBuffer[bufferIndex].index && "invalid frame index");
    auto fcn = [this, bufferIndex] {
        ImageYuv& fr = imageBuffer[bufferIndex];
        //fr.writeText(std::to_string(fr.index), 10, 10, 2, 3, ColorYuv::BLACK, ColorYuv::WHITE);
        //scale and put into av_frame
        uint8_t* src[] = { fr.plane(0), fr.plane(1), fr.plane(2), nullptr };
        int strides[] = { fr.stride, fr.stride, fr.stride, 0 }; //if only three values are provided, we get a warning "data not aligned"
        int sliceHeight = sws_scale(sws_scaler_ctx, src, strides, 0, fr.h, av_frame->data, av_frame->linesize);

        //set pts into frame to later name packet
        av_frame->pts = fr.index;

        //generate and write packet
        int result = sendFFmpegFrame(av_frame);
        while (result >= 0) {
            result = writeFFmpegPacket(av_frame);
        }
    };

    //enqueue new task and wait for oldest task
    encodingQueue.push_back(encoderPool.add(fcn));
    encodingQueue.front().wait();
    encodingQueue.pop_front();

    this->frameIndex++;
}


void FFmpegWriter::write(const FrameExecutor& executor) {
    assert(frameIndex == this->frameIndex && "invalid frame index");
    int idx = frameIndex % imageBufferSize;
    write(idx);
}


//flush encoder buffer
bool FFmpegWriter::startFlushing() {
    for (auto& f : encodingQueue) f.wait();
    int result = sendFFmpegFrame(nullptr);
    return result >= 0;
}


bool FFmpegWriter::flush() {
    return writeFFmpegPacket(nullptr) >= 0;
}


//clean up encoder stuff
FFmpegWriter::~FFmpegWriter() {
    av_frame_free(&av_frame);
    avcodec_free_context(&codec_ctx);
    sws_freeContext(sws_scaler_ctx);
}
