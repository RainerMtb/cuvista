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

#include "MovieWriter.hpp"
#include "Util.hpp"
#include <filesystem>


//set up ffmpeg encoder
void FFmpegWriter::open(EncodingOption videoCodec) {
    open(videoCodec, mData.w, mData.h, mData.fileOut);
}


//relay output to buffer
OutputContext FFmpegWriter::getOutputContext() {
    return { true, false, &outputFrame, nullptr };
}

void FFmpegWriter::openEncoder(const AVCodec* codec, const std::string& sourceName) {
    int result = 0;

    //open encoder
    result = avcodec_open2(codec_ctx, codec, NULL);
    if (result < 0)
        throw AVException(av_make_error(result, "error opening codec"));

    result = avcodec_parameters_from_context(videoStream->codecpar, codec_ctx);
    if (result < 0)
        throw AVException(av_make_error(result, "error setting codec parameters"));

    result = avio_open(&fmt_ctx->pb, fmt_ctx->url, AVIO_FLAG_WRITE);
    if (result < 0)
        throw AVException(std::string("error opening file '") + sourceName + "'");

    result = avformat_write_header(fmt_ctx, NULL);
    if (result < 0)
        throw AVException(av_make_error(result, "error writing file header"));
    else
        this->headerWritten = true; //store info for proper closing

    //av_dump_format(fmt_ctx, 0, fmt_ctx->url, 1);

    videoPacket = av_packet_alloc();
    if (!videoPacket)
        throw AVException("Could not allocate encoder packet");

    //allocate av_frames to be cycled through on encoding
    for (int i = 0; i < writeBufferSize; i++) {
        AVFrame* av_frame = av_frame_alloc();
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

        av_frames.push_back(av_frame);
    }
}


//set up ffmpeg encoder
void FFmpegWriter::open(EncodingOption videoCodec, int w, int h, const std::string& sourceName) {
    const AVCodec* codec = avcodec_find_encoder(codecMap[videoCodec.codec]);
    //const AVCodec* codec = avcodec_find_encoder_by_name("libsvtav1");
    if (!codec) 
        throw AVException("Could not find encoder");

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) 
        throw AVException("Could not allocate encoder context");

    //open container format
    FFmpegFormatWriter::open(videoCodec, sourceName);

    codec_ctx->width = w;
    codec_ctx->height = h;
    codec_ctx->pix_fmt = pixfmt;
    //codec_ctx->framerate = { data.inputCtx.fpsNum, data.inputCtx.fpsDen };
    codec_ctx->time_base = { (int) mReader.timeBaseNum, (int) mReader.timeBaseDen };
    codec_ctx->gop_size = GOP_SIZE;
    codec_ctx->max_b_frames = 4;
    //av_opt_set(codec_ctx->priv_data, "preset", "slow", 0);
    av_opt_set(codec_ctx->priv_data, "profile", "main", 0);
    av_opt_set(codec_ctx->priv_data, "x265-params", "log-level=error", 0);
    av_opt_set(codec_ctx->priv_data, "crf", std::to_string(mData.crf).c_str(), 0); //?????

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    openEncoder(codec, sourceName);

    sws_scaler_ctx = sws_getContext(w, h, AV_PIX_FMT_YUV444P, w, h, pixfmt, SWS_BICUBIC, NULL, NULL, NULL);
    if (!sws_scaler_ctx) 
        throw AVException("Could not get scaler context");
}


AVFrame* FFmpegWriter::putAVFrame(ImageYuv& fr) {
    //get storage
    size_t idx = fr.index % writeBufferSize;
    AVFrame* av_frame = av_frames[idx];

    //fr.writeText(std::to_string(status.frameWriteIndex), 10, 10, 2, 3, ColorYuv::BLACK, ColorYuv::WHITE);
    //scale and put into av_frame
    uint8_t* src[] = { fr.plane(0), fr.plane(1), fr.plane(2), nullptr };
    int strides[] = { fr.stride, fr.stride, fr.stride, 0 }; //if only three values are provided, we get a warning "data not aligned"
    int sliceHeight = sws_scale(sws_scaler_ctx, src, strides, 0, fr.h, av_frame->data, av_frame->linesize);

    //set pts into frame to later name packet
    av_frame->pts = this->frameIndex;

    return av_frame;
}


int FFmpegWriter::sendFFmpegFrame(AVFrame* av_frame) {
    //util::ConsoleTimer timer("send ffmpeg");
    int result = avcodec_send_frame(codec_ctx, av_frame);
    if (result < 0)
        ffmpeg_log_error(result, "error encoding #1");
    return result;
}


int FFmpegWriter::writeFFmpegPacket(AVFrame* av_frame) {
    int result = avcodec_receive_packet(codec_ctx, videoPacket);
    if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) { 
        //do not report error here, need more frame data or end of file

    } else if (result < 0) { 
        //report error, something wrong
        ffmpeg_log_error(result, "error encoding #2");

    } else { 
        //write packet to output
        //packet pts starts at 0 and is incremented, but here packets arrive in dts order
        videoPacket->stream_index = videoStream->index;
        writePacket(videoPacket, videoPacket->pts, videoPacket->dts, av_frame == nullptr);
    }
    return result;
}


void FFmpegWriter::write() {
    write(outputFrame);
}


void FFmpegWriter::write(ImageYuv& frame) {
    //put yuv frame into ffmpeg frame storage
    AVFrame* av_frame = putAVFrame(frame);

    //generate and write packet
    int result = sendFFmpegFrame(av_frame);
    while (result >= 0) {
        result = writeFFmpegPacket(av_frame);
    }
    this->frameIndex++;
}


std::future<void> FFmpegWriter::writeAsync() {
    //put yuv frame into ffmpeg frame storage
    AVFrame* av_frame = putAVFrame(outputFrame);

    //enqueue ffmpeg process
    auto fcn = [this, av_frame] {
        int result = sendFFmpegFrame(av_frame);
        while (result >= 0) {
            result = writeFFmpegPacket(av_frame);
        }
    };
    this->frameIndex++;
    return std::async(std::launch::async, fcn);
}


//flush encoder buffer
bool FFmpegWriter::startFlushing() {
    int result = sendFFmpegFrame(nullptr);
    return result >= 0;
}


bool FFmpegWriter::flush() {
    return writeFFmpegPacket(nullptr) >= 0;
}


//clean up encoder stuff
FFmpegWriter::~FFmpegWriter() {
    for (AVFrame* af : av_frames) av_frame_free(&af);
    avcodec_close(codec_ctx);
    avcodec_free_context(&codec_ctx);
    sws_freeContext(sws_scaler_ctx);
}
