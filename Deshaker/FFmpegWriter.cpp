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

//construct ffmpeg encoder
void FFmpegWriter::open(EncodingOption videoCodec) {
    int result = 0;

    const AVCodec* codec = avcodec_find_encoder(codecMap[videoCodec.codec]);
    //const AVCodec* codec = avcodec_find_encoder_by_name("libsvtav1");
    if (!codec) 
        throw AVException("Could not find encoder");

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) 
        throw AVException("Could not allocate encoder context");

    //open container format
    FFmpegFormatWriter::open(videoCodec);

    codec_ctx->width = mData.w;
    codec_ctx->height = mData.h;
    codec_ctx->pix_fmt = pixfmt;
    //codec_ctx->framerate = { data.inputCtx.fpsNum, data.inputCtx.fpsDen };
    codec_ctx->time_base = { (int) mData.inputCtx.timeBaseNum, (int) mData.inputCtx.timeBaseDen };
    codec_ctx->gop_size = GOP_SIZE;
    codec_ctx->max_b_frames = 4;
    //av_opt_set(codec_ctx->priv_data, "preset", "slow", 0);
    av_opt_set(codec_ctx->priv_data, "profile", "main", 0);
    av_opt_set(codec_ctx->priv_data, "x265-params", "log-level=error", 0);
    av_opt_set(codec_ctx->priv_data, "crf", std::to_string(mData.crf).c_str(), 0); //?????

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    //open encoder
    result = avcodec_open2(codec_ctx, codec, NULL);
    if (result < 0) 
        throw AVException(av_make_error(result, "error opening codec"));

    result = avcodec_parameters_from_context(videoStream->codecpar, codec_ctx);
    if (result < 0) 
        throw AVException(av_make_error(result, "error setting codec parameters"));

    result = avio_open(&fmt_ctx->pb, fmt_ctx->url, AVIO_FLAG_WRITE);
    if (result < 0) 
        throw AVException(std::string("error opening file '") + mData.fileOut + "'");

    result = avformat_write_header(fmt_ctx, NULL);
    if (result < 0)
        throw AVException(av_make_error(result, "error writing file header"));
    else
        this->headerWritten = true; //store info for proper closing

    //av_dump_format(fmt_ctx, 0, fmt_ctx->url, 1);

    videoPacket = av_packet_alloc();
    if (!videoPacket) 
        throw AVException("Could not allocate encoder packet");

    frame = av_frame_alloc();
    if (!frame) 
        throw AVException("Could not allocate video frame");

    frame->format = codec_ctx->pix_fmt;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;

    result = av_frame_get_buffer(frame, 0);
    if (result < 0) 
        throw AVException("Could not get frame buffer");

    result = av_frame_make_writable(frame);
    if (result < 0) 
        throw AVException("Could not make frame writable");

    sws_scaler_ctx = sws_getContext(mData.w, mData.h, AV_PIX_FMT_YUV444P, mData.w, mData.h, pixfmt, SWS_BICUBIC, NULL, NULL, NULL);
    if (!sws_scaler_ctx) 
        throw AVException("Could not get scaler context");
}


int FFmpegWriter::sendFFmpegFrame(AVFrame* frame) {
    //util::ConsoleTimer timer("write");
    int result = avcodec_send_frame(codec_ctx, frame);
    if (result < 0)
        ffmpeg_log_error(result, "error encoding #1");
    return result;
}


int FFmpegWriter::writeFFmpegPacket() {
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
        writePacket(videoPacket, videoPacket->pts, videoPacket->dts, frame == nullptr);
    }
    return result;
}


void FFmpegWriter::write() {
    //outputFrame.writeText(std::to_string(status.frameWriteIndex), 10, 10, 2, 3, ColorYuv::BLACK, ColorYuv::WHITE);
    ImageYuv& fr = outputFrame;
    uint8_t* src[] = { fr.plane(0), fr.plane(1), fr.plane(2), nullptr };
    int strides[] = { fr.stride, fr.stride, fr.stride, 0 }; //if only three values are provided, we get a warning "data not aligned"
    int sliceHeight = sws_scale(sws_scaler_ctx, src, strides, 0, fr.h, frame->data, frame->linesize);

    //set pts into frame to later name packet
    frame->pts = mStatus.frameWriteIndex;
    //frame->coded_picture_number = status.frameWriteIndex; //will not be set in output??

    //generate and write packet
    int result = sendFFmpegFrame(frame);
    while (result >= 0) {
        result = writeFFmpegPacket();
    }
}


//flush encoder buffer
bool FFmpegWriter::terminate(bool init) {
    int result = 0;
    if (init) {
        result = sendFFmpegFrame(nullptr);

    } else {
        result = writeFFmpegPacket();
    }
    return result >= 0;
}


//clean up encoder stuff
FFmpegWriter::~FFmpegWriter() {
    av_frame_free(&frame);
    avcodec_close(codec_ctx);
    avcodec_free_context(&codec_ctx);
    sws_freeContext(sws_scaler_ctx);
}
