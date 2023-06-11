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
#include <filesystem>

AVStream* FFmpegFormatWriter::newStream(AVFormatContext* fmt_ctx, AVStream* inStream) {
    AVStream* st = avformat_new_stream(fmt_ctx, NULL); //docs say AVCodec parameter is not used
    if (!st)
        throw AVException("could not create stream");

    st->time_base = inStream->time_base;
    return st;
}

//setup output format
void FFmpegFormatWriter::open() {
    //av_log_set_level(AV_LOG_ERROR);
    //custom callback to log ffmpeg errors
    av_log_set_callback(ffmpeg_log);

    //setup output file
    int result = avformat_alloc_output_context2(&fmt_ctx, NULL, NULL, data.fileOut.c_str());
    if (result < 0) 
        throw AVException(av_make_error(result));

    //setup streams
    AVStream* videoIn = data.inputCtx.videoStream;
    for (StreamContext& sc : status.inputStreams) {
        AVStream* inStream = sc.inputStream;
        int codecSupported = avformat_query_codec(fmt_ctx->oformat, inStream->codecpar->codec_id, FF_COMPLIANCE_NORMAL);

        if (inStream->index == videoIn->index) {
            if (codecSupported) {
                videoStream = newStream(fmt_ctx, inStream);
                sc.outputStream = videoStream;
                sc.handling = StreamHandling::STREAM_STABILIZE;

            } else {
                throw AVException(std::format("cannot write codec '{}' to output", avcodec_get_name(inStream->codecpar->codec_id)));
            }

        } else {
            if (codecSupported) {
                sc.outputStream = newStream(fmt_ctx, inStream);
                sc.handling = StreamHandling::STREAM_COPY;

                int retval = avcodec_parameters_copy(sc.outputStream->codecpar, inStream->codecpar);
                if (retval < 0)
                    throw AVException("cannot copy context for stream");

                //https://ffmpeg.org/doxygen/trunk/remuxing_8c-example.html
                //setting tag to 0 seems to avoid "Tag xxxx incompatible with output codec id as shown in ffmpeg example
                sc.outputStream->codecpar->codec_tag = 0;

            } else if (inStream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                sc.outputStream = newStream(fmt_ctx, inStream);
                sc.handling = StreamHandling::STREAM_TRANSCODE;

                //set up audio transcoding

            } else {
                sc.handling = StreamHandling::STREAM_IGNORE;
            }
        }
    }
}

//close ffmpeg format
FFmpegFormatWriter::~FFmpegFormatWriter() {
    if (fmt_ctx != nullptr && headerWritten) {
        int result = av_write_trailer(fmt_ctx);
        if (result < 0)
            errorLogger.logError(av_make_error(result, "error writing trailer"));
    }

    if (fmt_ctx != nullptr && fmt_ctx->pb != nullptr && headerWritten) {
        int result = avio_close(fmt_ctx->pb);
        if (result < 0)
            errorLogger.logError(av_make_error(result, "error closing output"));
    }

    av_packet_free(&videoPacket);
    avformat_free_context(fmt_ctx);
}


//----------------------------------
//-------- write packets
//----------------------------------

//write packet to output
void FFmpegFormatWriter::writePacket(AVPacket* packet) {
    int result = av_interleaved_write_frame(fmt_ctx, packet); //write_frame also does unref packet
    if (result == 0) {
        status.inputStreams[packet->stream_index].packetsWritten++;

    } else {
        errorLogger.logError(av_make_error(result, "error writing packet"));
    }
}

int64_t multiplyFrameTime(int64_t timeValue, const AVRational& framerate, const AVRational& timebase) {
    return timeValue * framerate.den * timebase.den / framerate.num / timebase.num;
}

int64_t rescaleTime(int64_t timeValue, const AVRational& src, const AVRational& dest) {
    return timeValue * src.num * dest.den / src.den / dest.num;
}

//rescale packet and apply offset to base stream
Timings rescale_ts_and_offset(const AVPacket* pkt, const AVRational& timebaseSrc, const AVRational& timebaseDest, const AVStream* ref) {
    int64_t offset = rescaleTime(ref->start_time, ref->time_base, timebaseDest);
    int64_t pts = rescaleTime(pkt->pts, timebaseSrc, timebaseDest) - offset;
    int64_t dts = rescaleTime(pkt->dts, timebaseSrc, timebaseDest) - offset;
    int64_t duration = rescaleTime(pkt->duration, timebaseSrc, timebaseDest);
    return { pts, dts, duration, 1.0 * pts * timebaseDest.num / timebaseDest.den, 1.0 * dts * timebaseDest.num / timebaseDest.den };
}

//write packets to output
void FFmpegFormatWriter::writePacket(AVPacket* pkt, int64_t ptsIdx, int64_t dtsIdx, bool terminate) {
    AVStream* videoInputStream = data.inputCtx.videoStream;

    // look up pts value for this frame in input, can be weird
    // sometimes frames are not in pts order in input - what to do then? see MovieReader
    // 
    // overall plan:
    // lookup pts from input to use for this output frame
    // then offset such that output will start at pts=0
    // then rescale to potentially different output timescale
    auto compareFunc = [&] (const VideoPacketContext& ctx) { return ctx.readIndex == ptsIdx; }; //sometimes wrong, check for increasing pts values
    auto vpc = std::find_if(status.packetList.cbegin(), status.packetList.cend(), compareFunc);
    status.encodedFrame = *vpc; //store for later analysis

    //set pts and dts with respect to input timebase
    uint64_t pts = vpc->pts - videoInputStream->start_time;
    pkt->dts = pts - multiplyFrameTime(ptsIdx - dtsIdx, videoInputStream->avg_frame_rate, videoInputStream->time_base); 
    pkt->pts = pts;
    pkt->duration = vpc->duration;
    //std::printf("pktpts=%zd pktdts=%zd dur=%zd\n", pkt->pts, pkt->dts, pkt->duration);

    //rescale input timebase to output timebase
    av_packet_rescale_ts(pkt, videoInputStream->time_base, videoStream->time_base);

    //process secondary streams
    double dtsTime = 1.0 * pkt->dts * videoStream->time_base.num / videoStream->time_base.den;
    for (StreamContext& sc : status.inputStreams) {
        for (auto it = sc.packets.begin(); it != sc.packets.end(); ) {
            AVPacket* sidePacket = *it;
            //compare dts value
            Timings timings = rescale_ts_and_offset(sidePacket, sc.inputStream->time_base, sc.outputStream->time_base, videoInputStream);
            //std::cout << timings << std::endl;

            //write side packets before this video packet or when terminating the file
            if (timings.dtsTime < dtsTime || terminate) {
                if (sc.handling == StreamHandling::STREAM_COPY) { //copy packet directly to output stream
                    sidePacket->stream_index = sc.outputStream->index;
                    sidePacket->pts = timings.pts;
                    sidePacket->dts = timings.dts;
                    sidePacket->duration = timings.duration;
                    writePacket(sidePacket);

                } else if (sc.handling == StreamHandling::STREAM_TRANSCODE) { //transcode audio

                }

                //remove packet from buffer
                av_packet_free(&sidePacket);
                it = sc.packets.erase(it);

            } else {
                it++;
            }
        }
    }

    //static std::ofstream testFile("f:/test.h265", std::ios::binary);
    //testFile.write(reinterpret_cast<char*>(videoPacket->data), videoPacket->size);

    //delete from input list
    status.packetList.erase(vpc);

    //store stats
    status.encodedBytes = pkt->size;
    status.encodedDts = pkt->dts;
    status.encodedPts = pkt->pts;

    //write packet to output
    writePacket(pkt);
    status.encodedBytesTotal += status.encodedBytes;
    status.outputBytesWritten = status.encodedBytesTotal; //how to get proper progress from AVFormatContext

    //advance encoder counter
    status.frameEncodeIndex++;
}

//----------------------------------
//-------- FFMPEG Encoding
//----------------------------------

//construct ffmpeg encoder
void FFmpegWriter::open() {
    int result = 0;

    //init format
    FFmpegFormatWriter::open();

    const AVCodec* codec = avcodec_find_encoder(codecMap[data.videoCodec]);
    if (!codec) 
        throw AVException("Could not find encoder");

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) 
        throw AVException("Could not allocate encoder context");

    codec_ctx->width = data.w;
    codec_ctx->height = data.h;
    codec_ctx->pix_fmt = pixfmt;
    //codec_ctx->framerate = { data.inputCtx.fpsNum, data.inputCtx.fpsDen };
    codec_ctx->time_base = { (int) data.inputCtx.timeBaseNum, (int) data.inputCtx.timeBaseDen };
    codec_ctx->gop_size = GOP_SIZE;
    codec_ctx->max_b_frames = 4;
    av_opt_set(codec_ctx->priv_data, "preset", "slow", 0);
    av_opt_set(codec_ctx->priv_data, "profile", "main", 0);
    av_opt_set(codec_ctx->priv_data, "x265-params", "log-level=none", 0);
    av_opt_set(codec_ctx->priv_data, "crf", std::to_string(data.crf).c_str(), 0); //?????

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    result = avcodec_open2(codec_ctx, codec, NULL);
    if (result < 0) 
        throw AVException(av_make_error(result, "error opening codec"));

    result = avcodec_parameters_from_context(videoStream->codecpar, codec_ctx);
    if (result < 0) 
        throw AVException(av_make_error(result, "error setting codec parameters"));

    result = avio_open(&fmt_ctx->pb, fmt_ctx->url, AVIO_FLAG_WRITE);
    if (result < 0) 
        throw AVException(std::string("error opening file '") + data.fileOut + "'");

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

    sws_scaler_ctx = sws_getContext(data.w, data.h, AV_PIX_FMT_YUV444P, data.w, data.h, pixfmt, SWS_BICUBIC, NULL, NULL, NULL);
    if (!sws_scaler_ctx) 
        throw AVException("Could not get scaler context");
}


int FFmpegWriter::sendFFmpegFrame(AVFrame* frame) {
    int result = avcodec_send_frame(codec_ctx, frame);
    if (result < 0)
        errorLogger.logError(av_make_error(result, "error encoding #1"));
    return result;
}


int FFmpegWriter::writeFFmpegPacket() {
    int result = avcodec_receive_packet(codec_ctx, videoPacket);
    if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) { 
        //do not report error here, need more frame data or end of file

    } else if (result < 0) { 
        //report error, something wrong
        errorLogger.logError(av_make_error(result, "error encoding #2"));

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

    //set pts into frame to later identify packet
    frame->pts = status.frameWriteIndex;
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
    avcodec_free_context(&codec_ctx);
    sws_freeContext(sws_scaler_ctx);
}
