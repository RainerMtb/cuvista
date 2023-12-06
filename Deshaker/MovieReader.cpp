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

#include "MovieReader.hpp"
#include "Util.hpp"

#include <fstream>
#include <iostream>

 //----------------------------------
 //-------- Movie Reader Main
 //----------------------------------

std::future<void> MovieReader::readAsync(ImageYuv& inputFrame, Stats& status) {
    return std::async(std::launch::async, [&] () { read(inputFrame, status); });
}


//----------------------------------
//-------- Placeholder Class
//----------------------------------

InputContext NullReader::open(std::string_view source) {
    return {};
}

void NullReader::read(ImageYuv& frame, Stats& status) {
    frame.setValues(ColorYuv { 0, 0, 0 });
    status.endOfInput = false;
}


//----------------------------------
//-------- FFmpeg Reader
//----------------------------------


//constructor opens ffmpeg file
InputContext FFmpegReader::open(std::string_view source) {
    //av_log_set_level(AV_LOG_ERROR);
    av_log_set_callback(ffmpeg_log);
    InputContext input = {};

    // Allocate format context
    av_format_ctx = avformat_alloc_context();
    if (av_format_ctx == nullptr) 
        throw AVException("could not create AVFormatContext");
        
    // Open the file using libavformat
    int err = avformat_open_input(&av_format_ctx, source.data(), NULL, NULL);
    if (err < 0) 
        throw AVException(av_make_error(err, "could not open input video file"));

    //without find_stream_info width or height might be 0
    err = avformat_find_stream_info(av_format_ctx, NULL);
    if (err < 0) 
        throw AVException(av_make_error(err, "could not get stream info"));

    //search streams
    const AVCodec* av_codec = nullptr;
    for (size_t i = 0; i < av_format_ctx->nb_streams; i++) {
        AVStream* stream = av_format_ctx->streams[i];

        if (av_stream == nullptr && stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            //store first video stream
            av_codec = avcodec_find_decoder(stream->codecpar->codec_id);
            if (av_codec) {
                input.videoStream = av_stream = stream;
            }
        }

        //store every stream found in input
        input.inputStreams.push_back(stream);
    }
    //continue only when there is a video stream to decode
    if (av_stream == nullptr || av_codec == nullptr) 
        throw AVException("could not find a valid video stream");

    // Set up a codec context for the decoder
    if ((av_codec_ctx = avcodec_alloc_context3(av_codec)) == nullptr) 
        throw AVException("could not create AVCodecContext");
    if (avcodec_parameters_to_context(av_codec_ctx, av_stream->codecpar) < 0) 
        throw AVException("could not initialize AVCodecContext");

    //enable multi threading for decoder
    av_codec_ctx->thread_count = 0;

    //open decoder
    if (avcodec_open2(av_codec_ctx, av_codec, NULL) < 0) 
        throw AVException("could not open codec");

    av_frame = av_frame_alloc();
    if (!av_frame) 
        throw AVException("could not allocate AVFrame");
    av_packet = av_packet_alloc();
    if (!av_packet) 
        throw AVException("could not allocate AVPacket");

    //set values in InputContext object
    input.avformatDuration = av_format_ctx->duration;
    input.fpsNum = av_stream->avg_frame_rate.num;
    input.fpsDen = av_stream->avg_frame_rate.den;
    input.timeBaseNum = av_stream->time_base.num;
    input.timeBaseDen = av_stream->time_base.den;
    input.h = av_codec_ctx->height;
    input.w = av_codec_ctx->width;
    input.frameCount = av_stream->nb_frames;
    input.source = source;
    //av_dump_format(av_format_ctx, av_stream->index, av_format_ctx->url, 0); //uses av_log
    return input;
}

//read one frame from ffmpeg
void FFmpegReader::read(ImageYuv& frame, Stats& status) {
    //util::ConsoleTimer timer("read");
    status.endOfInput = true;
    while (true) {
        av_packet_unref(av_packet); //unref old packet

        int response = av_read_frame(av_format_ctx, av_packet); //read new packet from input format
        if (av_packet->size > 0) {
            int sidx = av_packet->stream_index;
            StreamHandling sh = sidx < status.inputStreams.size() ? status.inputStreams[sidx].handling : StreamHandling::STREAM_IGNORE;

            if (sidx == av_stream->index) {
                //we have a video packet
                response = avcodec_send_packet(av_codec_ctx, av_packet); //send packet to decoder
                if (response < 0) {
                    errorLogger.logError(av_make_error(response, "Failed to send packet"));
                    break;
                }

                response = avcodec_receive_frame(av_codec_ctx, av_frame); //try to get a frame from decoder
                if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) { //we have to send more packets before getting next frame
                    continue;

                } else if (response < 0) { //something wrong
                    errorLogger.logError(av_make_error(response, "Failed to receive frame"));
                    break;

                } else { //we got a frame
                    status.endOfInput = false;
                    break;
                }

            } else if (sh == StreamHandling::STREAM_COPY || sh == StreamHandling::STREAM_TRANSCODE) {
                //we should store a packet from a secondary stream for processing
                AVPacket* pktcopy = av_packet_clone(av_packet);
                status.inputStreams[sidx].packets.push_back(pktcopy);
            }

        } else { //nothing left in input format, terminate the process, dump frames from decoder buffer
            response = avcodec_send_packet(av_codec_ctx, NULL); //send terminating signal to decoder
            response = avcodec_receive_frame(av_codec_ctx, av_frame);
            if (response == AVERROR_EOF) { //really the end of the file
                break;

            } else if (response < 0) {
                errorLogger.logError(av_make_error(response, "Failed to receive frame"));
                break;

            } else { //we still got a frame
                status.endOfInput = false;
                break;
            }
        }
    }

    if (!status.endOfInput) {
        //convert to YUV444 data
        int64_t idx = -1;
        frame.frameIdx = -1; // av_frame->coded_picture_number; //deprecated
        frame.index = status.frameReadIndex;
        int w = av_codec_ctx->width;
        int h = av_codec_ctx->height;

        //set up sws scaler after first frame has been decoded
        if (!sws_scaler_ctx) {
            sws_scaler_ctx = sws_getContext(w, h, av_codec_ctx->pix_fmt, w, h, AV_PIX_FMT_YUV444P, SWS_BILINEAR, NULL, NULL, NULL);
        }
        if (!sws_scaler_ctx) {
            errorLogger.logError("failed to initialize ffmpeg scaler");
        }

        //scale image data
        uint8_t* frame_buffer[] = { frame.plane(0), frame.plane(1), frame.plane(2), nullptr };
        int linesizes[] = { frame.stride, frame.stride, frame.stride, 0 };
        sws_scale(sws_scaler_ctx, av_frame->data, av_frame->linesize, 0, av_frame->height, frame_buffer, linesizes);

        //store parameters for writer
        status.packetList.emplace_back(status.frameReadIndex, idx, av_frame->pts, av_frame->pkt_dts, av_frame->duration);

        //in some cases pts values are not in proper sequence, but actual footage seems to be in order
        //in that case just reorder pts values
        //maybe this is a bug in ffmpeg?
        auto it = status.packetList.end();
        it--;
        while (it != status.packetList.begin() && it->pts < std::prev(it)->pts) {
            std::swap(it->pts, std::prev(it)->pts);
            it--;
        }

        //stamp frame index into image
        //frame.writeText(std::to_string(status.frameReadIndex), 100, 100, 3, 3, ColorYuv::WHITE, ColorYuv::GRAY);
        //frame.saveAsColorBMP(std::format("c:/temp/im{:03d}.bmp", status.frameReadIndex));
    }
}

void FFmpegReader::seek(double fraction) {
    int64_t target = av_format_ctx->start_time + int64_t(av_format_ctx->duration * fraction);

    int response = avformat_seek_file(av_format_ctx, -1, INT_MIN, target, target, 0); //always use min_ts = INT_MIN
    if (response < 0) {
        errorLogger.logError(av_make_error(response, "faild to seek in input"));
    }
    avcodec_flush_buffers(av_codec_ctx);
}

void FFmpegReader::rewind() {
    seek(0.0);
}

void FFmpegReader::close() {
    sws_freeContext(sws_scaler_ctx);
    avcodec_close(av_codec_ctx);
    avcodec_free_context(&av_codec_ctx);
    av_packet_free(&av_packet);
    av_frame_free(&av_frame);
    avformat_close_input(&av_format_ctx);
    avformat_free_context(av_format_ctx);

    av_stream = nullptr;
    sws_scaler_ctx = nullptr;
}

FFmpegReader::~FFmpegReader() {
    close();
}
