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
    int result = avformat_alloc_output_context2(&fmt_ctx, NULL, NULL, mData.fileOut.c_str());
    if (result < 0)
        throw AVException(av_make_error(result));

    //setup streams
    AVStream* videoIn = mData.inputCtx.videoStream;
    for (StreamContext& sc : mStatus.inputStreams) {
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
            codecSupported = false; //force transcode for debugging <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
                //open reader
                sc.audioInCodec = avcodec_find_decoder(inStream->codecpar->codec_id);
                if (!sc.audioInCodec)
                    throw AVException(std::format("cannot find audio decoder for '{}'", avcodec_get_name(inStream->codecpar->codec_id)));
                sc.audioInCtx = avcodec_alloc_context3(sc.audioInCodec);
                if (!sc.audioInCtx)
                    throw AVException("cannot allocate audio decoder context");
                int retval = avcodec_parameters_to_context(sc.audioInCtx, inStream->codecpar);
                if (retval < 0)
                    throw AVException(av_make_error(retval, "cannot copy audio parameters to input context"));
                retval = avcodec_open2(sc.audioInCtx, sc.audioInCodec, NULL);
                if (retval < 0)
                    throw AVException(av_make_error(retval, "cannot open audio decoder"));
                sc.audioInCtx->time_base = inStream->time_base;

                //open writer
                AVCodecID id = inStream->codecpar->codec_id;
                sc.audioOutCodec = avcodec_find_encoder(id);
                if (!sc.audioOutCodec)
                    throw AVException(std::format("cannot find audio encoder for '{}'", avcodec_get_name(id)));
                sc.audioOutCtx = avcodec_alloc_context3(sc.audioOutCodec);
                if (!sc.audioOutCtx)
                    throw AVException("cannot allocate audio encoder context");
                retval = av_channel_layout_copy(&sc.audioOutCtx->ch_layout, &sc.audioInCtx->ch_layout);
                if (retval < 0)
                    throw AVException("cannot copy audio channel layout");
                sc.audioOutCtx->sample_rate = sc.audioInCtx->sample_rate;
                sc.audioOutCtx->sample_fmt = sc.audioInCtx->sample_fmt;
                sc.audioOutCtx->bit_rate = sc.audioInCtx->bit_rate;

                if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
                    sc.audioOutCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

                retval = avcodec_open2(sc.audioOutCtx, sc.audioOutCodec, NULL);
                if (retval < 0)
                    throw AVException("cannot open audio encoder");

                //init resampler
                retval = swr_alloc_set_opts2(&sc.resampleCtx, 
                    &sc.audioOutCtx->ch_layout, sc.audioOutCtx->sample_fmt, sc.audioOutCtx->sample_rate,
                    &sc.audioInCtx->ch_layout, sc.audioInCtx->sample_fmt, sc.audioInCtx->sample_rate, 
                    0, NULL);
                if (retval < 0)
                    throw AVException("cannot set resampler options");
                retval = swr_init(sc.resampleCtx);
                if (retval < 0)
                    throw AVException(av_make_error(retval, "cannot init audio resampler"));

                //av_samples_alloc_array_and_samples

                //stream timebase
                sc.outputStream->time_base.den = sc.audioInCtx->sample_rate;
                sc.outputStream->time_base.num = 1;

                retval = avcodec_parameters_from_context(sc.outputStream->codecpar, sc.audioOutCtx);
                if (retval < 0)
                    throw AVException("cannot copy audio parameters to output stream");

                //allocate packet
                sc.outpkt = av_packet_alloc();
                if (!sc.outpkt)
                    errorLogger.logError("cannot allocate audio packet");

                //allocate frame
                sc.frame = av_frame_alloc();
                if (!sc.frame)
                    errorLogger.logError("cannot allocate audio frame");

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

    for (StreamContext& sc : mStatus.inputStreams) {
        if (sc.audioInCtx) {
            avcodec_close(sc.audioInCtx);
            avcodec_free_context(&sc.audioInCtx);
        }
        if (sc.audioOutCtx) {
            avcodec_close(sc.audioOutCtx);
            avcodec_free_context(&sc.audioOutCtx);
        }
        if (sc.outpkt) {
            av_packet_free(&sc.outpkt);
        }
        if (sc.frame) {
            av_frame_free(&sc.frame);
        }
        if (sc.resampleCtx) {
            swr_free(&sc.resampleCtx);
        }
    }

    av_packet_free(&videoPacket);
    avformat_free_context(fmt_ctx);
}


//----------------------------------
//-------- write packets
//----------------------------------

//write packet to output
int FFmpegFormatWriter::writePacket(AVPacket* packet) {
    StreamContext sc = mStatus.inputStreams[packet->stream_index];
    //std::printf("stream %d pts [sec] %.5f\n", packet->stream_index, 1.0 * packet->pts * sc.outputStream->time_base.num / sc.outputStream->time_base.den);
    
    int result = av_interleaved_write_frame(fmt_ctx, packet); //write_frame also does unref packet
    if (result == 0) {
        sc.packetsWritten++;

    } else {
        errorLogger.logError(av_make_error(result, "error writing packet"));
    }
    return result;
}

void FFmpegFormatWriter::rescaleAudioPacket(StreamContext& sc, AVPacket* pkt) {
    AVRational& tbin = sc.inputStream->time_base;
    AVRational& tbout = sc.outputStream->time_base;
    pkt->pts = av_rescale_delta(tbin, pkt->pts - sc.inputStream->start_time, tbin, (int) pkt->duration, &sc.lastPts, tbout);
    pkt->dts = pkt->pts;
    pkt->duration = 0;
}

//transcode pending audio packet and write to output
void FFmpegFormatWriter::transcodeAudio(AVPacket* pkt, StreamContext& sc, bool terminate) {
    //send packet to decoder
    int retval = 0;

    retval = avcodec_send_packet(sc.audioInCtx, pkt);
    if (retval == AVERROR_EOF) {
        //input packet was nullptr at least for the second time

    } else if (retval < 0) {
        errorLogger.logError("cannot send audio packet to decoder");
        return;
    }

    //get frames from decoder
    while (true) {
        //receive frame from decoder
        AVFrame* frameToEncode = sc.frame;
        retval = avcodec_receive_frame(sc.audioInCtx, sc.frame);
        if (retval == AVERROR(EAGAIN)) {
            break;

        } else if (retval == AVERROR_EOF) {
            frameToEncode = nullptr; //end of input, signal termination to encoder;

        } else if (retval < 0) {
            errorLogger.logError(av_make_error(retval, "error decoding audio packet"));
            break;

        } else {
            //we retrieved a frame, now resample it
            //static std::ofstream outfile("f:/audio.raw", std::ios::binary);
            //outfile.write(reinterpret_cast<char*>(frameToEncode->data[0]), frameToEncode->linesize[0] / frameToEncode->ch_layout.nb_channels);
        }

        //send frame to encoder
        retval = avcodec_send_frame(sc.audioOutCtx, frameToEncode);
        if (retval == AVERROR_EOF) {
            //flush signal was sent to encoder

        } else if (retval < 0) {
            errorLogger.logError(av_make_error(retval, "cannot send audio frame to encoder"));
            break;
        }

        //receive packet from encoder
        retval = avcodec_receive_packet(sc.audioOutCtx, sc.outpkt);
        if (retval == AVERROR(EAGAIN)) {
            continue; //see if decoder has any more frames available

        } else if (retval == AVERROR_EOF) {
            break; //really the end of the stream

        } else if (retval < 0) {
            errorLogger.logError(av_make_error(retval, "error encoding audio packet"));
            break;

        } else {
            //write packet to output
            rescaleAudioPacket(sc, sc.outpkt);
            sc.outpkt->stream_index = sc.outputStream->index;
            writePacket(sc.outpkt);
        }
    }
}

//write packets to output
void FFmpegFormatWriter::writePacket(AVPacket* pkt, int64_t ptsIdx, int64_t dtsIdx, bool terminate) {
    AVStream* videoInputStream = mData.inputCtx.videoStream;

    // look up pts value for this frame in input, can be weird
    // sometimes frames are not in pts order in input - what to do then? see MovieReader
    // 
    // overall plan:
    // lookup pts from input to use for this output frame
    // then offset such that output will start at pts=0
    // then rescale to potentially different output timescale
    auto compareFunc = [&] (const VideoPacketContext& ctx) { return ctx.readIndex == ptsIdx; }; //sometimes wrong, check for increasing pts values
    auto vpc = std::find_if(mStatus.packetList.cbegin(), mStatus.packetList.cend(), compareFunc);
    mStatus.encodedFrame = *vpc; //store for later analysis

    //set pts and dts with respect to input timebase
    uint64_t pts = vpc->pts - videoInputStream->start_time;
    pkt->dts = pts - av_rescale_q(ptsIdx - dtsIdx, videoInputStream->avg_frame_rate, videoInputStream->time_base);
    pkt->pts = pts;
    pkt->duration = vpc->duration;
    //std::printf("pktpts=%zd pktdts=%zd dur=%zd\n", pkt->pts, pkt->dts, pkt->duration);

    //rescale input timebase to output timebase
    av_packet_rescale_ts(pkt, videoInputStream->time_base, videoStream->time_base);

    //process secondary streams
    double dtsTime = 1.0 * pkt->dts * videoStream->time_base.num / videoStream->time_base.den;
    for (StreamContext& sc : mStatus.inputStreams) {
        for (auto it = sc.packets.begin(); it != sc.packets.end(); ) {
            AVPacket* sidePacket = *it;
            int comp = av_compare_ts(sidePacket->dts, sc.inputStream->time_base, vpc->dts, videoInputStream->time_base);
            if (comp < 0 || terminate) {
                if (sc.handling == StreamHandling::STREAM_COPY) { //copy packet directly to output stream
                    rescaleAudioPacket(sc, sidePacket);
                    writePacket(sidePacket);

                } else if (sc.handling == StreamHandling::STREAM_TRANSCODE) { //transcode audio
                    transcodeAudio(sidePacket, sc, false);
                }

                //remove packet from buffer
                av_packet_free(&sidePacket);
                it = sc.packets.erase(it);

            } else {
                it++;
            }
        }

        if (terminate && sc.handling == StreamHandling::STREAM_TRANSCODE) {
            transcodeAudio(nullptr, sc, true);
        }
    }

    //static std::ofstream testFile("f:/test.h265", std::ios::binary);
    //testFile.write(reinterpret_cast<char*>(videoPacket->data), videoPacket->size);

    //delete from input list
    mStatus.packetList.erase(vpc);

    //store stats
    mStatus.encodedBytes = pkt->size;
    mStatus.encodedDts = pkt->dts;
    mStatus.encodedPts = pkt->pts;

    //write packet to output
    writePacket(pkt);
    mStatus.encodedBytesTotal += mStatus.encodedBytes;
    mStatus.outputBytesWritten = mStatus.encodedBytesTotal; //how to get proper progress from AVFormatContext

    //advance encoder counter
    mStatus.frameEncodeIndex++;
}
