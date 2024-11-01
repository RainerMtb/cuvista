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
#include "MovieReader.hpp"
#include <filesystem>

AVStream* FFmpegFormatWriter::newStream(AVFormatContext* fmt_ctx, AVStream* inStream) {
    AVStream* st = avformat_new_stream(fmt_ctx, NULL); //docs say AVCodec parameter is not used
    if (!st)
        throw AVException("could not create stream");

    st->time_base = inStream->time_base;
    return st;
}

//setup output format
void FFmpegFormatWriter::open(EncodingOption videoCodec, const std::string& sourceName, int queueSize) {

    //av_log_set_level(AV_LOG_ERROR);
    //custom callback to log ffmpeg errors
    av_log_set_callback(ffmpeg_log);

    for (int i = 0; i < queueSize - 1; i++) {
        encodingQueue.push_back(std::async([] {}));
    }

    //setup output file
    int result = avformat_alloc_output_context2(&fmt_ctx, NULL, NULL, sourceName.c_str());
    if (result < 0)
        throw AVException(av_make_error(result));

    //setup streams
    AVStream* videoIn = mReader.videoStream;
    for (StreamContext& sc : mReader.inputStreams) {
        AVStream* inStream = sc.inputStream;

        if (inStream->index == videoIn->index) {
            AVCodecID codec = codecMap[videoCodec.codec];
            int codecSupported = avformat_query_codec(fmt_ctx->oformat, codec, FF_COMPLIANCE_STRICT);
            if (codecSupported == 1) {
                videoStream = newStream(fmt_ctx, inStream);
                sc.outputStream = videoStream;
                sc.handling = StreamHandling::STREAM_STABILIZE;

            } else {
                throw AVException(std::format("cannot write codec '{}' to output", avcodec_get_name(inStream->codecpar->codec_id)));
            }
            
        } else {
            int codecSupported = avformat_query_codec(fmt_ctx->oformat, inStream->codecpar->codec_id, FF_COMPLIANCE_STRICT);
            //codecSupported = false; //force transcode for debugging <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            if (codecSupported == 1) {
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
                //AVCodecID id = inStream->codecpar->codec_id;
                AVCodecID id = fmt_ctx->oformat->audio_codec; //default codec for output format
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
                sc.audioOutCtx->sample_fmt = sc.audioOutCodec->sample_fmts[0]; //deprecated
                //const AVSampleFormat* sampleFmts = nullptr;
                //int nFmts;
                //avcodec_get_supported_config(sc.audioOutCtx, sc.audioOutCodec, AV_CODEC_CONFIG_SAMPLE_FORMAT, 0, (const void**) &sampleFmts, &nFmts);
                //sc.audioOutCtx->sample_fmt = (sampleFmts)[0];
                sc.audioOutCtx->bit_rate = 48000LL * sc.audioOutCtx->ch_layout.nb_channels;

                if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
                    sc.audioOutCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

                retval = avcodec_open2(sc.audioOutCtx, sc.audioOutCodec, NULL);
                if (retval < 0)
                    throw AVException("cannot open audio encoder");

                //stream timebase
                sc.outputStream->time_base.den = sc.audioOutCtx->sample_rate;
                sc.outputStream->time_base.num = 1;

                retval = avcodec_parameters_from_context(sc.outputStream->codecpar, sc.audioOutCtx);
                if (retval < 0)
                    throw AVException("cannot copy audio parameters to output stream");

                //allocate packet
                sc.outpkt = av_packet_alloc();
                if (!sc.outpkt)
                    throw AVException("cannot allocate audio packet");

                //allocate input frame
                sc.frameIn = av_frame_alloc();
                if (!sc.frameIn)
                    throw AVException("cannot allocate audio input frame");

                //allocate output frame
                sc.frameOut = av_frame_alloc();
                if (!sc.frameOut)
                    throw AVException("cannot allocate audio output frame");
                retval = av_channel_layout_copy(&sc.frameOut->ch_layout, &sc.audioOutCtx->ch_layout);
                if (retval < 0)
                    throw AVException("cannot copy channel layout");
                sc.frameOut->format = sc.audioOutCtx->sample_fmt;
                sc.frameOut->nb_samples = sc.audioOutCtx->frame_size;
                retval = av_frame_get_buffer(sc.frameOut, 0);
                if (retval < 0)
                    throw AVException(av_make_error(retval, "cannot allocate audio frame buffers"));

                //resampler
                retval = swr_alloc_set_opts2(&sc.resampleCtx, 
                    &sc.audioOutCtx->ch_layout, sc.audioOutCtx->sample_fmt, sc.audioOutCtx->sample_rate,
                    &sc.audioInCtx->ch_layout, sc.audioInCtx->sample_fmt, sc.audioInCtx->sample_rate, 
                    0, NULL);
                if (retval < 0)
                    throw AVException("cannot set audio resampler options");
                retval = swr_init(sc.resampleCtx);
                if (retval < 0)
                    throw AVException(av_make_error(retval, "cannot init audio resampler"));

                //sample buffer
                int n = std::max(sc.audioInCtx->frame_size, sc.audioOutCtx->frame_size) * 2;
                sc.fifo = av_audio_fifo_alloc(sc.audioOutCtx->sample_fmt, sc.audioOutCtx->ch_layout.nb_channels, n);
                if (!sc.fifo)
                    throw AVException("cannot allocate fifo");

            } else {
                sc.handling = StreamHandling::STREAM_IGNORE;
            }
        }
    }
}

//----------------------------------
//-------- write packets
//----------------------------------

//write packet to output
int FFmpegFormatWriter::writePacket(AVPacket* packet) {
    StreamContext& sc = mReader.inputStreams[packet->stream_index];
    //std::printf("stream %d pts [sec] %.5f\n", packet->stream_index, 1.0 * packet->pts * sc.outputStream->time_base.num / sc.outputStream->time_base.den);
    
    int result = av_interleaved_write_frame(fmt_ctx, packet); //write_frame also does unref packet
    if (result == 0) {
        sc.packetsWritten++;

    } else {
        errorLogger.logError(av_make_error(result, "error writing packet"));
    }
    return result;
}

//transcode pending audio packet and write to output
void FFmpegFormatWriter::transcodeAudio(AVPacket* pkt, StreamContext& sc, bool terminate) {
    //send packet to decoder
    int retval = 0;

    retval = avcodec_send_packet(sc.audioInCtx, pkt);
    if (retval == AVERROR_EOF) {
        //input packet was nullptr at least for the second time, ignore

    } else if (retval < 0) {
        ffmpeg_log_error(retval, "cannot send audio packet to decoder");
        return;
    }

    bool doLoop = true;
    while (doLoop) {
        bool eof = false;

        //decode and convert and store in buffer
        while (av_audio_fifo_size(sc.fifo) < sc.audioOutCtx->frame_size) {
            retval = avcodec_receive_frame(sc.audioInCtx, sc.frameIn);
            if (retval == AVERROR(EAGAIN)) {
                doLoop = false; //decoder needs more packets
                break;

            } else if (retval == AVERROR_EOF) {
                eof = true;
                break;

            } else {
                //static std::ofstream outfile("f:/audio.raw", std::ios::binary);
                //outfile.write(reinterpret_cast<char*>(sc.frameIn->data[0]), sc.frameIn->linesize[0] / sc.frameIn->ch_layout.nb_channels);
                int sampleCount = swr_get_out_samples(sc.resampleCtx, sc.frameIn->nb_samples);
                uint8_t** samplesArray;
                int linesize;
                int ch = sc.audioOutCtx->ch_layout.nb_channels;
                int bufsiz = av_samples_alloc_array_and_samples(&samplesArray, &linesize, ch, sampleCount, sc.audioOutCtx->sample_fmt, 0);
                if (bufsiz < 0)
                    ffmpeg_log_error(retval, "cannot allocate samples");

                sampleCount = swr_convert(sc.resampleCtx, samplesArray, sampleCount, sc.frameIn->extended_data, sc.frameIn->nb_samples);
                if (sampleCount < 0)
                    ffmpeg_log_error(sampleCount, "cannot convert samples");

                retval = av_audio_fifo_realloc(sc.fifo, av_audio_fifo_size(sc.fifo) + sampleCount);
                if (retval < 0)
                    ffmpeg_log_error(retval, "cannot resize fifo");

                retval = av_audio_fifo_write(sc.fifo, (void**) samplesArray, sampleCount);
                if (retval != sampleCount)
                    ffmpeg_log_error(retval, "cannot write to fifo");

                if (samplesArray) av_freep(samplesArray);
                av_freep(&samplesArray);
            }
        } //decoder loop

        //encode from buffer
        while (av_audio_fifo_size(sc.fifo) >= sc.audioOutCtx->frame_size || eof) {
            void** ptrs = (void**) sc.frameOut->extended_data;
            int sampleCount = av_audio_fifo_read(sc.fifo, ptrs, sc.audioOutCtx->frame_size);
            AVFrame* av_frame = nullptr;

            if (sampleCount < 0) {
                ffmpeg_log_error(retval, "cannot read from fifo");

            } else if (sampleCount > 0) {
                sc.frameOut->pts = sc.pts;
                sc.frameOut->pkt_dts = sc.pts;
                sc.frameOut->duration = 0;
                sc.pts += sampleCount;
                av_frame = sc.frameOut;
            }

            retval = avcodec_send_frame(sc.audioOutCtx, av_frame); //send flush signal when no more samples to encode
            if (retval == AVERROR_EOF) {
                //flush signal was sent at least for the second time, ignore here

            } else if (retval < 0) {
                ffmpeg_log_error(retval, "cannot send audio frame to encoder");
            }

            retval = avcodec_receive_packet(sc.audioOutCtx, sc.outpkt);
            if (retval == AVERROR(EAGAIN)) {
                break;

            } else if (retval == AVERROR_EOF) {
                doLoop = false; //nothing more to do
                break;

            } else if (retval < 0) {
                ffmpeg_log_error(retval, "cannot receive audio packet from encodeer");

            } else {
				//write packet to output
				sc.outpkt->stream_index = sc.outputStream->index;
				writePacket(sc.outpkt);
            }
        } //encoder loop
    } //doLoop
}

//write packets to output
void FFmpegFormatWriter::writePacket(AVPacket* pkt, int64_t ptsIdx, int64_t dtsIdx, bool terminate) {
    AVStream* videoInputStream = mReader.videoStream;
    
    /*
    overall plan:
    lookup pts value from input for this frame index to use for output frame
    then offset such that output will start at pts=0
    then rescale from input stream to potentially different output timescale
    */

    //STEP 1: looking at input stream
    //set pts and dts with respect to input timebase
    auto compareFunc = [&] (const VideoPacketContext& ctx) { return ctx.readIndex == ptsIdx; }; //sometimes wrong, check for increasing pts values
    auto vpc = std::find_if(mReader.packetList.cbegin(), mReader.packetList.cend(), compareFunc);
    encodedFrame = *vpc; //store for debugging

    int64_t pts = vpc->pts - videoInputStream->start_time;
    //offset pts to always start at 0
    pkt->pts = pts;
    //calculate dts with offset to pts coming from current encoder
    AVRational r1 = { videoInputStream->avg_frame_rate.den, videoInputStream->avg_frame_rate.num };
    AVRational r2 = videoInputStream->time_base;
    pkt->dts = pts - av_rescale_q(ptsIdx - dtsIdx, r1, r2);
    //copy duration from input
    pkt->duration = vpc->duration;
    //std::printf("pktpts=%zd pktdts=%zd dur=%zd\n", pkt->pts, pkt->dts, pkt->duration);

    //STEP 2: convert to output timebase
    //rescale packet from input timebase to output timebase
    av_packet_rescale_ts(pkt, videoInputStream->time_base, videoStream->time_base);

    //process secondary streams
    double dtsTime = 1.0 * pkt->dts * videoStream->time_base.num / videoStream->time_base.den;
    for (StreamContext& sc : mReader.inputStreams) {
        for (auto it = sc.packets.begin(); it != sc.packets.end(); ) {
            AVPacket* sidePacket = *it;
            int comp = av_compare_ts(sidePacket->dts, sc.inputStream->time_base, vpc->dts, videoInputStream->time_base);
            if (comp < 0 || terminate) {
                if (sc.handling == StreamHandling::STREAM_COPY) { //copy packet directly to output stream
                    //rescale audio packet
                    AVRational& tbin = sc.inputStream->time_base;
                    AVRational& tbout = sc.outputStream->time_base;
                    int64_t ts = sidePacket->pts - sc.inputStream->start_time;
                    sidePacket->pts = av_rescale_delta(tbin, ts, tbin, (int) sidePacket->duration, &sc.lastPts, tbout);
                    sidePacket->dts = sidePacket->pts;
                    sidePacket->duration = 0;
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
        
        if (terminate && sc.handling == StreamHandling::STREAM_TRANSCODE) { //flush transcoding buffers
            transcodeAudio(nullptr, sc, true);
        }
    }

    //static std::ofstream testFile("f:/test.h265", std::ios::binary);
    //testFile.write(reinterpret_cast<char*>(videoPacket->data), videoPacket->size);

    //delete from input list
    mReader.packetList.erase(vpc);

    //store stats
    int encodedBytes = pkt->size;
    encodedDts = pkt->dts;
    encodedPts = pkt->pts;
    //static std::ofstream time("f:/time.txt");
    //time << frameEncoded << " " << pkt->pts << " " << pkt->dts << std::endl;

    //write packet to output
    writePacket(pkt);
    encodedBytesTotal += encodedBytes;
    outputBytesWritten = avio_tell(fmt_ctx->pb);

    //advance encoded counter
    this->frameEncoded++;
}

//close ffmpeg format
FFmpegFormatWriter::~FFmpegFormatWriter() {
    if (fmt_ctx != nullptr && headerWritten) {
        int result = av_write_trailer(fmt_ctx);
        if (result < 0) {
            errorLogger.logError(av_make_error(result, "error writing trailer"));
        }
        outputBytesWritten = avio_tell(fmt_ctx->pb);
    }

    if (fmt_ctx != nullptr && fmt_ctx->pb != nullptr && headerWritten) {
        int result = avio_close(fmt_ctx->pb);
        if (result < 0) {
            errorLogger.logError(av_make_error(result, "error closing output"));
        }
    }
    av_packet_free(&videoPacket);
    avformat_free_context(fmt_ctx);
}
