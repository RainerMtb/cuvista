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

//setup output format from codec
void FFmpegFormatWriter::openFormat(AVCodecID codecId, const std::string& sourceName, int queueSize) {
    av_log_set_callback(ffmpeg_log);

    //setup output context
    AVFormatContext* ctx = nullptr;
    int result = avformat_alloc_output_context2(&ctx, NULL, NULL, sourceName.c_str());
    if (result < 0)
        throw AVException(av_make_error(result, "cannot allocate output format"));

    //continue open format
    openFormat(codecId, ctx, queueSize);

    //open output for writing
    result = avio_open(&fmt_ctx->pb, fmt_ctx->url, AVIO_FLAG_WRITE);
    if (result < 0)
        throw AVException("error opening output file '" + mData.fileOut + "'");
}

//setup output format from format context
void FFmpegFormatWriter::openFormat(AVCodecID codecId, AVFormatContext* ctx, int queueSize) {
    //set format context into class
    fmt_ctx = ctx;

    //set default stream handling
    for (StreamContext& sc : mReader.mInputStreams) {
        auto osc = std::make_shared<OutputStreamContext>();
        osc->inputStream = sc.inputStream;

        if (sc.inputStream->index == mReader.videoStream->index) {
            osc->handling = StreamHandling::STREAM_STABILIZE;

        } else {
            int codecSupported = avformat_query_codec(fmt_ctx->oformat, sc.inputStream->codecpar->codec_id, FF_COMPLIANCE_NORMAL);
            //codecSupported = false; //force transcode for debugging <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            if (codecSupported == 1) {
                osc->handling = StreamHandling::STREAM_COPY;

            } else if (sc.inputStream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                osc->handling = StreamHandling::STREAM_TRANSCODE;

            } else {
                osc->handling = StreamHandling::STREAM_IGNORE;
            }
        }
        sc.outputStreams.push_back(osc);
        outputStreams.push_back(osc);
    }

    //saturate encoding buffer
    for (int i = 0; i < queueSize - 1; i++) {
        encodingQueue.push_back(std::async([] {}));
    }

    //allocate ffmpeg stuff
    openFormat(codecId);
}

void FFmpegFormatWriter::openFormat(AVCodecID codecId) {
    //av_log_set_level(AV_LOG_ERROR);
    //custom callback to log ffmpeg errors
    av_log_set_callback(ffmpeg_log);

    //setup streams
    for (std::shared_ptr<OutputStreamContext> posc : outputStreams) {
        OutputStreamContext& osc = *posc;
        AVStream* inStream = osc.inputStream;

        if (osc.handling == StreamHandling::STREAM_STABILIZE) {
            int codecSupported = avformat_query_codec(fmt_ctx->oformat, codecId, FF_COMPLIANCE_STRICT);
            if (codecSupported == 1) {
                osc.outputStream = videoStream = createNewStream(fmt_ctx, inStream);

            } else {
                throw AVException(std::format("cannot write codec '{}' to output", avcodec_get_name(codecId)));
            }
            
        } else if (osc.handling == StreamHandling::STREAM_COPY) {
            osc.outputStream = createNewStream(fmt_ctx, inStream);

            int retval = avcodec_parameters_copy(osc.outputStream->codecpar, inStream->codecpar);
            if (retval < 0)
                throw AVException("cannot copy context for stream");

            //setting tag to 0 seems to avoid "Tag xxxx incompatible with output codec id as shown in ffmpeg example
            //https://ffmpeg.org/doxygen/trunk/remuxing_8c-example.html
            osc.outputStream->codecpar->codec_tag = 0;

            //create default channel layout if unspecified
            if (osc.outputStream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && 
                osc.outputStream->codecpar->ch_layout.nb_channels > 0 &&
                osc.outputStream->codecpar->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
                av_channel_layout_default(&osc.outputStream->codecpar->ch_layout, osc.outputStream->codecpar->ch_layout.nb_channels);
            }

        } else if (osc.handling == StreamHandling::STREAM_TRANSCODE) {
            osc.outputStream = createNewStream(fmt_ctx, inStream);

            //set up audio transcoding
            //open reader
            osc.audioInCodec = avcodec_find_decoder(inStream->codecpar->codec_id);
            if (!osc.audioInCodec)
                throw AVException(std::format("cannot find audio decoder for '{}'", avcodec_get_name(inStream->codecpar->codec_id)));
            osc.audioInCtx = avcodec_alloc_context3(osc.audioInCodec);
            if (!osc.audioInCtx)
                throw AVException("cannot allocate audio decoder context");
            int retval = avcodec_parameters_to_context(osc.audioInCtx, inStream->codecpar);
            if (retval < 0)
                throw AVException(av_make_error(retval, "cannot copy audio parameters to input context"));

            //create default channel layout if unspecified
            if (osc.audioInCtx->ch_layout.nb_channels > 0 && osc.audioInCtx->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
                av_channel_layout_default(&osc.audioInCtx->ch_layout, osc.audioInCtx->ch_layout.nb_channels);
            }
            retval = avcodec_open2(osc.audioInCtx, osc.audioInCodec, NULL);
            if (retval < 0)
                throw AVException(av_make_error(retval, "cannot open audio decoder"));
            osc.audioInCtx->time_base = inStream->time_base;

            //open writer
            //AVCodecID id = inStream->codecpar->codec_id;
            AVCodecID id = fmt_ctx->oformat->audio_codec; //default codec for output format
            osc.audioOutCodec = avcodec_find_encoder(id);
            if (!osc.audioOutCodec)
                throw AVException(std::format("cannot find audio encoder for '{}'", avcodec_get_name(id)));
            osc.audioOutCtx = avcodec_alloc_context3(osc.audioOutCodec);
            if (!osc.audioOutCtx)
                throw AVException("cannot allocate audio encoder context");
            retval = av_channel_layout_copy(&osc.audioOutCtx->ch_layout, &osc.audioInCtx->ch_layout);
            if (retval < 0)
                throw AVException("cannot copy audio channel layout");
            osc.audioOutCtx->sample_rate = osc.audioInCtx->sample_rate;
            
            osc.audioOutCtx->sample_fmt = osc.audioOutCodec->sample_fmts[0]; //deprecated
            //const AVSampleFormat* sampleFmts = nullptr;
            //int nFmts;
            //avcodec_get_supported_config(osc.audioOutCtx, osc.audioOutCodec, AV_CODEC_CONFIG_SAMPLE_FORMAT, 0, (const void**) &sampleFmts, &nFmts);
            //osc.audioOutCtx->sample_fmt = (sampleFmts)[0];

            if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
                osc.audioOutCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

            retval = avcodec_open2(osc.audioOutCtx, osc.audioOutCodec, NULL);
            if (retval < 0)
                throw AVException("cannot open audio encoder");

            //stream timebase
            osc.outputStream->time_base.den = osc.audioOutCtx->sample_rate;
            osc.outputStream->time_base.num = 1;

            retval = avcodec_parameters_from_context(osc.outputStream->codecpar, osc.audioOutCtx);
            if (retval < 0)
                throw AVException("cannot copy audio parameters to output stream");

            //allocate packet
            osc.outpkt = av_packet_alloc();
            if (!osc.outpkt)
                throw AVException("cannot allocate audio packet");

            //allocate input frame
            osc.frameIn = av_frame_alloc();
            if (!osc.frameIn)
                throw AVException("cannot allocate audio input frame");

            //allocate output frame
            osc.frameOut = av_frame_alloc();
            if (!osc.frameOut)
                throw AVException("cannot allocate audio output frame");
            retval = av_channel_layout_copy(&osc.frameOut->ch_layout, &osc.audioOutCtx->ch_layout);
            if (retval < 0)
                throw AVException("cannot copy channel layout");
            osc.frameOut->format = osc.audioOutCtx->sample_fmt;
            osc.frameOut->nb_samples = osc.audioOutCtx->frame_size;
            retval = av_frame_get_buffer(osc.frameOut, 0);
            if (retval < 0)
                throw AVException(av_make_error(retval, "cannot allocate audio frame buffers"));

            //resampler
            retval = swr_alloc_set_opts2(&osc.resampleCtx, 
                &osc.audioOutCtx->ch_layout, osc.audioOutCtx->sample_fmt, osc.audioOutCtx->sample_rate,
                &osc.audioInCtx->ch_layout, osc.audioInCtx->sample_fmt, osc.audioInCtx->sample_rate, 
                0, NULL);
            if (retval < 0)
                throw AVException("cannot set audio resampler options");
            retval = swr_init(osc.resampleCtx);
            if (retval < 0)
                throw AVException(av_make_error(retval, "cannot init audio resampler"));

            //sample buffer
            int n = std::max(osc.audioInCtx->frame_size, osc.audioOutCtx->frame_size) * 2;
            osc.fifo = av_audio_fifo_alloc(osc.audioOutCtx->sample_fmt, osc.audioOutCtx->ch_layout.nb_channels, n);
            if (!osc.fifo)
                throw AVException("cannot allocate fifo");
        }
    }
}

AVStream* FFmpegFormatWriter::createNewStream(AVFormatContext* fmt_ctx, AVStream* inStream) {
    AVStream* stream = avformat_new_stream(fmt_ctx, NULL); //docs say AVCodec parameter is not used
    if (!stream)
        throw AVException("could not create stream");

    stream->time_base = inStream->time_base;
    stream->start_time = 0;
    stream->duration = inStream->duration;
    return stream;
}

//----------------------------------
//-------- write packets
//----------------------------------

//write packet to output
int FFmpegFormatWriter::writePacket(AVPacket* packet) {
    auto& osc = outputStreams[packet->stream_index];
    //std::printf("stream %d pts [sec] %.5f\n", packet->stream_index, 1.0 * packet->pts * osc->outputStream->time_base.num / osc->outputStream->time_base.den);
    
    int result = av_interleaved_write_frame(fmt_ctx, packet); //write_frame also does unref packet
    if (result == 0) {
        osc->packetsWritten++;

    } else {
        ffmpeg_log_error(result, "error writing packet", ErrorSource::WRITER);
    }
    return result;
}

//transcode pending audio packet and write to output
void FFmpegFormatWriter::transcodeAudio(AVPacket* pkt, OutputStreamContext& osc, bool terminate) {
    //send packet to decoder
    int retval = 0;

    retval = avcodec_send_packet(osc.audioInCtx, pkt);
    if (retval == AVERROR_EOF) {
        //input packet was nullptr at least for the second time, ignore

    } else if (retval < 0) {
        ffmpeg_log_error(retval, "cannot send audio packet to decoder", ErrorSource::WRITER);
        return;
    }

    bool doLoop = true;
    while (doLoop) {
        bool eof = false;

        //decode and convert and store in buffer
        while (av_audio_fifo_size(osc.fifo) < osc.audioOutCtx->frame_size) {
            retval = avcodec_receive_frame(osc.audioInCtx, osc.frameIn);
            if (retval == AVERROR(EAGAIN)) {
                doLoop = false; //decoder needs more packets
                break;

            } else if (retval == AVERROR_EOF) {
                eof = true;
                break;

            } else {
                //static std::ofstream outfile("f:/audio.raw", std::ios::binary);
                //outfile.write(reinterpret_cast<char*>(sc.frameIn->data[0]), sc.frameIn->linesize[0] / sc.frameIn->ch_layout.nb_channels);
                int sampleCount = swr_get_out_samples(osc.resampleCtx, osc.frameIn->nb_samples);
                uint8_t** samplesArray;
                int linesize;
                int ch = osc.audioOutCtx->ch_layout.nb_channels;
                int bufsiz = av_samples_alloc_array_and_samples(&samplesArray, &linesize, ch, sampleCount, osc.audioOutCtx->sample_fmt, 0);
                if (bufsiz < 0)
                    ffmpeg_log_error(retval, "cannot allocate samples", ErrorSource::WRITER);

                const uint8_t** indata = (const uint8_t**) (osc.frameIn->extended_data);
                sampleCount = swr_convert(osc.resampleCtx, samplesArray, sampleCount, indata, osc.frameIn->nb_samples);
                if (sampleCount < 0)
                    ffmpeg_log_error(sampleCount, "cannot convert samples", ErrorSource::WRITER);

                retval = av_audio_fifo_realloc(osc.fifo, av_audio_fifo_size(osc.fifo) + sampleCount);
                if (retval < 0)
                    ffmpeg_log_error(retval, "cannot resize fifo", ErrorSource::WRITER);

                retval = av_audio_fifo_write(osc.fifo, (void**) samplesArray, sampleCount);
                if (retval != sampleCount)
                    ffmpeg_log_error(retval, "cannot write to fifo", ErrorSource::WRITER);

                if (samplesArray) av_freep(samplesArray);
                av_freep(&samplesArray);
            }
        } //decoder loop

        //encode from buffer
        while (av_audio_fifo_size(osc.fifo) >= osc.audioOutCtx->frame_size || eof) {
            void** ptrs = (void**) osc.frameOut->extended_data;
            int sampleCount = av_audio_fifo_read(osc.fifo, ptrs, osc.audioOutCtx->frame_size);
            AVFrame* av_frame = nullptr;

            if (sampleCount < 0) {
                ffmpeg_log_error(retval, "cannot read from fifo", ErrorSource::WRITER);

            } else if (sampleCount > 0) {
                osc.frameOut->pts = osc.pts;
                osc.frameOut->pkt_dts = osc.pts;
                osc.frameOut->duration = 0;
                osc.pts += sampleCount;
                av_frame = osc.frameOut;
            }

            retval = avcodec_send_frame(osc.audioOutCtx, av_frame); //send flush signal when no more samples to encode
            if (retval == AVERROR_EOF) {
                //flush signal was sent at least for the second time, ignore here

            } else if (retval < 0) {
                ffmpeg_log_error(retval, "cannot send audio frame to encoder", ErrorSource::WRITER);
            }

            retval = avcodec_receive_packet(osc.audioOutCtx, osc.outpkt);
            if (retval == AVERROR(EAGAIN)) {
                break;

            } else if (retval == AVERROR_EOF) {
                doLoop = false; //nothing more to do
                break;

            } else if (retval < 0) {
                ffmpeg_log_error(retval, "cannot receive audio packet from encodeer", ErrorSource::WRITER);

            } else {
				//write packet to output
				osc.outpkt->stream_index = osc.outputStream->index;
				writePacket(osc.outpkt);
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
    VideoPacketContext vpc = {};
    {
        std::unique_lock<std::mutex> lock(mReader.mVideoPacketMutex);
        //search for packet info
        auto vpcIter = std::find_if(mReader.mVideoPacketList.cbegin(), mReader.mVideoPacketList.cend(), compareFunc);
        //store
        vpc = *vpcIter;
        //delete from input list
        mReader.mVideoPacketList.erase(vpcIter);
    }

    int64_t pts = vpc.pts - videoInputStream->start_time;
    //offset pts to always start at 0
    pkt->pts = pts;
    //calculate dts with offset to pts coming from current encoder
    AVRational r1 = { videoInputStream->avg_frame_rate.den, videoInputStream->avg_frame_rate.num };
    AVRational r2 = videoInputStream->time_base;
    pkt->dts = pts - av_rescale_q(ptsIdx - dtsIdx, r1, r2);
    //copy duration from input
    pkt->duration = vpc.duration;
    //std::printf("pktpts=%zd pktdts=%zd dur=%zd\n", pkt->pts, pkt->dts, pkt->duration);

    //STEP 2: convert to output timebase
    //rescale packet from input timebase to output timebase
    av_packet_rescale_ts(pkt, videoInputStream->time_base, videoStream->time_base);

    //process secondary streams
    //write packets from other streams that were read before this video frame
    for (std::shared_ptr<OutputStreamContext> posc : outputStreams) {
        std::unique_lock<std::mutex> lock(posc->mMutexSidePackets);
        for (auto it = posc->sidePackets.begin(); it != posc->sidePackets.end(); ) {
            SidePacket& sidePacket = *it;
            if (sidePacket.frameIndex <= frameEncoded || terminate) {
                if (posc->handling == StreamHandling::STREAM_COPY) { //copy packet directly to output stream
                    //rescale audio packet
                    AVRational& tbin = posc->inputStream->time_base;
                    AVRational& tbout = posc->outputStream->time_base;
                    AVPacket* avpkt = sidePacket.packet;
                    int64_t ts = avpkt->pts - posc->inputStream->start_time;
                    avpkt->pts = av_rescale_delta(tbin, ts, tbin, (int) avpkt->duration, &posc->lastPts, tbout);
                    avpkt->dts = avpkt->pts;
                    avpkt->duration = 0;
                    writePacket(avpkt);

                } else if (posc->handling == StreamHandling::STREAM_TRANSCODE) { //transcode audio
                    transcodeAudio(sidePacket.packet, *posc, false);
                }

                //remove packet from buffer
                it = posc->sidePackets.erase(it);

            } else {
                it++;
            }
        }
        
        if (terminate && posc->handling == StreamHandling::STREAM_TRANSCODE) { //flush transcoding buffers
            transcodeAudio(nullptr, *posc, true);
        }
    }

    //static std::ofstream testFile("f:/test.h265", std::ios::binary);
    //testFile.write(reinterpret_cast<char*>(videoPacket->data), videoPacket->size);

    //store stats
    int encodedBytes = pkt->size;
    //static std::ofstream time("f:/time.txt"); time << frameEncoded << " " << pkt->pts << " " << pkt->dts << std::endl;

    //write packet to output, this will unref packet data
    writePacket(pkt);

    //update stats
    std::unique_lock<std::mutex> lock(mStatsMutex);
    encodedBytesTotal += encodedBytes;
    outputBytesWritten = avio_tell(fmt_ctx->pb);
    frameEncoded++;
}

//close ffmpeg format
void FFmpegFormatWriter::close() {
    if (fmt_ctx && isHeaderWritten) {
        int result = av_write_trailer(fmt_ctx);
        if (result < 0) {
            errorLogger().logError(av_make_error(result, "error writing trailer"), ErrorSource::WRITER);
        }
        outputBytesWritten = avio_tell(fmt_ctx->pb);
    }
    if (fmt_ctx) {
        avformat_close_input(&fmt_ctx); //free_context does not properly close the output file
    }
}

FFmpegFormatWriter::~FFmpegFormatWriter() {
    if (av_avio) {
        avio_context_free(&av_avio);
    }
    if (videoPacket) {
        av_packet_free(&videoPacket);
    }
    if (fmt_ctx) {
        avformat_free_context(fmt_ctx); //release all resources of the output file
    }
}
