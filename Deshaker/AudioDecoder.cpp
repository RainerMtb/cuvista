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

#include "AudioDecoder.hpp"
#include "AVException.hpp"

#include <format>
#include <fstream>

void AudioDecoder::openFFmpeg(StreamContext* sc, double audioBufferSecs) {
    //set stream attributes
    mStreamCtx = sc;

    sc->handling = StreamHandling::STREAM_DECODE;
    AVSampleFormat bufferSampleFormat = AV_SAMPLE_FMT_FLTP;
    AVChannelLayout bufferChannelFormat = AV_CHANNEL_LAYOUT_STEREO;
    mBytesPerSample = av_get_bytes_per_sample(bufferSampleFormat);

    //read input format
    AVCodecID id = sc->inputStream->codecpar->codec_id;
    sc->audioInCodec = avcodec_find_decoder(id);
    if (!sc->audioInCodec)
        throw AVException(std::format("cannot find audio decoder for '{}'", avcodec_get_name(id)));
    sc->audioInCtx = avcodec_alloc_context3(sc->audioInCodec);
    if (!sc->audioInCtx)
        throw AVException("cannot allocate audio decoder context");
    int retval = avcodec_parameters_to_context(sc->audioInCtx, sc->inputStream->codecpar);
    if (retval < 0)
        throw AVException(av_make_error(retval, "cannot copy audio parameters to input context"));
    retval = avcodec_open2(sc->audioInCtx, sc->audioInCodec, NULL);
    if (retval < 0)
        throw AVException(av_make_error(retval, "cannot open audio decoder"));
    sc->audioInCtx->time_base = sc->inputStream->time_base;

    //converter to float stereo
    retval = swr_alloc_set_opts2(&sc->resampleCtx,
        &bufferChannelFormat, bufferSampleFormat, sc->audioInCtx->sample_rate,
        &sc->audioInCtx->ch_layout, sc->audioInCtx->sample_fmt, sc->audioInCtx->sample_rate,
        0, NULL);
    if (retval < 0)
        throw AVException("cannot set audio resampler options");
    retval = swr_init(sc->resampleCtx);
    if (retval < 0)
        throw AVException(av_make_error(retval, "cannot init audio resampler"));

    //buffer size
    mBuffer.resize(5 * 1024 * 1024);
    //mBuffer.resize(size_t(sc->audioInCtx->sample_rate * mBytesPerSample * audioBufferSecs));
    mBufferLimit = mBuffer.size() / mBytesPerSample;

    //allocate output frame
    sc->frameIn = av_frame_alloc();
    if (!sc->frameIn)
        throw AVException("cannot allocate audio output frame");
}

void AudioDecoder::decodePackets() {
    for (AVPacket* pkt : mStreamCtx->packets) {
        int retval = 0;
        retval = avcodec_send_packet(mStreamCtx->audioInCtx, pkt);
        if (retval == AVERROR_EOF) {
            //ignore end of input message

        } else if (retval < 0) {
            ffmpeg_log_error(retval, "cannot send audio packet to decoder");
            return;
        }

        while (retval >= 0) {
            retval = avcodec_receive_frame(mStreamCtx->audioInCtx, mStreamCtx->frameIn);
            if (retval == AVERROR(EAGAIN) || retval == AVERROR_EOF)
                break;

            else if (retval < 0) {
                ffmpeg_log_error(retval, "cannot decode audio");
                return;
            }
            
            //no sample rate conversion makes conversion functions simpler
            int sampleCount = swr_get_out_samples(mStreamCtx->resampleCtx, mStreamCtx->frameIn->nb_samples);

            //write samples to buffer tread safe
            std::unique_lock<std::mutex> lock(mMutex);
            if (sampleCount * mBytesPerSample > mBuffer.size() - mWriteIndex * mBytesPerSample) {
                mBufferLimit = mWriteIndex;
                mWriteIndex = 0;
            }
            uint8_t* dest[AV_NUM_DATA_POINTERS] = { mBuffer.data() + mWriteIndex }; //must be an array of values
            int converted = swr_convert(mStreamCtx->resampleCtx, dest, sampleCount, mStreamCtx->frameIn->data, mStreamCtx->frameIn->nb_samples);
            //static std::ofstream file("f:/audio.raw", std::ios::binary);
            //file.write(reinterpret_cast<char*>(*dest), sampleCount * mBytesPerSample);
            mWriteIndex += converted;
            mSamplesWritten += converted;
        }

        av_packet_free(&pkt);
    }
    mStreamCtx->packets.clear();
}

void AudioDecoder::setAudioLimit(std::optional<int64_t> millis) {
    if (millis.has_value()) {
        mPlayerLimit = *millis * mStreamCtx->audioInCtx->sample_rate / 1000;
    }
}