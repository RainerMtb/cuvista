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
#include "ErrorLogger.hpp"

#include <fstream>
#include <numeric>


//----------------------------------
//-------- Movie Reader Main -------
//----------------------------------


std::future<void> MovieReader::readAsync(ImageYuv& inputFrame) {
    return std::async(std::launch::async, [&] { read(inputFrame); });
}

std::optional<int64_t> MovieReader::ptsForFrameAsMillis(int64_t frameIndex) {
    auto fcn = [&] (const VideoPacketContext& vpc) { return vpc.readIndex == frameIndex; };
    auto result = std::find_if(videoPacketList.cbegin(), videoPacketList.cend(), fcn);
    if (result != videoPacketList.end()) {
        return result->bestTimestamp * 1000 * videoStream->time_base.num / videoStream->time_base.den;
    } else {
        return std::nullopt;
    }
}

double MovieReader::ptsForFrame(int64_t frameIndex) {
    auto millis = ptsForFrameAsMillis(frameIndex);
    return millis.has_value() ? (millis.value() / 1000.0) : std::numeric_limits<double>::quiet_NaN();
}

std::optional<std::string> MovieReader::ptsForFrameAsString(int64_t frameIndex) {
    auto millis = ptsForFrameAsMillis(frameIndex);
    if (millis.has_value()) {
        return timeString(millis.value());
    } else {
        return std::nullopt;
    }
}


//----------------------------------
//-------- Placeholder Class -------
//----------------------------------

void NullReader::read(ImageYuv& frame) {
    frameIndex++;
    endOfInput = false;
    frame.setValues(im::ColorYuv { 0, 0, 0 });
}


//----------------------------------
//-------- FFmpeg Reader -----------
//----------------------------------


void FFmpegReader::open(const std::string& source) {
    //av_log_set_level(AV_LOG_ERROR);
    av_log_set_callback(ffmpeg_log);

    // Allocate format context
    AVFormatContext* av_fmt = avformat_alloc_context();
    if (av_fmt == nullptr)
        throw AVException("could not create AVFormatContext");

    openInput(av_fmt, source.data());
}


//open audio decoding
int FFmpegReader::openAudioDecoder(int streamIndex) {
    StreamContext& sc = inputStreams[streamIndex];
    
    //input format and codec
    AVCodecID id = sc.inputStream->codecpar->codec_id;
    sc.audioInCodec = avcodec_find_decoder(id);
    if (!sc.audioInCodec)
        throw AVException(std::format("cannot find audio decoder for '{}'", avcodec_get_name(id)));
    sc.audioInCtx = avcodec_alloc_context3(sc.audioInCodec);
    if (!sc.audioInCtx)
        throw AVException("cannot allocate audio decoder context");
    int retval = avcodec_parameters_to_context(sc.audioInCtx, sc.inputStream->codecpar);
    if (retval < 0)
        throw AVException(av_make_error(retval, "cannot copy audio parameters to input context"));
    retval = avcodec_open2(sc.audioInCtx, sc.audioInCodec, NULL);
    if (retval < 0)
        throw AVException(av_make_error(retval, "cannot open audio decoder"));
    sc.audioInCtx->time_base = sc.inputStream->time_base;

    //converter to float stereo
    AVChannelLayout bufferChannelFormat = AV_CHANNEL_LAYOUT_STEREO;
    int sampleRate = sc.audioInCtx->sample_rate;
    retval = swr_alloc_set_opts2(&sc.resampleCtx,
        &bufferChannelFormat, decodingSampleFormat, sampleRate,
        &sc.audioInCtx->ch_layout, sc.audioInCtx->sample_fmt, sampleRate,
        0, NULL);
    if (retval < 0)
        throw AVException("cannot set audio resampler options");
    retval = swr_init(sc.resampleCtx);
    if (retval < 0)
        throw AVException(av_make_error(retval, "cannot init audio resampler"));

    //allocate output frame
    sc.frameIn = av_frame_alloc();
    if (!sc.frameIn)
        throw AVException("cannot allocate audio output frame");

    return sampleRate;
}


//open ffmpeg file
void FFmpegFormatReader::openInput(AVFormatContext* fmt, const char* source) {
    av_format_ctx = fmt;
    int err;

    // Open the file using libavformat
    err = avformat_open_input(&av_format_ctx, source, NULL, NULL);
    isFormatOpen = err == 0;
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

        if (videoStream == nullptr && stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            //store first video stream
            av_codec = avcodec_find_decoder(stream->codecpar->codec_id);
            if (av_codec) {
                videoStream = stream;
            }
        }

        //store every stream found in input
        inputStreams.emplace_back(stream);
    }
    //continue only when there is a video stream to decode
    if (videoStream == nullptr || av_codec == nullptr)
        throw AVException("could not find a valid video stream");

    // Set up a codec context for the decoder
    if ((av_codec_ctx = avcodec_alloc_context3(av_codec)) == nullptr) 
        throw AVException("could not create AVCodecContext");
    if (avcodec_parameters_to_context(av_codec_ctx, videoStream->codecpar) < 0)
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
    videoDuration = videoStream->duration;
    fpsNum = videoStream->avg_frame_rate.num;
    fpsDen = videoStream->avg_frame_rate.den;
    timeBaseNum = videoStream->time_base.num;
    timeBaseDen = videoStream->time_base.den;
    h = av_codec_ctx->height;
    w = av_codec_ctx->width;
    sourceName = source;
    frameCount = videoStream->nb_frames;
    //av_dump_format(av_format_ctx, av_stream->index, av_format_ctx->url, 0); //uses av_log
}


void FFmpegFormatReader::close() {
    avcodec_free_context(&av_codec_ctx);
    av_packet_free(&av_packet);
    av_frame_free(&av_frame);
    if (isFormatOpen) avformat_close_input(&av_format_ctx);
    if (isFormatOpen) avformat_free_context(av_format_ctx);
}


FFmpegFormatReader::~FFmpegFormatReader() {
    close();
}


//read one frame from ffmpeg
void FFmpegReader::read(ImageYuv& frame) {
    //util::ConsoleTimer timer("read");
    frameIndex++;
    endOfInput = true;
    int response;

    while (true) {
        //we may have a stored packet from last run
        if (isStoredPacket) {
            isStoredPacket = false;

        } else {
            av_packet_unref(av_packet); //unref old packet
            response = av_read_frame(av_format_ctx, av_packet); //read new packet from input format
        }

        //decode packet
        if (av_packet->size > 0) {
            int sidx = av_packet->stream_index;
            StreamHandling sh = sidx < inputStreams.size() ? inputStreams[sidx].handling : StreamHandling::STREAM_IGNORE;

            if (sidx == videoStream->index) {
                //we have a video packet
                response = avcodec_send_packet(av_codec_ctx, av_packet); //send packet to decoder
                if (response == AVERROR(EAGAIN)) { //we have to receive frames before we can send again
                    isStoredPacket = true;

                } else if (response < 0) {
                    errorLogger.logError(av_make_error(response, "failed to send packet"));
                    break;
                }

                response = avcodec_receive_frame(av_codec_ctx, av_frame); //try to get a frame from decoder
                if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) { //we have to send more packets before getting next frame
                    continue;

                } else if (response < 0) { //something wrong
                    errorLogger.logError(av_make_error(response, "failed to receive frame"));
                    break;

                } else { //we got a frame
                    endOfInput = false;
                    break;
                }

            } else if (storePackets == false) {
                //do not store any secondary packets

            } else if (sh == StreamHandling::STREAM_COPY || sh == StreamHandling::STREAM_TRANSCODE) {
                //we should store a packet from a secondary stream for processing
                std::list<SidePacket>& packetList = inputStreams[sidx].packets;
                packetList.emplace_back(frameIndex, av_packet);

                //limiting packets in memory
                auto fcn = [] (int sum, const SidePacket& pkt) { return sum + pkt.packet->size; };
                int siz = std::accumulate(packetList.begin(), packetList.end(), 0, fcn);
                while (siz > sideDataMaxSize) {
                    packetList.pop_front();
                    siz = std::accumulate(packetList.begin(), packetList.end(), 0, fcn);
                }

            } else if (sh == StreamHandling::STREAM_DECODE) {
                //decode the packets and store raw bytes
                StreamContext& sc = inputStreams[sidx];
                response = avcodec_send_packet(sc.audioInCtx, av_packet);
                if (response == AVERROR_EOF) {
                    break;

                } else if (response < 0) {
                    ffmpeg_log_error(response, "cannot send audio packet to decoder");
                    break;
                }

                double timestamp = 1.0 * (av_packet->pts - sc.inputStream->start_time) * sc.inputStream->time_base.num / sc.inputStream->time_base.den;
                SidePacket sidePacket(frameIndex, timestamp);
                int bytesPerSample = av_get_bytes_per_sample(decodingSampleFormat);
                while (response >= 0) {
                    response = avcodec_receive_frame(sc.audioInCtx, sc.frameIn);
                    if (response == AVERROR(EAGAIN) || response == AVERROR_EOF)
                        break;

                    else if (response < 0) {
                        ffmpeg_log_error(response, "cannot receive audio packet from decoder");
                        break;
                    }

                    //no sample rate conversion makes conversion functions simpler
                    int sampleCount = swr_get_out_samples(sc.resampleCtx, sc.frameIn->nb_samples);
                    int byteCount = sampleCount * bytesPerSample * 2; //two output channels in packed format
                    std::vector<uint8_t> decodedSamples(byteCount);
                    if (sampleCount > 0) {
                        uint8_t* dest[AV_NUM_DATA_POINTERS] = { decodedSamples.data() }; //must be an array of values
                        const uint8_t** indata = (const uint8_t**) (sc.frameIn->data);
                        int converted = swr_convert(sc.resampleCtx, dest, sampleCount, indata, sc.frameIn->nb_samples);
                        if (converted > 0) {
                            byteCount = converted * bytesPerSample * 2;
                            std::copy(decodedSamples.cbegin(), decodedSamples.cbegin() + byteCount, std::back_inserter(sidePacket.audioData));
                        }
                    }
                }

                //static std::ofstream file("f:/audio.raw", std::ios::binary);
                //file.write(reinterpret_cast<char*>(sidePacket.audioData.data()), sidePacket.audioData.size());
                inputStreams[sidx].packets.push_back(std::move(sidePacket));
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
                endOfInput = false;
                break;
            }
        }
    }

    if (endOfInput == false) {
        //convert to YUV444 data
        frame.index = frameIndex;
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
        int64_t timestamp = av_frame->best_effort_timestamp - videoStream->start_time;
        videoPacketList.emplace_back(frameIndex, av_frame->pts, av_frame->pkt_dts, av_frame->duration, timestamp);

        //in some cases pts values are not in proper sequence, but frames decoded by ffmpeg are indeed in correct order
        //in that case just reorder pts values -- bug in ffmpeg
        auto it = videoPacketList.end();
        it--;
        while (it != videoPacketList.begin() && it->pts < std::prev(it)->pts) {
            std::swap(it->pts, std::prev(it)->pts);
            it--;
        }

        //stamp frame index into image
        //frame.writeText(std::to_string(frameIndex), 100, 100, 3, 3, ColorYuv::WHITE, ColorYuv::GRAY);
        // frame.saveAsColorBMP(std::format("f:/im{:03d}.bmp", frameIndex));
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
    frameIndex = -1;
    endOfInput = true;
    videoPacketList.clear();
    for (StreamContext& s : inputStreams) {
        s.packets.clear();
        s.packetsWritten = 0;
        s.lastPts = 0;
        s.pts = 0;
    }
}

void FFmpegReader::close() {
    sws_freeContext(sws_scaler_ctx);

    videoStream = nullptr;
    sws_scaler_ctx = nullptr;
    videoPacketList.clear();
    inputStreams.clear();
}

FFmpegReader::~FFmpegReader() {
    close();
}


//-----------------------------------
//-------- Memory FFmpeg Reader -----
//-----------------------------------


MemoryFFmpegReader::MemoryFFmpegReader(std::span<unsigned char> movieData) :
    mData { movieData } {}

void MemoryFFmpegReader::open(const std::string& source) {
    int bufsiz = 4 * 1024;
    mBuffer = (unsigned char*) av_malloc(bufsiz);
    av_avio = avio_alloc_context(mBuffer, bufsiz, 0, this, &MemoryFFmpegReader::readBuffer, nullptr, &MemoryFFmpegReader::seekBuffer);
    av_fmt = avformat_alloc_context();
    av_fmt->pb = av_avio;
    av_fmt->flags |= AVFMT_FLAG_CUSTOM_IO;

    openInput(av_fmt, "");
}

int MemoryFFmpegReader::readBuffer(void* opaque, unsigned char* buf, int bufsiz) {
    //std::cout << "read" << std::endl;
    MemoryFFmpegReader* ptr = static_cast<MemoryFFmpegReader*>(opaque);
    int i = 0;
    int64_t siz = ptr->mData.size();
    while (ptr->mDataPos < siz && i < bufsiz) {
        buf[i] = ptr->mData[ptr->mDataPos];
        ptr->mDataPos++;
        i++;
    }
    return i;
}

int64_t MemoryFFmpegReader::seekBuffer(void* opaque, int64_t offset, int whence) {
    //std::cout << "seek" << std::endl;
    MemoryFFmpegReader* ptr = static_cast<MemoryFFmpegReader*>(opaque);
    int64_t out = 0;
    switch (whence) {
    case AVSEEK_SIZE:
        out = ptr->mData.size();
        break;

    case SEEK_CUR:
        ptr->mDataPos += offset;
        out = ptr->mDataPos;
        break;

    case SEEK_END:
        ptr->mDataPos = ptr->mData.size() + offset;
        out = ptr->mDataPos;
        break;

    case SEEK_SET:
        ptr->mDataPos = offset;
        out = ptr->mDataPos;
    }
    return out;
}

MemoryFFmpegReader::~MemoryFFmpegReader() {
    avformat_close_input(&av_fmt);
    avformat_free_context(av_fmt);
    avio_context_free(&av_avio); //seems to also free the buffer
}
