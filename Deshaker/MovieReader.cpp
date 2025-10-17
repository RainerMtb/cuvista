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
#include <regex>
#include <iostream>


//----------------------------------
//-------- Movie Reader Main -------
//----------------------------------


std::future<void> MovieReader::readAsync(ImageYuv& inputFrame) {
    return std::async(std::launch::async, [&] { read(inputFrame); });
}

std::optional<int64_t> MovieReader::ptsForFrameAsMillis(int64_t frameIndex) {
    auto fcn = [&] (const VideoPacketContext& vpc) { return vpc.readIndex == frameIndex; };
    std::unique_lock<std::mutex> lock(mVideoPacketMutex);
    auto result = std::find_if(mVideoPacketList.cbegin(), mVideoPacketList.cend(), fcn);
    if (result != mVideoPacketList.end()) {
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
        return util::millisToTimeString(millis.value());
    } else {
        return std::nullopt;
    }
}

std::string MovieReader::videoStreamSummary() const {
    std::string str = frameCount == 0 ? "unknown" : std::to_string(frameCount);
    return std::format("video {} x {} px @{:.3f} fps ({}:{})\nvideo frames: {}\n", w, h, fps(), fpsNum, fpsDen, str);
}


//----------------------------------
//-------- Placeholder Class -------
//----------------------------------

bool NullReader::read(ImageYuv& inputFrame) {
    frameIndex++;
    inputFrame.setColor(Color::BLACK);
    endOfInput = false;
    return false;
}


//----------------------------------
//-------- Image Reader ------------
//----------------------------------
bool ImageReader::readImage(ImageYuv& inputFrame, const ImageYuv& sourceImage) {
    frameIndex++;
    sourceImage.copyTo(inputFrame);
    endOfInput = false;
    return false;
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

    openInput(av_fmt, source);
}


//open audio decoding
int FFmpegReader::openAudioDecoder(OutputStreamContext& osc) {
    //input format and codec
    AVCodecID id = osc.inputStream->codecpar->codec_id;
    osc.audioInCodec = avcodec_find_decoder(id);
    if (!osc.audioInCodec)
        throw AVException(std::format("cannot find audio decoder for '{}'", avcodec_get_name(id)));
    osc.audioInCtx = avcodec_alloc_context3(osc.audioInCodec);
    if (!osc.audioInCtx)
        throw AVException("cannot allocate audio decoder context");
    int retval = avcodec_parameters_to_context(osc.audioInCtx, osc.inputStream->codecpar);
    if (retval < 0)
        throw AVException(av_make_error(retval, "cannot copy audio parameters to input context"));
    retval = avcodec_open2(osc.audioInCtx, osc.audioInCodec, NULL);
    if (retval < 0)
        throw AVException(av_make_error(retval, "cannot open audio decoder"));
    osc.audioInCtx->time_base = osc.inputStream->time_base;

    //converter to float stereo
    AVChannelLayout bufferChannelFormat = AV_CHANNEL_LAYOUT_STEREO;
    int sampleRate = osc.audioInCtx->sample_rate;
    retval = swr_alloc_set_opts2(&osc.resampleCtx,
        &bufferChannelFormat, decodingSampleFormat, sampleRate,
        &osc.audioInCtx->ch_layout, osc.audioInCtx->sample_fmt, sampleRate,
        0, NULL);
    if (retval < 0)
        throw AVException("cannot set audio resampler options");
    retval = swr_init(osc.resampleCtx);
    if (retval < 0)
        throw AVException(av_make_error(retval, "cannot init audio resampler"));

    //allocate output frame
    osc.frameIn = av_frame_alloc();
    if (!osc.frameIn)
        throw AVException("cannot allocate audio output frame");

    return sampleRate;
}


//open ffmpeg file
void FFmpegFormatReader::openInput(AVFormatContext* fmt, const std::string& source) {
    av_format_ctx = fmt;
    int err;

    // Open the file using libavformat
    err = avformat_open_input(&av_format_ctx, source.c_str(), NULL, NULL);
    if (err < 0)
        throw AVException(av_make_error(err, "", std::format("could not open file '{}'", source)));
    else
        isFormatOpen = true;

    //without find_stream_info width or height might be 0
    err = avformat_find_stream_info(av_format_ctx, NULL);
    if (err < 0) 
        throw AVException(av_make_error(err, "could not get stream info"));

    //search streams
    const AVCodec* av_codec = nullptr;
    int64_t frameCountEstimate = 0;
    for (size_t i = 0; i < av_format_ctx->nb_streams; i++) {
        AVStream* stream = av_format_ctx->streams[i];

        int64_t millis = -1;
        const AVDictionaryEntry* entry = av_dict_get(stream->metadata, "DURATION", NULL, 0);
        if (entry != nullptr) {
            std::string str = entry->value;
            std::regex pattern("(\\d+):(\\d+):(\\d+)[\\.,](\\d{3})\\d*");
            std::smatch matcher;
            if (std::regex_match(str, matcher, pattern)) {
                millis = std::stoll(matcher[1]) * 3'600'000 + std::stoll(matcher[2]) * 60'000 + std::stoll(matcher[3]) * 1000 + std::stoll(matcher[4]);
            }
        }

        if (videoStream == nullptr && stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            //store first video stream
            av_codec = avcodec_find_decoder(stream->codecpar->codec_id);
            if (av_codec) {
                videoStream = stream;
            }
            //calculate frame count on duration tag
            AVRational rate = stream->avg_frame_rate;
            if (millis != -1 && rate.den > 0) {
                frameCountEstimate = rate.num * millis / rate.den / 1000;
            }
        }
        
        //store every stream found in input
        mInputStreams.emplace_back(stream, millis);
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
    fpsNum = videoStream->avg_frame_rate.num;
    fpsDen = videoStream->avg_frame_rate.den;
    timeBaseNum = videoStream->time_base.num;
    timeBaseDen = videoStream->time_base.den;
    h = av_codec_ctx->height;
    w = av_codec_ctx->width;
    sourceName = source;

    //find the best number for frame count
    if (videoStream->nb_frames > 0) {
        frameCount = videoStream->nb_frames;

    } else if (videoStream->duration > 0) {
        frameCount = videoStream->duration * fpsNum * timeBaseNum / fpsDen / timeBaseDen;

    } else if (frameCountEstimate > 0) {
        frameCount = frameCountEstimate;

    } else {
        frameCount = 0;
    }
    //av_dump_format(av_format_ctx, av_stream->index, av_format_ctx->url, 0); //uses av_log
}


void FFmpegFormatReader::close() {
    avcodec_free_context(&av_codec_ctx);
    av_packet_free(&av_packet);
    av_frame_free(&av_frame);
    if (isFormatOpen) {
        avformat_close_input(&av_format_ctx);
        avformat_free_context(av_format_ctx);
    }
}


FFmpegFormatReader::~FFmpegFormatReader() {
    close();
}


//read one frame from ffmpeg
bool FFmpegReader::read(ImageYuv& inputFrame) {
    //util::ConsoleTimer timer("read");
    frameIndex++;
    endOfInput = true;
    int response = 0;

    while (true) {
        //we may have a stored packet from last run
        if (isStoredPacket) {
            isStoredPacket = false;

        } else {
            av_packet_unref(av_packet); //unref old packet
            response = av_read_frame(av_format_ctx, av_packet); //read new packet from input format
        }

        if (response == AVERROR_EOF) {
            //termination signal means we still need to drain the decoder buffer
            response = avcodec_send_packet(av_codec_ctx, NULL); //send termination, can be done repeatedly
            response = avcodec_receive_frame(av_codec_ctx, av_frame);
            if (response == AVERROR_EOF) { //really the end of the file
                break;

            } else if (response < 0) {
                errorLogger().logError(av_make_error(response, "failed to receive frame"), ErrorSource::READER);
                break;

            } else { //we still got a frame
                endOfInput = false;
                break;
            }

        } else {
            //decode a packet
            int sidx = av_packet->stream_index;
            if (sidx == videoStream->index) {
                //we have a video packet
                response = avcodec_send_packet(av_codec_ctx, av_packet); //send packet to decoder
                if (response == AVERROR(EAGAIN)) { //we have to receive frames first before we can send new data
                    isStoredPacket = true;

                } else if (response < 0) {
                    ffmpeg_log_error(response, "failed to send packet", ErrorSource::READER);
                    break;
                }

                response = avcodec_receive_frame(av_codec_ctx, av_frame); //try to get a frame from decoder
                if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) { //we have to send more packets before getting next frame
                    continue;

                } else if (response < 0) { //something wrong
                    ffmpeg_log_error(response, "failed to receive frame", ErrorSource::READER);
                    break;

                } else { //we got a frame
                    endOfInput = false;
                    break;
                }

            } else if (mStoreSidePackets == false) {
                // do nothing here

            } else {
                //provide side packets to writers
                for (std::shared_ptr<OutputStreamContext> posc : mInputStreams[sidx].outputStreams) {
                    if (posc->handling == StreamHandling::STREAM_COPY || posc->handling == StreamHandling::STREAM_TRANSCODE) {
                        std::unique_lock<std::mutex> lock(posc->mMutexSidePackets);
                        //we should store a packet from a secondary stream for processing
                        posc->sidePackets.emplace_back(frameIndex, av_packet);

                        //limiting packets in memory
                        auto fcn = [] (int sum, const SidePacket& pkt) { return sum + pkt.packet->size; };
                        int siz = std::accumulate(posc->sidePackets.begin(), posc->sidePackets.end(), 0, fcn);
                        while (siz > sideDataMaxSize) {
                            siz -= posc->sidePackets.front().packet->size;
                            posc->sidePackets.pop_front();
                        }

                    } else if (posc->handling == StreamHandling::STREAM_DECODE) {
                        //decode the packets and store raw bytes
                        StreamContext& sc = mInputStreams[sidx];
                        response = avcodec_send_packet(posc->audioInCtx, av_packet);
                        if (response == AVERROR_EOF) {
                            break;

                        } else if (response < 0) {
                            ffmpeg_log_error(response, "cannot send audio packet to decoder", ErrorSource::READER);
                            break;
                        }

                        double timestamp = 1.0 * (av_packet->pts - sc.inputStream->start_time) *
                            sc.inputStream->time_base.num / sc.inputStream->time_base.den;

                        int bytesPerSample = av_get_bytes_per_sample(decodingSampleFormat);
                        while (response >= 0) {
                            response = avcodec_receive_frame(posc->audioInCtx, posc->frameIn);
                            if (response == AVERROR(EAGAIN) || response == AVERROR_EOF)
                                break;

                            else if (response < 0) {
                                ffmpeg_log_error(response, "cannot receive audio packet from decoder", ErrorSource::READER);
                                break;
                            }

                            //no sample rate conversion makes conversion functions simpler
                            int sampleCount = swr_get_out_samples(posc->resampleCtx, posc->frameIn->nb_samples);
                            int byteCount = sampleCount * bytesPerSample * 2; //two output channels in packed format
                            SidePacket sidePacket(frameIndex, timestamp, byteCount);
                            if (sampleCount > 0) {
                                uint8_t* dest[AV_NUM_DATA_POINTERS] = { sidePacket.audioData.data() }; //must be an array of values
                                const uint8_t** indata = (const uint8_t**) (posc->frameIn->data);
                                int converted = swr_convert(posc->resampleCtx, dest, sampleCount, indata, posc->frameIn->nb_samples);
                                if (converted > 0) {
                                    byteCount = converted * bytesPerSample * 2;
                                    sidePacket.audioData.resize(byteCount);

                                    //static std::ofstream file("f:/audio.raw", std::ios::binary);
                                    //file.write(reinterpret_cast<char*>(sidePacket.audioData.data()), sidePacket.audioData.size());
                                    std::unique_lock<std::mutex> lock(posc->mMutexSidePackets);
                                    posc->sidePackets.push_back(std::move(sidePacket));
                                }
                            }
                        }
                    } //end audio decoding
                } //end loop output streams
            }
        }
    }

    if (endOfInput == false) {
        //convert to YUV444 data
        inputFrame.index = frameIndex;
        int w = av_codec_ctx->width;
        int h = av_codec_ctx->height;
        //std::cout << w << " " << h << std::endl;

        //set up sws scaler after first frame has been decoded
        if (!sws_scaler_ctx) {
            sws_scaler_ctx = sws_getContext(w, h, av_codec_ctx->pix_fmt, w, h, AV_PIX_FMT_YUV444P, SWS_BILINEAR, NULL, NULL, NULL);
        }
        if (!sws_scaler_ctx) {
            ffmpeg_log_error(0, "failed to initialize ffmpeg scaler", ErrorSource::READER);
        }

        //scale image data
        uint8_t* frame_buffer[] = { inputFrame.plane(0), inputFrame.plane(1), inputFrame.plane(2), nullptr };
        int linesizes[] = { inputFrame.stride, inputFrame.stride, inputFrame.stride, 0 };
        sws_scale(sws_scaler_ctx, av_frame->data, av_frame->linesize, 0, av_frame->height, frame_buffer, linesizes);

        //store parameters for writer
        int64_t timestamp = av_frame->best_effort_timestamp - videoStream->start_time;
        std::unique_lock<std::mutex> lock(mVideoPacketMutex);
        mVideoPacketList.emplace_back(frameIndex, av_frame->pts, av_frame->pkt_dts, av_frame->duration, timestamp);

        //in some cases pts values are not in proper sequence, but frames decoded by ffmpeg are indeed in correct order
        //in that case just reorder pts values -- bug in ffmpeg
        auto it = mVideoPacketList.end();
        it--;
        while (it != mVideoPacketList.begin() && it->pts < std::prev(it)->pts) {
            std::swap(it->pts, std::prev(it)->pts);
            it--;
        }

        //stamp frame index into image
        //inputFrame.writeText(std::to_string(frameIndex), 20, 20, 3, 3, im::TextAlign::TOP_LEFT, im::ColorYuv::WHITE, im::ColorYuv::GRAY);
        //inputFrame.saveAsColorBMP(std::format("f:/im{:03d}.bmp", frameIndex));
    }

    return endOfInput == false;
}

bool FFmpegReader::seek(double fraction) {
    int64_t target = av_format_ctx->start_time + int64_t(av_format_ctx->duration * fraction);
    int64_t min_ts = INT_MIN; //always use min_ts = INT_MIN
    int64_t max_ts = INT_MAX;

    int response = avformat_seek_file(av_format_ctx, -1, min_ts, target, max_ts, 0); 
    if (response < 0) {
        ffmpeg_log_error(response, "faild to seek in input", ErrorSource::READER);

    } else {
        avcodec_flush_buffers(av_codec_ctx);
    }

    return response >= 0;
}

void FFmpegReader::rewind() {
    seek(0.0);
    frameIndex = -1;
    endOfInput = false;
    startOfInput = true;
    mVideoPacketList.clear();
    for (StreamContext& s : mInputStreams) {
        for (auto& ptr : s.outputStreams) {
            ptr->sidePackets.clear();
            ptr->packetsWritten = 0;
            ptr->lastPts = 0;
            ptr->pts = 0;
        }
    }
}

void FFmpegReader::close() {
    sws_freeContext(sws_scaler_ctx);

    videoStream = nullptr;
    sws_scaler_ctx = nullptr;
    mVideoPacketList.clear();
    mInputStreams.clear();
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
    AVFormatContext* av_fmt = avformat_alloc_context();
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
    avio_context_free(&av_avio); //seems to also free the buffer
}
