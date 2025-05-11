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

#include "MovieFrame.hpp"
#include "MovieReader.hpp"
#include "Util.hpp"
#include "ThreadPool.hpp"
#include "DeviceInfo.hpp"
#include "CudaWriter.hpp"
#include "CudaInterface.hpp"

#if defined(BUILD_CUDA) && BUILD_CUDA == 0

CudaFFmpegWriter::CudaFFmpegWriter(MainData& data, MovieReader& reader) : FFmpegFormatWriter(data, reader) {}
CudaFFmpegWriter::~CudaFFmpegWriter() {}

void CudaFFmpegWriter::writePacketToFile(const NvPacket& nvpkt, bool terminate) {}
void CudaFFmpegWriter::writePacketsToFile(std::list<NvPacket> nvpkts, bool terminate) {}
void CudaFFmpegWriter::encodePackets() {}

void CudaFFmpegWriter::open(EncodingOption videoCodec) {}
void CudaFFmpegWriter::prepareOutput(FrameExecutor& executor) {}
void CudaFFmpegWriter::write(const FrameExecutor& executor) {}
bool CudaFFmpegWriter::startFlushing() { return false; }
bool CudaFFmpegWriter::flush() { return false; }

#else

std::map<GUID, AVCodecID> guidToCodecMap = {
    { NV_ENC_CODEC_H264_GUID, AV_CODEC_ID_H264 },
    { NV_ENC_CODEC_HEVC_GUID, AV_CODEC_ID_HEVC },
};

std::map<Codec, GUID> codecToGuidMap = {
    { Codec::H264, NV_ENC_CODEC_H264_GUID },
    { Codec::H265, NV_ENC_CODEC_HEVC_GUID },
};


CudaFFmpegWriter::CudaFFmpegWriter(MainData& data, MovieReader& reader) :
    FFmpegFormatWriter(data, reader),
    nvPackets { std::make_unique<std::list<NvPacket>>() } {}

//cuda encoding contructor
void CudaFFmpegWriter::open(EncodingOption videoCodec) {
    int result;

    //select codec
    const DeviceInfoBase* dev = mData.deviceList[mData.deviceSelected];
    const DeviceInfoCuda* dic;
    if (dev->type == DeviceType::CUDA) dic = static_cast<const DeviceInfoCuda*>(dev);
    else dic = &mData.cudaInfo.devices[0];

    if (videoCodec.codec == Codec::AUTO) videoCodec.codec = dic->encodingOptions[0].codec;
    GUID guid = codecToGuidMap[videoCodec.codec];
    nvenc = dic->nvenc;

    //open ffmpeg output format
    AVCodecID id = codecToCodecIdMap[videoCodec.codec];
    FFmpegFormatWriter::open(id, mData.fileOut, 4);

    //setup nvenc class
    nvenc->createEncoder(mReader.w, mReader.h, mReader.fpsNum, mReader.fpsDen, gopSize, mData.crf, guid);

    //setup codec parameters for ffmpeg format output
    AVCodecParameters* params = videoStream->codecpar;
    params->codec_type = AVMEDIA_TYPE_VIDEO;
    params->codec_id = guidToCodecMap[guid];
    params->width = mData.w;
    params->height = mData.h;
    params->extradata_size = nvenc->mExtradataSize;
    params->extradata = (uint8_t*) av_mallocz(0ull + nvenc->mExtradataSize + AV_INPUT_BUFFER_PADDING_SIZE);
    std::memcpy(params->extradata, nvenc->mExtradata.data(), nvenc->mExtradataSize);

    result = avformat_init_output(fmt_ctx, NULL);
    if (result < 0)
        throw AVException(av_make_error(result, "error initializing output"));

    result = avformat_write_header(fmt_ctx, NULL);
    if (result < 0)
        throw AVException(av_make_error(result, "error writing file header"));
    else
        this->isHeaderWritten = true; //set for proper closing

    videoPacket = av_packet_alloc();
    if (!videoPacket)
        throw AVException("could not allocate encoder packet");
}


void CudaFFmpegWriter::prepareOutput(FrameExecutor& executor) {
    executor.getOutput(frameIndex, reinterpret_cast<unsigned char*>(nvenc->getNextInputFramePtr()), nvenc->cudaPitch);
}


//write packets to ffmpeg stream
void CudaFFmpegWriter::writePacketToFile(const NvPacket& nvpkt, bool terminate) {
    //put data from NvPacket into ffmpeg video packet
    std::vector<uint8_t> byteData = nvpkt.packet; //actual byte stream
    videoPacket->data = byteData.data();
    videoPacket->size = (int) byteData.size();
    videoPacket->stream_index = videoStream->index;

    //static std::ofstream out("f:/test.hevc", std::ios::binary);
    //out.write(reinterpret_cast<char*>(byteData.data()), byteData.size());

    //copy timing from input assuming same timebase for input and output streams
    const NV_ENC_LOCK_BITSTREAM& info = nvpkt.bitstreamData;
    videoPacket->flags = (info.pictureType == NV_ENC_PIC_TYPE_I || info.pictureType == NV_ENC_PIC_TYPE_IDR) ? AV_PKT_FLAG_KEY : 0;

    //write to container
    writePacket(videoPacket, info.frameIdx, info.frameIdx, terminate);
}


void CudaFFmpegWriter::writePacketsToFile(std::list<NvPacket> nvpkts, bool terminate) {
    for (const NvPacket& nvpkt : nvpkts) {
        writePacketToFile(nvpkt, terminate);
    }
}


void CudaFFmpegWriter::encodePackets() {
    try {
        nvenc->encodeFrame(*nvPackets);

    } catch (const AVException& e) {
        errorLogger().logError("error writing: ", e.what());
    }
}


void CudaFFmpegWriter::writeOutput(const FrameExecutor& executor) {
    encodePackets();
    encodingQueue.push_back(encoderPool.add([this, pkts = *nvPackets] { writePacketsToFile(pkts, false); }));
    encodingQueue.front().wait();
    encodingQueue.pop_front();
    this->frameIndex++;
}


//flush encoder buffer
bool CudaFFmpegWriter::startFlushing() {
    for (auto& futs : encodingQueue) futs.wait();
    nvenc->endEncode();
    return nvenc->hasBufferedFrame();
}


bool CudaFFmpegWriter::flush() {
    writePacketToFile(nvenc->getBufferedFrame(), true);
    return nvenc->hasBufferedFrame();
}


CudaFFmpegWriter::~CudaFFmpegWriter() {
    nvenc->endEncode();
    nvenc->destroyEncoder();
}

#endif