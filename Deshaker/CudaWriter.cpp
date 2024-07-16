#include "CudaWriter.hpp"
#include "Util.hpp"
#include "NvEncoder.hpp"
#include "ThreadPool.hpp"
#include "MovieFrame.hpp"

std::map<GUID, AVCodecID> guidToCodecMap = {
    { NV_ENC_CODEC_H264_GUID, AV_CODEC_ID_H264 },
    { NV_ENC_CODEC_HEVC_GUID, AV_CODEC_ID_HEVC },
};

std::map<Codec, GUID> guidMap = {
    { Codec::H264, NV_ENC_CODEC_H264_GUID },
    { Codec::H265, NV_ENC_CODEC_HEVC_GUID },
};


CudaFFmpegWriter::CudaFFmpegWriter(MainData& data, MovieReader& reader) :
    FFmpegFormatWriter(data, reader),
    nvenc { std::make_unique<NvEncoder>(data.w, data.h) },
    nvPackets { std::make_unique<std::list<NvPacket>>() } {}

//cuda encoding contructor
void CudaFFmpegWriter::open(EncodingOption videoCodec) {
    int result;

    //select codec
    const DeviceInfoCuda* dic = &mData.cudaInfo.devices[0];
    if (mData.deviceList[mData.deviceSelected]->type == DeviceType::CUDA) {
        dic = static_cast<DeviceInfoCuda*>(mData.deviceList[mData.deviceSelected]);
    }
    if (videoCodec.codec == Codec::AUTO) videoCodec.codec = dic->encodingOptions[0].codec;
    GUID guid = guidMap[videoCodec.codec];

    //open ffmpeg output format
    FFmpegFormatWriter::open(videoCodec, mData.fileOut, 4);

    //setup nvenc class
    nvenc->createEncoder(mReader.fpsNum, mReader.fpsDen, GOP_SIZE, mData.crf, guid, dic->cudaIndex);

    //setup codec parameters for ffmpeg format output
    AVCodecParameters* params = videoStream->codecpar;
    params->codec_type = AVMEDIA_TYPE_VIDEO;
    params->codec_id = guidToCodecMap[guid];
    params->width = mData.w;
    params->height = mData.h;
    params->extradata_size = nvenc->mExtradataSize;
    params->extradata = (uint8_t*) av_mallocz(0ull + nvenc->mExtradataSize + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(params->extradata, nvenc->mExtradata.data(), nvenc->mExtradataSize);

    result = avio_open(&fmt_ctx->pb, fmt_ctx->url, AVIO_FLAG_WRITE);
    if (result < 0)
        throw AVException("error opening file '" + mData.fileOut + "'");
    
    result = avformat_init_output(fmt_ctx, NULL);
    if (result < 0)
        throw AVException(av_make_error(result, "error initializing output"));

    result = avformat_write_header(fmt_ctx, NULL);
    if (result < 0)
        throw AVException(av_make_error(result, "error writing file header"));
    else
        this->headerWritten = true; //set for proper closing

    videoPacket = av_packet_alloc();
    if (!videoPacket)
        throw AVException("could not allocate encoder packet");
}


void CudaFFmpegWriter::prepareOutput(MovieFrame& frame) {
    frame.getOutput(frame.mWriter.frameIndex, reinterpret_cast<unsigned char*>(nvenc->getNextInputFramePtr()), nvenc->cudaPitch);
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
        errorLogger.logError("error writing: ", e.what());
    }
}


void CudaFFmpegWriter::write(const MovieFrame& frame) {
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
    nvenc->destroyEncoder();
}
