#include "CudaWriter.hpp"

//cuda encoding contructor
void CudaFFmpegWriter::open() {
    //init format
    FFmpegFormatWriter::open();

    //check device
    int deviceIndex = data.deviceNum == -1 ? data.deviceNumBest : data.deviceNum;
    if (deviceIndex == -1)
        throw AVException("no gpu device present for encoding");

    //select first codec from list
    OutputCodec codec = data.videoCodec == OutputCodec::AUTO ? data.cuda.supportedCodecs[deviceIndex][0] : data.videoCodec;
    GUID guid = guidMap[codec];

    //setup nvenc class
    nvenc.createEncoder(data.inputCtx.fpsNum, data.inputCtx.fpsDen, GOP_SIZE, data.crf, guid, data.deviceNumBest);

    //setup ffmpeg format output
    AVCodecParameters* params = videoStream->codecpar;
    params->codec_type = AVMEDIA_TYPE_VIDEO;
    params->codec_id = guidToCodecMap[guid];
    params->width = data.w;
    params->height = data.h;

    int result = avio_open(&fmt_ctx->pb, fmt_ctx->url, AVIO_FLAG_WRITE);
    if (result < 0)
        throw AVException("error opening file '" + data.fileOut + "'");

    result = avformat_write_header(fmt_ctx, NULL);
    if (result < 0)
        throw AVException(av_make_error(result, "error writing file header"));
    else
        this->headerWritten = true; //set for proper closing

    videoPacket = av_packet_alloc();
    if (!videoPacket)
        throw AVException("could not allocate encoder packet");
}


OutputContext CudaFFmpegWriter::getOutputData() {
    return { false, true, &outputFrame, reinterpret_cast<unsigned char*>(nvenc.getNextInputFramePtr()), (int) nvenc.pitch };
}


//write packets to ffmpeg stream
void CudaFFmpegWriter::writePacketToFile(const NvPacket& nvpkt, bool terminate) {
    //put data from NvPacket into ffmpeg video packet
    std::vector<uint8_t> byteData = nvpkt.packet; //actual byte stream
    videoPacket->data = byteData.data();
    videoPacket->size = (int) byteData.size();
    videoPacket->stream_index = videoStream->index;

    //static std::ofstream out("f:/test.video", std::ios::binary);
    //out.write(reinterpret_cast<char*>(byteData.data()), byteData.size());

    //copy timing from input assuming same timebase for input and output streams
    const NV_ENC_LOCK_BITSTREAM& info = nvpkt.bitstreamData;
    videoPacket->flags = (info.pictureType == NV_ENC_PIC_TYPE_I || info.pictureType == NV_ENC_PIC_TYPE_IDR) ? AV_PKT_FLAG_KEY : 0;

    //write to container
    writePacket(videoPacket, info.frameIdx, info.frameIdx, terminate);
}

void CudaFFmpegWriter::write() {
    try {
        nvenc.encodeFrame(nvPackets);
        for (NvPacket& nvpkt : nvPackets) {
            writePacketToFile(nvpkt, false);
        }

    } catch (const AVException& e) {
        errorLogger.logError("error writing: " + std::string(e.what()));
    }
}


//flush encoder buffer
bool CudaFFmpegWriter::terminate(bool init) {
    if (init) {
        //notify encoder to flush
        nvenc.endEncode();

    } else {
        //get remaining frames one by one
        writePacketToFile(nvenc.getBufferedFrame(), true);
    }
    return nvenc.hasBufferedFrame();
}


CudaFFmpegWriter::~CudaFFmpegWriter() {
    nvenc.destroyEncoder();
}


//so GUID can be used as key in a map
bool CudaFFmpegWriter::FunctorLess::operator () (const GUID& g1, const GUID& g2) const {
    return std::tie(g1.Data1, g1.Data2, g1.Data3, *g1.Data4) < std::tie(g2.Data1, g2.Data2, g2.Data3, *g2.Data4);
}
