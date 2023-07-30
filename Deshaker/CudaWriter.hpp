#pragma once

#include "MovieWriter.hpp"
#include "NvEncoder.hpp"

//-----------------------------------------------------------------------------------
class CudaFFmpegWriter : public FFmpegFormatWriter {

private:
	class FunctorLess {
	public:
		bool operator () (const GUID& g1, const GUID& g2) const;
	};
	std::map<GUID, AVCodecID, FunctorLess> guidToCodecMap = {
		{ NV_ENC_CODEC_H264_GUID, AV_CODEC_ID_H264 },
		{ NV_ENC_CODEC_HEVC_GUID, AV_CODEC_ID_HEVC },
	};

	NvEncoder nvenc;
	std::list<NvPacket> nvPackets; //encoded packets

	void writePacketToFile(const NvPacket& nvpkt, bool terminate);
	void encodePackets();

public:
	CudaFFmpegWriter(MainData& data) : FFmpegFormatWriter(data), nvenc { NvEncoder(data.w, data.h) } {}
	virtual ~CudaFFmpegWriter() override;

	void open(OutputCodec videoCodec) override;
	OutputContext getOutputData() override;
	void write() override;
	std::future<void> writeAsync() override;
	bool terminate(bool init) override;
};