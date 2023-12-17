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
	CudaFFmpegWriter(MainData& data, MovieReader& reader) :
		FFmpegFormatWriter(data, reader), 
		nvenc { NvEncoder(data.w, data.h) } 
	{}

	virtual ~CudaFFmpegWriter() override;

	void open(EncodingOption videoCodec) override;
	OutputContext getOutputContext() override;
	void write() override;
	std::future<void> writeAsync() override;
	bool startFlushing() override;
	bool flush() override;
};