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

struct NvPacket;
class NvEncoder;

class CudaFFmpegWriter : public FFmpegFormatWriter {

private:
	std::shared_ptr<NvEncoder> nvenc = nullptr;
	std::unique_ptr<std::list<NvPacket>> nvPackets; //encoded packets

	void writePacketToFile(const NvPacket& nvpkt, bool terminate);
	void writePacketsToFile(std::list<NvPacket> nvpkts, bool terminate);
	void encodePackets();
	ImageNV12 outputNV12;

public:
	CudaFFmpegWriter(MainData& data, MovieReader& reader);
	~CudaFFmpegWriter() override;

	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
	bool flush() override;
};
