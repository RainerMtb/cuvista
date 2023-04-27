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

#include "AVException.hpp"
#include "nvEncodeAPI.h"
#include <cuda.h>

#include <vector>
#include <list>

struct NvPacket {
	std::vector<uint8_t> packet;
	NV_ENC_LOCK_BITSTREAM bitstreamData = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
};

class NvEncoder {

private:
	NV_ENC_BUFFER_FORMAT bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;

	int h, w;
	void* encoder = nullptr;
	int32_t encBufferSize = 0;
	int32_t outputDelay = 0;
	CUcontext cuctx = nullptr;
	int32_t frameToSend = 0;
	int32_t frameGot = 0;

	NV_ENCODE_API_FUNCTION_LIST encFuncList {};
	std::vector<NV_ENC_INPUT_PTR> mappedInputBuffers;
	std::vector<NV_ENC_OUTPUT_PTR> bitstreamOutputBuffer;
	std::vector<NV_ENC_REGISTERED_PTR> registeredResources;
	std::vector<CUdeviceptr> inputFrames;

	void getEncodedPackets(std::vector<NV_ENC_OUTPUT_PTR>& outputBuffer, std::list<NvPacket>& nvPackets, bool outputDelay);
	NvPacket getEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR>& outputBuffer);

public:
	size_t pitch = 0;

	NvEncoder(int w, int h) : h { h }, w { w } {}

	void createEncoder(int fpsNum, int fpsDen, uint32_t gopLen, uint8_t crf, GUID guid, int deviceNum);
	void createEncoder(int fpsNum, int fpsDen, uint32_t gopLen, uint8_t crf, GUID guid, CUcontext cuctx);
	void destroyEncoder();

	CUdeviceptr getNextInputFramePtr();
	void encodeFrame(std::list<NvPacket>& nvPackets);
	void endEncode();
	bool hasBufferedFrame();
	NvPacket getBufferedFrame();
};