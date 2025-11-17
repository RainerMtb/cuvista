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

#include <vector>
#include <list>
#include <map>
#include <optional>
#include <cuda.h>

#include "AVException.hpp"
#include "nvEncodeAPI.h"

#undef min
#undef max

bool operator < (const GUID& g1, const GUID& g2);
bool operator == (const GUID& g1, const GUID& g2);

class DeviceInfoCuda;

struct NvPacket {
	std::vector<uint8_t> packet;
	NV_ENC_LOCK_BITSTREAM bitstreamData = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
};

class NvEncoder {

private:
	int mCudaIndex = -1;
	CUcontext mCuContext = nullptr;
	void* mEncoder = nullptr;
	int32_t mEncoderBufferSize = 0;
	int32_t mOutputDelay = 0;
	int32_t mFrameToSend = 0;
	int32_t mFrameGot = 0;
	int h = 0;
	int w = 0;

	NV_ENCODE_API_FUNCTION_LIST encFuncList = { NV_ENCODE_API_FUNCTION_LIST_VER };
	std::vector<CUdeviceptr> inputFrames;
	std::vector<NV_ENC_REGISTERED_PTR> registeredResources;
	std::vector<NV_ENC_OUTPUT_PTR> bitstreamOutputBuffer;

	void getEncodedPackets(std::vector<NV_ENC_OUTPUT_PTR>& outputBuffer, std::list<NvPacket>& nvPackets, bool outputDelay);
	NvPacket getEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR>& outputBuffer);

public:
	int cudaPitch = 0;
	std::vector<uint8_t> mExtradata;
	uint32_t mExtradataSize = 0;

	NvEncoder(int cudaIndex) :
		mCudaIndex { cudaIndex } {}

	void init();
	void probeEncoding(uint32_t* nvencVersionApi, uint32_t* nvencVersionDriver);
	void probeSupportedCodecs(DeviceInfoCuda& deviceInfoCuda);
	void createEncoder(int w, int h, int fpsNum, int fpsDen, uint32_t gopLen, uint8_t crf, GUID guid);
	void destroyEncoder();

	CUdeviceptr getNextInputFramePtr();
	void encodeFrame(std::list<NvPacket>& nvPackets);
	void endEncode();
	bool hasBufferedFrame();
	NvPacket getBufferedFrame();
};
