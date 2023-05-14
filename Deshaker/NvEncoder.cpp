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

#include "NvEncoder.hpp"

void handleResult(bool isError, std::string&& msg) {
	if (isError) throw AVException(msg);
}


void handleResult(NVENCSTATUS status, std::string&& msg) {
	if (status != NV_ENC_SUCCESS) throw AVException("encoder error " + std::to_string(status) + ": " + msg);
}


void handleResult(CUresult result, std::string&& msg) {
	if (result != CUDA_SUCCESS) {
		const char* custr;
		cuGetErrorString(result, &custr);
		throw AVException("cuda error: " + std::string(custr));
	}
}


void NvEncoder::probeEncoding(CudaInfo& cudaInfo) {
	//check supported encoder version on this system
	cudaInfo.nvencVersionApi = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION; //api version for the libraries
	handleResult(NvEncodeAPIGetMaxSupportedVersion(&cudaInfo.nvencVersionDriver), "cannot get max supported version"); //max version supported by driver
}


void NvEncoder::probeSupportedCodecs(CudaInfo& cudaInfo) {
	//create instance
	NV_ENCODE_API_FUNCTION_LIST encFuncList = { NV_ENCODE_API_FUNCTION_LIST_VER };
	handleResult(NvEncodeAPICreateInstance(&encFuncList), "cannot create api instance");
	handleResult(encFuncList.nvEncOpenEncodeSession == NULL, "error opening encode session");

	//check supported codecs for all devices
	cudaInfo.supportedCodecs.clear();
	for (int i = 0; i < cudaInfo.cudaProps.size(); i++) {
		void* encoder = nullptr;

		//create context per device
		CUcontext cuctx;
		handleResult(cuCtxCreate_v2(&cuctx, 0, i), "cannot create device context");

		//open session
		NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
		encodeSessionExParams.device = cuctx;
		encodeSessionExParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
		encodeSessionExParams.apiVersion = NVENCAPI_VERSION;
		handleResult(encFuncList.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &encoder), "cannot open encoder session");

		//check available guid
		uint32_t guidCount;
		handleResult(encFuncList.nvEncGetEncodeGUIDCount(encoder, &guidCount), "cannot get guid count");

		uint32_t guidSupportCount;
		std::vector<GUID> guids(guidCount);
		handleResult(encFuncList.nvEncGetEncodeGUIDs(encoder, guids.data(), guidCount, &guidSupportCount), "cannot get guids");

		//order by automatic selection
		std::vector<OutputCodec> codecs;
		if (std::find(guids.cbegin(), guids.cend(), NV_ENC_CODEC_HEVC_GUID) != guids.cend()) codecs.push_back(OutputCodec::H265);
		if (std::find(guids.cbegin(), guids.cend(), NV_ENC_CODEC_H264_GUID) != guids.cend()) codecs.push_back(OutputCodec::H264);
		cudaInfo.supportedCodecs.push_back(codecs);

		handleResult(cuCtxDestroy_v2(cuctx), "cannot destroy context");
	}
}


void NvEncoder::createEncoder(int fpsNum, int fpsDen, uint32_t gopLen, uint8_t crf, GUID guid, int deviceNum) {
	CUcontext cuctx;
	handleResult(cuCtxGetCurrent(&cuctx), "cannot get device context");
	if (cuctx == NULL) 
		handleResult(cuCtxCreate_v2(&cuctx, 0, deviceNum), "cannot create device context");
	createEncoder(fpsNum, fpsDen, gopLen, crf, guid, cuctx);
}


void NvEncoder::createEncoder(int fpsNum, int fpsDen, uint32_t gopLen, uint8_t crf, GUID guid, CUcontext cuctx) {
	this->cuctx = cuctx;
	handleResult(NvEncodeAPICreateInstance(&encFuncList), "cannot create api instance");
	handleResult(encFuncList.nvEncOpenEncodeSession == NULL, "error opening encode session");

	//open session
	NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
	encodeSessionExParams.device = cuctx;
	encodeSessionExParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
	encodeSessionExParams.apiVersion = NVENCAPI_VERSION;
	handleResult(encFuncList.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &encoder), "cannot open encoder session");

	//check support for given guid
	uint32_t guidCount;
	handleResult(encFuncList.nvEncGetEncodeGUIDCount(encoder, &guidCount), "cannot get guid count");

	uint32_t guidSupportCount;
	std::vector<GUID> guids(guidCount);
	handleResult(encFuncList.nvEncGetEncodeGUIDs(encoder, guids.data(), guidCount, &guidSupportCount), "cannot get guids");
	handleResult(std::find(guids.cbegin(), guids.cend(), guid) == guids.cend(), "guid is not supported");

	//check if input format is supported
	uint32_t fmtCount;
	handleResult(encFuncList.nvEncGetInputFormatCount(encoder, guid, &fmtCount), "cannot get format count");

	uint32_t fmtSupportedCount;
	std::vector<NV_ENC_BUFFER_FORMAT> fmts(fmtCount);
	handleResult(encFuncList.nvEncGetInputFormats(encoder, guid, fmts.data(), fmtCount, &fmtSupportedCount), "cannot get formats");
	handleResult(std::find(fmts.cbegin(), fmts.cend(), mBufferFormat) == fmts.end(), "input format not supported");

	//set init parameters structure
	NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
	NV_ENC_INITIALIZE_PARAMS initParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	initParams.encodeConfig = &encodeConfig;
	initParams.encodeGUID = guid;
	initParams.presetGUID = NV_ENC_PRESET_P3_GUID;
	initParams.encodeWidth = w;
	initParams.encodeHeight = h;
	initParams.darWidth = w;
	initParams.darHeight = h;
	initParams.maxEncodeWidth = w;
	initParams.maxEncodeHeight = h;
	initParams.frameRateNum = fpsNum;
	initParams.frameRateDen = fpsDen;
	initParams.enablePTD = 1; //send input in display order
	initParams.enableMEOnlyMode = false;
	initParams.enableOutputInVidmem = false;
	initParams.enableEncodeAsync = false;
	initParams.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
	
	NV_ENC_PRESET_CONFIG presetConfig = { NV_ENC_PRESET_CONFIG_VER, { NV_ENC_CONFIG_VER } };
	encFuncList.nvEncGetEncodePresetConfigEx(encoder, initParams.encodeGUID, initParams.presetGUID, initParams.tuningInfo, &presetConfig);
	memcpy(&encodeConfig, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));
	encodeConfig.profileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
	encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
	encodeConfig.gopLength = gopLen;
	encodeConfig.rcParams.targetQuality = crf;
	encodeConfig.frameIntervalP = 1; //picture pattern
	encodeConfig.rcParams.targetQuality = crf;
	encodeConfig.rcParams.enableLookahead = 1;
	encodeConfig.rcParams.lookaheadDepth = 8;

	//encodeCodecConfig is a union
	if (guid == NV_ENC_CODEC_HEVC_GUID) {
		encodeConfig.encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 = 0; //set to 0 for 8bit, set to 2 for 10bit input data
		encodeConfig.encodeCodecConfig.hevcConfig.chromaFormatIDC = 1; //for yuv444 formats = 3

	} else if (guid == NV_ENC_CODEC_H264_GUID) {
		encodeConfig.encodeCodecConfig.h264Config.chromaFormatIDC = 1;
	}

	//initialize encoder
	initParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
	handleResult(encFuncList.nvEncInitializeEncoder(encoder, &initParams), "cannot initialize encoder");

	//size of buffer vector
	const int32_t extraDelay = 4; //taken from samples
	encBufferSize = encodeConfig.frameIntervalP + encodeConfig.rcParams.lookaheadDepth + extraDelay;
	outputDelay = encBufferSize - 1;
	mappedInputBuffers.resize(encBufferSize, nullptr);

	for (size_t i = 0; i < encBufferSize; i++) {
		NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBuffer = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
		handleResult(encFuncList.nvEncCreateBitstreamBuffer(encoder, &createBitstreamBuffer), "cannot create bitstream buffer");
		bitstreamOutputBuffer.push_back(createBitstreamBuffer.bitstreamBuffer);
	}

	//allocate input buffers on device through driver api
	int h_image = h * 3 / 2; //for yuv444 formats h * 3
	for (int i = 0; i < encBufferSize; i++) {
		CUdeviceptr pDeviceFrame;
		handleResult(cuMemAllocPitch_v2(&pDeviceFrame, &pitch, w, h_image, 16), "error allocating input buffers");
		inputFrames.push_back(pDeviceFrame);
	}

	//register input resources
	for (size_t i = 0; i < inputFrames.size(); i++) {
		NV_ENC_REGISTER_RESOURCE registerResource = { NV_ENC_REGISTER_RESOURCE_VER };
		registerResource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
		registerResource.resourceToRegister = (void*) inputFrames[i];
		registerResource.width = w;
		registerResource.height = h;
		registerResource.pitch = (int) pitch;
		registerResource.bufferFormat = mBufferFormat;
		registerResource.bufferUsage = NV_ENC_INPUT_IMAGE;
		handleResult(encFuncList.nvEncRegisterResource(encoder, &registerResource), "cannot register resource");
		NV_ENC_REGISTERED_PTR registeredPtr = registerResource.registeredResource;
		registeredResources.push_back(registeredPtr);
	}
}

CUdeviceptr NvEncoder::getNextInputFramePtr() {
	int32_t i = frameToSend % encBufferSize;
	return inputFrames[i];
}


NvPacket NvEncoder::getEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR>& outputBuffer) {
	int32_t fridx = frameGot % encBufferSize;

	NvPacket pkg;
	pkg.bitstreamData.outputBitstream = outputBuffer[fridx];
	pkg.bitstreamData.doNotWait = false;
	handleResult(encFuncList.nvEncLockBitstream(encoder, &pkg.bitstreamData), "cannot lock bitstream");

	uint8_t* pData = (uint8_t*) pkg.bitstreamData.bitstreamBufferPtr;
	std::vector<uint8_t>& data = pkg.packet;
	data.insert(data.end(), pData, pData + pkg.bitstreamData.bitstreamSizeInBytes);
	handleResult(encFuncList.nvEncUnlockBitstream(encoder, pkg.bitstreamData.outputBitstream), "cannot unlock bitstream");

	if (mappedInputBuffers[fridx]) {
		handleResult(encFuncList.nvEncUnmapInputResource(encoder, mappedInputBuffers[fridx]), "cannot unmap resource");
		mappedInputBuffers[fridx] = nullptr;
	}
	return pkg;
}


void NvEncoder::getEncodedPackets(std::vector<NV_ENC_OUTPUT_PTR>& outputBuffer, std::list<NvPacket>& nvPackets, bool delay) {
	nvPackets.clear();
	int32_t frameEnd = frameToSend;
	if (delay) frameEnd = frameToSend - outputDelay;

	for (; frameGot < frameEnd; frameGot++) {
		nvPackets.push_back(getEncodedPacket(outputBuffer));
	}
}


void NvEncoder::encodeFrame(std::list<NvPacket>& nvPackets) {
	int32_t fridx = frameToSend % encBufferSize;
	cuCtxPushCurrent_v2(cuctx);

	NV_ENC_MAP_INPUT_RESOURCE mapInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };
	mapInputResource.registeredResource = registeredResources[fridx];
	handleResult(encFuncList.nvEncMapInputResource(encoder, &mapInputResource), "cannot map input resource");
	mappedInputBuffers[fridx] = mapInputResource.mappedResource;

	NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
	picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
	picParams.inputBuffer = mappedInputBuffers[fridx];
	picParams.bufferFmt = mBufferFormat;
	picParams.inputWidth = w;
	picParams.inputHeight = h;
	picParams.outputBitstream = bitstreamOutputBuffer[fridx];
	picParams.completionEvent = nullptr; //do not use events, do it synchronous

	NVENCSTATUS nvStatus = encFuncList.nvEncEncodePicture(encoder, &picParams);
	if (nvStatus == NV_ENC_ERR_NEED_MORE_INPUT) {
		frameToSend++;

	} else if (nvStatus == NV_ENC_SUCCESS) {
		frameToSend++;
		getEncodedPackets(bitstreamOutputBuffer, nvPackets, true);

	} else {
		throw AVException("cannot encode frame, status=" + std::to_string(nvStatus));
	}
	cuCtxPopCurrent_v2(nullptr);
}


void NvEncoder::endEncode() {
	NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
	picParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
	picParams.completionEvent = nullptr;
	handleResult(encFuncList.nvEncEncodePicture(encoder, &picParams), "cannot encode picture");
}


bool NvEncoder::hasBufferedFrame() {
	return frameGot < frameToSend;
}


NvPacket NvEncoder::getBufferedFrame() {
	NvPacket out = getEncodedPacket(bitstreamOutputBuffer);
	frameGot++;
	return out;
}


void NvEncoder::destroyEncoder() {
	cuCtxPushCurrent_v2(cuctx);

	for (size_t i = 0; i < mappedInputBuffers.size(); i++) {
		if (mappedInputBuffers[i]) {
			encFuncList.nvEncUnmapInputResource(encoder, mappedInputBuffers[i]);
		}
	}
	mappedInputBuffers.clear();

	for (size_t i = 0; i < registeredResources.size(); i++) {
		if (registeredResources[i]) {
			encFuncList.nvEncUnregisterResource(encoder, registeredResources[i]);
		}
	}
	registeredResources.clear();

	for (size_t i = 0; i < inputFrames.size(); i++) {
		if (inputFrames[i]) cuMemFree_v2(inputFrames[i]);
	}
	inputFrames.clear();

	for (size_t i = 0; i < bitstreamOutputBuffer.size(); i++) {
		if (bitstreamOutputBuffer[i]) {
			encFuncList.nvEncDestroyBitstreamBuffer(encoder, bitstreamOutputBuffer[i]);
		}
	}
	bitstreamOutputBuffer.clear();

	if (encoder != nullptr) {
		encFuncList.nvEncDestroyEncoder(encoder);
	}
}