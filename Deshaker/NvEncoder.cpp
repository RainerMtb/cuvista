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

#include "Util.hpp"
#include "DeviceInfo.hpp"

#if defined(BUILD_CUDA) && BUILD_CUDA == 0
#else

#include "NvEncoder.hpp"

static std::strong_ordering compareGuid(const GUID& g1, const GUID& g2) {
	auto t1 = std::tie(g1.Data1, g1.Data2, g1.Data3, g1.Data4[0], g1.Data4[1], g1.Data4[2], g1.Data4[3]);
	auto t2 = std::tie(g2.Data1, g2.Data2, g2.Data3, g2.Data4[0], g2.Data4[1], g2.Data4[2], g2.Data4[3]);
	return t1 <=> t2;
}

//on windows we already have an == operator
#ifndef _WIN64
bool operator == (const GUID& g1, const GUID& g2) {
	return compareGuid(g1, g2) == std::strong_ordering::equal;
}
#endif

bool operator < (const GUID& g1, const GUID& g2) {
	return compareGuid(g1, g2) == std::strong_ordering::less;
}


static void handleResult(bool isError, std::string&& msg) {
	if (isError) 
		throw AVException(msg);
}


static void handleResult(NVENCSTATUS status, std::string&& msg) {
	if (status != NV_ENC_SUCCESS) 
		throw AVException("encoder error " + std::to_string(status) + ": " + msg);
}


static void handleResult(CUresult result, std::string&& msg) {
	if (result != CUDA_SUCCESS) {
		const char* custr;
		cuGetErrorString(result, &custr);
		throw AVException("cuda error: " + std::string(custr));
	}
}


void NvEncoder::probeEncoding(uint32_t* nvencVersionApi, uint32_t* nvencVersionDriver) {
	//check supported encoder version on this system
	*nvencVersionApi = NVENCAPI_MAJOR_VERSION * 1000 + NVENCAPI_MINOR_VERSION * 10; //api version for the libraries
	uint32_t versionDriver = 0;
	handleResult(NvEncodeAPIGetMaxSupportedVersion(&versionDriver), "cannot get max supported version"); //max version supported by driver
	*nvencVersionDriver = versionDriver / 16 * 1000 + versionDriver % 16 * 10;
}


void NvEncoder::init() {
	//create instance
	handleResult(NvEncodeAPICreateInstance(&encFuncList), "cannot create api instance");
	handleResult(encFuncList.nvEncOpenEncodeSession == NULL, "cannot create api instance");

	//create context per device
	CUdevice dev;
	handleResult(cuCtxGetCurrent(&mCuContext), "cannot get device context");
	if (mCuContext == NULL) {
		handleResult(cuDeviceGet(&dev, mCudaIndex), "cannot get device");
		CUctxCreateParams params = {};
		handleResult(cuCtxCreate_v4(&mCuContext, &params, 0, dev), "cannot create device context"); //changing api for cuda 13.0
	}

	//open session
	NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
	encodeSessionExParams.device = mCuContext;
	encodeSessionExParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
	encodeSessionExParams.apiVersion = NVENCAPI_VERSION;
	handleResult(encFuncList.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &mEncoder), "cannot open encoder session");
}


void NvEncoder::probeSupportedCodecs(DeviceInfoCuda& deviceInfoCuda) {
	init();

	//check available guid
	uint32_t guidCount;
	handleResult(encFuncList.nvEncGetEncodeGUIDCount(mEncoder, &guidCount), "cannot get guid count");

	uint32_t guidSupportCount;
	std::vector<GUID> guids(guidCount);
	handleResult(encFuncList.nvEncGetEncodeGUIDs(mEncoder, guids.data(), guidCount, &guidSupportCount), "cannot get guids");

	//order best codec first
	if (std::find(guids.cbegin(), guids.cend(), NV_ENC_CODEC_AV1_GUID) != guids.cend())
		deviceInfoCuda.encodingOptions.emplace_back(EncodingDevice::NVENC, Codec::AV1);
	if (std::find(guids.cbegin(), guids.cend(), NV_ENC_CODEC_HEVC_GUID) != guids.cend())
		deviceInfoCuda.encodingOptions.emplace_back(EncodingDevice::NVENC, Codec::H265);
	if (std::find(guids.cbegin(), guids.cend(), NV_ENC_CODEC_H264_GUID) != guids.cend())
		deviceInfoCuda.encodingOptions.emplace_back(EncodingDevice::NVENC, Codec::H264);

	encFuncList.nvEncDestroyEncoder(mEncoder);
}


void NvEncoder::createEncoder(int w, int h, int fpsNum, int fpsDen, uint32_t gopLen, std::optional<uint8_t> crf, GUID guid) {
	this->h = h;
	this->w = w;
	init();

	//check support for given guid
	uint32_t guidCount;
	handleResult(encFuncList.nvEncGetEncodeGUIDCount(mEncoder, &guidCount), "cannot get guid count");

	uint32_t guidSupportCount;
	std::vector<GUID> guids(guidCount);
	handleResult(encFuncList.nvEncGetEncodeGUIDs(mEncoder, guids.data(), guidCount, &guidSupportCount), "cannot get guids");
	handleResult(std::find(guids.cbegin(), guids.cend(), guid) == guids.cend(), "guid is not supported");

	//check if input format is supported
	uint32_t fmtCount;
	handleResult(encFuncList.nvEncGetInputFormatCount(mEncoder, guid, &fmtCount), "cannot get format count");

	uint32_t fmtSupportedCount;
	std::vector<NV_ENC_BUFFER_FORMAT> fmts(fmtCount);
	handleResult(encFuncList.nvEncGetInputFormats(mEncoder, guid, fmts.data(), fmtCount, &fmtSupportedCount), "cannot get formats");
	handleResult(std::find(fmts.cbegin(), fmts.cend(), mBufferFormat) == fmts.end(), "input format not supported");

	NV_ENC_CAPS_PARAM capsParam = {};
	capsParam.capsToQuery = NV_ENC_CAPS_SUPPORT_LOOKAHEAD;
	int hasEnableLookahead;
	handleResult(encFuncList.nvEncGetEncodeCaps(mEncoder, guid, &capsParam, &hasEnableLookahead), "cannot query lookahead");
	capsParam.capsToQuery = NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE;
	int hasBframesRefMode;
	handleResult(encFuncList.nvEncGetEncodeCaps(mEncoder, guid, &capsParam, &hasBframesRefMode), "cannot query ref mode");

	//create encoder preset
	NV_ENC_TUNING_INFO tuning = NV_ENC_TUNING_INFO_HIGH_QUALITY;
	GUID presetGuid = NV_ENC_PRESET_P5_GUID;
	NV_ENC_PRESET_CONFIG presetConfig = { NV_ENC_PRESET_CONFIG_VER, 0, { NV_ENC_CONFIG_VER } };
	encFuncList.nvEncGetEncodePresetConfigEx(mEncoder, guid, presetGuid, tuning, &presetConfig);

	NV_ENC_CONFIG encoderConfig = presetConfig.presetCfg;
	encoderConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
	encoderConfig.gopLength = gopLen;
	if (crf) encoderConfig.rcParams.targetQuality = *crf;
	encoderConfig.frameIntervalP = 1; //picture pattern
	encoderConfig.rcParams.enableLookahead = hasEnableLookahead;
	encoderConfig.rcParams.lookaheadDepth = 8;

	////encodeCodecConfig is a union
	NV_ENC_BFRAME_REF_MODE refmode = hasBframesRefMode ? NV_ENC_BFRAME_REF_MODE_MIDDLE : NV_ENC_BFRAME_REF_MODE_DISABLED;
	if (guid == NV_ENC_CODEC_HEVC_GUID) {
		encoderConfig.encodeCodecConfig.hevcConfig.chromaFormatIDC = 1; //for yuv444 formats = 3
		encoderConfig.encodeCodecConfig.hevcConfig.useBFramesAsRef = refmode;

	} else if (guid == NV_ENC_CODEC_H264_GUID) {
		encoderConfig.encodeCodecConfig.h264Config.chromaFormatIDC = 1;
		encoderConfig.encodeCodecConfig.h264Config.useBFramesAsRef = refmode;
	}

	//set init parameters structure
	NV_ENC_INITIALIZE_PARAMS initParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	initParams.encodeConfig = &encoderConfig;
	initParams.encodeGUID = guid;
	initParams.presetGUID = presetGuid;
	initParams.encodeWidth = w;
	initParams.encodeHeight = h;
	initParams.frameRateNum = fpsNum;
	initParams.frameRateDen = fpsDen;
	initParams.enablePTD = 1; //send input in display order
	initParams.tuningInfo = tuning;

	//initialize encoder
	handleResult(encFuncList.nvEncInitializeEncoder(mEncoder, &initParams), "cannot initialize encoder");

	//get sequence parameters needed for extradata field in ffmpeg stream
	NV_ENC_SEQUENCE_PARAM_PAYLOAD seq = { NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER };
	mExtradata.assign(NV_MAX_SEQ_HDR_LEN, 0);
	seq.spsppsBuffer = mExtradata.data();
	seq.inBufferSize = NV_MAX_SEQ_HDR_LEN;
	seq.outSPSPPSPayloadSize = &mExtradataSize;
	handleResult(encFuncList.nvEncGetSequenceParams(mEncoder, &seq), "cannot get sequence parameters");

	//size of buffer vector
	const int32_t extraDelay = 4; //taken from samples
	mEncoderBufferSize = encoderConfig.frameIntervalP + encoderConfig.rcParams.lookaheadDepth + extraDelay;
	mOutputDelay = mEncoderBufferSize - 1;

	size_t pitch;
	size_t h_image = h * 3 / 2; //for yuv444 formats h * 3
	for (size_t i = 0; i < mEncoderBufferSize; i++) {
		//output buffers
		NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBuffer = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
		handleResult(encFuncList.nvEncCreateBitstreamBuffer(mEncoder, &createBitstreamBuffer), "cannot create bitstream buffer");
		bitstreamOutputBuffer.push_back(createBitstreamBuffer.bitstreamBuffer);

		CUdeviceptr pDeviceFrame;
		handleResult(cuMemAllocPitch_v2(&pDeviceFrame, &pitch, w, h_image, 16), "error allocating input buffers");
		inputFrames.push_back(pDeviceFrame);

		NV_ENC_REGISTER_RESOURCE registerResource = { NV_ENC_REGISTER_RESOURCE_VER };
		registerResource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
		registerResource.resourceToRegister = (void*) pDeviceFrame;
		registerResource.width = w;
		registerResource.height = h;
		registerResource.pitch = (int) pitch;
		registerResource.bufferFormat = mBufferFormat;
		registerResource.bufferUsage = NV_ENC_INPUT_IMAGE;
		handleResult(encFuncList.nvEncRegisterResource(mEncoder, &registerResource), "cannot register resource");
		NV_ENC_REGISTERED_PTR registeredPtr = registerResource.registeredResource;
		registeredResources.push_back(registeredPtr);
	}
	cudaPitch = (int) pitch;
}

CUdeviceptr NvEncoder::getNextInputFramePtr() {
	int32_t i = mFrameToSend % mEncoderBufferSize;
	return inputFrames[i];
}


NvPacket NvEncoder::getEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR>& outputBuffer) {
	int32_t fridx = mFrameGot % mEncoderBufferSize;

	NvPacket pkg;
	pkg.bitstreamData.outputBitstream = outputBuffer[fridx];
	pkg.bitstreamData.doNotWait = false;
	handleResult(encFuncList.nvEncLockBitstream(mEncoder, &pkg.bitstreamData), "cannot lock bitstream");

	uint8_t* pData = (uint8_t*) pkg.bitstreamData.bitstreamBufferPtr;
	std::vector<uint8_t>& data = pkg.packet;
	data.insert(data.end(), pData, pData + pkg.bitstreamData.bitstreamSizeInBytes);
	handleResult(encFuncList.nvEncUnlockBitstream(mEncoder, pkg.bitstreamData.outputBitstream), "cannot unlock bitstream");

	return pkg;
}


void NvEncoder::getEncodedPackets(std::vector<NV_ENC_OUTPUT_PTR>& outputBuffer, std::list<NvPacket>& nvPackets, bool delay) {
	nvPackets.clear();
	int32_t frameEnd = mFrameToSend;
	if (delay) frameEnd = mFrameToSend - mOutputDelay;

	for (; mFrameGot < frameEnd; mFrameGot++) {
		nvPackets.push_back(getEncodedPacket(outputBuffer));
	}
}


void NvEncoder::encodeFrame(std::list<NvPacket>& nvPackets) {
	int32_t i = mFrameToSend % mEncoderBufferSize;
	cuCtxPushCurrent(mCuContext);

	//util::ConsoleTimer ct("encode");
	NV_ENC_MAP_INPUT_RESOURCE mapInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };
	mapInputResource.registeredResource = registeredResources[i];
	handleResult(encFuncList.nvEncMapInputResource(mEncoder, &mapInputResource), "cannot map input resource");

	NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
	picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
	picParams.inputBuffer = mapInputResource.mappedResource;
	picParams.bufferFmt = mBufferFormat;
	picParams.inputWidth = w;
	picParams.inputHeight = h;
	picParams.outputBitstream = bitstreamOutputBuffer[i];
	picParams.completionEvent = nullptr; //do not use events, do it synchronous

	NVENCSTATUS stat;
	stat = encFuncList.nvEncEncodePicture(mEncoder, &picParams);
	if (stat == NV_ENC_ERR_NEED_MORE_INPUT) {
		mFrameToSend++;

	} else if (stat == NV_ENC_SUCCESS) {
		mFrameToSend++;
		getEncodedPackets(bitstreamOutputBuffer, nvPackets, true);

	} else {
		throw AVException("cannot encode frame, status=" + std::to_string(stat));
	}
	cuCtxPopCurrent(nullptr);
}


void NvEncoder::endEncode() {
	NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
	picParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
	picParams.completionEvent = nullptr;
	handleResult(encFuncList.nvEncEncodePicture(mEncoder, &picParams), "cannot encode picture");
}


bool NvEncoder::hasBufferedFrame() {
	return mFrameGot < mFrameToSend;
}


NvPacket NvEncoder::getBufferedFrame() {
	NvPacket out = getEncodedPacket(bitstreamOutputBuffer);
	mFrameGot++;
	return out;
}


void NvEncoder::destroyEncoder() {
	cuCtxPushCurrent(mCuContext);

	//clear input buffers
	for (size_t i = 0; i < registeredResources.size(); i++) {
		if (registeredResources[i]) {
			encFuncList.nvEncUnregisterResource(mEncoder, registeredResources[i]);
		}
	}
	registeredResources.clear();

	for (size_t i = 0; i < inputFrames.size(); i++) {
		if (inputFrames[i]) {
			cuMemFree_v2(inputFrames[i]);
		}
	}
	inputFrames.clear();

	//clear output buffers
	for (size_t i = 0; i < bitstreamOutputBuffer.size(); i++) {
		if (bitstreamOutputBuffer[i]) {
			encFuncList.nvEncDestroyBitstreamBuffer(mEncoder, bitstreamOutputBuffer[i]);
		}
	}
	bitstreamOutputBuffer.clear();

	//destroy encoder
	if (mEncoder != nullptr) {
		encFuncList.nvEncDestroyEncoder(mEncoder);
	}

	cuCtxPopCurrent(nullptr);
	//do not destroy while cuda is still executing, will result in errors in destructor of CudaExecutor
	//cuCtxDestroy_v2(cuctx);
}

#endif