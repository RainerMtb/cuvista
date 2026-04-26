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

#if defined(BUILD_CUDA) && BUILD_CUDA == 0 //cuda disabled

#else //cuda enabled

#include "NvEncoder.hpp"

NvEncoder::NvEncoder(int cudaIndex) :
	mCudaIndex { cudaIndex }
{
	cuInit(0);
}

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
	std::string strStatus;
	switch (status) {
		case NV_ENC_SUCCESS: strStatus = "NV_ENC_SUCCESS"; break;
		case NV_ENC_ERR_NO_ENCODE_DEVICE: strStatus = "NV_ENC_ERR_NO_ENCODE_DEVICE"; break;
		case NV_ENC_ERR_UNSUPPORTED_DEVICE: strStatus = "NV_ENC_ERR_UNSUPPORTED_DEVICE"; break;
		case NV_ENC_ERR_INVALID_ENCODERDEVICE: strStatus = "NV_ENC_ERR_INVALID_ENCODERDEVICE"; break;
		case NV_ENC_ERR_INVALID_DEVICE: strStatus = "NV_ENC_ERR_INVALID_DEVICE"; break;
		case NV_ENC_ERR_DEVICE_NOT_EXIST: strStatus = "NV_ENC_ERR_DEVICE_NOT_EXIST"; break;
		case NV_ENC_ERR_INVALID_PTR: strStatus = "NV_ENC_ERR_INVALID_PTR"; break;
		case NV_ENC_ERR_INVALID_EVENT: strStatus = "NV_ENC_ERR_INVALID_EVENT"; break;
		case NV_ENC_ERR_INVALID_PARAM: strStatus = "NV_ENC_ERR_INVALID_PARAM"; break;
		case NV_ENC_ERR_INVALID_CALL: strStatus = "NV_ENC_ERR_INVALID_CALL"; break;
		case NV_ENC_ERR_OUT_OF_MEMORY: strStatus = "NV_ENC_ERR_OUT_OF_MEMORY"; break;
		case NV_ENC_ERR_ENCODER_NOT_INITIALIZED: strStatus = "NV_ENC_ERR_ENCODER_NOT_INITIALIZED"; break;
		case NV_ENC_ERR_UNSUPPORTED_PARAM: strStatus = "NV_ENC_ERR_UNSUPPORTED_PARAM"; break;
		case NV_ENC_ERR_LOCK_BUSY: strStatus = "NV_ENC_ERR_LOCK_BUSY"; break;
		case NV_ENC_ERR_NOT_ENOUGH_BUFFER: strStatus = "NV_ENC_ERR_NOT_ENOUGH_BUFFER"; break;
		case NV_ENC_ERR_INVALID_VERSION: strStatus = "NV_ENC_ERR_INVALID_VERSION"; break;
		case NV_ENC_ERR_MAP_FAILED: strStatus = "NV_ENC_ERR_MAP_FAILED"; break;
		case NV_ENC_ERR_NEED_MORE_INPUT: strStatus = "NV_ENC_ERR_NEED_MORE_INPUT"; break;
		case NV_ENC_ERR_ENCODER_BUSY: strStatus = "NV_ENC_ERR_ENCODER_BUSY"; break;
		case NV_ENC_ERR_EVENT_NOT_REGISTERD: strStatus = "NV_ENC_ERR_EVENT_NOT_REGISTERD"; break;
		case NV_ENC_ERR_GENERIC: strStatus = "NV_ENC_ERR_GENERIC"; break;
		case NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY: strStatus = "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY"; break;
		case NV_ENC_ERR_UNIMPLEMENTED: strStatus = "NV_ENC_ERR_UNIMPLEMENTED"; break;
		case NV_ENC_ERR_RESOURCE_REGISTER_FAILED: strStatus = "NV_ENC_ERR_RESOURCE_REGISTER_FAILED"; break;
		case NV_ENC_ERR_RESOURCE_NOT_REGISTERED: strStatus = "NV_ENC_ERR_RESOURCE_NOT_REGISTERED"; break;
		case NV_ENC_ERR_RESOURCE_NOT_MAPPED: strStatus = "NV_ENC_ERR_RESOURCE_NOT_MAPPED"; break;
		case NV_ENC_ERR_NEED_MORE_OUTPUT: strStatus = "NV_ENC_ERR_NEED_MORE_OUTPUT"; break;
		default: strStatus = "unknown nvenc error"; break;
	}
	if (status != NV_ENC_SUCCESS) {
		throw AVException("encoder error, " + strStatus + ", " + msg);
	}
}


static void handleResult(CUresult result, std::string&& msg) {
	if (result != CUDA_SUCCESS) {
		const char* custr;
		cuGetErrorString(result, &custr);
		throw AVException("encoder error, " + std::string(custr) + ", " + msg);
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
	handleResult(cuCtxGetCurrent(&mCuContext), "cannot get device context");
	if (mCuContext == NULL) {
		handleResult(cuDeviceGet(&mDevice, mCudaIndex), "cannot get device");
		handleResult(cuDevicePrimaryCtxRetain(&mCuContext, mDevice), "cannot retain device context");
		//handleResult(cuCtxCreate_v4(&mCuContext, NULL, 0, dev), "cannot create device context"); //changing api for create with cuda 13.0
	}

	//create instance
	handleResult(NvEncodeAPICreateInstance(&encFuncList), "cannot create api instance");
	handleResult(encFuncList.nvEncOpenEncodeSession == NULL, "cannot create api instance");

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

	//order codecs
	if (std::find(guids.cbegin(), guids.cend(), NV_ENC_CODEC_AV1_GUID) != guids.cend())
		deviceInfoCuda.videoEncodingOptions.push_back(OutputOption::NVENC_AV1);
	if (std::find(guids.cbegin(), guids.cend(), NV_ENC_CODEC_HEVC_GUID) != guids.cend())
		deviceInfoCuda.videoEncodingOptions.push_back(OutputOption::NVENC_HEVC);
	if (std::find(guids.cbegin(), guids.cend(), NV_ENC_CODEC_H264_GUID) != guids.cend())
		deviceInfoCuda.videoEncodingOptions.push_back(OutputOption::NVENC_H264);

	encFuncList.nvEncDestroyEncoder(mEncoder);
	mEncoder = nullptr;
}


void NvEncoder::createEncoder(int w, int h, int fpsNum, int fpsDen, uint32_t gopLen, uint8_t crf, GUID guid) {
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
	handleResult(std::find(fmts.cbegin(), fmts.cend(), NV_ENC_BUFFER_FORMAT_NV12) == fmts.end(), "input format not supported");

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
	encoderConfig.frameIntervalP = 1; //picture pattern
	encoderConfig.gopLength = gopLen;
	encoderConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
	encoderConfig.rcParams.targetQuality = crf;
	encoderConfig.rcParams.maxBitRate = 85'000'000; //bits per second
	encoderConfig.rcParams.enableLookahead = hasEnableLookahead;
	encoderConfig.rcParams.lookaheadDepth = 8;

	////encodeCodecConfig is a union
	NV_ENC_BFRAME_REF_MODE refmode = hasBframesRefMode ? NV_ENC_BFRAME_REF_MODE_MIDDLE : NV_ENC_BFRAME_REF_MODE_DISABLED;
	if (guid == NV_ENC_CODEC_HEVC_GUID) {
		encoderConfig.encodeCodecConfig.hevcConfig.chromaFormatIDC = 1; //for yuv444 formats = 3
		encoderConfig.encodeCodecConfig.hevcConfig.useBFramesAsRef = refmode;
		encoderConfig.encodeCodecConfig.hevcConfig.inputBitDepth = NV_ENC_BIT_DEPTH_8;

	} else if (guid == NV_ENC_CODEC_H264_GUID) {
		encoderConfig.encodeCodecConfig.h264Config.chromaFormatIDC = 1;
		encoderConfig.encodeCodecConfig.h264Config.useBFramesAsRef = refmode;
		encoderConfig.encodeCodecConfig.h264Config.inputBitDepth = NV_ENC_BIT_DEPTH_8;

	} else if (guid == NV_ENC_CODEC_AV1_GUID) {
		encoderConfig.encodeCodecConfig.av1Config.chromaFormatIDC = 1;
		encoderConfig.encodeCodecConfig.av1Config.useBFramesAsRef = refmode;
		encoderConfig.encodeCodecConfig.av1Config.inputBitDepth = NV_ENC_BIT_DEPTH_8;
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
	initParams.enablePTD = 1;
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
	int imageRows = h * 3 / 2; //for nv12 format h * 3 / 2, for yuv444 formats h * 3
	
	// ????
	// memory allocation through driver api requires pushing context
	// later writing to that memory through runtime api requires retained context
	// ????
	cuCtxPushCurrent(mCuContext);
	for (size_t i = 0; i < mEncoderBufferSize; i++) {
		//output buffers
		NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBuffer = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
		handleResult(encFuncList.nvEncCreateBitstreamBuffer(mEncoder, &createBitstreamBuffer), "cannot create bitstream buffer");
		bitstreamOutputBuffer.push_back(createBitstreamBuffer.bitstreamBuffer);

		//allocate pitched memory through driver api
		CUdeviceptr pDeviceFrame;
		handleResult(cuMemAllocPitch_v2(&pDeviceFrame, &pitch, w, imageRows, 16), "error allocating input buffers");
		inputFrames.push_back(pDeviceFrame);

		NV_ENC_REGISTER_RESOURCE registerResource = { NV_ENC_REGISTER_RESOURCE_VER };
		registerResource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
		registerResource.resourceToRegister = (void*) pDeviceFrame;
		registerResource.width = w;
		registerResource.height = h;
		registerResource.pitch = pitch;
		registerResource.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
		registerResource.bufferUsage = NV_ENC_INPUT_IMAGE;
		handleResult(encFuncList.nvEncRegisterResource(mEncoder, &registerResource), "cannot register resource");
		NV_ENC_REGISTERED_PTR registeredPtr = registerResource.registeredResource;
		registeredResources.push_back(registeredPtr);
	}
	cuDevicePrimaryCtxRetain(&mCuContext, mDevice);
	mCudaPitch = (int) pitch;
}

std::span<uint8_t> NvEncoder::getExtraData() {
	return { mExtradata.data(), mExtradataSize };
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
	pkg.packet = { pData, pData + pkg.bitstreamData.bitstreamSizeInBytes };
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


void NvEncoder::encodeFrame(std::list<NvPacket>& nvPackets, int64_t frameIndex) {
	int32_t i = mFrameToSend % mEncoderBufferSize;
	cuCtxPushCurrent(mCuContext);

	//util::ConsoleTimer ct("encode");
	NV_ENC_MAP_INPUT_RESOURCE mappedInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };
	mappedInputResource.registeredResource = registeredResources[i];
	handleResult(encFuncList.nvEncMapInputResource(mEncoder, &mappedInputResource), "cannot map input resource");

	NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
	picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
	picParams.inputBuffer = mappedInputResource.mappedResource;
	picParams.bufferFmt = mappedInputResource.mappedBufferFmt;
	picParams.inputWidth = w;
	picParams.inputHeight = h;
	picParams.outputBitstream = bitstreamOutputBuffer[i];
	picParams.completionEvent = nullptr; //do not use events, do it synchronous
	picParams.inputTimeStamp = frameIndex;
	picParams.frameIdx = frameIndex;

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
	handleResult(encFuncList.nvEncUnmapInputResource(mEncoder, mappedInputResource.mappedResource), "cannot unmap input resource");
	cuDevicePrimaryCtxRetain(&mCuContext, mDevice);
}


void NvEncoder::encodeNvData(const unsigned char* nv12data, int siz, unsigned char* nvencPtr) {
	CUdeviceptr ptr = (CUdeviceptr) nvencPtr;
	cuCtxPushCurrent(mCuContext);
	handleResult(cuMemcpyHtoD_v2(ptr, nv12data, siz), "cannot copy nv12 data to device");
	cuDevicePrimaryCtxRetain(&mCuContext, mDevice);
}


void NvEncoder::endEncode() {
	cuCtxPushCurrent(mCuContext);
	NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
	picParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
	picParams.completionEvent = nullptr;
	handleResult(encFuncList.nvEncEncodePicture(mEncoder, &picParams), "cannot encode picture");
	cuDevicePrimaryCtxRetain(&mCuContext, mDevice);
}


bool NvEncoder::hasBufferedFrame() const {
	return mFrameGot < mFrameToSend;
}


NvPacket NvEncoder::getBufferedFrame() {
	NvPacket out = getEncodedPacket(bitstreamOutputBuffer);
	mFrameGot++;
	return out;
}


void NvEncoder::destroyEncoder() {
	//destroy encoder before resources, otherwise nvEncDestroyEncoder() will throw exeption ???
	if (mEncoder != nullptr) {
		encFuncList.nvEncDestroyEncoder(mEncoder);
		mEncoder = nullptr;
	}

	//clear input buffers
	for (void* res : registeredResources) {
		if (res != nullptr) {
			encFuncList.nvEncUnmapInputResource(mEncoder, res);
			encFuncList.nvEncUnregisterResource(mEncoder, res);
		}
	}
	registeredResources.clear();

	for (CUdeviceptr ptr : inputFrames) {
		if (ptr) {
			cuMemFree_v2(ptr);
		}
	}
	inputFrames.clear();

	//clear output buffers
	for (void* ptr : bitstreamOutputBuffer) {
		if (ptr != nullptr) {
			encFuncList.nvEncUnlockBitstream(mEncoder, ptr);
			encFuncList.nvEncDestroyBitstreamBuffer(mEncoder, ptr);
		}
	}
	bitstreamOutputBuffer.clear();

	cuDevicePrimaryCtxRelease(mDevice);
}

#endif