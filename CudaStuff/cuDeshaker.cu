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

#include "cuDeshaker.cuh"
#include "cuNPP.cuh"
#include "Image.hpp"

#include <algorithm>
#include <fstream>

//parameter structure
//all values must be initialized to be used as __constant__ variable in device code, no constructor calls
__constant__ CoreData d_core;
CudaData cudaData;

//-------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------- HOST CODE ------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------

static void handleStatus(cudaError_t status, std::string&& title) {
	if (status != cudaSuccess) {
		errorLogger().logError(title + ": " + cudaGetErrorString(status));
	}
}

static cudaTextureObject_t prepareComputeTexture(float* src, int w, int h, int pitch) {
	cudaResourceDesc resDesc {};
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = src;
	resDesc.res.pitch2D.width = w;
	resDesc.res.pitch2D.height = h;
	resDesc.res.pitch2D.pitchInBytes = pitch;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();

	// Specify texture object parameters
	cudaTextureDesc texDesc {};
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;

	cudaTextureObject_t texObj;
	handleStatus(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL), "error @compute 10");
	return texObj;
}

void ComputeTextures::create(int64_t idx, int64_t idxPrev, const CoreData& core, float* pyrBase) {
	size_t pyramidSize = 1ull * core.pyramidRowCount * cudaData.strideFloatN; //size of one full pyramid in elements
	float* ptrPrev = pyrBase + pyramidSize * idxPrev;
	Y[0] = prepareComputeTexture(ptrPrev, core.w, core.pyramidRowCount, cudaData.strideFloat);
	float* ptrCur = pyrBase + pyramidSize * idx;
	Y[1] = prepareComputeTexture(ptrCur, core.w, core.pyramidRowCount, cudaData.strideFloat);
}

void ComputeTextures::destroy() const {
	cudaDestroyTextureObject(Y[0]);
	cudaDestroyTextureObject(Y[1]);
}

//allocate cuda memory and store pointers
template <class T> void allocSafe(T* ptr, size_t size) {
	handleStatus(cudaMalloc(ptr, size), "error @init allocating memory");
}

template <class T> void allocDeviceIndices(T*** indexArray, T* srcptr, size_t offset, size_t count) {
	std::vector<T*> idxarr(count);
	size_t siz = sizeof(T*) * count;
	for (size_t i = 0; i < count; i++) idxarr[i] = srcptr + i * offset;
	allocSafe(indexArray, siz);
	handleStatus(cudaMemcpy(*indexArray, idxarr.data(), siz, cudaMemcpyDefault), "error @init copy");
}

bool checkKernelParameters(int3 threads, int3 blocks, size_t shdsize, const cudaDeviceProp& cudaProps) {
	bool out = true;
	out &= threads.x <= cudaProps.maxThreadsDim[0];
	out &= threads.y <= cudaProps.maxThreadsDim[1];
	out &= threads.z <= cudaProps.maxThreadsDim[2];
	out &= blocks.x <= cudaProps.maxGridSize[0];
	out &= blocks.y <= cudaProps.maxGridSize[1];
	out &= blocks.z <= cudaProps.maxGridSize[2];
	out &= shdsize <= cudaProps.sharedMemPerBlock;
	out &= threads.x * threads.y * threads.z <= cudaProps.maxThreadsPerBlock;
	return out;
}

bool checkKernelParameters(int3 threads, int3 blocks, const cudaDeviceProp& cudaProps) {
	return checkKernelParameters(threads, blocks, 0, cudaProps);
}

bool checkKernelParameters(const CudaData& core, const cudaDeviceProp& cudaProps) {
	return checkKernelParameters(core.computeThreads, core.computeBlocks, core.computeSharedMem, cudaProps);
}

//write data from device pointer to file for debugging
template <class T> void writeDeviceDataToFile(const T* devData, size_t h, size_t wCount, size_t stride, const std::string& path) {
	cudaDeviceSynchronize();
	std::vector<T> hostData(h * wCount);
	cudaMemcpy2D(hostData.data(), sizeof(T) * wCount, devData, sizeof(T) * stride, sizeof(T) * wCount, h, cudaMemcpyDeviceToHost);
	std::ofstream file(path, std::ios::binary);
	file.write(reinterpret_cast<char*>(&h), sizeof(size_t));
	file.write(reinterpret_cast<char*>(&wCount), sizeof(size_t));
	size_t sizT = sizeof(T);
	file.write(reinterpret_cast<char*>(&sizT), sizeof(size_t));
	file.write(reinterpret_cast<char*>(hostData.data()), hostData.size() * sizeof(T));
}

//write string into image given by device pointer
void writeText(const std::string& text, int x0, int y0, int scaleX, int scaleY, float* deviceData, const CoreData& core) {
	//create Image<float>
	int imh = 10 * scaleY;
	int siz = imh * cudaData.strideFloat;
	im::ImageBase<float> im(imh, core.w, cudaData.strideFloatN, 3);

	//copy three horizontal stripes into host memory
	for (size_t z = 0; z < 3; z++) {
		float* src = deviceData + (y0 + z * core.h) * cudaData.strideFloatN;
		float* dst = im.plane(z);
		cudaMemcpy(dst, src, siz, cudaMemcpyDefault);
	}

	//write text
	im.writeText(text, x0, 0, scaleX, scaleY, im::TextAlign::TOP_LEFT); //write into host memory

	//copy YUV planes back into device memory
	for (size_t z = 0; z < 3; z++) {
		float* src = im.plane(z);
		float* dst = deviceData + (y0 + z * core.h) * cudaData.strideFloatN;
		cudaMemcpy(dst, src, siz, cudaMemcpyDefault);
	}
}

//----------------------------------
//-------- INIT --------------------
//----------------------------------

//check for cuda runtime installation, this only needs link to cudart_static.lib
CudaProbeResult cudaProbeRuntime() {
	CudaProbeResult out;
	//absence of cuda will report error "CUDA driver is insufficient for CUDA runtime version"
	cudaRuntimeGetVersion(&out.runtimeVersion);
	cudaDriverGetVersion(&out.driverVersion);

	//if we found a proper cuda installation, ask for list of devices
	int deviceCount = 0;
	if (out.driverVersion > 0) {
		handleStatus(cudaGetDeviceCount(&deviceCount), "error probing cuda devices");
		for (int i = 0; i < deviceCount; i++) {
			cudaDeviceProp devProp;
			handleStatus(cudaGetDeviceProperties(&devProp, i), "error getting device properties");
			out.props.push_back(devProp);
		}

		//query npp version numbers, this loads nvcuda.dll
		//const NppLibraryVersion* libVer = nppGetLibVersion(); //nppc.lib
		//cudaInfo.nppMajor = libVer->major;
		//cudaInfo.nppMinor = libVer->minor;
		//cudaInfo.nppBuild = libVer->build;
	}
	return out;
}


//----------------------------------
//-------- CLASS DEFINITION --------
//----------------------------------


CudaExecutor::CudaExecutor(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
	FrameExecutor(data, deviceInfo, frame, pool) {}


void CudaExecutor::cudaInit(CoreData& core, int devIdx, const cudaDeviceProp& prop, ImageYuv& yuvFrame) {
	//copy device prop structure
	props = prop;

	//pin memory of transfer object
	registeredMemPtr = yuvFrame.data();
	handleStatus(cudaHostRegister(registeredMemPtr, yuvFrame.sizeInBytes(), cudaHostRegisterDefault), "error @init #2");

	const size_t h = core.h;
	const size_t w = core.w;

	handleStatus(cudaSetDevice(devIdx), "cannot set device");
	//cudaFree(0); //seems necessary in order to get a context later;

	//sum up required shared memory for compute kernel
	int doublesCount = 0
		+ core.iw * core.iw * 6   //sd
		+ core.iw * core.iw * 1	  //delta
		+ 6 * 6		//S
		+ 6 * 6     //g
		+ 3 * 3  	//wp
		+ 3 * 3     //dwp;
		+ 6 * 1		//b
		+ 6 * 1     //eta
		+ 6 * 1		//temp
		;
	cudaData.computeSharedMem = 0
		+ doublesCount * sizeof(double)   //number of double values in shared memory
		+ 6 * sizeof(double*)             //additional double pointers
		;

	//compute kernel configuration
	cudaData.computeBlocks = { core.ixCount, core.iyCount };
	int rows = std::max(core.iw, 6);
	int ws = prop.warpSize;
	cudaData.computeThreads = { ws / rows, rows };

	//determine memory requirements
	//size_t texAlign = prop.texturePitchAlignment;
	size_t pitch = 0;
	uchar* d_ptr8;
	cudaMallocPitch(&d_ptr8, &pitch, w, 1);
	cudaData.strideChar = (int) pitch;
	cudaFree(d_ptr8);

	float* d_ptr32;
	cudaMallocPitch(&d_ptr32, &pitch, w * sizeof(float), 1);
	cudaData.strideFloat = (int) pitch;
	cudaData.strideFloatN = cudaData.strideFloat / sizeof(float);
	cudaFree(d_ptr32);

	float4* d_ptr128;
	cudaMallocPitch(&d_ptr128, &pitch, w * sizeof(float4), 1);
	cudaData.strideFloat4 = (int) pitch;
	cudaData.strideFloat4N = cudaData.strideFloat4 / sizeof(float4);
	cudaFree(d_ptr128);

	//compute required heap size
	size_t frameSize8 = 3ull * cudaData.strideChar * h;  //bytes for yuv444 images
	size_t heapRequired = 0;
	heapRequired += frameSize8 * core.bufferCount;       //yuv input storage
	heapRequired += frameSize8;						     //yuv out
	heapRequired += frameSize8;						     //rgb out
	heapRequired += 2ull * cudaData.strideFloat * h;     //filter buffers
	heapRequired += cudaData.strideFloat * h * core.pyramidLevels * core.pyramidCount;     //pyramid of Y frames
	heapRequired += cudaData.strideFloat4 * h * cudaData.outBufferCount;                   //output buffer in floats
	heapRequired += sizeof(CudaPointResult) * core.resultCount;                            //array of results structure
	heapRequired += 10ull * 1024 * 1024;

	//set memory limit
	size_t heap = 0;
	handleStatus(cudaDeviceGetLimit(&heap, cudaLimitMallocHeapSize), "error @init #10");
	if (heapRequired < heap) {
		handleStatus(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapRequired), "error @init #20");
	}

	size_t yuvRowCount = 3ull * h * core.bufferCount;
	size_t memtotal, memfree1, memfree2;
	handleStatus(cudaMemGetInfo(&memfree1, &memtotal), "error @init #30");

	//allocate debug storage
	allocSafe(&debugData.d_data, debugData.maxSize);
	handleStatus(cudaMemset(debugData.d_data, 0, debugData.maxSize), "error @init #32");

	//allocate frameResult arrays
	allocSafe(&d_results, sizeof(CudaPointResult) * core.resultCount);
	h_results = new CudaPointResult[core.resultCount];

	//allocate output yuv array
	allocSafe(&d_yuvOut, frameSize8);
	allocSafe(&d_rgba, 4ull * w * h);

	//allocate memory for yuv input data in char format [0..255]
	allocSafe(&d_yuvData, frameSize8 * core.bufferCount);
	allocDeviceIndices(&d_yuvRows, d_yuvData, cudaData.strideChar, yuvRowCount);
	allocDeviceIndices(&d_yuvPlanes, d_yuvRows, h, core.bufferCount * 3ull);
	frameIndizes.assign(core.bufferCount, -1);

	//check pyramid indizes
	pyramidIndizes.assign(core.pyramidCount, -1);

	//allocate float buffers
	allocSafe(&out.data, cudaData.strideFloat4 * h * cudaData.outBufferCount);
	//name individual parts for convenience
	size_t outSize = h * cudaData.strideFloat4N;
	out.start = out.data;
	out.warped = out.start + outSize;
	out.filterH = out.warped + outSize;
	out.filterV = out.filterH + outSize;
	out.final = out.filterV + outSize;
	out.background = out.final + outSize;

	//float filter buffers
	allocSafe(&d_bufferH, cudaData.strideFloat * h);
	allocSafe(&d_bufferV, cudaData.strideFloat * h);

	//initialize background color in output buffer
	float4 bgval = { core.bgcolorYuv[0], core.bgcolorYuv[1], core.bgcolorYuv[2] };
	std::vector<float4> bg(w * h, bgval);
	//write to static background
	size_t siz = w * sizeof(float4);
	handleStatus(cudaMemcpy2D(out.background, cudaData.strideFloat4, bg.data(), siz, siz, h, cudaMemcpyDefault), "error @init 60");
	//write to first image
	handleStatus(cudaMemcpy2D(out.warped, cudaData.strideFloat4, bg.data(), siz, siz, h, cudaMemcpyDefault), "error @init 61");

	//allocate image pyramids, all the same strided width but increasingly shorter
	//number of rows through all three pyramids, Y, DX, DY
	size_t pyrTotalRows = 1ull * core.pyramidRowCount * core.pyramidCount;
	allocSafe(&d_pyrData, cudaData.strideFloat * pyrTotalRows);
	allocDeviceIndices(&d_pyrRows, d_pyrData, cudaData.strideFloatN, pyrTotalRows);

	//set up cuda streams
	cs.assign(2, 0);
	for (size_t i = 0; i < cs.size(); i++) {
		handleStatus(cudaStreamCreate(&cs[i]), "error @init #70");
	}

	//set up compute kernel
	allocSafe(&d_interrupt, 1);

	//memory statistics
	handleStatus(cudaMemGetInfo(&memfree2, &memtotal), "error @init #80");
	cudaData.cudaMemTotal = memtotal;
	cudaData.cudaUsedMem = memfree1 - memfree2;

	//copy core struct to device
	const void* coreptr = &d_core;
	cudaMemcpyToSymbol(coreptr, &core, sizeof(core));

	//final error checks
	handleStatus(cudaDeviceSynchronize(), "error @init #90");
	handleStatus(cudaGetLastError(), "error @init #92");
}


//----------------------------------
//-------- READ
//----------------------------------

//copy yuv input to device
void CudaExecutor::inputData(int64_t frameIndex, const ImageYuv& inputFrame) {
	int64_t fr = frameIndex % mData.bufferCount;
	frameIndizes[fr] = frameIndex;
	size_t frameSizeBytes = 3ull * cudaData.strideChar * mData.h;
	unsigned char* d_frame = d_yuvData + fr * frameSizeBytes;
	handleStatus(cudaMemcpy2D(d_frame, cudaData.strideChar, inputFrame.data(), inputFrame.stride, 
		mData.w, 3ull * mData.h, cudaMemcpyDefault), "error @read #10");
	handleStatus(cudaGetLastError(), "error @read #20");
}


//----------------------------------
//-------- PYRAMID
//----------------------------------

//create image pyramid
void CudaExecutor::createPyramid(int64_t frameIndex, AffineDataFloat trf, bool warp) {
	int w = mData.w;
	int h = mData.h;

	//get to the start of this yuv image
	int64_t frIdx = frameIndex % mData.bufferCount;
	unsigned char* yuvStart = d_yuvData + frIdx * cudaData.strideChar * h * 3;

	//get to the start of this pyramid
	int64_t pyrIdx = frameIndex % mData.pyramidCount;
	float* pyrStart = d_pyrData + pyrIdx * mData.pyramidRowCount * cudaData.strideFloatN;

	//to keep track of things
	pyramidIndizes[pyrIdx] = frameIndex;

	//first level of pyramid Y data
	if (warp) {
		cu::set_32f(pyrStart, cudaData.strideFloatN, w, h, 0);
		cu::scale_8u32f(yuvStart, cudaData.strideChar, d_bufferH, cudaData.strideFloatN, w, h);
		cu::warp_back_32f(d_bufferH, cudaData.strideFloatN, pyrStart, cudaData.strideFloatN, w, h, trf);

	} else {
		cu::scale_8u32f(yuvStart, cudaData.strideChar, pyrStart, cudaData.strideFloatN, w, h);
		cu::filter_32f_h(pyrStart, d_bufferH, cudaData.strideFloatN, w, h, 0);
		cu::filter_32f_v(d_bufferH, pyrStart, cudaData.strideFloatN, w, h, 0);
	}

	//lower levels
	float* src = pyrStart;
	float* dest = pyrStart + 1ull * cudaData.strideFloatN * h;
	for (int z = 1; z <= mData.zMax; z++) {
		cu::remap_downsize_32f(src, cudaData.strideFloatN, dest, cudaData.strideFloatN, w, h);
		w /= 2;
		h /= 2;
		src = dest;
		dest += 1ull * cudaData.strideFloatN * h;
	}

	handleStatus(cudaGetLastError(), "error @pyramid");
}


//----------------------------------
//-------- COMPUTE
//----------------------------------

void CudaExecutor::computeStart(int64_t frameIndex, std::vector<PointResult>& results) {
	int64_t pyrIdx = frameIndex % mData.pyramidCount;
	int64_t pyrIdxPrev = (frameIndex - 1) % mData.pyramidCount;
	assert(frameIndex > 0 && pyramidIndizes[pyrIdx] == pyramidIndizes[pyrIdxPrev] + 1 && "wrong frames to compute"); 

	//prepare kernel
	assert(checkKernelParameters(cudaData, props) && "invalid kernel parameters");
	computeTexture.create(pyrIdx, pyrIdxPrev, mData, d_pyrData);

	//reset computed flags
	handleStatus(cudaMemsetAsync(d_results, 0, sizeof(CudaPointResult) * mData.resultCount, cs[0]), "error @compute #20");

	//issue the call
	ComputeKernelParam param = { 
		debugData.d_data,
		debugData.maxSize,
		cudaData.computeBlocks,
		cudaData.computeThreads,
		cudaData.computeSharedMem,
		cs[0], 
		frameIndex, 
		d_interrupt
	};
	kernelComputeCall(param, computeTexture, d_results);

	//cudaStreamQuery(cs[0]);
	handleStatus(cudaGetLastError(), "error @compute #20");
}

void CudaExecutor::computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) {
	//reset interrupt signal
	handleStatus(cudaMemsetAsync(d_interrupt, 0, sizeof(char), cs[1]), "error @compute #50");

	//restart kernel
	ComputeKernelParam param = {
		debugData.d_data,
		debugData.maxSize,
		cudaData.computeBlocks,
		cudaData.computeThreads,
		cudaData.computeSharedMem,
		cs[0],
		frameIndex,
		d_interrupt
	};
	kernelComputeCall(param, computeTexture, d_results);

	//get results from device
	handleStatus(cudaMemcpy(h_results, d_results, sizeof(CudaPointResult) * mData.resultCount, cudaMemcpyDefault), "error @compute #100");

	//translate to host structure
	for (int i = 0; i < mData.resultCount; i++) {
		const CudaPointResult& hr = h_results[i];
		double x0 = hr.xm - mData.w / 2.0 + hr.u * hr.direction;
		double y0 = hr.ym - mData.h / 2.0 + hr.v * hr.direction;
		double fdir = 1.0 - 2.0 * hr.direction;
		results[i] = { hr.idx, hr.ix0, hr.iy0, x0, y0, hr.u * fdir, hr.v * fdir, hr.result, hr.z, hr.direction };
	}

	//shutdown
	computeTexture.destroy();
	handleStatus(cudaGetLastError(), "error @compute #100");
}


//----------------------------------
//-------- OUTPUT
//----------------------------------

void CudaExecutor::outputData(int64_t frameIndex, AffineDataFloat trf) {
	//ConsoleTimer timer;
	//interrupt compute kernel
	handleStatus(cudaMemsetAsync(d_interrupt, 1, sizeof(char), cs[1]), "error @output #10");

	int h = mData.h;
	int w = mData.w;
	int64_t fr = frameIndex % mData.bufferCount;
	assert(frameIndizes[fr] == frameIndex && "invalid frame in buffer");

	//size of all pixel data in bytes in yuv including padding
	size_t frameSize8 = 3ull * cudaData.strideChar * h;
	//start of input yuv data
	unsigned char* yuvSrc = d_yuvData + fr * frameSize8;

	cu::scale_8u32f_3(yuvSrc, cudaData.strideChar, out.start, cudaData.strideFloat4N, w, h, cs[1]);
	//fill static background when requested
	if (mData.bgmode == BackgroundMode::COLOR) {
		cu::copy_32f_3(out.background, cudaData.strideFloat4N, out.warped, cudaData.strideFloat4N, w, h, cs[1]);
	}
	//warp input
	cu::warp_back_32f_3(out.start, cudaData.strideFloat4N, out.warped, cudaData.strideFloat4N, w, h, trf, cs[1]);
	//writeDeviceDataToFile(out.start, h, w, cudaData.strideFloat4N, "f:/cuda.dat");
	//first filter pass
	cu::filter_32f_h_3(out.warped, out.filterH, cudaData.strideFloat4N, w, h, cs[1]);
	//second filter pass
	cu::filter_32f_v_3(out.filterH, out.filterV, cudaData.strideFloat4N, w, h, cs[1]);
	//combine unsharp mask
	cu::unsharp_32f_3(out.warped, out.filterV, out.final, cudaData.strideFloat4N, w, h, cs[1]);
}

void CudaExecutor::getOutputYuv(int64_t frameIndex, ImageYuv& image) const {
	cu::outputHost(out.final, cudaData.strideFloat4N, d_yuvOut, cudaData.strideChar, mData.w, mData.h, cs[1]);
	for (int planeSize = cudaData.strideChar * mData.h, i = 0; i < 3; i++) {
		cu::copy_32f_3(d_yuvOut + i * planeSize, cudaData.strideChar, image.addr(i, 0, 0), image.strideInBytes(), mData.w, mData.h, cs[1]);
	}
	image.setIndex(frameIndex);
	handleStatus(cudaStreamSynchronize(cs[1]), "error @output #90");
	handleStatus(cudaGetLastError(), "error @output #91");
}

void CudaExecutor::getOutputImage(int64_t frameIndex, ImageBaseRgb& image) const {
	const std::vector<int>& idx = image.indexRgba();
	cu::yuv_to_rgba(out.final, cudaData.strideFloat4N, d_rgba, -1, mData.w, mData.h, { idx[0], idx[1], idx[2], idx[3] }, cs[1]);
	handleStatus(cudaMemcpyAsync(image.plane(0), d_rgba, 4ull * mData.w * mData.h, cudaMemcpyDefault, cs[1]), "error @output #94");
	image.setIndex(frameIndex);
	handleStatus(cudaStreamSynchronize(cs[1]), "error @output #92");
	handleStatus(cudaGetLastError(), "error @output #93");
}

void CudaExecutor::getOutputNvenc(int64_t frameIndex, ImageNV12& image, unsigned char* cudaNv12ptr) const {
	cu::outputNvenc(out.final, cudaData.strideFloat4N, cudaNv12ptr, image.stride, mData.w, mData.h, cs[1]);
	handleStatus(cudaStreamSynchronize(cs[1]), "error @output #95");
	handleStatus(cudaGetLastError(), "error @output #96");
}

void CudaExecutor::cudaGetTransformedOutput(float* warped) const {
	std::vector<float4> data(1ull * mData.w * mData.h);
	size_t wbytes = mData.w * sizeof(float4);
	handleStatus(cudaMemcpy2D(data.data(), wbytes, out.warped, cudaData.strideFloat4, wbytes, mData.h, cudaMemcpyDefault), "error @transformedOutput");

	for (int i = 0; i < mData.w * mData.h; i++) {
		warped[i] = data[i].x;
		warped[i + mData.w * mData.h] = data[i].y;
		warped[i + mData.w * mData.h * 2] = data[i].z;
	}
}

void CudaExecutor::cudaGetPyramid(int64_t frameIndex, float* data) const {
	int pyrIdx = frameIndex % mData.pyramidCount;
	float* devptr = d_pyrData + pyrIdx * mData.pyramidRowCount * cudaData.strideFloatN;
	size_t wbytes = mData.w * sizeof(float);

	handleStatus(cudaMemcpy2D(data, wbytes, devptr, cudaData.strideFloat, wbytes, mData.pyramidRowCount, cudaMemcpyDefault), "error @getPyramid");
}

void CudaExecutor::getInput(int64_t frameIndex, ImageYuv& image) const {
	int fr = frameIndex % mData.bufferCount;
	//start of input yuv data
	unsigned char* yuvSrc = d_yuvData + fr * 3 * mData.h * cudaData.strideChar;
	//copy 2D data without stride
	handleStatus(cudaMemcpy2D(image.data(), image.stride, yuvSrc, cudaData.strideChar, image.w, 3ll * image.h, cudaMemcpyDefault), "error @getInput");
}

void CudaExecutor::getInput(int64_t frameIndex, ImageRGBA& image) const {
	int fridx = frameIndex % mData.bufferCount;
	assert(frameIndizes[fridx] == frameIndex && "invalid frame in buffer");
	unsigned char* yuvSrc = d_yuvData + fridx * 3ull * mData.h * cudaData.strideChar;
	cu::yuv_to_rgba(yuvSrc, cudaData.strideChar, d_rgba, -1, mData.w, mData.h);
	handleStatus(cudaMemcpy(image.plane(0), d_rgba, 4ull * mData.w * mData.h, cudaMemcpyDefault), "error @progress input");
}

void CudaExecutor::getWarped(int64_t frameIndex, ImageRGBA& image) {
	int4 index = { 0, 1, 2, 3 };
	cu::yuv_to_rgba(out.warped, cudaData.strideFloat4N, d_rgba, -1, mData.w, mData.h, index);
	handleStatus(cudaMemcpy(image.plane(0), d_rgba, 4ull * mData.w * mData.h, cudaMemcpyDefault), "error @progress output");
}


void encodeNvData(const ImageNV12& image, unsigned char* nvencPtr) {
	handleStatus(cudaMemcpy(nvencPtr, image.addr(0, 0, 0), image.sizeInBytes(), cudaMemcpyHostToDevice), "error @simple encode #1 cannot copy to device");
}

void getNvData(std::vector<unsigned char>& nv12, unsigned char* cudaNv12ptr) {
	handleStatus(cudaMemcpy(nv12.data(), cudaNv12ptr, nv12.size(), cudaMemcpyDeviceToHost), "error getting nv12 data");
}

/*
void cudaSynchronize() {
	handleStatus(cudaDeviceSynchronize(), "error @synchronize #10");
	handleStatus(cudaGetLastError(), "error @synchronize #20");
}
*/


//----------------------------------
//-------- SHUTDOWN
//----------------------------------

void CudaExecutor::getDebugData(const CoreData& core, const std::string& imageFile, std::function<void(size_t, size_t, double*)> fcn) {
	std::vector<double> data(debugData.maxSize / sizeof(double));
	handleStatus(cudaMemcpy(data.data(), debugData.d_data, debugData.maxSize, cudaMemcpyDefault), "error @shutdown #5 copy debug data");

	double* ptr = data.data() + 1;
	double* ptrEnd = data.data() + size_t(data[0]) + 1;
	while (ptr != ptrEnd) {
		size_t h = (size_t) *ptr++;
		size_t w = (size_t) *ptr++;
		fcn(h, w, ptr);
		ptr += h * w;
	}

	//get image of kernel timing values
	int h = core.resultCount;
	int w = 6'000;
	ImageBGR kernelTimerImage = ImageBGR(h, w);
	auto fcnMin = [] (CudaPointResult& r1, CudaPointResult& r2) { return r1.timeStart < r2.timeStart; };
	auto minTime = std::min_element(h_results, h_results + core.resultCount, fcnMin);
	auto fcnMax = [] (CudaPointResult& r1, CudaPointResult& r2) { return r1.timeStop < r2.timeStop; };
	auto maxTime = std::max_element(h_results, h_results + core.resultCount, fcnMax);
	int64_t delta = maxTime->timeStop - minTime->timeStart;
	if (delta > 0) {
		double f = delta / (w - 1.0);
		for (int i = 0; i < h; i++) {
			CudaPointResult& r = h_results[i];
			int t1 = int((r.timeStart - minTime->timeStart) / f);
			int t2 = int((r.timeStop - minTime->timeStart) / f);
			for (int k = t1; k <= t2; k++) {
				kernelTimerImage.at(0, i, k) = 255;
			}
		}
	}
	kernelTimerImage.saveAsColorBMP(imageFile);
}

CudaExecutor::~CudaExecutor() {
	
	//delete device memory
	void* d_arr[] = { d_results, d_yuvOut, d_rgba, d_yuvData, d_yuvRows, d_yuvPlanes, 
		out.data, d_bufferH, d_bufferV, d_pyrData, d_pyrRows, 
		debugData.d_data, d_interrupt
	};

	for (void* ptr : d_arr) {
		handleStatus(cudaFree(ptr), "error @shutdown #10 shutting down memory");
	}

	//delete streams
	for (int i = 0; i < cs.size(); i++) {
		handleStatus(cudaStreamDestroy(cs[i]), "error @shutdown #20 shutting down streams");
	}

	//delete host memory
	delete[] h_results;

	//unregister memory
	handleStatus(cudaHostUnregister(registeredMemPtr), "error @shutdown #30 unregister");

	//do not reset device while nvenc is still active
	//handleStatus(cudaDeviceReset(), "error @shutdown #90", errorList);
	handleStatus(cudaGetLastError(), "error @shutdown #100");
}
