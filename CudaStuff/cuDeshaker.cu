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
#include "cuKernels.cuh"
#include "ImageClasses.hpp"
#include "ErrorLogger.hpp"
#include "DeviceInfoBase.hpp"

#include <algorithm>
#include <fstream>

//parameter structure used in device code
//all values must be initialized to be used as __constant__ variable in device code, no constructor calls
__constant__ CoreData d_core;

//number of threads for reading textures is set via init function
uint numCudaThreads = 0;

dim3 configThreads() {
	return { numCudaThreads, numCudaThreads };
}

dim3 configBlocks(dim3 threads, int width, int height) {
	uint bx = (width + threads.x - 1) / threads.x;
	uint by = (height + threads.y - 1) / threads.y;
	return { bx, by };
}

int3 CudaExecutor::computeBlocks() const {
	return { mData.ixCount, mData.iyCount };
}

int3 CudaExecutor::computeThreads() const {
	int rows = std::max(mData.iw, 6);
	int ws = props.warpSize;
	return { ws / rows, rows };
}

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

void ComputeTextures::create(int64_t idx, int64_t idxPrev, float* pyrBase, const CoreData& core) {
	size_t pyramidSize = 1ull * core.pyramidRowCount * core.strideFloat / sizeof(float); //size of one full pyramid in elements
	float* ptrPrev = pyrBase + pyramidSize * idxPrev;
	Y[0] = prepareComputeTexture(ptrPrev, core.w, core.pyramidRowCount, core.strideFloat);
	float* ptrCur = pyrBase + pyramidSize * idx;
	Y[1] = prepareComputeTexture(ptrCur, core.w, core.pyramidRowCount, core.strideFloat);
}

void ComputeTextures::destroy() const {
	cudaDestroyTextureObject(Y[0]);
	cudaDestroyTextureObject(Y[1]);
}

//allocate cuda memory and store pointers
template <class T> void allocSafe(T* ptr, size_t size) {
	handleStatus(cudaMalloc(ptr, size), "error @init allocating memory");
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

//write image from device to disk for debugging
void writeDeviceDataToImage(const uchar* devData, int h, int w, int stride, const std::string& path) {
	cudaDeviceSynchronize();
	im::ImageVuyx image(h, w, stride);
	handleStatus(cudaMemcpy2D(image.data(), image.stride(), devData, stride, w * 4, h, cudaMemcpyDeviceToHost), "error copy");
	image.saveBmpPlanes(path);
}

//write image from device to disk for debugging
void writeDeviceDataToImage(const float* devData, int h, int w, int stride, const std::string& path) {
	cudaDeviceSynchronize();
	im::ImageY<float> image(h, w, stride, 1.0f);
	handleStatus(cudaMemcpy2D(image.data(), image.strideInBytes(), devData, stride * sizeof(float), w * sizeof(float), h, cudaMemcpyDeviceToHost), "error copy");
	image.saveBmpPlanes(path);
}

//write image from device to disk for debugging
void writeDeviceDataToImage(const float4* devData, int h, int w, int stride, const std::string& path) {
	cudaDeviceSynchronize();
	im::ImageVuyxFloat image(h, w);
	handleStatus(cudaMemcpy2D(image.data(), image.strideInBytes(), devData, stride * 4 * sizeof(float), w * 4 * sizeof(float), h, cudaMemcpyDeviceToHost), "error copy");
	image.saveBmpPlanes(path);
}

bool CudaExecutor::checkKernelParameters(int3 threads, int3 blocks, size_t shdsize) const {
	bool out = true;
	out &= threads.x <= props.maxThreadsDim[0];
	out &= threads.y <= props.maxThreadsDim[1];
	out &= threads.z <= props.maxThreadsDim[2];
	out &= blocks.x <= props.maxGridSize[0];
	out &= blocks.y <= props.maxGridSize[1];
	out &= blocks.z <= props.maxGridSize[2];
	out &= shdsize <= props.sharedMemPerBlock;
	out &= threads.x * threads.y * threads.z <= props.maxThreadsPerBlock;
	return out;
}

bool CudaExecutor::checkKernelParameters(int3 threads, int3 blocks) const {
	return checkKernelParameters(threads, blocks, 0);
}

bool CudaExecutor::checkKernelParameters() const {
	return checkKernelParameters(computeThreads(), computeBlocks(), mData.cudaComputeSharedMem);
}

//write string into image given by device pointer
void CudaExecutor::writeText(const std::string& text, int x0, int y0, float4* deviceData, int devicePitch) {
	ImageVuyxFloat im(mData.h, mData.w);

	//copy to host memory
	cudaMemcpy2D(im.data(), im.strideInBytes(), deviceData, devicePitch, sizeof(float) * mData.w * 4, mData.h, cudaMemcpyDefault);

	//write text in host memory
	im.writeText(text, x0, y0);

	//copy back into device memory
	cudaMemcpy(deviceData, im.data(), im.sizeInBytes(), cudaMemcpyDefault);
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
	}
	return out;
}


//----------------------------------
//-------- CLASS DEFINITION --------
//----------------------------------


CudaExecutor::CudaExecutor(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
	FrameExecutor(data, deviceInfo, frame, pool) {}


void CudaExecutor::init() {
	assert(mDeviceInfo.getType() == DeviceType::CUDA && "device type must be CUDA here");
	const DeviceInfoCudaBase* device = static_cast<const DeviceInfoCudaBase*>(&mDeviceInfo);

	//copy device prop structure
	props = *device->props;

	//number of threads
	numCudaThreads = mData.cudaThreads;

	const size_t h = mData.h;
	const size_t w = mData.w;

	handleStatus(cudaSetDevice(device->cudaIndex), "cannot set device");
	//cudaFree(0); //seems necessary in order to get a context later;

	//sum up required shared memory for compute kernel
	int iw = mData.iw;
	int doublesCount = 0
		+ iw * iw * 6  //sd
		+ iw * iw * 1  //delta
		+ 6 * 6        //S
		+ 6 * 6        //g
		+ 3 * 3        //wp
		+ 3 * 3        //dwp;
		+ 6 * 1        //b
		+ 6 * 1        //eta
		+ 6 * 1        //temp
		;
	mData.cudaComputeSharedMem = 0
		+ doublesCount * sizeof(double)   //number of double values in shared memory
		+ 6 * sizeof(double*)             //additional double pointers
		;

	//determine memory requirements
	//probe cuda pitch values by allocating one row
	size_t pitch = 0;

	uchar* d_ptr1;
	cudaMallocPitch(&d_ptr1, &pitch, w, 1);
	mData.stride = (int) pitch;
	cudaFree(d_ptr1);

	uchar* d_ptr4;
	cudaMallocPitch(&d_ptr4, &pitch, w * 4, 1);
	mData.stride4 = (int) pitch;
	cudaFree(d_ptr4);

	float* d_ptrf;
	cudaMallocPitch(&d_ptrf, &pitch, w * sizeof(float), 1);
	mData.strideFloat = (int) pitch;
	cudaFree(d_ptrf);

	float4* d_ptrf4;
	cudaMallocPitch(&d_ptrf4, &pitch, w * sizeof(float4), 1);
	mData.strideFloat4 = (int) pitch;
	cudaFree(d_ptrf4);

	size_t memtotal, memfree1, memfree2;
	handleStatus(cudaMemGetInfo(&memfree1, &memtotal), "error @init #30");

	//allocate input frame storage with proper cuda stride
	h_input = ImageYuv(mData.h, mData.w, mData.stride);
	allocSafe(&d_input, 3ull * mData.stride * mData.h);

	//pin memory of transfer object
	registeredMemPtr = h_input.data();
	handleStatus(cudaHostRegister(registeredMemPtr, h_input.sizeInBytes(), cudaHostRegisterDefault), "error @init #31");

	//allocate image pyramids, all the same strided width but increasingly shorter
	//number of rows through all three pyramids, Y, DX, DY
	size_t pyrTotalRows = 1ull * mData.pyramidRowCount * mData.pyramidCount;
	allocSafe(&d_pyrData, mData.strideFloat * pyrTotalRows);

	//allocate debug storage
	allocSafe(&debugData.d_data, debugData.maxSize);
	handleStatus(cudaMemset(debugData.d_data, 0, debugData.maxSize), "error @init #32");

	//allocate frameResult arrays
	allocSafe(&d_results, sizeof(CudaPointResult) * mData.resultCount);
	h_results.resize(mData.resultCount);

	//allocate output vuyx data
	size_t frameSize4 = mData.stride4 * h;               //bytes for vuyx images
	allocSafe(&d_output, frameSize4);

	//allocate memory for luma sum
	allocSafe(&d_luma, mData.w * sizeof(int64_t));

	//allocate memory for vuyx input data in char format [0..255]
	allocSafe(&d_vuyxData, frameSize4 * mData.bufferCount);
	frameIndizes.assign(mData.bufferCount, -1);

	//check pyramid indizes
	pyramidIndizes.assign(mData.pyramidCount, -1);

	//allocate float4 buffers
	allocSafe(&out.data, mData.strideFloat4 * h * mData.cudaOutBufferCount);
	//name individual parts for convenience
	size_t outSize = h * mData.strideFloat4 / sizeof(float4);
	out.start = out.data;
	out.warped = out.start + outSize;
	out.filterH = out.warped + outSize;
	out.filterV = out.filterH + outSize;
	out.output = out.filterV + outSize;
	out.background = out.output + outSize;

	//float filter buffers
	allocSafe(&d_bufferH, mData.strideFloat * h);
	allocSafe(&d_bufferV, mData.strideFloat * h);

	//initialize background color in output buffer
	float4 bgval = { mData.bgcol4[0], mData.bgcol4[1], mData.bgcol4[2], mData.bgcol4[3] };
	std::vector<float4> bg(w * h, bgval);
	//write to static background
	size_t siz = w * sizeof(float4);
	handleStatus(cudaMemcpy2D(out.background, mData.strideFloat4, bg.data(), siz, siz, h, cudaMemcpyDefault), "error @init 60");
	//write to first image
	handleStatus(cudaMemcpy2D(out.warped, mData.strideFloat4, bg.data(), siz, siz, h, cudaMemcpyDefault), "error @init 61");

	//set up cuda streams
	cs.assign(2, 0);
	for (size_t i = 0; i < cs.size(); i++) {
		handleStatus(cudaStreamCreate(&cs[i]), "error @init #70");
	}

	//set up compute kernel
	allocSafe(&d_interrupt, 1);

	//memory statistics
	handleStatus(cudaMemGetInfo(&memfree2, &memtotal), "error @init #80");
	mData.cudaMemTotal = memtotal;
	mData.cudaMemUsed = memfree1 - memfree2;

	//copy core struct to device
	const void* coreptr = &d_core;
	cudaMemcpyToSymbol(coreptr, &mData, sizeof(mData));

	//final error checks
	handleStatus(cudaDeviceSynchronize(), "error @init #90");
	handleStatus(cudaGetLastError(), "error @init #92");
}


//----------------------------------
//-------- READ
//----------------------------------

//host image to decode into
Image8& CudaExecutor::inputDestination(int64_t frameIndex) {
	return h_input;
}

//copy yuv input to device
void CudaExecutor::inputData(int64_t frameIndex) {
	handleStatus(cudaMemcpy(d_input, h_input.data(), h_input.sizeInBytes(), cudaMemcpyDefault), "error @input #10");
	cudaError_t err;

	int64_t idx = frameIndex % mData.bufferCount;
	frameIndizes[idx] = frameIndex;
	unsigned char* d_ptr = d_vuyxData + idx * mData.stride4 * mData.h;
	err = cu::input(d_input, mData.stride, mData.w, d_ptr, mData.stride4, mData.w * 4, mData.h);

	handleStatus(err, "error @input #11");
	handleStatus(cudaGetLastError(), "error @input #12");
}


//----------------------------------
//-------- PYRAMID
//----------------------------------

//create image pyramid
int64_t CudaExecutor::createPyramid(int64_t frameIndex, AffineDataFloat trf, bool warp) {
	//util::ConsoleTimer timer("cuda pyramid " + std::to_string(frameIndex));
	int w = mData.w;
	int h = mData.h;
	int strideFloatCount = mData.strideFloat / sizeof(float);
	cudaError_t err = cudaSuccess;

	//get to the start of this yuv image
	int64_t frIdx = frameIndex % mData.bufferCount;
	unsigned char* vuyxStart = d_vuyxData + frIdx * mData.stride4 * h;
	//debugLogger().format("pyr {} {}", frameIndex, static_cast<void*>(vuyxStart));

	//get to the start of this pyramid
	int64_t pyrIdx = frameIndex % mData.pyramidCount;
	float* pyrStart = d_pyrData + pyrIdx * mData.pyramidRowCount * mData.strideFloat / sizeof(float);

	//to keep track of things
	pyramidIndizes[pyrIdx] = frameIndex;
	
	//first level of pyramid Y data
	if (warp) {
		cu::set_32f(pyrStart, strideFloatCount, w, h, 0);
		err = cu::scale_8u32f(vuyxStart, mData.stride4, w * 4, d_bufferH, strideFloatCount, w, h, d_luma);
		cu::warp_back_32f(d_bufferH, strideFloatCount, pyrStart, strideFloatCount, w, h, trf);

	} else {
		err = cu::scale_8u32f(vuyxStart, mData.stride4, w * 4, pyrStart, strideFloatCount, w, h, d_luma);
		cu::filter_32f_h(pyrStart, d_bufferH, strideFloatCount, w, h, 0);
		cu::filter_32f_v(d_bufferH, pyrStart, strideFloatCount, w, h, 0);
	}

	//lower levels
	float* src = pyrStart;
	float* dest = pyrStart + 1ull * strideFloatCount * h;
	for (int z = 1; z <= mData.zMax; z++) {
		cu::remap_downsize_32f(src, strideFloatCount, dest, strideFloatCount, w, h);
		w /= 2;
		h /= 2;
		src = dest;
		dest += 1ull * strideFloatCount * h;
	}

	int64_t lumaSum = cu::lumaSum(d_luma, mData.w);
	handleStatus(err, "error @pyramid #1");
	handleStatus(cudaGetLastError(), "error @pyramid #2");
	return lumaSum;
}

void CudaExecutor::adjustPyramid(int64_t frameIndex, float gamma) {

}


//----------------------------------
//-------- COMPUTE
//----------------------------------

void CudaExecutor::computeStart(int64_t frameIndex, std::span<PointResult> results) {
	int64_t pyrIdx = frameIndex % mData.pyramidCount;
	int64_t pyrIdxPrev = (frameIndex - 1) % mData.pyramidCount;
	assert(frameIndex > 0 && pyramidIndizes[pyrIdx] == pyramidIndizes[pyrIdxPrev] + 1 && "wrong frames to compute"); 

	//prepare kernel
	assert(checkKernelParameters() && "invalid kernel parameters");
	computeTexture.create(pyrIdx, pyrIdxPrev, d_pyrData, mData);

	//reset computed flags
	handleStatus(cudaMemsetAsync(d_results, 0, sizeof(CudaPointResult) * mData.resultCount, cs[0]), "error @compute #20");

	//issue the call
	ComputeKernelParam param = { 
		debugData.d_data,
		debugData.maxSize,
		computeBlocks(),
		computeThreads(),
		mData.cudaComputeSharedMem,
		cs[0], 
		frameIndex, 
		d_interrupt
	};
	kernelComputeCall(param, computeTexture, d_results);

	//cudaStreamQuery(cs[0]);
	handleStatus(cudaGetLastError(), "error @compute #20");
}

void CudaExecutor::computeTerminate(int64_t frameIndex, std::span<PointResult> results) {
	//reset interrupt signal
	handleStatus(cudaMemsetAsync(d_interrupt, 0, sizeof(char), cs[1]), "error @compute #50");

	//restart kernel
	ComputeKernelParam param = {
		debugData.d_data,
		debugData.maxSize,
		computeBlocks(),
		computeThreads(),
		mData.cudaComputeSharedMem,
		cs[0],
		frameIndex,
		d_interrupt
	};
	kernelComputeCall(param, computeTexture, d_results);

	//get results from device
	handleStatus(cudaMemcpy(h_results.data(), d_results, sizeof(CudaPointResult) * mData.resultCount, cudaMemcpyDefault), "error @compute #100");

	//translate to host structure
	for (int i = 0; i < mData.resultCount; i++) {
		const CudaPointResult& hr = h_results[i];
		double x0 = hr.xm - mData.w / 2.0 + hr.u * hr.direction;
		double y0 = hr.ym - mData.h / 2.0 + hr.v * hr.direction;
		double fdir = 1.0 - 2.0 * hr.direction;
		results[i] = { hr.idx, hr.ix0, hr.iy0, x0, y0, hr.u * fdir, hr.v * fdir, hr.result, hr.z, hr.direction, hr.length };
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
	handleStatus(cudaMemsetAsync(d_interrupt, 1, sizeof(char), cs[1]), "error @output #90");

	int h = mData.h;
	int w = mData.w;
	int strideFloat4N = mData.strideFloat4 / sizeof(float4);
	int64_t idx = frameIndex % mData.bufferCount;
	assert(frameIndizes[idx] == frameIndex && "invalid frame in buffer");

	//size of all pixel data in bytes in vuyx including padding
	int frameSize4 = mData.stride4 * h;
	//start of input yuv data
	unsigned char* vuyxSrc = d_vuyxData + idx * frameSize4;

	cu::scale_8u32f_3(vuyxSrc, mData.stride4, w * 4, out.start, strideFloat4N, w, h, cs[1]);
	//fill static background when requested
	if (mData.bgmode == BackgroundMode::COLOR) {
		handleStatus(cudaMemcpyAsync(out.warped, out.background, 1ull * mData.strideFloat4 * h, cudaMemcpyDefault, cs[1]), "error @output #91");
	}
	//warp input
	cu::warp_back_32f_3(out.start, strideFloat4N, out.warped, strideFloat4N, w, h, trf, cs[1]);
	//first filter pass
	cu::filter_32f_h_3(out.warped, out.filterH, strideFloat4N, w, h, cs[1]);
	//second filter pass
	cu::filter_32f_v_3(out.filterH, out.filterV, strideFloat4N, w, h, cs[1]);
	//combine unsharp mask
	cu::unsharp_32f_3(out.warped, out.filterV, out.output, strideFloat4N, w, h, cs[1]);

	//writeDeviceDataToImage(out.output, h, w, strideFloat4N, "f:/test.bmp");
	//writeText(std::to_string(frameIndex), 10, 50, out.output, mData.strideFloat4);
}

//get output data to host in different formats
void CudaExecutor::getOutput(int64_t frameIndex, Image8& image) const {
	int srcStep = mData.strideFloat4 / sizeof(float4);
	cudaError_t err = cudaSuccess;

	if (image.imageType() == ImageType::VUYX) {
		err = cu::outputHost(out.output, srcStep, d_output, mData.stride4, mData.w, mData.h, cs[1]);
		handleStatus(cudaMemcpy2DAsync(image.data(), image.stride(), d_output, mData.stride4, mData.w * 4ull, mData.h, cudaMemcpyDefault, cs[1]), "error @output #92");

	} else if (image.imageType() == ImageType::YUV) {
		err = cu::outputHostYuv(out.output,srcStep, d_output, mData.w, mData.w, mData.h, cs[1]);
		handleStatus(cudaMemcpy2DAsync(image.data(), image.stride(), d_output, mData.w, mData.w, mData.h * 3ull, cudaMemcpyDefault, cs[1]), "error @output #93");

	} else if (image.colorBase() == ColorBase::RGB) {
		auto idx = image.colorIndex();
		err = cu::yuv_to_rgba(out.output, srcStep, d_output, mData.stride4, mData.w, mData.h, { idx[0], idx[1], idx[2], idx[3] }, cs[1]);
		handleStatus(cudaMemcpy2DAsync(image.data(), image.stride(), d_output, mData.stride4, mData.w * 4ull, mData.h, cudaMemcpyDefault, cs[1]), "error @output #94");
	}

	image.setIndex(frameIndex);
	handleStatus(err, "error @output #95");
	handleStatus(cudaStreamSynchronize(cs[1]), "error @output #96");
	handleStatus(cudaGetLastError(), "error @output #97");
}

//get output data in nv12 format
bool CudaExecutor::getOutput(int64_t frameIndex, Image8& image, int cudaNv12stride, unsigned char* cudaNv12ptr) const {
	bool needsCopy;
	int srcStep = mData.strideFloat4 / sizeof(float4);
	cudaError_t err;

	if (cudaNv12ptr == nullptr) {
		assert(image.imageType() == ImageType::NV12 && "invalid frame format");
		err = cu::outputNvenc(out.output, srcStep, d_output, image.stride(), mData.w, mData.h, cs[1]);
		handleStatus(cudaMemcpyAsync(image.data(), d_output, image.sizeInBytes(), cudaMemcpyDefault, cs[1]), "error @output #100");
		needsCopy = true;

	} else {
		err = cu::outputNvenc(out.output, srcStep, cudaNv12ptr, cudaNv12stride, mData.w, mData.h, cs[1]);
		needsCopy = false;
	}

	image.setIndex(frameIndex);
	handleStatus(err, "error @output #101");
	handleStatus(cudaStreamSynchronize(cs[1]), "error @output #102");
	handleStatus(cudaGetLastError(), "error @output #103");
	return needsCopy;
}

//get input data
void CudaExecutor::getInput(int64_t frameIndex, Image8& image) const {
	int64_t fridx = frameIndex % mData.bufferCount;
	assert(frameIndizes[fridx] == frameIndex && "invalid frame in buffer");
	size_t offset = fridx * mData.h * mData.stride4;
	unsigned char* d_src = d_vuyxData + offset;

	unsigned char* ptr = nullptr;
	cudaError_t err = cudaSuccess;
	if (image.imageType() == ImageType::VUYX) {
		ptr = d_src;

	} else if (image.colorBase() == ColorBase::RGB) {
		std::span<int> idx = image.colorIndex();
		err = cu::yuv_to_rgba(d_src, mData.stride4, d_output, mData.stride4, mData.w, mData.h, { idx[0], idx[1], idx[2], idx[3] }, cs[1]);
		ptr = d_output;
	}

	image.setIndex(frameIndex);
	handleStatus(err, "error @getInput #10");
	handleStatus(cudaMemcpy2DAsync(image.data(), image.stride(), ptr, mData.stride4, image.w() * 4ull, image.h(), cudaMemcpyDefault, cs[1]), "error @getInput #11");
	handleStatus(cudaStreamSynchronize(cs[1]), "error @getInput #12");
	handleStatus(cudaGetLastError(), "error @getInput #13");
}

void CudaExecutor::cudaGetPyramid(int64_t frameIndex, float* data) const {
	int pyrIdx = frameIndex % mData.pyramidCount;
	float* devptr = d_pyrData + pyrIdx * mData.pyramidRowCount * mData.strideFloat / sizeof(float);
	size_t wbytes = mData.w * sizeof(float);
	handleStatus(cudaMemcpy2D(data, wbytes, devptr, mData.strideFloat, wbytes, mData.pyramidRowCount, cudaMemcpyDefault), "error @getPyramid");
}

void CudaExecutor::cudaGetTransformedOutput(float* warpedData, size_t h, size_t w) const {
	size_t wbytes = w * sizeof(float);
	handleStatus(cudaMemcpy2D(warpedData, wbytes, out.warped, mData.strideFloat4, wbytes, mData.h, cudaMemcpyDefault), "error @transformedOutput");
}


//----------------------------------
//-------- SHUTDOWN
//----------------------------------

void CudaExecutor::getDebugData(const std::string& imageFile, std::function<void(size_t, size_t, double*)> fcn) {
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
	int h = mData.resultCount;
	int w = 6'000;
	ImageBgr kernelTimerImage = ImageBgr(h, w);
	auto fcnMin = [] (const CudaPointResult& r1, const CudaPointResult& r2) { return r1.timeStart < r2.timeStart; };
	auto minTime = std::min_element(h_results.cbegin(), h_results.cend(), fcnMin);
	auto fcnMax = [] (const CudaPointResult& r1, const CudaPointResult& r2) { return r1.timeStop < r2.timeStop; };
	auto maxTime = std::max_element(h_results.cbegin(), h_results.cend(), fcnMax);
	int64_t delta = maxTime->timeStop - minTime->timeStart;
	if (delta > 0) {
		double f = delta / (w - 1.0);
		for (int i = 0; i < h; i++) {
			const CudaPointResult& r = h_results[i];
			int t1 = int((r.timeStart - minTime->timeStart) / f);
			int t2 = int((r.timeStop - minTime->timeStart) / f);
			for (int k = t1; k <= t2; k++) {
				kernelTimerImage.at(0, i, k) = 255;
			}
		}
	}
	kernelTimerImage.saveBmpColor(imageFile);
}

CudaExecutor::~CudaExecutor() {
	//unregister memory
	handleStatus(cudaHostUnregister(registeredMemPtr), "error @shutdown #10 unregister");

	//delete device memory
	void* d_arr[] = { d_input, d_results, d_vuyxData, d_output, out.data, d_pyrData, d_bufferH, d_bufferV, debugData.d_data, d_interrupt, d_luma };
	for (void* ptr : d_arr) {
		handleStatus(cudaFree(ptr), "error @shutdown #20 delete memory");
	}

	//delete streams
	for (int i = 0; i < cs.size(); i++) {
		handleStatus(cudaStreamDestroy(cs[i]), "error @shutdown #30 delete streams");
	}

	//do not reset device while nvenc is still active
	//handleStatus(cudaDeviceReset(), "error @shutdown #90", errorList);
	handleStatus(cudaGetLastError(), "error @shutdown #50");
}
