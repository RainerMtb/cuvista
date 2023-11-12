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

unsigned char* d_yuvData;			     //continuous array of all pixel values in yuv format, allocated on device
unsigned char** d_yuvRows;			     //index into rows of pixels, allocated on device
unsigned char*** d_yuvPlanes;		     //index into Y-U-V planes of frames, allocated on device 

unsigned char* d_yuvOut;   //image data for encoding on host
unsigned char* d_rgb;      //image data for progress update

struct {
	float4* data;

	float4* start;
	float4* warped;
	float4* filterH;
	float4* filterV;
	float4* final;
	float4* background;
} out;

float* d_bufferH;
float* d_bufferV;

float* d_pyrData;
float** d_pyrRows;

//results from compute kernel
PointResult* d_results;

//init cuda streams
std::vector<cudaStream_t> cs(2);

//data output from kernels for later analysis
cu::DebugData debugData = {};

//registered memory
void* registeredMemPtr = nullptr;

//textures used in compute kernel
ComputeTextures compTex;

//array of time captures for compute kernel
KernelTimer* d_kernelTimer = nullptr;

//signal to interrupt compute kernel
char* d_interrupt;
//array to keep track of already computed blocks
char* d_computed;

//parameter structure
__constant__ CudaData d_core;


__device__ void KernelTimer::start() {
	block = blockIdx;
	thread = threadIdx;
	cu::globaltimer(&timeStart);
}

__device__ void KernelTimer::stop() {
	cu::globaltimer(&timeStop);
}

__device__ const CudaData& getCudaData() {
	return d_core;
}

//-------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------- HOST CODE ------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------

void handleStatus(cudaError_t status, std::string&& title) {
	if (status != cudaSuccess) {
		errorLogger.logError(title + ": " + cudaGetErrorString(status));
	}
}

cudaTextureObject_t prepareComputeTexture(float* src, int w, int h, int pitch) {
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

void ComputeTextures::create(int64_t idx, int64_t idxPrev, const CudaData& core) {
	size_t pyramidSize = 1ull * core.pyramidRowCount * core.strideFloatN; //size of one full pyramid in elements
	float* ptr1 = d_pyrData + 3 * pyramidSize * idx;
	Ycur = prepareComputeTexture(ptr1, core.w, core.pyramidRowCount, core.strideFloat);

	float* ptr2 = d_pyrData + 3 * pyramidSize * idxPrev;
	Yprev = prepareComputeTexture(ptr2, core.w, core.pyramidRowCount, core.strideFloat);
	DXprev = prepareComputeTexture(ptr2 + pyramidSize, core.w, core.pyramidRowCount, core.strideFloat);
	DYprev = prepareComputeTexture(ptr2 + 2 * pyramidSize, core.w, core.pyramidRowCount, core.strideFloat);
}

void ComputeTextures::destroy() {
	cudaDestroyTextureObject(Ycur);
	cudaDestroyTextureObject(Yprev);
	cudaDestroyTextureObject(DXprev);
	cudaDestroyTextureObject(DYprev);
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
template <class T> void writeDeviceDataToFile(const T* devData, size_t h, size_t wCount, size_t strideFloatN, const std::string& path) {
	std::vector<T> hostData(h * wCount);
	cudaMemcpy2D(hostData.data(), sizeof(T) * wCount, devData, sizeof(T) * strideFloatN, sizeof(T) * wCount, h, cudaMemcpyDeviceToHost);
	std::ofstream file(path, std::ios::binary);
	file.write(reinterpret_cast<char*>(&h), sizeof(size_t));
	file.write(reinterpret_cast<char*>(&wCount), sizeof(size_t));
	size_t sizT = sizeof(T);
	file.write(reinterpret_cast<char*>(&sizT), sizeof(size_t));
	file.write(reinterpret_cast<char*>(hostData.data()), hostData.size() * sizeof(T));
}

//write string into image given by device pointer
void writeText(const std::string& text, int x0, int y0, int scaleX, int scaleY, float* deviceData, const CudaData& core) {
	//create Image<float>
	int imh = 10 * scaleY;
	int siz = imh * core.strideFloat;
	ImagePlanar<float> im(imh, core.w, core.strideFloatN, 3);

	//copy three horizontal stripes into host memory
	for (size_t z = 0; z < 3; z++) {
		float* src = deviceData + (y0 + z * core.h) * core.strideFloatN;
		float* dst = im.plane(z);
		cudaMemcpy(dst, src, siz, cudaMemcpyDefault);
	}

	//write text
	im.writeText(text, x0, 0, scaleX, scaleY, ColorNorm::WHITE, ColorNorm::BLACK); //write into host memory

	//copy YUV planes back into device memory
	for (size_t z = 0; z < 3; z++) {
		float* src = im.plane(z);
		float* dst = deviceData + (y0 + z * core.h) * core.strideFloatN;
		cudaMemcpy(dst, src, siz, cudaMemcpyDefault);
	}
}

int align(size_t base, size_t alignment) {
	return (int) ((base + alignment - 1) / alignment * alignment);
}


//----------------------------------
//-------- INIT
//----------------------------------

//check for cuda runtime installation, this only needs link to cudart_static.lib
std::vector<cudaDeviceProp> cudaProbeRuntime(CudaInfo& cudaInfo) {
	//do not check cudaError_t here, absence of cuda will report error "CUDA driver is insufficient for CUDA runtime version"
	cudaRuntimeGetVersion(&cudaInfo.cudaRuntimeVersion);
	cudaDriverGetVersion(&cudaInfo.cudaDriverVersion);

	//if we found a proper cuda installation, ask for list of devices
	int deviceCount = 0;
	std::vector<cudaDeviceProp> props;
	if (cudaInfo.cudaRuntimeVersion > 0) {
		handleStatus(cudaGetDeviceCount(&deviceCount), "error probing cuda devices");
		for (int i = 0; i < deviceCount; i++) {
			cudaDeviceProp devProp;
			handleStatus(cudaGetDeviceProperties(&devProp, i), "error getting device properties");
			props.push_back(devProp);
		}

		//query npp version numbers, this loads nvcuda.dll
		//const NppLibraryVersion* libVer = nppGetLibVersion(); //nppc.lib
		//cudaInfo.nppMajor = libVer->major;
		//cudaInfo.nppMinor = libVer->minor;
		//cudaInfo.nppBuild = libVer->build;
	}
	return props;
}

void cudaInit(CudaData& core, int devIdx, const cudaDeviceProp& prop, ImageYuv& yuvFrame) {
	//pin memory of transfer object
	registeredMemPtr = yuvFrame.data();
	handleStatus(cudaHostRegister(registeredMemPtr, yuvFrame.dataSizeInBytes(), cudaHostRegisterDefault), "error @init #2");

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
		+ 6 * 1		//Apiv
		+ 6 * 1		//b
		+ 6 * 1     //eta
		+ 3 * 3  	//wp
		+ 3 * 3     //dwp;
		;
	core.computeSharedMem = doublesCount * sizeof(double);

	//compute kernel configuration
	core.computeBlocks = { core.ixCount, core.iyCount };
	int rows = std::max(core.iw, 6);
	int ws = prop.warpSize;
	core.computeThreads = { ws / rows, rows };

	//allocate storage for compute timings
	handleStatus(cudaMalloc(&d_kernelTimer, sizeof(KernelTimer) * core.computeBlocks.x * core.computeBlocks.y), "error @int #5");
	
	//determine memory requirements
	//size_t texAlign = prop.texturePitchAlignment;
	size_t pitch = 0;
	uchar* d_ptr8;
	cudaMallocPitch(&d_ptr8, &pitch, w, 1);
	core.strideChar = (int) pitch;
	cudaFree(d_ptr8);

	float* d_ptr32;
	cudaMallocPitch(&d_ptr32, &pitch, w * sizeof(float), 1);
	core.strideFloat = (int) pitch;
	core.strideFloatN = core.strideFloat / sizeof(float);
	cudaFree(d_ptr32);

	float4* d_ptr128;
	cudaMallocPitch(&d_ptr128, &pitch, w * sizeof(float4), 1);
	core.strideFloat4 = (int) pitch;
	core.strideFloat4N = core.strideFloat4 / sizeof(float4);
	cudaFree(d_ptr128);

	//compute required heap size
	size_t frameSize8 = 3ull * core.strideChar * h;		//bytes for yuv444 images
	size_t heapRequired = 0;
	heapRequired += frameSize8 * core.bufferCount;		//yuv input storage
	heapRequired += frameSize8;						    //yuv out
	heapRequired += frameSize8;						    //rgb out
	heapRequired += 2ull * core.strideFloat * h;        //filter buffers
	heapRequired += 3ull * core.strideFloat * h * core.pyramidLevels * core.pyramidCount;		//pyramid mit Y, DX, DY
	heapRequired += 1ull * core.strideFloat4 * h * core.outBufferCount;						    //output buffer in floats
	heapRequired += sizeof(PointResult) * core.resultCount;									    //array of results structure
	heapRequired += 10 * 1024 * 1024;

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

	//allocate frameResult array on device
	allocSafe(&d_results, sizeof(PointResult) * core.resultCount);
	//allocate output yuv array
	allocSafe(&d_yuvOut, frameSize8);
	allocSafe(&d_rgb, 3ull * w * h);

	//allocate memory for yuv input data in char format [0..255]
	allocSafe(&d_yuvData, frameSize8 * core.bufferCount);
	allocDeviceIndices(&d_yuvRows, d_yuvData, core.strideChar, yuvRowCount);
	allocDeviceIndices(&d_yuvPlanes, d_yuvRows, h, core.bufferCount * 3ull);

	//allocate float buffers
	allocSafe(&out.data, core.strideFloat4 * h * core.outBufferCount);
	//name individual parts for convenience
	size_t outSize = h * core.strideFloat4N;
	out.start = out.data;
	out.warped = out.start + outSize;
	out.filterH = out.warped + outSize;
	out.filterV = out.filterH + outSize;
	out.final = out.filterV + outSize;
	out.background = out.final + outSize;

	//float filter buffers
	allocSafe(&d_bufferH, core.strideFloat * h);
	allocSafe(&d_bufferV, core.strideFloat * h);

	//initialize background color in output buffer
	float4 bgval = { core.bgcol_yuv.colors[0], core.bgcol_yuv.colors[1], core.bgcol_yuv.colors[2] };
	std::vector<float4> bg(w * h, bgval);
	//write to static background
	size_t siz = w * sizeof(float4);
	handleStatus(cudaMemcpy2D(out.background, core.strideFloat4, bg.data(), siz, siz, h, cudaMemcpyDefault), "error @init 60");
	//write to first image
	handleStatus(cudaMemcpy2D(out.warped, core.strideFloat4, bg.data(), siz, siz, h, cudaMemcpyDefault), "error @init 61");

	//allocate image pyramids, all the same strided width but increasingly shorter
	//number of rows through all three pyramids, Y, DX, DY
	size_t pyrTotalRows = 3ull * core.pyramidRowCount * core.pyramidCount;
	allocSafe(&d_pyrData, core.strideFloat * pyrTotalRows);
	allocDeviceIndices(&d_pyrRows, d_pyrData, core.strideFloatN, pyrTotalRows);

	//set up cuda streams
	for (size_t i = 0; i < cs.size(); i++) {
		handleStatus(cudaStreamCreate(&cs[i]), "error @init #70");
	}

	//set up compute kernel
	allocSafe(&d_interrupt, 1);
	allocSafe(&d_computed, 1ll * core.ixCount * core.iyCount);

	//memory statistics
	handleStatus(cudaMemGetInfo(&memfree2, &memtotal), "error @init #80");
	core.cudaMemTotal = memtotal;
	core.cudaUsedMem = memfree1 - memfree2;

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
void cudaReadFrame(int64_t frameIdx, const CudaData& core, const ImageYuv& inputFrame) {
	int64_t fr = frameIdx % core.bufferCount;
	size_t frameSizeBytes = 3ull * core.strideChar * core.h;
	unsigned char* d_frame = d_yuvData + fr * frameSizeBytes;
	handleStatus(cudaMemcpy2D(d_frame, core.strideChar, inputFrame.data(), inputFrame.stride, core.w, 3ull * core.h, cudaMemcpyDefault), "error @read #10");
	handleStatus(cudaGetLastError(), "error @read #20");
}


//----------------------------------
//-------- PYRAMID
//----------------------------------

//create image pyramid for Y, DX, DY
void cudaCreatePyramid(int64_t frameIdx, const CudaData& core) {
	int w = core.w;
	int h = core.h;
	int64_t frIdx = frameIdx % core.bufferCount;
	unsigned char* yuvStart = d_yuvData + frIdx * core.strideChar * h * 3; //get to the start of this yuv image

	int64_t pyrIdx = frameIdx % core.pyramidCount;
	float* pyrStart = d_pyrData + pyrIdx * core.pyramidRowCount * 3 * core.strideFloatN; //get to the start of this pyramid
	size_t planeOffset = 1ull * core.strideFloatN * core.pyramidRowCount;

	//first level of pyramid
	//Y data
	cu::scale_8u32f(yuvStart, core.strideChar, pyrStart, core.strideFloatN, w, h);
	//DX data
	cu::filter_32f_h(pyrStart, pyrStart + planeOffset, core.strideFloatN, w, h, 3);
	//DY data
	cu::filter_32f_v(pyrStart, pyrStart + planeOffset * 2, core.strideFloatN, w, h, 3);

	//lower levels
	float* pyrNext = pyrStart + 1ull * core.strideFloatN * h;
	for (int z = 1; z <= core.zMax; z++) {
		cu::filter_32f_h(pyrStart, d_bufferH, core.strideFloatN, w, h, 0);
		cu::filter_32f_v(d_bufferH, d_bufferV, core.strideFloatN, w, h, 0);
		cu::remap_downsize_32f(d_bufferV, core.strideFloatN, pyrNext, core.strideFloatN, w, h);
		w /= 2;
		h /= 2;
		pyrStart = pyrNext;
		pyrNext += 1ull * core.strideFloatN * h;
		//DX
		cu::filter_32f_h(pyrStart, pyrStart + planeOffset, core.strideFloatN, w, h, 3);
		//DY
		cu::filter_32f_v(pyrStart, pyrStart + planeOffset * 2, core.strideFloatN, w, h, 3);
	}

	handleStatus(cudaGetLastError(), "error @pyramid");
}


//----------------------------------
//-------- COMPUTE
//----------------------------------

void cudaCompute1(int64_t frameIdx, const CudaData& core, const cudaDeviceProp& props) {
	assert(frameIdx > 0 && "invalid pyramid index");
	int64_t pyrIdx = frameIdx % core.pyramidCount;
	int64_t pyrIdxPrev = (frameIdx - 1) % core.pyramidCount;

	assert(checkKernelParameters(core, props) && "invalid kernel parameters");
	compTex.create(pyrIdx, pyrIdxPrev, core);
	//reset computed flags
	handleStatus(cudaMemsetAsync(d_computed, 0, 1ll * core.ixCount * core.iyCount, cs[0]), "error @compute #20");
	ComputeKernelParam param = { 
		core.computeBlocks, 
		core.computeThreads, 
		core.computeSharedMem, 
		cs[0], 
		&debugData, 
		d_kernelTimer, 
		frameIdx, 
		d_interrupt, 
		d_computed 
	};
	kernelComputeCall(param, compTex, d_results);

	//cudaStreamQuery(cs[0]);
	handleStatus(cudaGetLastError(), "error @compute #20");
}

void cudaCompute2(int64_t frameIdx, const CudaData& core) {
	//reset interrupt signal
	handleStatus(cudaMemsetAsync(d_interrupt, 0, sizeof(char), cs[1]), "error @compute #50");
	//restart kernel
	ComputeKernelParam param = {
		core.computeBlocks,
		core.computeThreads,
		core.computeSharedMem,
		cs[0],
		&debugData,
		d_kernelTimer,
		frameIdx,
		d_interrupt,
		d_computed
	};
	kernelComputeCall(param, compTex, d_results);
	handleStatus(cudaGetLastError(), "error @compute #30");
}

void cudaComputeTerminate(const CudaData& core, std::vector<PointResult>& results) {
	//handleStatus(cudaMemcpyAsync(results.data(), d_results, sizeof(PointResult) * results.size(), cudaMemcpyDefault, cs1), "error @compute #40", err);
	handleStatus(cudaMemcpy(results.data(), d_results, sizeof(PointResult) * results.size(), cudaMemcpyDefault), "error @compute #100");
	compTex.destroy();
	handleStatus(cudaGetLastError(), "error @compute #100");
}


//----------------------------------
//-------- OUTPUT
//----------------------------------

void cudaOutput(int64_t frameIdx, const CudaData& core, OutputContext outCtx, std::array<double, 6> trf) {
	//ConsoleTimer timer;
	//interrupt compute kernel
	handleStatus(cudaMemsetAsync(d_interrupt, 1, sizeof(char), cs[1]), "error @output #10");

	int h = core.h;
	int w = core.w;
	int64_t fr = frameIdx % core.bufferCount;

	//size of all pixel data in bytes in yuv including padding
	size_t frameSize8 = 3ull * core.strideChar * h;
	//start of input yuv data
	unsigned char* yuvSrc = d_yuvData + fr * frameSize8;

	cu::scale_8u32f_3(yuvSrc, core.strideChar, out.start, core.strideFloat4N, w, h, cs[1]);
	//fill static background when requested
	if (core.bgmode == BackgroundMode::COLOR) {
		cu::copy_32f_3(out.background, core.strideFloat4N, out.warped, core.strideFloat4N, w, h, cs[1]);
	}
	//warp input
	cu::Affine cutrf = { trf[0], trf[1], trf[2], trf[3], trf[4], trf[5] };
	cu::warp_back_32f_3(out.start, core.strideFloat4N, out.warped, core.strideFloat4N, w, h, cutrf, cs[1]);
	//first filter pass
	cu::filter_32f_h_3(out.warped, out.filterH, core.strideFloat4N, w, h, cs[1]);
	//second filter pass
	cu::filter_32f_v_3(out.filterH, out.filterV, core.strideFloat4N, w, h, cs[1]);
	//combine unsharp mask
	cu::unsharp_32f_3(out.warped, out.filterV, out.final, core.strideFloat4N, w, h, cs[1]);
	//blend input frame on top of output when requested
	const BlendInput& bi = core.blendInput;
	if (bi.blendWidth > 0) {
		cu::copy_32f_3(out.start + bi.blendStart, core.strideFloat4N, out.final + bi.blendStart, core.strideFloat4N, bi.blendWidth, h, cs[1]);
		cu::copy_32f_3(out.background + bi.separatorStart, core.strideFloat4N, out.final + bi.separatorStart, core.strideFloat4N, bi.separatorWidth, h, cs[1]);
	}
	//output to host
	if (outCtx.encodeCpu) {
		cu::outputHost(out.final, core.strideFloat4N, d_yuvOut, core.strideChar, w, h, cs[1]);
		ImageYuv* im = outCtx.outputFrame;
		cu::copy_32f_3(d_yuvOut, core.strideChar, im->data(), im->stride, w, h * 3, cs[1]);
		outCtx.outputFrame->frameIdx = frameIdx;
	}
	//output to nvenc
	if (outCtx.encodeCuda) {
		cu::outputNvenc(out.final, core.strideFloat4N, outCtx.cudaNv12ptr, outCtx.cudaPitch, w, h, cs[1]);
	}

	//writeText(std::to_string(frameIdx), 10, 10, 2, 3, bufferFrames[18], core);

	handleStatus(cudaStreamSynchronize(cs[1]), "error @output #99");
	handleStatus(cudaGetLastError(), "error @output #100");
}

void encodeNvData(const std::vector<unsigned char>& nv12, unsigned char* nvencPtr) {
	handleStatus(cudaMemcpy(nvencPtr, nv12.data(), nv12.size(), cudaMemcpyHostToDevice), "error @simple encode #1 cannot copy to device");
}

void getNvData(std::vector<unsigned char>& nv12, OutputContext outCtx) {
	handleStatus(cudaMemcpy(nv12.data(), outCtx.cudaNv12ptr, nv12.size(), cudaMemcpyDeviceToHost), "error getting nv12 data");
}


void cudaGetTransformedOutput(float* warpedData, const CudaData& core) {
	std::vector<float4> data(1ull * core.w * core.h);
	size_t wbytes = core.w * sizeof(float4);
	cudaMemcpy2D(data.data(), wbytes, out.warped, core.strideFloat4, wbytes, core.h, cudaMemcpyDefault);

	for (int i = 0; i < core.w * core.h; i++) {
		warpedData[i] = data[i].x;
		warpedData[i + core.w * core.h] = data[i].y;
		warpedData[i + core.w * core.h * 2] = data[i].z;
	}
}

void cudaGetPyramid(float* pyramid, size_t idx, const CudaData& core) {
	size_t pyrIdx = idx % core.pyramidCount;
	float* devptr = d_pyrData + pyrIdx * core.pyramidRowCount * 3 * core.strideFloatN;
	size_t wbytes = core.w * sizeof(float);
	cudaMemcpy2D(pyramid, wbytes, devptr, core.strideFloat, wbytes, core.pyramidRowCount * 3ull, cudaMemcpyDefault);
}

ImageYuv cudaGetInput(int64_t index, const CudaData& core) {
	ImageYuv out(core.h, core.w, core.w);
	int64_t fr = index % core.bufferCount;
	//start of input yuv data
	unsigned char* yuvSrc = d_yuvData + fr * 3 * core.h * core.strideChar;
	//copy 2D data without stride
	cudaMemcpy2D(out.data(), out.w, yuvSrc, core.strideChar, out.w, 3ll * out.h, cudaMemcpyDefault);
	return out;
}

void cudaGetCurrentInputFrame(ImagePPM& image, const CudaData& core, int64_t idx) {
	int fridx = idx % core.bufferCount;
	unsigned char* yuvSrc = d_yuvData + fridx * 3ull * core.h * core.strideChar;
	cu::yuv_to_rgb(yuvSrc, core.strideChar, d_rgb, core.w, core.w, core.h);
	handleStatus(cudaMemcpy(image.data(), d_rgb, 3ull * core.w * core.h, cudaMemcpyDefault), "error @progress input");
}

void cudaGetCurrentOutputFrame(ImagePPM& image, const CudaData& core) {
	cu::yuv_to_rgb(out.warped, core.strideFloat4N, d_rgb, core.w, core.w, core.h);
	handleStatus(cudaMemcpy(image.data(), d_rgb, 3ull * core.w * core.h, cudaMemcpyDefault), "error @progress output");
}


//----------------------------------
//-------- SYNCHRONIZE
//----------------------------------

void cudaSynchronize() {
	handleStatus(cudaDeviceSynchronize(), "error @synchronize #10");
	handleStatus(cudaGetLastError(), "error @synchronize #20");
}


//----------------------------------
//-------- SHUTDOWN
//----------------------------------

DebugData cudaShutdown(const CudaData& core) {
	//get debug data
	std::vector<double> outDebug(debugData.maxSize / sizeof(double));
	handleStatus(cudaMemcpy(outDebug.data(), debugData.d_data, debugData.maxSize, cudaMemcpyDeviceToHost), "error @shutdown #5 copy debug data");

	//get image of kernel timing values
	int siz = core.computeBlocks.x * core.computeBlocks.y;
	std::vector<KernelTimer> kernelTimer(siz);
	int h = (int) kernelTimer.size();
	int w = 8'000;
	ImageBGR kernelTimerImage(h, w);
	
	handleStatus(cudaMemcpy(kernelTimer.data(), d_kernelTimer, sizeof(KernelTimer) * kernelTimer.size(), cudaMemcpyDefault), "error @shutdown #110");

	auto fcnMin = [] (KernelTimer& kt1, KernelTimer& kt2) { return kt1.timeStart < kt2.timeStart; };
	auto minTime = std::min_element(kernelTimer.begin(), kernelTimer.end(), fcnMin);
	auto fcnMax = [] (KernelTimer& kt1, KernelTimer& kt2) { return kt1.timeStop < kt2.timeStop; };
	auto maxTime = std::max_element(kernelTimer.begin(), kernelTimer.end(), fcnMax);
	int64_t delta = maxTime->timeStop - minTime->timeStart;
	if (delta > 0) {
		double f = delta / (w - 1.0);
		for (int i = 0; i < core.computeBlocks.x * core.computeBlocks.y; i++) {
			KernelTimer& kt = kernelTimer[i];
			int t1 = int((kt.timeStart - minTime->timeStart) / f);
			int t2 = int((kt.timeStop - minTime->timeStart) / f);
			for (int k = t1; k <= t2; k++) {
				kernelTimerImage.at(0, i, k) = 255;
			}
		}
	}
	
	//delete device memory
	void* d_arr[] = { d_results, d_yuvOut, d_rgb, d_yuvData, d_yuvRows, d_yuvPlanes, 
		out.data, d_bufferH, d_bufferV, d_pyrData, d_pyrRows, 
		debugData.d_data, d_kernelTimer, d_interrupt, d_computed 
	};

	for (void* ptr : d_arr) {
		handleStatus(cudaFree(ptr), "error @shutdown #10 shutting down memory");
	}

	//delete streams
	for (int i = 0; i < cs.size(); i++) {
		handleStatus(cudaStreamDestroy(cs[i]), "error @shutdown #20 shutting down streams");
	}

	//unregister memory
	handleStatus(cudaHostUnregister(registeredMemPtr), "error @shutdown #30 unregister");

	//do not reset device while nvenc is still active
	//handleStatus(cudaDeviceReset(), "error @shutdown #90", errorList);
	handleStatus(cudaGetLastError(), "error @shutdown #100");

	return { outDebug, kernelTimerImage };
}
