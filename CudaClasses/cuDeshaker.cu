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

float* d_bufferData;
std::vector<float*> bufferFrames; //index to one buffer frame, allocated on host

float* d_pyrData;
float** d_pyrRows;

//declare memory for index lookup during filter operations
__constant__ float constFilterKernels[] = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0.25f, 0.5f, 0.25f, -0.5f, 0.0f, 0.5f };
float* filterKernelGauss[3];
int filterKernelGaussSizes[] = { 5, 3, 3 };
float* filterKernelDifference;

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

__device__ void KernelTimer::start() {
	block = blockIdx;
	thread = threadIdx;
	cu::globaltimer(&timeStart);
}

__device__ void KernelTimer::stop() {
	cu::globaltimer(&timeStop);
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

void ComputeTextures::create(int64_t idx, int64_t idxPrev, const CoreData& core) {
	size_t pyramidSize = 1ull * core.pyramidRows * core.strideCount; //size of one full pyramid in elements
	float* ptr1 = d_pyrData + 3 * pyramidSize * idx;
	Ycur = prepareComputeTexture(ptr1, core.w, core.pyramidRows, core.strideFloatBytes);

	float* ptr2 = d_pyrData + 3 * pyramidSize * idxPrev;
	Yprev = prepareComputeTexture(ptr2, core.w, core.pyramidRows, core.strideFloatBytes);
	DXprev = prepareComputeTexture(ptr2 + pyramidSize, core.w, core.pyramidRows, core.strideFloatBytes);
	DYprev = prepareComputeTexture(ptr2 + 2 * pyramidSize, core.w, core.pyramidRows, core.strideFloatBytes);
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

bool checkKernelParameters(const CoreData& core, const cudaDeviceProp& cudaProps) {
	return checkKernelParameters(core.computeThreads, core.computeBlocks, core.computeSharedMem, cudaProps);
}

//write data from device pointer to file for debugging
template <class T> void writeDeviceDataToFile(const T* devData, size_t h, size_t wCount, size_t strideCount, const std::string& path) {
	std::vector<T> hostData(h * wCount);
	cudaMemcpy2D(hostData.data(), sizeof(T) * wCount, devData, sizeof(T) * strideCount, sizeof(T) * wCount, h, cudaMemcpyDeviceToHost);
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
	int siz = imh * core.strideFloatBytes;
	ImagePlanar<float> im(imh, core.w, core.strideCount, 3);

	//copy three horizontal stripes into host memory
	for (size_t z = 0; z < 3; z++) {
		float* src = deviceData + (y0 + z * core.h) * core.strideCount;
		float* dst = im.plane(z);
		cudaMemcpy(dst, src, siz, cudaMemcpyDefault);
	}

	//write text
	im.writeText(text, x0, 0, scaleX, scaleY, ColorNorm::WHITE, ColorNorm::BLACK); //write into host memory

	//copy YUV planes back into device memory
	for (size_t z = 0; z < 3; z++) {
		float* src = im.plane(z);
		float* dst = deviceData + (y0 + z * core.h) * core.strideCount;
		cudaMemcpy(dst, src, siz, cudaMemcpyDefault);
	}
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

void cudaInit(CoreData& core, int devIdx, const cudaDeviceProp& prop, ImageYuv& yuvFrame) {
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

	//get stride values for byte and float data
	void* d_ptr = nullptr;
	size_t pitchval;
	handleStatus(cudaMallocPitch(&d_ptr, &pitchval, core.w, 1), "error @init 6 probing pitch value");
	handleStatus(cudaFree(d_ptr), "error @init 7 freeing memory");

	handleStatus(cudaMallocPitch(&d_ptr, &pitchval, core.w * sizeof(float), 1), "error @init 8 probing float stride value");
	core.strideFloatBytes = (int) pitchval;
	handleStatus(cudaFree(d_ptr), "error @init 9 freeing memory");
	core.strideCount = core.strideFloatBytes / sizeof(float);
	core.cudapitch = (int) pitchval;

	//set memory limit
	size_t heap = 0;
	size_t heapRequired = 0;
	handleStatus(cudaDeviceGetLimit(&heap, cudaLimitMallocHeapSize), "error @init #10");

	size_t frameSize8 = 3ull * core.cudapitch * h;			//bytes for yuv444 images
	heapRequired += frameSize8 * core.bufferCount;		//yuv input storage
	heapRequired += frameSize8 * 2;						//yuv out
	heapRequired += 3ull * core.strideFloatBytes * h * (core.zMax + 1ull) * core.pyramidCount;		//pyramid mit Y, DX, DY
	heapRequired += 1ull * core.strideFloatBytes * h * core.BUFFER_COUNT;						    //output buffer in floats
	heapRequired += sizeof(PointResult) * core.resultCount;										    //array of results structure
	heapRequired += 10 * 1024 * 1024;

	if (heapRequired < heap)
		handleStatus(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapRequired), "error @init #20");

	size_t yuvRowCount = 3ull * h * core.bufferCount;
	size_t memtotal, memfree1, memfree2;
	handleStatus(cudaMemGetInfo(&memfree1, &memtotal), "error @init #30");

	//allocate debug storage
	allocSafe(&debugData.d_data, debugData.maxSize);
	handleStatus(cudaMemset(debugData.d_data, 0, debugData.maxSize), "error @init #32");

	//setup filter kernel pointers
	float* constKernels;
	float** symbolAddress = &constKernels;
	handleStatus(cudaGetSymbolAddress((void**) (symbolAddress), constFilterKernels), "error @init #40");
	filterKernelGauss[0] = constKernels;
	filterKernelGauss[1] = constKernels + 5;
	filterKernelGauss[2] = constKernels + 5;
	filterKernelDifference = constKernels + 8;

	//allocate frameResult array on device
	allocSafe(&d_results, sizeof(PointResult) * core.resultCount);
	//allocate output yuv array
	allocSafe(&d_yuvOut, frameSize8);
	allocSafe(&d_rgb, 3ull * w * h);

	//allocate memory for yuv input data in char format [0..255]
	allocSafe(&d_yuvData, frameSize8 * core.bufferCount);
	allocDeviceIndices(&d_yuvRows, d_yuvData, core.cudapitch, yuvRowCount);
	allocDeviceIndices(&d_yuvPlanes, d_yuvRows, h, core.bufferCount * 3ull);

	//allocate several buffers in float [0..1]
	allocSafe(&d_bufferData, core.strideFloatBytes * core.BUFFER_COUNT * h);

	bufferFrames.resize(core.BUFFER_COUNT);
	for (size_t i = 0; i < bufferFrames.size(); i++) bufferFrames[i] = d_bufferData + core.strideCount * h * i;

	//initialize background color in output buffer
	size_t siz = w * sizeof(float);
	for (size_t i = 0; i < 3; i++) {
		std::vector<float> bg(w * h, core.bgcol_yuv.colors[i]);
		//buffer 6-7-8 static blank background
		handleStatus(cudaMemcpy2D(bufferFrames[6 + i], core.strideFloatBytes, bg.data(), siz, siz, h, cudaMemcpyHostToDevice), "error @init #60");
		//buffer 9-10-11 first initialize to background
		handleStatus(cudaMemcpy2D(bufferFrames[9 + i], core.strideFloatBytes, bg.data(), siz, siz, h, cudaMemcpyHostToDevice), "error @init #61");
	}

	//allocate image pyramids, all the same strided width but increasingly shorter
	//number of rows through all three pyramids, Y, DX, DY
	size_t pyrTotalRows = 3ull * core.pyramidRows * core.pyramidCount;
	allocSafe(&d_pyrData, core.strideFloatBytes * pyrTotalRows);
	allocDeviceIndices(&d_pyrRows, d_pyrData, core.strideCount, pyrTotalRows);

	//set up cuda streams
	for (size_t i = 0; i < cs.size(); i++) {
		handleStatus(cudaStreamCreate(&cs[i]), "error @init #70");
	}

	//set up compute kernel
	computeInit(core);
	allocSafe(&d_interrupt, 1);
	allocSafe(&d_computed, 1ll * core.ixCount * core.iyCount);

	//memory statistics
	handleStatus(cudaMemGetInfo(&memfree2, &memtotal), "error @init #80");
	core.cudaMemTotal = memtotal;
	core.cudaUsedMem = memfree1 - memfree2;

	//final error checks
	handleStatus(cudaDeviceSynchronize(), "error @init #90");
	handleStatus(cudaGetLastError(), "error @init #92");
}


//----------------------------------
//-------- READ
//----------------------------------

//copy yuv input to device
void cudaReadFrame(int64_t frameIdx, const CoreData& core, const ImageYuv& inputFrame) {
	int64_t fr = frameIdx % core.bufferCount;
	size_t frameSizeBytes = 3ull * core.cudapitch * core.h;
	unsigned char* d_frame = d_yuvData + fr * frameSizeBytes;
	handleStatus(cudaMemcpy2D(d_frame, core.cudapitch, inputFrame.data(), inputFrame.stride, core.w, 3ull * core.h, cudaMemcpyDefault), "error @read copy");
	handleStatus(cudaGetLastError(), "error @read unspecified");
}


//----------------------------------
//-------- PYRAMID
//----------------------------------

//create image pyramid for Y, DX, DY
void cudaCreatePyramid(int64_t frameIdx, const CoreData& core) {
	int w = core.w;
	int h = core.h;
	int64_t frIdx = frameIdx % core.bufferCount;
	unsigned char* yuvStart = d_yuvData + frIdx * core.cudapitch * h * 3; //get to the start of this yuv image

	int64_t pyrIdx = frameIdx % core.pyramidCount;
	float* pyrStart = d_pyrData + pyrIdx * core.pyramidRows * 3 * core.strideCount; //get to the start of this pyramid
	size_t planeOffset = 1ull * core.strideCount * core.pyramidRows;

	//first level of pyramid
	//Y data
	cu::scale_8u32f(yuvStart, core.cudapitch, pyrStart, core.strideCount, w, h);
	//DX data
	cu::filter_32f(pyrStart, pyrStart + planeOffset, core.strideFloatBytes, w, h, filterKernelDifference, 3, cu::FilterDim::FILTER_HORIZONZAL);
	//DY data
	cu::filter_32f(pyrStart, pyrStart + planeOffset * 2, core.strideFloatBytes, w, h, filterKernelDifference, 3, cu::FilterDim::FILTER_VERTICAL);

	//lower levels
	float* pyrNext = pyrStart + 1ull * core.strideCount * h;
	for (int z = 1; z <= core.zMax; z++) {
		cu::filter_32f(pyrStart, bufferFrames[13], core.strideFloatBytes, w, h, filterKernelGauss[0], filterKernelGaussSizes[0], cu::FilterDim::FILTER_HORIZONZAL);
		cu::filter_32f(bufferFrames[13], bufferFrames[12], core.strideFloatBytes, w, h, filterKernelGauss[0], filterKernelGaussSizes[0], cu::FilterDim::FILTER_VERTICAL);
		cu::remap_downsize_32f(bufferFrames[12], core.strideFloatBytes, pyrNext, core.strideCount, w, h);
		w /= 2;
		h /= 2;
		pyrStart = pyrNext;
		pyrNext += 1ull * core.strideCount * h;
		cu::filter_32f(pyrStart, pyrStart + planeOffset, core.strideFloatBytes, w, h, filterKernelDifference, 3, cu::FilterDim::FILTER_HORIZONZAL);
		cu::filter_32f(pyrStart, pyrStart + planeOffset * 2, core.strideFloatBytes, w, h, filterKernelDifference, 3, cu::FilterDim::FILTER_VERTICAL);
	}

	handleStatus(cudaGetLastError(), "error @pyramid");
}


//----------------------------------
//-------- COMPUTE
//----------------------------------

void cudaCompute1(int64_t frameIdx, const CoreData& core, const cudaDeviceProp& props) {
	int64_t pyrIdx = frameIdx % core.pyramidCount;
	size_t pyrIdxPrev = (pyrIdx == 0 ? core.pyramidCount : pyrIdx) - 1;

	assert(checkKernelParameters(core, props) && "invalid kernel parameters");
	compTex.create(pyrIdx, pyrIdxPrev, core);
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

void cudaCompute2(int64_t frameIdx, const CoreData& core) {
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

void cudaComputeTerminate(const CoreData& core, std::vector<PointResult>& results) {
	//handleStatus(cudaMemcpyAsync(results.data(), d_results, sizeof(PointResult) * results.size(), cudaMemcpyDefault, cs1), "error @compute #40", err);
	handleStatus(cudaMemcpy(results.data(), d_results, sizeof(PointResult) * results.size(), cudaMemcpyDefault), "error @compute #100");
	compTex.destroy();
	handleStatus(cudaGetLastError(), "error @compute #100");
}


//----------------------------------
//-------- OUTPUT
//----------------------------------

/*
buffer frames usage, each frame holds one plane Y, U, V in float format
	0,  1,  2: input converted from yuv
	3,  4,  5: start with background, then put warped data from input there
	6,  7,  8: background color
	9, 10, 11: background to use for blending next frame
	12, 13, 14: temporary buffer for gauss filtering
	15, 16, 17: gauss filter result
	18, 19, 20: output to encoder
*/
void cudaOutput(int64_t frameIdx, const CoreData& core, OutputContext outCtx, cu::Affine trf) {
	//interrupt compute kernel
	handleStatus(cudaMemsetAsync(d_interrupt, 1, sizeof(char), cs[1]), "error @output #10 memset");

	int h = core.h;
	int w = core.w;
	int64_t fr = frameIdx % core.bufferCount;

	//size of all pixel data in bytes in yuv including padding
	size_t frameSize8 = 3ull * core.cudapitch * h;
	//start of input yuv data
	unsigned char* yuvSrc = d_yuvData + fr * frameSize8;

	//handle three frames at once, convert input image from 8bit to float
	cu::scale_8u32f(yuvSrc, core.cudapitch, bufferFrames[0], core.strideCount, w, h * 3, cs[1]);

	//handle individual frames
	const BlendInput& bi = core.blendInput;
	for (size_t i = 0; i < 3; i++) {
		float* in = bufferFrames[i];
		float* bg = bufferFrames[6 + i];
		float* warped = bufferFrames[9 + i];
		float* temp = bufferFrames[12 + i];
		float* buffer = bufferFrames[15 + i];
		float* out = bufferFrames[18 + i];

		//reset background with static color if requested
		if (core.bgmode == BackgroundMode::COLOR) {
			cu::copy_32f(bg, core.strideFloatBytes, warped, core.strideCount, w, h);
		}
		//transform input on top of background
		cu::warp_back_32f(in, core.strideFloatBytes, warped, core.strideCount, w, h, trf, cs[1]);
		//first filter pass
		cu::filter_32f(warped, temp, core.strideFloatBytes, w, h, filterKernelGauss[i], filterKernelGaussSizes[i], cu::FilterDim::FILTER_HORIZONZAL, cs[1]);
		//second filter pass
		cu::filter_32f(temp, buffer, core.strideFloatBytes, w, h, filterKernelGauss[i], filterKernelGaussSizes[i], cu::FilterDim::FILTER_VERTICAL, cs[1]);
		//combine unsharp mask
		cu::unsharp_32f(warped, buffer, core.strideFloatBytes, out, core.strideCount, w, h, core.unsharp[i], cs[1]);

		//blend input frame on top of output when requested
		if (bi.blendWidth > 0) {
			cu::copy_32f(in + bi.blendStart, core.strideFloatBytes, out + bi.blendStart, core.strideCount, bi.blendWidth, h);
			cu::copy_32f(bg + bi.separatorStart, core.strideFloatBytes, out + bi.separatorStart, core.strideCount, bi.separatorWidth, h);
		}
	}

	//if (frameIdx < 10) {
	//	writeDeviceDataToFile(bufferFrames[0], h, w, core.strideCount, "d:/inY-" + std::to_string(frameIdx) + ".dat");
	//	writeDeviceDataToFile(bufferFrames[9], h, w, core.strideCount, "d:/outY-" + std::to_string(frameIdx) + ".dat");
	//}

	//writeText(std::to_string(frameIdx), 10, 10, 2, 3, bufferFrames[18], core);

	//output to nvenc buffer
	if (outCtx.encodeCuda) {
		//convert Y plane
		cu::scale_32f8u(bufferFrames[18], core.strideFloatBytes, outCtx.cudaNv12ptr, outCtx.cudaPitch, w, h, cs[1]);
		//convert and interleave U and V plane
		cu::uv_to_nv12(bufferFrames[19], core.strideFloatBytes, outCtx.cudaNv12ptr + 1ull * outCtx.cudaPitch * h, outCtx.cudaPitch, w, h, cs[1]);
	}

	//output to host
	if (outCtx.encodeCpu) {
		cu::scale_32f8u(bufferFrames[18], core.strideFloatBytes, d_yuvOut, core.cudapitch, w, h * 3, cs[1]);
		ImageYuv* im = outCtx.outputFrame;
		handleStatus(cudaMemcpy2D(im->data(), im->stride, d_yuvOut, core.cudapitch, w, 3ull * h, cudaMemcpyDeviceToHost), "error @output #50 memcopy");
		outCtx.outputFrame->frameIdx = frameIdx;
	}

	handleStatus(cudaStreamSynchronize(cs[1]), "error @output #99");
	handleStatus(cudaGetLastError(), "error @output #100");
}

void encodeNvData(const std::vector<unsigned char>& nv12, unsigned char* nvencPtr) {
	handleStatus(cudaMemcpy(nvencPtr, nv12.data(), nv12.size(), cudaMemcpyHostToDevice), "error @simple encode #1 cannot copy to device");
}

void getNvData(std::vector<unsigned char>& nv12, OutputContext outCtx) {
	handleStatus(cudaMemcpy(nv12.data(), outCtx.cudaNv12ptr, nv12.size(), cudaMemcpyDeviceToHost), "error getting nv12 data");
}


void cudaGetTransformedOutput(float* warpedData, const CoreData& core) {
	size_t width = core.w * sizeof(float);
	cudaMemcpy2D(warpedData, width, bufferFrames[9], core.strideFloatBytes, width, core.h * 3ll, cudaMemcpyDefault);
}

void cudaGetPyramid(float* pyramid, size_t idx, const CoreData& core) {
	size_t pyrIdx = idx % core.pyramidCount;
	float* devptr = d_pyrData + pyrIdx * core.pyramidRows * 3 * core.strideCount;
	size_t wbytes = core.w * sizeof(float);
	cudaMemcpy2D(pyramid, wbytes, devptr, core.strideFloatBytes, wbytes, core.pyramidRows * 3ull, cudaMemcpyDefault);
}

ImageYuv cudaGetInput(int64_t index, const CoreData& core) {
	ImageYuv out(core.h, core.w, core.w);
	int64_t fr = index % core.bufferCount;
	//start of input yuv data
	unsigned char* yuvSrc = d_yuvData + fr * 3 * core.h * core.cudapitch;
	//copy 2D data without stride
	cudaMemcpy2D(out.data(), out.w, yuvSrc, core.cudapitch, out.w, 3ll * out.h, cudaMemcpyDefault);
	return out;
}

void cudaGetCurrentInputFrame(ImagePPM& image, const CoreData& core, int idx) {
	unsigned char* yuvSrc = d_yuvData + idx * 3ll * core.h * core.cudapitch;
	cu::yuv_to_rgb(yuvSrc, core.cudapitch, d_rgb, core.strideCount, core.w, core.h);
	cudaMemcpy2D(image.data(), image.w, d_rgb, core.cudapitch, core.w, 3ull * core.h, cudaMemcpyDefault);
}

void cudaGetCurrentOutputFrame(ImagePPM& image, const CoreData& core) {
	cu::yuv_to_rgb(bufferFrames[9], core.strideFloatBytes, d_rgb, core.strideCount, core.w, core.h);
	cudaMemcpy2D(image.data(), image.w, d_rgb, core.cudapitch, core.w, 3ull * core.h, cudaMemcpyDefault);
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

DebugData cudaShutdown(const CoreData& core) {
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
	double f = (maxTime->timeStop - minTime->timeStart) / (w - 1.0);

	for (int i = 0; i < core.computeBlocks.x * core.computeBlocks.y; i++) {
		KernelTimer& kt = kernelTimer[i];
		int t1 = int((kt.timeStart - minTime->timeStart) / f);
		int t2 = int((kt.timeStop - minTime->timeStart) / f);
		for (int k = t1; k <= t2; k++) {
			kernelTimerImage.at(0, i, k) = 255;
		}
	}
	
	//delete device memory
	void* d_arr[] = { d_results, d_yuvOut, d_rgb, d_yuvData, d_yuvRows, d_yuvPlanes, 
		d_bufferData, d_pyrData, d_pyrRows, debugData.d_data, d_kernelTimer, d_interrupt, d_computed };

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
