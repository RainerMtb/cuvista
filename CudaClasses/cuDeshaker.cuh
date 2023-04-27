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

#include "cuDecompose.cuh"
#include "CoreData.cuh"

#include <sstream>
#include <fstream>
#include <iostream>


//pointer to data for a frame
struct DevicePointers {
	float** Ycur;		//pointer to first row of Y data for current frame
	float** Yprev;		//pointer to first row of Y data for previous frame
	float** DXprev;		//DX data for previous frame
	float** DYprev;		//DY data for previous frame

	//put memory locations from pyramid into this structure
	__host__ DevicePointers(int64_t idx, int64_t idxPrev, const CoreData& core);
	
	//move each pointer by delta rows
	__device__ void movePosition(int delta);
};

//parameters for kernel launch
struct kernelParam {
	dim3 blk;
	dim3 thr;
	size_t shdBytes;
	cudaStream_t stream;
};


void kernelComputeCall(kernelParam param, DevicePointers pointers, PointResult* d_results, int64_t frameIdx, cu::DebugData debugData);

void computeInit(const CoreData& core);

//----------------------------------
// interface to host callers
//----------------------------------


void cudaProbeRuntime(std::vector<cudaDeviceProp>& devicesList, CudaInfo& cudaInfo);

void cudaDeviceSetup(CoreData& core);

/*
@brief initialize cuda device
@param core: CoreData structure
@param yuvFrame: the object used later to transfer frame data
@param info: info about gpu memory
@param err: error info text stream
*/
void cudaInit(const CoreData& core, ImageYuv& yuvFrame);

/*
@brief read a frame into device memory
@param frameIdx: frame index to read
@param core: CoreData structure
@param inputFrame: YUV pixel data to load into device for processing
@param err: error info text stream
*/
void cudaReadFrame(int64_t frameIdx, const CoreData& core, const ImageYuv& inputFrame);

/*
@breif create image pyramid
*/
void cudaCreatePyramid(int64_t frameIdx, const CoreData& core);

/*
@brief compute displacements between frame and previous frame in video, compute resulting affine transformation
@param frameIdx: frame to compute transform for with respect to previous frame
@param core: CoreData structure
@param err: error info text stream
*/
void cudaComputeStart(int64_t frameIdx, const CoreData& core);

/*
@brief return vector of results from async computation
*/
void cudaComputeTerminate(const CoreData& core, std::vector<PointResult>& results);

/*
@brief transform a frame and output pixel data to host and/or device memory
@param frameIdx: frame to read from buffers
@param core: CoreData+
@param h_ptr: pointer to host frame for output
@param nvencPtr: pointer to device memory to writer output
@param nvencPitch: pitch value to use for encoding in cuda
@param trf: six affine transformation parameters
@param err: error info text stream
*/
void cudaOutput(int64_t frameIdx, const CoreData& core, OutputContext outReq, cu::Affine trf);

/*
@brief only encode given nv12 data
*/
void encodeNvData(std::vector<unsigned char>& nv12, unsigned char* nvencPtr);

/*
@brief get NV12 data prepared for cuda encoding
*/
void getNvData(std::vector<unsigned char>& nv12, OutputContext outReq);

/*
@brief shutdown cuda device
@return timing values and debug data
*/
void cudaShutdown(const CoreData& core, std::vector<int64_t>& timings, std::vector<double>& debugData);

/*
@brief get transformed float output for testing
*/
void cudaGetTransformedOutput(float* warpedData, const CoreData& core);

/*
@brief get pyramid image for given index
*/
void cudaGetPyramid(float* pyramid, size_t idx, const CoreData& core);

/*
@brief get input iomage from buffers
*/
ImageYuv cudaGetInput(int64_t index, const CoreData& core);

/*
@brief get current input frame for progress display
*/
void cudaGetCurrentInputFrame(ImagePPM& image, const CoreData& core, int idx);

/*
@brief get current output frame for progress display
*/
void cudaGetCurrentOutputFrame(ImagePPM& image, const CoreData& core);
