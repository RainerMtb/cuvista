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

#include "cuNPP.cuh"
#include "cuFilterKernel.cuh"
#include "Util.hpp"

#include <chrono>
#include <iostream>

struct KernelContext {
	cudaTextureObject_t texture;
	dim3 blocks;
	dim3 threads;
	cudaError_t status;
};

//declare here to mitigate red underlines
template<class T> __device__ T tex2D(cudaTextureObject_t tex, float x, float y);
template<class T> __device__ T tex2Dgather(cudaTextureObject_t tex, float x, float y, int comp);

KernelContext prepareTextureObject(cudaResourceDesc resDesc, cudaTextureDesc texDesc, int thrw, int thrh) {
	// Create texture object
	cudaTextureObject_t texObj;
	cudaError_t status = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	//kernel configuration
	dim3 threads = { cu::THREAD_COUNT, cu::THREAD_COUNT };
	uint bx = (thrw + threads.x - 1) / threads.x;
	uint by = (thrh + threads.y - 1) / threads.y;
	dim3 blocks = { bx, by };

	return { texObj, blocks, threads, status };
}

//KernelContext prepareTextureArray(float* src, int srcStep, int texw, int texh, int thrw, int thrh) {
//	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//	cudaError_t status = cudaMallocArray(&d_array, &channelDesc, texw, texh, cudaArrayTextureGather | cudaArraySurfaceLoadStore);
//	status = cudaMemcpy2DToArray(d_array, 0, 0, src, srcStep, texw * sizeof(float), texh, cudaMemcpyDefault);
//	cudaResourceDesc resDesc {};
//	resDesc.resType = cudaResourceTypeArray;
//	resDesc.res.array.array = d_array;
//
//	cudaTextureDesc texDesc {};
//	texDesc.addressMode[0] = cudaAddressModeClamp;
//	texDesc.addressMode[1] = cudaAddressModeClamp;
//	texDesc.filterMode = cudaFilterModeLinear;
//
//	return prepareTextureObject(resDesc, texDesc, thrw, thrh);
//}

//create texture for reading data in kernel
template <class T> KernelContext prepareTexture(T* src, int srcStep, int texw, int texh, int thrw, int thrh) {
	cudaResourceDesc resDesc {};
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = src;
	resDesc.res.pitch2D.width = texw;
	resDesc.res.pitch2D.height = texh;
	resDesc.res.pitch2D.pitchInBytes = srcStep;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();

	// Specify texture object parameters
	cudaTextureDesc texDesc {};
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;

	return prepareTextureObject(resDesc, texDesc, thrw, thrh);
}

template <class T> KernelContext prepareTexture(T* src, int srcStep, int texw, int texh) {
	return prepareTexture(src, srcStep, texw, texh, texw, texh);
}


//--------------- kernels

__device__ void yuv_to_rgb_func(float yf, float uf, float vf, uchar* r, uchar* g, uchar* b) {
	*r = (uchar) cu::clamp(yf + (1.370705f * (vf - 128.0f)), 0.0f, 255.0f);
	*g = (uchar) cu::clamp(yf - (0.337633f * (uf - 128.0f)) - (0.698001f * (vf - 128.0f)), 0.0f, 255.0f);
	*b = (uchar) cu::clamp(yf + (1.732446f * (uf - 128.0f)), 0.0f, 255.0f);
}

__device__ float tex2Dinterp(cudaTextureObject_t tex, float dx, float dy) {
	float f00 = tex2D<float>(tex, dx, dy);
	float f01 = tex2D<float>(tex, dx + 1, dy);
	float f10 = tex2D<float>(tex, dx, dy + 1);
	float f11 = tex2D<float>(tex, dx + 1, dy + 1);
	dx = dx - floorf(dx);
	dy = dy - floorf(dy);
	return (1.0f - dx) * (1.0f - dy) * f00 + (1.0f - dx) * dy * f10 + dx * (1.0f - dy) * f01 + dx * dy * f11;
}

__global__ void kernel_scale_8u32f(cudaTextureObject_t texObj, cuMatf dest) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	constexpr float f = 1.0f / 255.0f;

	if (x < dest.w && y < dest.h) {
		uchar val = tex2D<uchar>(texObj, x, y);
		dest.at(y, x) = val * f;
	}
}

__global__ void kernel_scale_32f8u(cudaTextureObject_t texObj, cuMatc dest) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < dest.w && y < dest.h) {
		float val = tex2D<float>(texObj, x, y);
		dest.at(y, x) = (uchar) roundf(val * 255.0f);
	}
}

__global__ void kernel_warp_back(cudaTextureObject_t texObj, cuMatf dest, cu::Affine trf) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	float xx = (float) (x * trf.m00 + y * trf.m01 + trf.m02);
	float yy = (float) (x * trf.m10 + y * trf.m11 + trf.m12);
	if (x < dest.w && y < dest.h && xx >= 0.0f && xx <= dest.w - 1 && yy >= 0.0f && yy <= dest.h - 1) {
		//dest.at(y, x) = tex2D<float>(texObj, xx + 0.5f, yy + 0.5f); //linear interpolation
		dest.at(y, x) = tex2Dinterp(texObj, xx, yy);
	}
}

__global__ void kernel_unsharp(cudaTextureObject_t texBase, cudaTextureObject_t texGauss, cuMatf dest, float factor) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < dest.w && y < dest.h) {
		float base = tex2D<float>(texBase, x, y);
		float gauss = tex2D<float>(texGauss, x, y);
		dest.at(y, x) = __saturatef(base + (base - gauss) * factor);
	}
}

__global__ void kernel_remap_downsize(cudaTextureObject_t texObj, cuMatf dest, cuMatf srcMat) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < dest.w && y < dest.h) {
		//dest.at(y, x) = tex2D<float>(texObj, x * 2 + 1, y * 2 + 1); //linear interpolation
		dest.at(y, x) = tex2Dinterp(texObj, x * 2.0f + 0.5f, y * 2.0f + 0.5f);
	}
}

__global__ void kernel_uv_to_nv12(cudaTextureObject_t texObj, uchar* nvencPtr, int cudaPitch, int w, int h) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w / 2 && y < h / 2) {
		//u
		float val = tex2D<float>(texObj, x * 2 + 1, y * 2 + 1);
		nvencPtr[cudaPitch * y + x * 2] = (uchar) (val * 255.0f);
		//v
		val = tex2D<float>(texObj, x * 2 + 1, y * 2 + h + 1);
		nvencPtr[cudaPitch * y + x * 2 + 1] = (uchar) (val * 255.0f);
	}
}

__global__ void kernel_filter_horizontal(cudaTextureObject_t texObj, cuMatf dest, size_t filterKernelIndex) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const FilterKernel& kernel = getFilterKernel(filterKernelIndex);

	if (x < dest.w && y < dest.h) {
		float result = 0.0f;
		for (int i = 0; i < kernel.siz; i++) {
			int ix = x - kernel.siz / 2 + i;
			result += kernel.k[i] * tex2D<float>(texObj, ix, y);
		}
		dest.at(y, x) = result;
	}
}

__global__ void kernel_filter_vertical(cudaTextureObject_t texObj, cuMatf dest, size_t filterKernelIndex) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const FilterKernel& kernel = getFilterKernel(filterKernelIndex);

	if (x < dest.w && y < dest.h) {
		float result = 0.0f;
		for (int i = 0; i < kernel.siz; i++) {
			int iy = y - kernel.siz / 2 + i;
			result += kernel.k[i] * tex2D<float>(texObj, x, iy);
		}
		dest.at(y, x) = result;
	}
}

__global__ void kernel_yuv8_to_rgb8(cudaTextureObject_t texObj, uchar* rgb, int w, int h) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		uchar colY = tex2D<uchar>(texObj, x, y);
		uchar colU = tex2D<uchar>(texObj, x, y + h);
		uchar colV = tex2D<uchar>(texObj, x, y + h + h);
		uchar* dest = rgb + 3ull * (y * w + x);
		yuv_to_rgb_func(colY, colU, colV, dest, dest + 1, dest + 2);
	}
}

__global__ void kernel_yuv32_to_rgb8(cudaTextureObject_t texObj, uchar* rgb, int w, int h) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		float colY = tex2D<float>(texObj, x, y);
		float colU = tex2D<float>(texObj, x, y + h);
		float colV = tex2D<float>(texObj, x, y + h + h);
		uchar* dest = rgb + 3ull * (y * w + x);
		yuv_to_rgb_func(colY * 255.0f, colU * 255.0f, colV * 255.0f, dest, dest + 1, dest + 2);
	}
}

//----------- callers

cudaError_t cu::scale_8u32f(uchar* src, int srcStep, float* dest, int destStep, int w, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatf destMat(dest, h, w, destStep);
	kernel_scale_8u32f <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, destMat);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::scale_32f8u(float* src, int srcStep, uchar* dest, int destStep, int w, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatc destMat(dest, h, w, destStep);
	kernel_scale_32f8u <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, destMat);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::copy_32f(float* src, int srcStep, float* dest, int destStep, int w, int h) {
	return cudaMemcpy2D(dest, destStep * sizeof(float), src, srcStep, w * sizeof(float), h, cudaMemcpyDeviceToDevice);
}

cudaError_t cu::warp_back_32f(float* src, int srcStep, float* dest, int destStep, int w, int h, cu::Affine trf, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatf destMat(dest, h, w, destStep);
	kernel_warp_back <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, destMat, trf);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::filter_32f_h(float* src, float* dest, int srcStep, int w, int h, size_t filterKernelIndex, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatf destMat(dest, h, w, srcStep / sizeof(float));
	kernel_filter_horizontal <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, destMat, filterKernelIndex);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::filter_32f_v(float* src, float* dest, int srcStep, int w, int h, size_t filterKernelIndex, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatf destMat(dest, h, w, srcStep / sizeof(float));
	kernel_filter_vertical << <ki.blocks, ki.threads, 0, cs >> > (ki.texture, destMat, filterKernelIndex);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::unsharp_32f(float* base, float* gauss, int srcStep, float* dest, int destStep, int w, int h, float factor, cudaStream_t cs) {
	KernelContext kiBase = prepareTexture(base, srcStep, w, h);
	KernelContext kiGauss = prepareTexture(gauss, srcStep, w, h);
	cuMatf destMat(dest, h, w, destStep);
	kernel_unsharp <<<kiBase.blocks, kiBase.threads, 0, cs>>> (kiBase.texture, kiGauss.texture, destMat, factor);
	cudaDestroyTextureObject(kiBase.texture);
	cudaDestroyTextureObject(kiGauss.texture);
	return cudaGetLastError();
}

cudaError_t cu::remap_downsize_32f(float* src, int srcStep, float* dest, int destStep, int wsrc, int hsrc) {
	int wdest = wsrc / 2;
	int hdest = hsrc / 2;
	KernelContext ki = prepareTexture(src, srcStep, wsrc, hsrc, wdest, hdest);
	cuMatf srcMat(src, hsrc, wsrc, destStep);
	cuMatf destMat(dest, hdest, wdest, destStep);
	kernel_remap_downsize <<<ki.blocks, ki.threads>>> (ki.texture, destMat, srcMat);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::uv_to_nv12(float* src, int srcStep, uchar* nvencPtr, int cudaPitch, int w, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h * 2, w / 2, h / 2);
	kernel_uv_to_nv12 <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, nvencPtr, cudaPitch, w, h);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::yuv_to_rgb(uchar* src, int srcStep, uchar* dest, int destStep, int w, int h) {
	KernelContext ki = prepareTexture(src, srcStep, w, 3 * h, w, h);
	kernel_yuv8_to_rgb8 <<<ki.blocks, ki.threads>>> (ki.texture, dest, w, h);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::yuv_to_rgb(float* src, int srcStep, uchar* dest, int destStep, int w, int h) {
	KernelContext ki = prepareTexture(src, srcStep, w, 3 * h, w, h);
	kernel_yuv32_to_rgb8 <<<ki.blocks, ki.threads>>> (ki.texture, dest, w, h);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}