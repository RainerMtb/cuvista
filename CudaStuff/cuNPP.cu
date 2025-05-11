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
#include "cuDeshaker.cuh"

#include <chrono>
#include <iostream>

extern __constant__ CoreData d_core;

struct KernelContext {
	cudaTextureObject_t texture;
	dim3 blocks;
	dim3 threads;
	cudaError_t status;
};

struct CudaFilterKernel {
	static const int maxSize = 8;
	int siz;
	float k[maxSize];
};

__constant__ CudaFilterKernel filterKernels[4] = {
	{5, {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f}},
	{5, {0.0f, 0.25f, 0.5f, 0.25f, 0.0f}},
	{5, {0.0f, 0.25f, 0.5f, 0.25f, 0.0f}},
	{3, {-0.5f, 0.0f, 0.5f}},
};

//declare here to mitigate red underlines
template<class T> __device__ T tex2D(cudaTextureObject_t tex, float x, float y);
template<class T> __device__ T tex2Dgather(cudaTextureObject_t tex, float x, float y, int comp);


dim3 configThreads() {
	return { cu::THREAD_COUNT, cu::THREAD_COUNT };
}

dim3 configBlocks(dim3 threads, int width, int height) {
	uint bx = (width + threads.x - 1) / threads.x;
	uint by = (height + threads.y - 1) / threads.y;
	return { bx, by };
}

//create texture for reading data in kernel
template <class T> KernelContext prepareTexture(T* src, int lineCount, int texw, int texh, int thrw, int thrh) {
	cudaResourceDesc resDesc {};
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = src;
	resDesc.res.pitch2D.width = texw;
	resDesc.res.pitch2D.height = texh;
	resDesc.res.pitch2D.pitchInBytes = lineCount * sizeof(T);
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();

	// Specify texture object parameters
	cudaTextureDesc texDesc {};
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;

	// Create texture object
	cudaTextureObject_t texObj;
	cudaError_t status = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	//kernel configuration
	dim3 threads = configThreads();
	dim3 blocks = configBlocks(threads, thrw, thrh);

	return { texObj, blocks, threads, status };
}

template <class T> KernelContext prepareTexture(T* src, int srcStep, int texw, int texh) {
	return prepareTexture(src, srcStep, texw, texh, texw, texh);
}


//--------------- kernels

__device__ void yuv_to_rgba_func(float yf, float uf, float vf, uchar* r, uchar* g, uchar* b, uchar* a) {
	*r = (uchar) rint(cu::clamp(yf + (1.370705f * (vf - 128.0f)), 0.0f, 255.0f));
	*g = (uchar) rint(cu::clamp(yf - (0.337633f * (uf - 128.0f)) - (0.698001f * (vf - 128.0f)), 0.0f, 255.0f));
	*b = (uchar) rint(cu::clamp(yf + (1.732446f * (uf - 128.0f)), 0.0f, 255.0f));
	*a = 0xFF;
}

__device__ float interp(float f00, float f01, float f10, float f11, float dx, float dy) {
	return (1.0f - dx) * (1.0f - dy) * f00 + (1.0f - dx) * dy * f10 + dx * (1.0f - dy) * f01 + dx * dy * f11;
}

__device__ float tex2Dinterp(cudaTextureObject_t tex, float dx, float dy) {
	float f00 = tex2D<float>(tex, dx, dy);
	float f01 = tex2D<float>(tex, dx + 1, dy);
	float f10 = tex2D<float>(tex, dx, dy + 1);
	float f11 = tex2D<float>(tex, dx + 1, dy + 1);
	dx = dx - floorf(dx);
	dy = dy - floorf(dy);
	return interp(f00, f01, f10, f11, dx, dy);
}

__device__ float4 tex2Dinterp_3(cudaTextureObject_t tex, float dx, float dy) {
	float4 f00 = tex2D<float4>(tex, dx, dy);
	float4 f01 = tex2D<float4>(tex, dx + 1, dy);
	float4 f10 = tex2D<float4>(tex, dx, dy + 1);
	float4 f11 = tex2D<float4>(tex, dx + 1, dy + 1);
	dx = dx - floorf(dx);
	dy = dy - floorf(dy);
	return { 
		interp(f00.x, f01.x, f10.x, f11.x, dx, dy), 
		interp(f00.y, f01.y, f10.y, f11.y, dx, dy), 
		interp(f00.z, f01.z, f10.z, f11.z, dx, dy) 
	};
}


//------------ KERNELS

__global__ void kernel_scale_8u32f_3(cudaTextureObject_t texObj, cuMatf4 dest) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	constexpr float f = 1.0f / 255.0f;

	if (x < dest.w && y < dest.h) {
		uchar yy = tex2D<uchar>(texObj, x, y);
		uchar uu = tex2D<uchar>(texObj, x, y + dest.h);
		uchar vv = tex2D<uchar>(texObj, x, y + 2 * dest.h);
		dest.at(y, x) = { yy * f, uu * f, vv * f };
	}
}

__global__ void kernel_warp_back_3(cudaTextureObject_t texObj, cuMatf4 dest, AffineCore trf) {
	double x = blockIdx.x * blockDim.x + threadIdx.x;
	double y = blockIdx.y * blockDim.y + threadIdx.y;

	//float xx = x * trf.m00 + y + trf.m01 * trf.m02;
	//float yy = x * trf.m10 + y + trf.m11 * trf.m12;
	float xx = (float) (__fma_rz(x, trf.m00, __fma_rz(y, trf.m01, trf.m02)));
	float yy = (float) (__fma_rz(x, trf.m10, __fma_rz(y, trf.m11, trf.m12)));
	if (x < dest.w && y < dest.h && xx >= 0.0f && xx <= dest.w - 1 && yy >= 0.0f && yy <= dest.h - 1) {
		//dest.at(y, x) = tex2D<float>(texObj, xx + 0.5f, yy + 0.5f); //linear interpolation does not match
		dest.at(y, x) = tex2Dinterp_3(texObj, xx, yy);
	}
}

__global__ void kernel_filter_3(cudaTextureObject_t texObj, cuMatf4 dest, int ks, int dx, int dy) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < dest.w && y < dest.h) {
		float4 result = {};
		int ix = x - dx * ks / 2;
		int iy = y - dy * ks / 2;
		for (int i = 0; i < ks; i++) {
			float4 val = tex2D<float4>(texObj, ix, iy);
			result.x += val.x * filterKernels[0].k[i];
			result.y += val.y * filterKernels[1].k[i];
			result.z += val.z * filterKernels[2].k[i];
			ix += dx;
			iy += dy;
		}
		dest.at(y, x) = result;
	}
}

__global__ void kernel_unsharp_3(float4* base, float4* gauss, float4* dest, int step, int w, int h) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		int idx = y * step + x;
		float4 valBase = base[idx];
		float4 valGauss = gauss[idx];
		dest[idx].x = __saturatef(valBase.x + (valBase.x - valGauss.x) * d_core.unsharp.y);
		dest[idx].y = __saturatef(valBase.y + (valBase.y - valGauss.y) * d_core.unsharp.u);
		dest[idx].z = __saturatef(valBase.z + (valBase.z - valGauss.z) * d_core.unsharp.v);
	}
}

__global__ void kernel_output_host(float4* src, int srcStep, uchar* dest, int destStep, int w, int h) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		float4 yuv = src[y * srcStep + x];
		int idx = y * destStep + x;
		dest[idx] = (uchar) rintf(yuv.x * 255.0f);
		idx += h * destStep;
		dest[idx] = (uchar) rintf(yuv.y * 255.0f);
		idx += h * destStep;
		dest[idx] = (uchar) rintf(yuv.z * 255.0f);
	}
}

__device__ void output(float4 val, uchar* nv12, int idx, float* u, float* v) {
	nv12[idx] = (uchar) rintf(val.x * 255.0f);
	*u += val.y;
	*v += val.z;
}

__global__ void kernel_output_nvenc(float4* src, int srcStep, uchar* cudaNv12ptr, int cudaPitch, int w, int h) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w / 2 && y < h / 2) {
		int idx = y * 2 * srcStep + x * 2;
		int outIdx = y * 2 * cudaPitch + x * 2;
		float u = 0.0f;
		float v = 0.0f;
		output(src[idx], cudaNv12ptr, outIdx, &u, &v);
		output(src[idx + 1], cudaNv12ptr, outIdx + 1, &u, &v);
		output(src[idx + w], cudaNv12ptr, outIdx + cudaPitch, &u, &v);
		output(src[idx + w + 1], cudaNv12ptr, outIdx + cudaPitch + 1, &u, &v);

		outIdx = (y + h) * cudaPitch + x * 2;
		cudaNv12ptr[outIdx] = (uchar) rintf(u / 4 * 255);
		cudaNv12ptr[outIdx + 1] = (uchar) rintf(v / 4 * 255);
	}
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
		dest.at(y, x) = (uchar) rintf(val * 255.0f);
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

__global__ void kernel_filter(cudaTextureObject_t texObj, cuMatf dest, size_t ki, int dx, int dy) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const CudaFilterKernel& kernel = filterKernels[ki];

	if (x < dest.w && y < dest.h) {
		float result = 0.0f;
		int ix = x - dx * kernel.siz / 2;
		int iy = y - dy * kernel.siz / 2;
		for (int i = 0; i < kernel.siz; i++) {
			result += tex2D<float>(texObj, ix, iy) * kernel.k[i];
			ix += dx;
			iy += dy;
		}
		dest.at(y, x) = result;
	}
}

__global__ void kernel_yuv8_to_rgba8(cudaTextureObject_t texObj, uchar* rgba, int w, int h) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		uchar colY = tex2D<uchar>(texObj, x, y);
		uchar colU = tex2D<uchar>(texObj, x, y + h);
		uchar colV = tex2D<uchar>(texObj, x, y + h + h);
		uchar* dest = rgba + 4ull * (y * w + x);
		yuv_to_rgba_func(colY, colU, colV, dest, dest + 1, dest + 2, dest + 3);
	}
}

__global__ void kernel_yuv128_to_rgba8(cudaTextureObject_t texObj, uchar* rgba, int w, int h) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		float4 yuv = tex2D<float4>(texObj, x, y);
		uchar* dest = rgba + 4ull * (y * w + x);
		yuv_to_rgba_func(yuv.x * 255.0f, yuv.y * 255.0f, yuv.z * 255.0f, dest, dest + 1, dest + 2, dest + 3);
	}
}

//----------- callers float4

cudaError_t cu::scale_8u32f_3(uchar* src, int srcStep, float4* dest, int destStep, int w, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h * 3, w, h);
	cuMatf4 destMat(dest, h, w, destStep);
	kernel_scale_8u32f_3 <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, destMat);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::warp_back_32f_3(float4* src, int srcStep, float4* dest, int destStep, int w, int h, const AffineCore& trf, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatf4 destMat(dest, h, w, destStep);
	kernel_warp_back_3 <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, destMat, trf);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::filter_32f_h_3(float4* src, float4* dest, int step, int w, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, step, w, h);
	cuMatf4 destMat(dest, h, w, step);
	kernel_filter_3 <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, destMat, 5, 1, 0);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::filter_32f_v_3(float4* src, float4* dest, int step, int w, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, step, w, h);
	cuMatf4 destMat(dest, h, w, step);
	kernel_filter_3 <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, destMat, 5, 0, 1);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::unsharp_32f_3(float4* base, float4* gauss, float4* dest, int step, int w, int h, cudaStream_t cs) {
	dim3 threads = configThreads();
	dim3 blocks = configBlocks(threads, w, h);
	kernel_unsharp_3 <<<blocks, threads, 0, cs>>> (base, gauss, dest, step, w, h);
	return cudaGetLastError();
}

cudaError_t cu::outputHost(float4* src, int srcStep, uchar* destYuv, int destStep, int w, int h, cudaStream_t cs) {
	dim3 threads = configThreads();
	dim3 blocks = configBlocks(threads, w, h);
	kernel_output_host <<<blocks, threads, 0, cs>>> (src, srcStep, destYuv, destStep, w, h);
	return cudaGetLastError();
}

cudaError_t cu::outputNvenc(float4* src, int srcStep, uchar* cudaNv12ptr, int cudaPitch, int w, int h, cudaStream_t cs) {
	dim3 threads = configThreads();
	dim3 blocks = configBlocks(threads, w / 2, h / 2);
	kernel_output_nvenc <<<blocks, threads, 0, cs>>> (src, srcStep, cudaNv12ptr, cudaPitch, w, h);
	return cudaGetLastError();
}

cudaError_t cu::copy_32f_3(float4* src, int srcStep, float4* dest, int destStep, int w, int h, cudaStream_t cs) {
	return cudaMemcpy2DAsync(dest, destStep * sizeof(float4), src, srcStep * sizeof(float4), w * sizeof(float4), h, cudaMemcpyDefault, cs);
}

//----------- callers float

cudaError_t cu::scale_8u32f(uchar* src, int srcStep, float* dest, int destStep, int w, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatf destMat(dest, h, w, destStep);
	kernel_scale_8u32f <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, destMat);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::copy_32f_3(uchar* src, int srcStep, uchar* dest, int destStep, int w, int h, cudaStream_t cs) {
	return cudaMemcpy2DAsync(dest, destStep, src, srcStep, w, h, cudaMemcpyDefault, cs);
}

cudaError_t cu::filter_32f_h(float* src, float* dest, int srcStep, int w, int h, size_t filterKernelIndex, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatf destMat(dest, h, w, srcStep);
	kernel_filter <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, destMat, filterKernelIndex, 1, 0);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::filter_32f_v(float* src, float* dest, int srcStep, int w, int h, size_t filterKernelIndex, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatf destMat(dest, h, w, srcStep);
	kernel_filter << <ki.blocks, ki.threads, 0, cs >> > (ki.texture, destMat, filterKernelIndex, 0, 1);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::remap_downsize_32f(float* src, int srcStep, float* dest, int destStep, int wsrc, int hsrc, cudaStream_t cs) {
	int wdest = wsrc / 2;
	int hdest = hsrc / 2;
	KernelContext ki = prepareTexture(src, srcStep, wsrc, hsrc, wdest, hdest);
	cuMatf srcMat(src, hsrc, wsrc, destStep);
	cuMatf destMat(dest, hdest, wdest, destStep);
	kernel_remap_downsize <<<ki.blocks, ki.threads, 0, cs>>> (ki.texture, destMat, srcMat);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::yuv_to_rgba(uchar* src, int srcStep, uchar* dest, int destStep, int w, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, 3 * h, w, h);
	kernel_yuv8_to_rgba8 <<<ki.blocks, ki.threads>>> (ki.texture, dest, w, h);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::yuv_to_rgba(float4* src, int srcStep, uchar* dest, int destStep, int w, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	kernel_yuv128_to_rgba8 <<<ki.blocks, ki.threads>>> (ki.texture, dest, w, h);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}