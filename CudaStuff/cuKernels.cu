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

#include "cuKernels.cuh"
#include "cuDeshaker.cuh"

#include <chrono>
#include <iostream>

 //declare here to mitigate red underlines
template<class T> __device__ T tex2D(cudaTextureObject_t tex, float x, float y);
template<class T> __device__ T tex2Dgather(cudaTextureObject_t tex, float x, float y, int comp);

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


//create texture for reading data in kernel
template <class T> KernelContext prepareTexture(T* src, int stride, int texw, int texh, int thrw, int thrh) {
	cudaResourceDesc resDesc {};
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = src;
	resDesc.res.pitch2D.width = texw;
	resDesc.res.pitch2D.height = texh;
	resDesc.res.pitch2D.pitchInBytes = stride * sizeof(T);
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


//--------------- device functions

__device__ void yuv_to_rgba_func(float yf, float uf, float vf, uchar* r, uchar* g, uchar* b, uchar* a) {
	float y = yf - 16.0f;
	float u = uf - 128.0f;
	float v = vf - 128.0f;
	*r = (uchar) rint(cu::clamp(1.164f * y + 1.596f * v, 0.0f, 255.0f));
	*g = (uchar) rint(cu::clamp(1.164f * y - 0.392f * u - 0.813f * v, 0.0f, 255.0f));
	*b = (uchar) rint(cu::clamp(1.164f * y + 2.017f * u, 0.0f, 255.0f));
	*a = 0xFF;
}

__device__ float interp(float f00, float f01, float f10, float f11, float dx, float dy) {
	return (1.0f - dx) * (1.0f - dy) * f00 + (1.0f - dx) * dy * f10 + dx * (1.0f - dy) * f01 + dx * dy * f11;
}

__device__ float tex2Dinterp(cudaTextureObject_t tex, float x, float y) {
	// attention: floor(x + 1) is not necessarily equal to floor(x) + 1
	// example:   int(1023.9994f + 1.0f) is actually 1025
	// therefore first flooring, then incrementing for the next vertex
	float flx = floorf(x);
	float fly = floorf(y);
	float f00 = tex2D<float>(tex, flx, fly);
	float f01 = tex2D<float>(tex, flx + 1, fly);
	float f10 = tex2D<float>(tex, flx, fly + 1);
	float f11 = tex2D<float>(tex, flx + 1, fly + 1);
	return interp(f00, f01, f10, f11, x - flx, y - fly);
}

__device__ float4 tex2Dinterp_3(cudaTextureObject_t tex, float x, float y) {
	float flx = floorf(x);
	float fly = floorf(y);
	float dx = x - flx;
	float dy = y - fly;
	float4 f00 = tex2D<float4>(tex, flx, fly);
	float4 f01 = tex2D<float4>(tex, flx + 1, fly);
	float4 f10 = tex2D<float4>(tex, flx, fly + 1);
	float4 f11 = tex2D<float4>(tex, flx + 1, fly + 1);
	return {
		interp(f00.x, f01.x, f10.x, f11.x, dx, dy),
		interp(f00.y, f01.y, f10.y, f11.y, dx, dy),
		interp(f00.z, f01.z, f10.z, f11.z, dx, dy),
		interp(f00.w, f01.w, f10.w, f11.w, dx, dy)
	};
}


//------------ KERNELS FOR FLOAT4

__global__ void kernel_scale_8u32f_3(cudaTextureObject_t texObj, cuMatf4 dest) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	constexpr float f = 1.0f / 255.0f;

	if (x < dest.w && y < dest.h) {
		uchar vv = tex2D<uchar>(texObj, x * 4 + 0, y);
		uchar uu = tex2D<uchar>(texObj, x * 4 + 1, y);
		uchar yy = tex2D<uchar>(texObj, x * 4 + 2, y);
		dest.at(y, x) = { vv * f, uu * f, yy * f, 1.0f };
	}
}

__global__ void kernel_warp_back_3(cudaTextureObject_t texObj, cuMatf4 dest, AffineDataFloat trf) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	float xx = fmaf(x, trf.m00, fmaf(y, trf.m01, trf.m02));
	float yy = fmaf(x, trf.m10, fmaf(y, trf.m11, trf.m12));
	if (x < dest.w && y < dest.h && xx >= 0.0f && xx <= dest.w - 1 && yy >= 0.0f && yy <= dest.h - 1) {
		dest.at(y, x) = tex2Dinterp_3(texObj, xx, yy);
	}
	//if (y == 228 && x == 1082) printf("\ncpu %16.12f\n", yy);
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
			result.x = fmaf(val.x, filterKernels[2].k[i], result.x);
			result.y = fmaf(val.y, filterKernels[1].k[i], result.y);
			result.z = fmaf(val.z, filterKernels[0].k[i], result.z);
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
		dest[idx].x = __saturatef(valBase.x + (valBase.x - valGauss.x) * d_core.unsharp4.v);
		dest[idx].y = __saturatef(valBase.y + (valBase.y - valGauss.y) * d_core.unsharp4.u);
		dest[idx].z = __saturatef(valBase.z + (valBase.z - valGauss.z) * d_core.unsharp4.y);
		dest[idx].w = __saturatef(valBase.w + (valBase.w - valGauss.w) * d_core.unsharp4.x);
	}
}

//------------ KERNELS FOR SINGLE PLANES

__global__ void kernel_output_host(float4* src, int srcStep, uchar* dest, int destStep, int w, int h) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		float4 xvuy = src[y * srcStep + x];
		int idx = y * destStep + x * 4;
		dest[idx + 0] = (uchar) rintf(xvuy.x * 255.0f);
		dest[idx + 1] = (uchar) rintf(xvuy.y * 255.0f);
		dest[idx + 2] = (uchar) rintf(xvuy.z * 255.0f);
		dest[idx + 3] = (uchar) rintf(xvuy.w * 255.0f);
	}
}

__global__ void kernel_output_host_yuv(float4* src, int srcStep, uchar* dest, int destStep, int w, int h) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		float4 xvuy = src[y * srcStep + x];
		uchar* ptr = dest + y * destStep + x;
		*ptr = (uchar) rintf(xvuy.z * 255.0f);
		ptr += h * destStep;
		*ptr = (uchar) rintf(xvuy.y * 255.0f);
		ptr += h * destStep;
		*ptr = (uchar) rintf(xvuy.x * 255.0f);
	}
}

__global__ void kernel_output_nvenc(float4* src, int srcStep, uchar* dest, int stride, int w, int h) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx;
	int outIdx;

	if (x < w / 2 && y < h / 2) {
		idx = y * 2 * srcStep + x * 2;
		outIdx = y * 2 * stride + x * 2;
		float4 f00 = src[idx + 0];
		float4 f01 = src[idx + 1];
		float4 f10 = src[idx + srcStep + 0];
		float4 f11 = src[idx + srcStep + 1];

		dest[outIdx + 0] = rintf(f00.z * 255.0f);
		dest[outIdx + 1] = rintf(f01.z * 255.0f);
		dest[outIdx + stride + 0] = rintf(f10.z * 255.0f);
		dest[outIdx + stride + 1] = rintf(f11.z * 255.0f);

		outIdx = (y + h) * stride + x * 2;
		float u = rintf(f00.y * 255.0f) + rintf(f01.y * 255.0f) + rintf(f10.y * 255.0f) + rintf(f11.y * 255.0f);
		float v = rintf(f00.x * 255.0f) + rintf(f01.x * 255.0f) + rintf(f10.x * 255.0f) + rintf(f11.x * 255.0f);
		dest[outIdx + 0] = u / 4;
		dest[outIdx + 1] = v / 4;
	}
}

__global__ void kernel_scale_8u32f(cudaTextureObject_t texObj, cuMatf dest, int64_t* luma) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint offsetY = 2;
	constexpr float f = 1.0f / 255.0f;

	if (x < dest.w && y < dest.h) {
		uchar val = tex2D<uchar>(texObj, x * 4 + offsetY, y);
		dest.at(y, x) = val * f;
	}

	//first row of blocks
	uint idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (blockIdx.y == 0 && idx < dest.w) {
		int sum = 0;
		for (int i = 0; i < dest.h; i++) {
			int val = tex2D<uchar>(texObj, idx * 4 + offsetY, i);
			sum += val * val;
		}
		luma[idx] = sum;
	}
}

__global__ void kernel_lumaSum(int64_t* luma, int w) {
	uint x = threadIdx.x;
	uint ws = blockDim.x;
	extern __shared__ int64_t shd64[];

	//32 threads
	shd64[x] = 0;
	for (int i = x; i < w; i += ws) shd64[x] += luma[i];

	//first thread
	if (x == 0) {
		for (int i = 1; i < ws; i++) shd64[0] += shd64[i];
		luma[0] = shd64[0];
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

__global__ void kernel_warp_back(cudaTextureObject_t texObj, cuMatf dest, AffineDataFloat trf) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	float xx = fmaf(x, trf.m00, fmaf(y, trf.m01, trf.m02));
	float yy = fmaf(x, trf.m10, fmaf(y, trf.m11, trf.m12));
	if (x < dest.w && y < dest.h && xx >= 0.0f && xx <= dest.w - 1 && yy >= 0.0f && yy <= dest.h - 1) {
		dest.at(y, x) = tex2Dinterp(texObj, xx, yy);
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
			float val = tex2D<float>(texObj, ix, iy);
			result = fmaf(val, kernel.k[i], result);
			ix += dx;
			iy += dy;
		}
		dest.at(y, x) = result;
	}
}

__global__ void kernel_yuv8_to_rgba8(cudaTextureObject_t texObj, uchar* rgba, int destStep, int w, int h, int4 index) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		uchar yy = tex2D<uchar>(texObj, x * 4 + 2, y);
		uchar uu = tex2D<uchar>(texObj, x * 4 + 1, y);
		uchar vv = tex2D<uchar>(texObj, x * 4 + 0, y);
		uchar* dest = rgba + y * destStep + x * 4;
		yuv_to_rgba_func(yy, uu, vv, dest + index.x, dest + index.y, dest + index.z, dest + index.w);
	}
}

__global__ void kernel_yuv128_to_rgba8(cudaTextureObject_t texObj, uchar* rgba, int destStep, int w, int h, int4 index) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		float4 yuv = tex2D<float4>(texObj, x, y);
		uchar* dest = rgba + y * destStep + x * 4;
		yuv_to_rgba_func(yuv.z * 255.0f, yuv.y * 255.0f, yuv.x * 255.0f, dest + index.x, dest + index.y, dest + index.z, dest + index.w);
	}
}

//----------- callers float4

cudaError_t cu::scale_8u32f_3(uchar* src, int srcStep, int srcWidth, float4* dest, int destStep, int destWidth, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, srcWidth, h, destWidth, h);
	cuMatf4 destMat(dest, h, destWidth, destStep);
	kernel_scale_8u32f_3 << <ki.blocks, ki.threads, 0, cs >> > (ki.texture, destMat);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::warp_back_32f_3(float4* src, int srcStep, float4* dest, int destStep, int w, int h, AffineDataFloat trf, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatf4 destMat(dest, h, w, destStep);
	kernel_warp_back_3 << <ki.blocks, ki.threads, 0, cs >> > (ki.texture, destMat, trf);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::filter_32f_h_3(float4* src, float4* dest, int step, int w, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, step, w, h);
	cuMatf4 destMat(dest, h, w, step);
	kernel_filter_3 << <ki.blocks, ki.threads, 0, cs >> > (ki.texture, destMat, 5, 1, 0);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::filter_32f_v_3(float4* src, float4* dest, int step, int w, int h, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, step, w, h);
	cuMatf4 destMat(dest, h, w, step);
	kernel_filter_3 << <ki.blocks, ki.threads, 0, cs >> > (ki.texture, destMat, 5, 0, 1);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::unsharp_32f_3(float4* base, float4* gauss, float4* dest, int step, int w, int h, cudaStream_t cs) {
	dim3 threads = configThreads();
	dim3 blocks = configBlocks(threads, w, h);
	kernel_unsharp_3 << <blocks, threads, 0, cs >> > (base, gauss, dest, step, w, h);
	return cudaGetLastError();
}

cudaError_t cu::outputHost(float4* src, int srcStep, uchar* destVuyx, int destStep, int w, int h, cudaStream_t cs) {
	dim3 threads = configThreads();
	dim3 blocks = configBlocks(threads, w, h);
	kernel_output_host << <blocks, threads, 0, cs >> > (src, srcStep, destVuyx, destStep, w, h);
	return cudaGetLastError();
}

cudaError_t cu::outputHostYuv(float4* src, int srcStep, uchar* destVuyx, int destStep, int w, int h, cudaStream_t cs) {
	dim3 threads = configThreads();
	dim3 blocks = configBlocks(threads, w, h);
	kernel_output_host_yuv << <blocks, threads, 0, cs >> > (src, srcStep, destVuyx, destStep, w, h);
	return cudaGetLastError();
}

cudaError_t cu::outputNvenc(float4* src, int srcStep, uchar* dest, int stride, int w, int h, cudaStream_t cs) {
	dim3 threads = configThreads();
	dim3 blocks = configBlocks(threads, w / 2, h / 2);
	kernel_output_nvenc << <blocks, threads, 0, cs >> > (src, srcStep, dest, stride, w, h);
	return cudaGetLastError();
}

//----------- callers float

cudaError_t cu::scale_8u32f(uchar* src, int srcStep, int srcWidth, float* dest, int destStep, int destWidth, int h, int64_t* d_luma, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, srcWidth, h);
	cuMatf destMat(dest, h, destWidth, destStep);
	kernel_scale_8u32f << <ki.blocks, ki.threads, 0, cs >> > (ki.texture, destMat, d_luma);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

int64_t cu::lumaSum(int64_t* d_luma, int w, cudaStream_t cs) {
	kernel_lumaSum << <1, 32, 32*8, cs >> > (d_luma, w);
	int64_t h_sum;
	cudaMemcpy(&h_sum, d_luma, sizeof(int64_t), cudaMemcpyDefault);
	return h_sum;
}

cudaError_t cu::warp_back_32f(float* src, int srcStep, float* dest, int destStep, int w, int h, AffineDataFloat trf, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatf destMat(dest, h, w, destStep);
	kernel_warp_back << <ki.blocks, ki.threads, 0, cs >> > (ki.texture, destMat, trf);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::set_32f(float* dest, int destStep, int w, int h, int value, cudaStream_t cs) {
	return cudaMemset2DAsync(dest, destStep * sizeof(float), value, w * sizeof(float), h, cs);
}

cudaError_t cu::filter_32f_h(float* src, float* dest, int srcStep, int w, int h, size_t filterKernelIndex, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	cuMatf destMat(dest, h, w, srcStep);
	kernel_filter << <ki.blocks, ki.threads, 0, cs >> > (ki.texture, destMat, filterKernelIndex, 1, 0);
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
	kernel_remap_downsize << <ki.blocks, ki.threads, 0, cs >> > (ki.texture, destMat, srcMat);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::yuv_to_rgba(uchar* src, int srcStep, uchar* dest, int destStep, int w, int h, int4 index, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w * 4, h, w, h);
	kernel_yuv8_to_rgba8 << <ki.blocks, ki.threads >> > (ki.texture, dest, destStep, w, h, index);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}

cudaError_t cu::yuv_to_rgba(float4* src, int srcStep, uchar* dest, int destStep, int w, int h, int4 index, cudaStream_t cs) {
	KernelContext ki = prepareTexture(src, srcStep, w, h);
	kernel_yuv128_to_rgba8 << <ki.blocks, ki.threads >> > (ki.texture, dest, destStep, w, h, index);
	cudaDestroyTextureObject(ki.texture);
	return cudaGetLastError();
}
