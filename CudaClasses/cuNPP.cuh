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

#include "cuUtil.cuh"

using uchar = unsigned char;
using uint = unsigned int;
using cuMatf = cu::Mat<float>;
using cuMatf4 = cu::Mat<float4>;
using cuMatc = cu::Mat<uchar>;

namespace cu {
	struct Affine {
		double m00, m01, m02, m10, m11, m12;
	};

	cudaError_t scale_8u32f(uchar* src, int srcStep, float* dest, int destStep, int w, int h, cudaStream_t cs = 0);
	cudaError_t scale_8u32f_3(uchar* src, int srcStep, float4* dest, int destStep, int w, int h, cudaStream_t cs = 0);

	cudaError_t copy_32f_3(float4* src, int srcStep, float4* dest, int destStep, int w, int h, cudaStream_t cs = 0);
	cudaError_t copy_32f_3(uchar* src, int srcStep, uchar* dest, int destStep, int w, int h, cudaStream_t cs = 0);

	cudaError_t warp_back_32f_3(float4* src, int srcStep, float4* dest, int destStep, int w, int h, Affine trf, cudaStream_t cs = 0);

	cudaError_t unsharp_32f_3(float4* base, float4* gauss, float4* dest, int step, int w, int h, cudaStream_t cs = 0);

	cudaError_t outputHost(float4* src, int srcStep, uchar* destYuv, int destStep, int w, int h, cudaStream_t cs = 0);
	cudaError_t outputNvenc(float4* src, int srcStep, uchar* cudaNv12ptr, int cudaPitch, int w, int h, cudaStream_t cs = 0);

	cudaError_t remap_downsize_32f(float* src, int srcStep, float* dest, int destStep, int wsrc, int hsrc, cudaStream_t cs = 0);

	cudaError_t filter_32f_h(float* src, float* dest, int srcStep, int w, int h, size_t filterKernelIndex, cudaStream_t cs = 0);
	cudaError_t filter_32f_h_3(float4* src, float4* dest, int step, int w, int h, cudaStream_t cs = 0);

	cudaError_t filter_32f_v(float* src, float* dest, int srcStep, int w, int h, size_t filterKernelIndex, cudaStream_t cs = 0);
	cudaError_t filter_32f_v_3(float4* src, float4* dest, int step, int w, int h, cudaStream_t cs = 0);

	cudaError_t yuv_to_rgb(uchar* src, int srcStep, uchar* dest, int destStep, int w, int h);

	cudaError_t yuv_to_rgb(float4* src, int srcStep, uchar* dest, int destStep, int w, int h);
}