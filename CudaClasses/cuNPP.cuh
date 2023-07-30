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

enum class FilterDim {
	FILTER_HORIZONZAL = 0,
	FILTER_VERTICAL = 1,
};

using uchar = unsigned char;
using uint = unsigned int;
using cuMatf = cu::Mat<float>;
using cuMatc = cu::Mat<uchar>;

cudaError_t npz_scale_8u32f(uchar* src, int srcStep, float* dest, int destStep, int w, int h);

cudaError_t npz_scale_32f8u(float* src, int srcStep, uchar* dest, int destStep, int w, int h);

cudaError_t npz_copy_32f(float* src, int srcStep, float* dest, int destStep, int w, int h);

cudaError_t npz_warp_back_32f(float* src, int srcStep, float* dest, int destStep, int w, int h, cu::Affine coeffs, cudaStream_t cs = 0);

cudaError_t npz_unsharp_32f(float* base, float* gauss, int srcStep, float* dest, int destStep, int w, int h, float factor, cudaStream_t cs = 0);

cudaError_t npz_remap_downsize_32f(float* src, int srcStep, float* dest, int destStep, int wsrc, int hsrc);

cudaError_t npz_filter_32f(float* src, float* dest, int srcStep, int w, int h, float* d_kernel, int kernelSize, FilterDim filterDim, cudaStream_t cs = 0);

cudaError_t npz_uv_to_nv12(float* src, int srcStep, uchar* nvencPtr, int cudaPitch, int w, int h);

cudaError_t npz_yuv_to_rgb(uchar* src, int srcStep, uchar* dest, int destStep, int w, int h);

cudaError_t npz_yuv_to_rgb(float* src, int srcStep, uchar* dest, int destStep, int w, int h);