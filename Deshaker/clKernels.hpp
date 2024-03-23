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

#include <string>

inline std::string kernelsInputOutput = R"(
#pragma OPENCL FP_CONTRACT OFF

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
const sampler_t downsampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;


__kernel void scale_8u32f_1(__read_only image2d_t src, __write_only image2d_t dest) {
	int c = get_global_id(0);
	int r = get_global_id(1);
	float f = 1.0f / 255.0f;

	int2 coords = (int2)(c, r);
	uchar val = read_imageui(src, coords).x;
	write_imagef(dest, coords, val * f);
}

__kernel void scale_8u32f_3(__read_only image2d_t src, __write_only image2d_t dest) {
	int c = get_global_id(0);
	int r = get_global_id(1);
	int h = get_global_size(1);
	float f = 1.0f / 255.0f;

	float4 val = (float4)(
		read_imageui(src, (int2)(c, r)).x,
		read_imageui(src, (int2)(c, r + h)).x,
		read_imageui(src, (int2)(c, r + h + h)).x,
		0
	);
	write_imagef(dest, (int2)(c, r), val * f);
}

__kernel void scale_32f8u_3(__read_only image2d_t src, __global uchar* dest, int pitch) {
	int c = get_global_id(0);
	int r = get_global_id(1);
	int h = get_global_size(1);

	float4 val = rint(read_imagef(src, (int2)(c, r)) * 255.0f);
	int idx = r * pitch + c;
	dest[idx] = (uchar)(val.x);
	idx += h * pitch;
	dest[idx] = (uchar)(val.y);
	idx += h * pitch;
	dest[idx] = (uchar)(val.z);
}

struct FilterKernel {
	int siz;
	float k[8];
};

__constant struct FilterKernel filterKernels[] = {
	{5, {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f}},
	{5, {0.0f, 0.25f, 0.5f, 0.25f, 0.0f}},
	{5, {0.0f, 0.25f, 0.5f, 0.25f, 0.0f}},
	{3, {-0.5f, 0.0f, 0.5f}},
};

__kernel void filter_32f_1(__read_only image2d_depth_t src, __write_only image2d_t dest, int filterIndex, int dx, int dy) {
	int c = get_global_id(0);
	int r = get_global_id(1);

	float result = 0.0f;
	int siz = filterKernels[filterIndex].siz;
	__constant float* k = filterKernels[filterIndex].k;
	int x = c - dx * siz / 2;
	int y = r - dy * siz / 2;
	for (int i = 0; i < siz; i++) {
		float val = read_imagef(src, sampler, (int2)(x, y));
		result += val * k[i];
		x += dx;
		y += dy;
	}
	write_imagef(dest, (int2)(c, r), result);
}

__kernel void filter_32f_3(__read_only image2d_t src, __write_only image2d_t dest, int filterIndex, int dx, int dy) {
	int c = get_global_id(0);
	int r = get_global_id(1);

	float4 result = 0.0f;
	int siz = filterKernels[0].siz;
	int x = c - dx * siz / 2;
	int y = r - dy * siz / 2;
	for (int i = 0; i < siz; i++) {
		float4 val = read_imagef(src, sampler, (int2)(x, y));
		result.x += val.x * filterKernels[0].k[i];
		result.y += val.y * filterKernels[1].k[i];
		result.z += val.z * filterKernels[2].k[i];
		x += dx;
		y += dy;
	}
	write_imagef(dest, (int2)(c, r), result);
}

float interpolate(__read_only image2d_t src, int c, int r, float dx, float dy) {
	float f00 = read_imagef(src, (int2)(c,     r)).x;
	float f01 = read_imagef(src, (int2)(c + 1, r)).x;
	float f10 = read_imagef(src, (int2)(c,     r + 1)).x;
	float f11 = read_imagef(src, (int2)(c + 1, r + 1)).x;
	return (1.0f - dx) * (1.0f - dy) * f00 + (1.0f - dx) * dy * f10 + dx * (1.0f - dy) * f01 + dx * dy * f11;
}


__kernel void remap_downsize_32f(__read_only image2d_t src, __write_only image2d_t dest) {
	int c = get_global_id(0);
	int r = get_global_id(1);

	//sampling produces different result to cpu code
	//float2 coords = (float2) (c * 2 + 0.5f, r * 2 + 0.5f);
	//float val = read_imagef(src, downsampler, coords).x;
	
	float val = interpolate(src, c * 2, r * 2, 0.5f, 0.5f);
	write_imagef(dest, (int2)(c, r), val);
}

__kernel void warp_back(__read_only image2d_t src, __write_only image2d_t dest, double8 trf) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);

	float xx = (float) (fma(x, trf.s0, fma(y, trf.s1, trf.s2)));
	float yy = (float) (fma(x, trf.s3, fma(y, trf.s4, trf.s5)));
	if (xx >= 0.0f && xx <= w - 1 && yy >= 0.0f && yy <= h - 1) {
		float4 f00 = read_imagef(src, sampler, (float2)(xx, yy));
		float4 f01 = read_imagef(src, sampler, (float2)(xx + 1, yy));
		float4 f10 = read_imagef(src, sampler, (float2)(xx, yy + 1));
		float4 f11 = read_imagef(src, sampler, (float2)(xx + 1, yy + 1));
		float dx = xx - floor(xx);
		float dy = yy - floor(yy);

		//matching result with cpu code only when separating sums
		float4 val = (1.0f - dx) * (1.0f - dy) * f00 + (1.0f - dx) * dy * f10 + dx * (1.0f - dy) * f01 + dx * dy * f11;
		write_imagef(dest, (int2)(x, y), val);
	}
}

__kernel void unsharp(__read_only image2d_t src, __write_only image2d_t dest, __read_only image2d_t gauss, float4 factor) {
	int c = get_global_id(0);
	int r = get_global_id(1);

	int2 coords = (int2)(c, r);
	float4 valBase = read_imagef(src, coords);
	float4 valGauss = read_imagef(gauss, coords);
	float4 val = clamp(valBase + (valBase - valGauss) * factor, 0.0f, 1.0f);
	write_imagef(dest, coords, val);
}

//input float values in range 0..255
void yuv_to_rgb_func(float yf, float uf, float vf, uchar* r, uchar* g, uchar* b) {
	*r = (uchar) clamp(yf + (1.370705f * (vf - 128.0f)), 0.0f, 255.0f);
	*g = (uchar) clamp(yf - (0.337633f * (uf - 128.0f)) - (0.698001f * (vf - 128.0f)), 0.0f, 255.0f);
	*b = (uchar) clamp(yf + (1.732446f * (uf - 128.0f)), 0.0f, 255.0f);
}

__kernel void yuv8u_to_rgb(__read_only image2d_t src, __global uchar* dest) {
	int c = get_global_id(0);
	int r = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);

	uchar y = read_imageui(src, (int2)(c, r)).x;
	uchar u = read_imageui(src, (int2)(c, r + h)).x;
	uchar v = read_imageui(src, (int2)(c, r + h + h)).x;
	uchar* ptr = dest + 3 * (r * w + c);
	yuv_to_rgb_func(y, u, v, ptr, ptr + 1, ptr + 2);
}

__kernel void yuv32f_to_rgb(__read_only image2d_t src, __global uchar* dest) {
	int c = get_global_id(0);
	int r = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);

	float4 yuv = read_imagef(src, (int2)(c, r)) * 255.0f;
	uchar* ptr = dest + 3 * (r * w + c);
	yuv_to_rgb_func(yuv.s0, yuv.s1, yuv.s2, ptr, ptr + 1, ptr + 2);
}

__kernel void scrap() {
}

)";

inline std::string luinvFunction = R"(
#pragma OPENCL FP_CONTRACT OFF

void luinv(double** Apiv, double* A0, double* temp, double* Ainv, int s, int r, int ci, int cols) {
	//step 1. decompose matrix A0
	if (r < s && ci == 0) {
		Apiv[r] = A0 + r * s;

		//iterate j over columns
		for (int j = 0; j < s; j++) {
			for (int i = 0; i < j; i++) {
				if (r > i) {
					Apiv[r][j] -= Apiv[r][i] * Apiv[i][j];
				}
			}
			temp[r] = fabs(Apiv[r][j]);

			if (r == 0) {
				int p = j;
				for (int i = j + 1; i < s; i++) {
					if (temp[i] > temp[p]) p = i;
				}
				double* dp = Apiv[p];
				Apiv[p] = Apiv[j];
				Apiv[j] = dp;
			}

			if (r > j) {
				Apiv[r][j] /= Apiv[j][j];
			}
		}
	}

	//step 2. solve decomposed matrix against identity mat
	if (r < s && ci < cols) {
		for (int c = ci; c < s; c += cols) {
			Ainv[r * s + c] = 0.0;
		}
		if (ci == 0) {
			Ainv[r * s + (Apiv[r] - A0) / s] = 1.0;
		}

		//forward substitution
		for (int k = 0; k < s; k++) {
			for (int c = ci; c < s; c += cols) {
				if (r > k) Ainv[r * s + c] -= Ainv[k * s + c] * Apiv[r][k];
			}
		}

		//backwards substitution
		for (int k = s - 1; k >= 0; k--) {
			double ap = Apiv[k][k];
			for (int c = ci; c < s; c += cols) {
				double ai = Ainv[k * s + c];
				if (r == k) Ainv[r * s + c] /= ap;
				if (r < k) Ainv[r * s + c] -= ai / ap * Apiv[r][k];
			}
		}
	}
}
)";

inline std::string norm1Function = R"(
#pragma OPENCL FP_CONTRACT OFF

double norm1(const double* mat, int m, int n, double* temp) {
	int i = get_local_id(0);
	int k = get_local_id(1);

	//find max per column
	if (i < n && k == 0) {
		temp[i] = fabs(mat[i]); //first row from mat
		for (int y = 1; y < m; y++) {
			temp[i] += fabs(mat[y * n + i]);
		}
	}

	//find max in row
	if (i == 0 && k == 0) {
		for (int x = 1; x < n; x++) {
			double val = temp[x];
			if (isnan(val) || val > temp[0]) temp[0] = val;
		}
	}
	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	//max is now first item
	return temp[0];
}
)";

inline std::string testKernels = R"(
__kernel void luinvTest(__global double* input, __global double* outAinv, __local double* temp) {
	int s = get_local_size(0);
	int r = get_local_id(0);
	int c = get_local_id(1);
	int cols = get_local_size(1);

	double** Apiv = (double**) (temp + s);
	luinv(Apiv, input, temp, outAinv, s, r, c, cols);
}

__kernel void luinvGroupTest(__global double* input, __global double* output, __local double* ptr) {
	int s = get_local_size(0);
	int offsetLocal = get_local_id(0) * s;
	int offsetGlobal = get_group_id(0) * s * s;
	int gid = offsetGlobal + offsetLocal;

	double* src = ptr;
	for (int i = 0; i < s; i++) {
		src[offsetLocal + i] = input[gid + i];
	}

	double* temp = src + s * s;
	double** Apiv = (double**)(temp + s);
	int r = get_local_id(0);
	int c = get_local_id(1);
	int cols = get_local_size(1);
	double* out = output + offsetGlobal;
	luinv(Apiv, src, temp, out, s, r, c, cols);
}

__kernel void norm1Test(__global double* mat, int m, int n, __local double* temp, __global double* result) {
	int idx = get_local_linear_id();	
	result[idx] = norm1(mat, m, n, temp);
}
)";