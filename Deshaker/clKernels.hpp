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
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

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

__kernel void scale_32f8u_3(__read_only image2d_t src, __write_only image2d_t dest) {
	int c = get_global_id(0);
	int r = get_global_id(1);
	int h = get_global_size(1);

	float4 val = round(read_imagef(src, (int2)(c, r)) * 255.0f);
	write_imageui(dest, (int2)(c, r), (uchar)(val.x));
	write_imageui(dest, (int2)(c, r + h), (uchar)(val.y));
	write_imageui(dest, (int2)(c, r + h + h), (uchar)(val.z));
}

__kernel void filter_32f_1(__read_only image2d_t src, __write_only image2d_t dest, __constant float4* filterKernel, int8 ix, int8 iy, int siz) {
	int c = get_global_id(0);
	int r = get_global_id(1);

	float result = 0.0f;
	for (int i = 0; i < siz; i++) {
		int x = c + ix[i];
		int y = r + iy[i];
		float val = read_imagef(src, sampler, (int2)(x, y)).x;
		result = fma(val, filterKernel[i][0], result);
	}
	write_imagef(dest, (int2)(c, r), result);
}

__kernel void filter_32f_3(__read_only image2d_t src, __write_only image2d_t dest, __constant float4* filterKernel, int8 ix, int8 iy, int siz) {
	int c = get_global_id(0);
	int r = get_global_id(1);

	float4 result = 0.0f;
	for (int i = 0; i < siz; i++) {
		int x = c + ix[i];
		int y = r + iy[i];
		float4 val = read_imagef(src, sampler, (int2)(x, y));
		result = fma(val, filterKernel[i], result);
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
	//int2 coords = (int2) (c * 2 + 0.5f, r * 2 + 0.5f);
	//float val = read_imagef(src, sampler, coords).x;
	
	float val = interpolate(src, c * 2, r * 2, 0.5f, 0.5f);
	write_imagef(dest, (int2)(c, r), val);
}

__kernel void warp_back(__read_only image2d_t src, __write_only image2d_t dest, double8 trf) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);

	float xx = (float) (x * trf.s0 + y * trf.s1 + trf.s2);
	float yy = (float) (x * trf.s3 + y * trf.s4 + trf.s5);
	if (xx >= 0.0f && xx <= w - 1 && yy >= 0.0f && yy <= h - 1) {
		float4 f00 = read_imagef(src, sampler, (float2)(xx, yy));
		float4 f01 = read_imagef(src, sampler, (float2)(xx + 1, yy));
		float4 f10 = read_imagef(src, sampler, (float2)(xx, yy + 1));
		float4 f11 = read_imagef(src, sampler, (float2)(xx + 1, yy + 1));
		float dx = xx - floor(xx);
		float dy = yy - floor(yy);
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

__kernel void scrap() {}

)";

inline std::string luinvFunction = R"(
void luinv(double** Apiv, double* A0, double* temp, double* Ainv, int s, int r, int ci, int cols) {
	//step 1. decompose matrix A0
	if (r < s && ci == 0) {
		Apiv[r] = A0 + 1ull * r * s;

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
			Ainv[1ull * r * s + (Apiv[r] - A0) / s] = 1.0;
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

inline std::string luinvTestKernel = R"(
__kernel void luinvTest(__global double* input, __global double* outAinv, int s, __local double* temp) {
	int c = get_global_id(0) / s;
	int cols = get_global_size(0) / s;
	int r = get_global_id(0) - c * s;

	double** Apiv = (double**) (temp + s);
	luinv(Apiv, input, temp, outAinv, s, r, c, cols);
}
)";
