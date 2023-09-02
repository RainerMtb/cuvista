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

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cassert>
#include <iostream>

void __syncthreads();
void __trap();
long long int clock64();
double fabs(double);
float __saturatef(float);
bool isnan(double);
double max(const double a, const double b);
double min(const double a, const double b);


namespace cu {
	const int THREAD_COUNT = 16;	//number of threads used in kernels to access textures

	//parameters of Affine Transform in struct so it can be used as an argument to a cuda kernel
	struct Affine {
		double m00, m01, m02, m10, m11, m12;
	};

	//check if first thread is running
	__device__ bool firstThread();

	//get time from register
	__device__ void globaltimer(int64_t* time);

	//memory copy on device
	__device__ void memcpy(void* dest, const void* src, size_t count);

	__device__ __host__ double sqr(double x);

	__device__ __host__ size_t clampUnsigned(size_t valueToAdd, size_t valueToSubtract, size_t lo, size_t hi);

	__device__ __host__ double clamp(double val, double lo, double hi);

	__device__ __host__ float clamp(float val, float lo, float hi);

	__device__ __host__ int64_t clamp(int64_t val, int64_t lo, int64_t hi);

	__device__ __host__ int clamp(int val, int lo, int hi);

	//-----------------------------------
	//matrix implementation in device code
	template <class T> class Mat {

		__device__ float roundTex(double d) const {
			return float(nextafter(d, signbit(d) ? -1e300 : 1e300));
		}

	public:
		int h, w, stride;
		T* data;

		__host__ __device__ Mat(T* data, int h, int w, int stride) : data { data }, h { h }, w { w }, stride { stride } {}

		__device__ T& at(int row, int col) {
			assert(row >= 0 && col >= 0 && row < h && col < w);
			return data[row * stride + col];
		}

		__device__ const T& at(int row, int col) const {
			assert(row >= 0 && col >= 0 && row < h && col < w);
			return data[row * stride + col];
		}

		__device__ T* addr(int row, int col) {
			assert(row >= 0 && col >= 0 && row < h && col < w);
			return data + row * stride + col;
		}

		__device__ const T* addr(int row, int col) const {
			assert(row >= 0 && col >= 0 && row < h && col < w);
			return data + row * stride + col;
		}

		__device__ T* row(int row) {
			assert(row >= 0 && row < h);
			return data + row * stride;
		}

		__device__ T interp2(int ix, int iy, T dx, T dy, T dxdy) const {
			T f00 = at(iy, ix);
			T f01 = dx == 0 ? f00 : at(iy, ix + 1);
			T f10 = dy == 0 ? f00 : at(iy + 1, ix);
			T f11 = dx == 0 || dy == 0 ? f00 : at(iy + 1, ix + 1);
			return roundTex((1 - dx) * (1 - dy) * f00) + roundTex((1 - dx) * dy * f10) + roundTex(dx * (1 - dy) * f01) + roundTex(dxdy * f11);
		}

		__device__ void toConsole(int digits = 6, const char* title = "") const {
			if (firstThread()) {
				if (*title) printf("\n%s = ", title); //print title if present
				printf("[\n");

				int d = clamp(digits, 0, 25);
				char fmt[] = " %.??f  "; //replace ?? with numbers
				fmt[3] = (char) (48 + d / 10);
				fmt[4] = (char) (48 + d % 10);
				for (int r = 0; r < h && r < 80; r++) {
					for (int c = 0; c < w && c < 30; c++) {
						double val = at(r, c);
						printf(val < 0.0 ? fmt + 1 : fmt, val); //positive values with a leading space
					}
					printf("\n");
				}
				printf("]\n");
			}
			__syncthreads();
		}
	};

	//------------------------------------------
	//simple string implementation in device code
	struct string {
	private:
		char* mStr = nullptr;

		__device__ string(size_t length, bool dummy) : mStr { new char[length] } {}

	public:

		//copy constructor
		__device__ string(const cu::string& other);

		//move constructor
		__device__ string(cu::string&& other) noexcept;

		//destructor
		__device__ ~string();

		//create from char array
		__device__ string(const char* str);

		//create from int value
		__device__ string(int value);

		//concatenate strings
		__device__ cu::string operator + (const cu::string& other);

		//get the underlying char array
		__device__ const char* str() const;

		//length of the string
		__device__ size_t size() const;
	};


	//store data from device for debugging on host
	struct DebugData {
		int64_t* d_timestamps;
		size_t n_timestamps = 0;
		double* d_data;
		size_t maxSize = 1024 * 128;
	};


	//store data for later debugging
	template <class T> __device__ bool storeDebugData(DebugData& debugData, size_t h, size_t w, T* data) {
		double siz = *debugData.d_data; //first value contains size in byte in double
		if (siz + h * w + 2 > debugData.maxSize) return false; //check max allowed size
		double* ptr = debugData.d_data + (size_t) siz + 1; //pointer to write this data
		*ptr++ = h; //first value per dataset is height
		*ptr++ = w; //second value per dataset is width
		for (int i = 0; i < h * w; i++) *ptr++ = data[i]; //actual data
		*debugData.d_data += h * w + 2; //update size
	}
}
