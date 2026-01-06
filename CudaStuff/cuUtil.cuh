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

#include <cassert>
#include <iostream>

 //define device functions here to avoid red underlines
void __syncthreads();
void __trap();
long long int clock64();
float __saturatef(float);
double min(double, double);
double max(double, double);

namespace cu {
	//check if first thread is running
	__device__ bool firstThread();

	//get time from register
	__device__ int64_t globaltimer();

	//memory copy on device
	__device__ void memcpy(void* dest, const void* src, size_t count);

	__device__ size_t clampUnsigned(size_t valueToAdd, size_t valueToSubtract, size_t lo, size_t hi);

	__device__ double clamp(double val, double lo, double hi);

	//-----------------------------------
	//matrix implementation in device code
	template <class T> class Mat {

	public:
		int h, w, stride;
		T* data;

		__host__ __device__ Mat(T* data, int h, int w, int stride) :
			data { data }, h { h }, w { w }, stride { stride }
		{}

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

		__device__ string(size_t length, bool dummy) :
			mStr { new char[length] }
		{}

	public:

		//copy constructor
		__device__ string(const string& other);

		//move constructor
		__device__ string(string&& other) noexcept;

		//destructor
		__device__ ~string();

		//create from char array
		__device__ string(const char* str);

		//create from int value
		__device__ string(int value);

		//concatenate strings
		__device__ string operator + (const string& other);

		//get the underlying char array
		__device__ const char* str() const;

		//length of the string
		__device__ size_t size() const;
	};


	//store data from device for debugging on host
	struct DebugData {
		double* d_data;
		size_t maxSize = 1024 * 256; //number of doubles
	};


	//store data for later debugging
	template <class T> __device__ bool storeDebugData(double* debugData, size_t maxSize, size_t h, size_t w, T* data) {
		double siz = debugData[0]; //first value contains size in byte in double
		if (siz + h * w + 2 > maxSize) return false; //check max allowed size
		double* ptr = debugData + (size_t) siz + 1; //pointer to start this data at
		*ptr++ = h; //first value per dataset is height
		*ptr++ = w; //second value per dataset is width
		for (int i = 0; i < h * w; i++) *ptr++ = data[i]; //actual data
		debugData[0] += h * w + 2; //update size
	}
}

using uchar = unsigned char;
using uint = unsigned int;
using cuMatf = cu::Mat<float>;
using cuMatf4 = cu::Mat<float4>;
using cuMatc = cu::Mat<uchar>;
