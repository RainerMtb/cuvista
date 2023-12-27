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

#include "cuUtil.cuh"

__device__ int64_t cu::globaltimer() {
	int64_t time = 0;
	asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(time));
	return time;
}

__device__ bool cu::firstThread() {
	return (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
}

__device__ void cu::memcpy(void* dest, const void* src, size_t count) {
	char* destptr = reinterpret_cast<char*>(dest);
	const char* srcptr = reinterpret_cast<const char*>(src);
	for (size_t i = 0; i < count; i++) {
		destptr[i] = srcptr[i];
	}
}

__device__ __host__ double cu::sqr(double x) {
	return x * x;
}

__device__ __host__ size_t cu::clampUnsigned(size_t valueToAdd, size_t valueToSubtract, size_t lo, size_t hi) {
	return (size_t) cu::clamp(int64_t(valueToAdd) - int64_t(valueToSubtract), int64_t(lo), int64_t(hi));
}

//-------------------
//CUDA STRING
//-------------------

//copy constructor
__device__ cu::string::string(const cu::string& other) {
	mStr = new char[other.size() + 1];
	memcpy(mStr, other.mStr, other.size());
}

//move constructor
__device__ cu::string::string(cu::string&& other) noexcept {
	mStr = other.mStr;
	other.mStr = nullptr;
}

//destructor
__device__ cu::string::~string() {
	delete[] mStr;
}

//create from char array
__device__ cu::string::string(const char* str) {
	const char* ptr = str;
	while (*ptr) ptr++;
	size_t siz = ptr - str + 1;
	mStr = new char[siz];
	memcpy(mStr, str, siz);
}

//create from int value
__device__ cu::string::string(int value) {
	size_t n = 1;
	int v = value / 10;
	while (v != 0) {
		n++;
		v /= 10;
	}
	char* first = nullptr;
	if (value < 0) {
		first = mStr = new char[n + 2];
		mStr[0] = '-';
		value = abs(value);

	} else {
		mStr = new char[n + 1];
		first = mStr - 1;
	}
	for (char* ptr = first + n; ptr != first; ptr--) {
		*ptr = 48 + value % 10;
		value /= 10;
	}
}

//get the underlying char array
__device__ const char* cu::string::str() const {
	return mStr;
}

//length of the string
__device__ size_t cu::string::size() const {
	char* ptr = mStr;
	while (*ptr) ptr++;
	return ptr - mStr;
}

//concatenate strings
__device__ cu::string cu::string::operator + (const cu::string& other) {
	size_t siz1 = size();
	size_t siz2 = other.size();
	cu::string out(siz1 + siz2 + 1, true);
	memcpy(out.mStr, mStr, siz1);
	memcpy(out.mStr + siz1, other.mStr, siz2);
	return out;
}
