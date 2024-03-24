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

#include "Mat.h"


//specialization of mat for use with avx
class AvxMatFloat : protected CoreMat<float> {
public:
	int64_t frameIndex = -1;

	AvxMatFloat() : CoreMat<float>() {}
	AvxMatFloat(int h, int w) : CoreMat<float>(h, w) {}
	AvxMatFloat(int h, int w, float value) : CoreMat<float>(h, w, value) {}

	int w() const { return int(CoreMat::w); }
	int h() const { return int(CoreMat::h); }

	float& at(int row, int col) { return CoreMat::at(row, col); }
	const float& at(int row, int col) const { return CoreMat::at(row, col); }
	float* addr(int row, int col) { return CoreMat::addr(row, col); }
	const float* addr(int row, int col) const { return CoreMat::addr(row, col); }
	float* data() { return CoreMat::data(); }
	const float* data() const { return CoreMat::data(); }

	float* row(int r) { return addr(r, 0); }
	const float* row(int r) const { return addr(r, 0); }

	void fill(float value) { std::fill(array, array + numel(), value); }
	void saveAsBinary(const std::string& filename) { Matf::fromArray(h(), w(), array, false).saveAsBinary(filename); }

	const CoreMat<float>& core() const{ return *this; }
};


//wrapper for _mm512
class VF16 {
public:
	__m512 a;

	VF16() : a { _mm512_setzero_ps() } {}

	VF16(float a) : a { _mm512_set1_ps(a) } {}

	VF16(__m512 a) : a { a } {}

	VF16(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7,
		float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15) :
		a { _mm512_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) } {}

	VF16(const float* data) : a { _mm512_loadu_ps(data) } {}

	VF16(const float* data, __mmask16 mask) : a { _mm512_maskz_load_ps(mask, data) } {}

	VF16(const unsigned char* data) : a { _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm_loadu_epi8(data))) } {}

	VF16(const unsigned char* data, __mmask16 mask) : a { _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

	VF16 operator + (VF16 other) { return _mm512_add_ps(a, other.a); }
	VF16 operator - (VF16 other) { return _mm512_sub_ps(a, other.a); }
	VF16 operator * (VF16 other) { return _mm512_mul_ps(a, other.a); }
	VF16 operator / (VF16 other) { return _mm512_div_ps(a, other.a); }
	VF16 add(VF16 other) { return _mm512_add_ps(a, other.a); }
	VF16 sub(VF16 other) { return _mm512_sub_ps(a, other.a); }
	VF16 mul(VF16 other) { return _mm512_mul_ps(a, other.a); }
	VF16 div(VF16 other) { return _mm512_div_ps(a, other.a); }

	float operator [] (size_t i) const { return at(i); }

	friend std::ostream& operator << (std::ostream& os, const VF16& vec) {
		for (int i = 0; i < 16; i++) os << vec[i] << " ";
		return os;
	}

	float at(size_t i) const {
		return a.m512_f32[i];
	}

	float sum(int from, int to) const {
		float sum = 0.0f;
		for (int i = from; i < to; i++) sum += at(i);
		return sum;
	}

	float sum() const {
		return sum(0, 16);
	}

	VF16 clamp(VF16 lo, VF16 hi) const {
		return _mm512_min_ps(_mm512_max_ps(a, lo.a), hi.a);
	}

	static VF16 clamp(VF16 value, VF16 lo, VF16 hi) {
		return value.clamp(lo, hi);
	}

	void storeu(float* dest) {
		_mm512_storeu_ps(dest, a);
	}

	void storeu(float* dest, __mmask16 mask) {
		_mm512_mask_storeu_ps(dest, mask, a);
	}
};


//wrapper for _mm256
class VF8 {
public:
	__m256 a;

	VF8() : a { _mm256_setzero_ps() } {}

	VF8(float a) : a { _mm256_set1_ps(a) } {}

	VF8(__m256 a) : a { a } {}

	VF8(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) :
		a { _mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7) } {}

	VF8(const float* data) : a { _mm256_loadu_ps(data) } {}

	VF8(const float* data, __mmask8 mask) : a { _mm256_maskz_load_ps(mask, data) } {}

	VF8(const unsigned char* data) : a { _mm256_cvtepu32_ps(_mm256_cvtepu8_epi32(_mm_maskz_loadu_epi8(0xFF, data))) } {}

	VF8(const unsigned char* data, __mmask8 mask) : a { _mm256_cvtepu32_ps(_mm256_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

	VF8 operator + (VF8 other) { return _mm256_add_ps(a, other.a); }
	VF8 operator - (VF8 other) { return _mm256_sub_ps(a, other.a); }
	VF8 operator * (VF8 other) { return _mm256_mul_ps(a, other.a); }
	VF8 operator / (VF8 other) { return _mm256_div_ps(a, other.a); }
	VF8 add(VF8 other) { return _mm256_add_ps(a, other.a); }
	VF8 sub(VF8 other) { return _mm256_sub_ps(a, other.a); }
	VF8 mul(VF8 other) { return _mm256_mul_ps(a, other.a); }
	VF8 div(VF8 other) { return _mm256_div_ps(a, other.a); }

	float operator [] (size_t i) const { return at(i); }

	friend std::ostream& operator << (std::ostream& os, const VF8& vec) {
		for (int i = 0; i < 8; i++) os << vec[i] << " ";
		return os;
	}

	float at(size_t i) const {
		return a.m256_f32[i];
	}

	float sum(int from, int to) const {
		float sum = 0.0f;
		for (int i = from; i < to; i++) sum += at(i);
		return sum;
	}

	float sum() const {
		return at(0) + at(1) + at(2) + at(3) + at(4) + at(5) + at(6) + at(7);
	}

	VF8 clamp(VF8 lo, VF8 hi) const {
		return _mm256_min_ps(_mm256_max_ps(a, lo.a), hi.a);
	}

	static VF8 clamp(VF8 value, VF8 lo, VF8 hi) {
		return value.clamp(lo, hi);
	}

	void storeu(float* dest) {
		_mm256_storeu_ps(dest, a);
	}

	void storeu(float* dest, __mmask8 mask) {
		_mm256_mask_storeu_ps(dest, mask, a);
	}
};


//wrapper for _mm128
class VF4 {
public:
	__m128 a;

	VF4() : a { _mm_setzero_ps() } {}

	VF4(float a) : a { _mm_set1_ps(a) } {}

	VF4(__m128 a) : a { a } {}

	VF4(float v0, float v1, float v2, float v3) :
		a { _mm_setr_ps(v0, v1, v2, v3) } {}

	VF4(const float* data) : a { _mm_loadu_ps(data) } {}

	VF4(const float* data, __mmask8 mask) : a { _mm_maskz_load_ps(mask, data) } {}

	VF4(const unsigned char* data) : a { _mm_cvtepu32_ps(_mm_cvtepu8_epi32(_mm_maskz_loadu_epi8(0xF, data))) } {}

	VF4(const unsigned char* data, __mmask8 mask) : a { _mm_cvtepu32_ps(_mm_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

	VF4 operator + (VF4 other) { return _mm_add_ps(a, other.a); }
	VF4 operator - (VF4 other) { return _mm_sub_ps(a, other.a); }
	VF4 operator * (VF4 other) { return _mm_mul_ps(a, other.a); }
	VF4 operator / (VF4 other) { return _mm_div_ps(a, other.a); }

	float operator [] (size_t i) const { return at(i); }

	friend std::ostream& operator << (std::ostream& os, const VF4& vec) {
		for (int i = 0; i < 4; i++) os << vec[i] << " ";
		return os;
	}

	float at(size_t i) const {
		return a.m128_f32[i];
	}

	float sum(int from, int to) const {
		float sum = 0.0f;
		for (int i = from; i < to; i++) sum += at(i);
		return sum;
	}

	float sum() const {
		return at(0) + at(1) + at(2) + at(3);
	}

	VF4 clamp(VF4 lo, VF4 hi) const {
		return _mm_min_ps(_mm_max_ps(a, lo.a), hi.a);
	}

	static VF4 clamp(VF4 value, VF4 lo, VF4 hi) {
		return value.clamp(lo, hi);
	}

	void storeu(float* dest) {
		_mm_storeu_ps(dest, a);
	}

	void storeu(float* dest, __mmask8 mask) {
		_mm_mask_storeu_ps(dest, mask, a);
	}
};