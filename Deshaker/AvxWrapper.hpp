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

#include <immintrin.h>
#include <iostream>

 //wrapper for _mm512
class VF16 {
	__m512 a;

public:
	VF16() : a { _mm512_setzero_ps() } {}

	VF16(float a) : a { _mm512_set1_ps(a) } {}

	VF16(__m512 a) : a { a } {}

	VF16(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7,
		float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15) :
		a { _mm512_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) } {}

	VF16(float a, float b) : VF16(a, b, a, b, a, b, a, b, a, b, a, b, a, b, a, b) {}

	VF16(float a, float b, float c, float d) : VF16(a, b, c, d, a, b, c, d, a, b, c, d, a, b, c, d) {}

	VF16(const float* data) : a { _mm512_loadu_ps(data) } {}

	VF16(const float* data, __mmask16 mask) : a { _mm512_maskz_loadu_ps(mask, data) } {}

	VF16(const unsigned char* data) : a { _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm_loadu_epi8(data))) } {}

	VF16(const unsigned char* data, __mmask16 mask) : a { _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

	VF16 operator + (VF16 other) { return _mm512_add_ps(a, other.a); }
	VF16 operator - (VF16 other) { return _mm512_sub_ps(a, other.a); }
	VF16 operator * (VF16 other) { return _mm512_mul_ps(a, other.a); }
	VF16 operator / (VF16 other) { return _mm512_div_ps(a, other.a); }
	VF16 operator += (VF16 other) { a = _mm512_add_ps(a, other.a); return *this; }
	VF16 operator -= (VF16 other) { a = _mm512_sub_ps(a, other.a); return *this; }
	VF16 operator *= (VF16 other) { a = _mm512_mul_ps(a, other.a); return *this; }
	VF16 operator /= (VF16 other) { a = _mm512_div_ps(a, other.a); return *this; }
	VF16 add(VF16 other) { return _mm512_add_ps(a, other.a); }
	VF16 sub(VF16 other) { return _mm512_sub_ps(a, other.a); }
	VF16 mul(VF16 other) { return _mm512_mul_ps(a, other.a); }
	VF16 div(VF16 other) { return _mm512_div_ps(a, other.a); }

	template <int i> VF16 rot() { return _mm512_castsi512_ps(_mm512_alignr_epi32(_mm512_castps_si512(a), _mm512_castps_si512(a), i)); }

	float operator [] (size_t i) const { return at(i); }
	float& operator [] (size_t i) { return at(i); }

#ifdef _MSC_VER
	float at(size_t i) const { return a.m512_f32[i]; }
	float& at(size_t i) { return a.m512_f32[i]; }
#else
	float at(size_t i) const { return a[i]; }
	float& at(size_t i) { return a[i]; }
#endif

	friend std::ostream& operator << (std::ostream& os, const VF16& vec) {
		for (int i = 0; i < 16; i++) os << vec[i] << " ";
		return os;
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

	void storeu(float* dest) const {
		_mm512_storeu_ps(dest, a);
	}

	void storeu(float* dest, __mmask16 mask) const {
		_mm512_mask_storeu_ps(dest, mask, a);
	}

	operator __m512() { return a; }
};


class VD8;

//wrapper for _mm256
class VF8 {
	__m256 a;

public:
	VF8() : a { _mm256_setzero_ps() } {}

	VF8(float a) : a { _mm256_set1_ps(a) } {}

	VF8(__m256 a) : a { a } {}

	VF8(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) :
		a { _mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7) } {}

	VF8(float a, float b) : VF8(a, b, a, b, a, b, a, b) {}

	VF8(const float* data) : a { _mm256_loadu_ps(data) } {}

	VF8(const float* data, __mmask8 mask) : a { _mm256_maskz_loadu_ps(mask, data) } {}

	VF8(const unsigned char* data) : a { _mm256_cvtepu32_ps(_mm256_cvtepu8_epi32(_mm_maskz_loadu_epi8(0xFF, data))) } {}

	VF8(const unsigned char* data, __mmask8 mask) : a { _mm256_cvtepu32_ps(_mm256_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

	VF8 operator + (VF8 other) { return _mm256_add_ps(a, other.a); }
	VF8 operator - (VF8 other) { return _mm256_sub_ps(a, other.a); }
	VF8 operator * (VF8 other) { return _mm256_mul_ps(a, other.a); }
	VF8 operator / (VF8 other) { return _mm256_div_ps(a, other.a); }
	VF8 operator += (VF8 other) { a = _mm256_add_ps(a, other.a); return *this; }
	VF8 operator -= (VF8 other) { a = _mm256_sub_ps(a, other.a); return *this; }
	VF8 operator *= (VF8 other) { a = _mm256_mul_ps(a, other.a); return *this; }
	VF8 operator /= (VF8 other) { a = _mm256_div_ps(a, other.a); return *this; }
	VF8 add(VF8 other) { return _mm256_add_ps(a, other.a); }
	VF8 sub(VF8 other) { return _mm256_sub_ps(a, other.a); }
	VF8 mul(VF8 other) { return _mm256_mul_ps(a, other.a); }
	VF8 div(VF8 other) { return _mm256_div_ps(a, other.a); }

	template <int i> VF8 rot() { return _mm256_castsi256_ps(_mm256_alignr_epi32(_mm256_castps_si256(a), _mm256_castps_si256(a), i)); }

	float operator [] (size_t i) const { return at(i); }
	float& operator [] (size_t i) { return at(i); }

#ifdef _MSC_VER
	float at(size_t i) const { return a.m256_f32[i]; }
	float& at(size_t i) { return a.m256_f32[i]; }
#else
	float at(size_t i) const { return a[i]; }
	float& at(size_t i) { return a[i]; }
#endif

	friend std::ostream& operator << (std::ostream& os, const VF8& vec) {
		for (int i = 0; i < 8; i++) os << vec[i] << " ";
		return os;
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

	void storeu(float* dest) const {
		_mm256_storeu_ps(dest, a);
	}

	void storeu(float* dest, __mmask8 mask) const {
		_mm256_mask_storeu_ps(dest, mask, a);
	}

	operator __m256() { return a; }
};


//wrapper for _m128 (128 bits - 4 floats)
class VF4 {
	__m128 a;

public:
	VF4() : a { _mm_setzero_ps() } {}

	VF4(float a) : a { _mm_set_ps1(a) } {}

	VF4(__m128 a) : a { a } {}

	VF4(float v0, float v1, float v2, float v3) :
		a { _mm_setr_ps(v0, v1, v2, v3) } {}

	VF4(float a, float b) : VF4(a, b, a, b) {}

	VF4(const float* data) : a { _mm_loadu_ps(data) } {}

	VF4(const float* data, __mmask8 mask) : a { _mm_maskz_loadu_ps(mask, data) } {}

	VF4(const unsigned char* data) : a { _mm_cvtepu32_ps(_mm_cvtepu8_epi32(_mm_maskz_loadu_epi8(0xF, data))) } {}

	VF4(const unsigned char* data, __mmask8 mask) : a { _mm_cvtepu32_ps(_mm_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

	VF4 operator + (VF4 other) { return _mm_add_ps(a, other.a); }
	VF4 operator - (VF4 other) { return _mm_sub_ps(a, other.a); }
	VF4 operator * (VF4 other) { return _mm_mul_ps(a, other.a); }
	VF4 operator / (VF4 other) { return _mm_div_ps(a, other.a); }
	VF4 operator += (VF4 other) { a = _mm_add_ps(a, other.a); return *this; }
	VF4 operator -= (VF4 other) { a = _mm_sub_ps(a, other.a); return *this; }
	VF4 operator *= (VF4 other) { a = _mm_mul_ps(a, other.a); return *this; }
	VF4 operator /= (VF4 other) { a = _mm_div_ps(a, other.a); return *this; }
	VF4 add(VF4 other) { return _mm_add_ps(a, other.a); }
	VF4 sub(VF4 other) { return _mm_sub_ps(a, other.a); }
	VF4 mul(VF4 other) { return _mm_mul_ps(a, other.a); }
	VF4 div(VF4 other) { return _mm_div_ps(a, other.a); }

	template <int i> VF4 rot() { return _mm_castsi128_ps(_mm_alignr_epi32(_mm_castps_si128(a), _mm_castps_si128(a), i)); }

	float operator [] (size_t i) const { return at(i); }
	float& operator [] (size_t i) { return at(i); }

#ifdef _MSC_VER
	float at(size_t i) const { return a.m128_f32[i]; }
	float& at(size_t i) { return a.m128_f32[i]; }
#else
	float at(size_t i) const { return a[i]; }
	float& at(size_t i) { return a[i]; }
#endif

	friend std::ostream& operator << (std::ostream& os, const VF4& vec) {
		for (int i = 0; i < 4; i++) os << vec[i] << " ";
		return os;
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

	void storeu(float* dest) const {
		_mm_storeu_ps(dest, a);
	}

	void storeu(float* dest, __mmask8 mask) const {
		_mm_mask_storeu_ps(dest, mask, a);
	}

	operator __m128() { return a; }
};


//wrapper for _mm512d
class VD8 {
	__m512d a;

public:
	VD8() : a { _mm512_setzero_pd() } {}

	VD8(double a) : a { _mm512_set1_pd(a) } {}

	VD8(double a, double b) : VD8(a, b, a, b, a, b, a, b) {}

	VD8(__m512d a) : a { a } {}

	VD8(__m256 a) : a { _mm512_cvtps_pd(a) } {}

	VD8(VF8 a) : a { _mm512_cvtps_pd(a) } {}

	VD8(double v0, double v1, double v2, double v3, double v4, double v5, double v6, double v7) :
		a { _mm512_setr_pd(v0, v1, v2, v3, v4, v5, v6, v7) } {}

	VD8(const double* data) : a { _mm512_loadu_pd(data) } {}

	VD8(const double* data, __mmask8 mask) : a { _mm512_maskz_loadu_pd(mask, data) } {}

	VD8(const unsigned char* data) : a { _mm512_cvtepu64_pd(_mm512_cvtepu8_epi64(_mm_maskz_loadu_epi8(0xFF, data))) } {}

	VD8(const unsigned char* data, __mmask8 mask) : a { _mm512_cvtepu64_pd(_mm512_cvtepu8_epi64(_mm_maskz_loadu_epi8(mask, data))) } {}

	VD8 operator + (VD8 other) { return _mm512_add_pd(a, other.a); }
	VD8 operator - (VD8 other) { return _mm512_sub_pd(a, other.a); }
	VD8 operator * (VD8 other) { return _mm512_mul_pd(a, other.a); }
	VD8 operator / (VD8 other) { return _mm512_div_pd(a, other.a); }
	VD8 operator += (VD8 other) { a = _mm512_add_pd(a, other.a); return *this; }
	VD8 operator -= (VD8 other) { a = _mm512_sub_pd(a, other.a); return *this; }
	VD8 operator *= (VD8 other) { a = _mm512_mul_pd(a, other.a); return *this; }
	VD8 operator /= (VD8 other) { a = _mm512_div_pd(a, other.a); return *this; }
	VD8 add(VD8 other) { return _mm512_add_pd(a, other.a); }
	VD8 sub(VD8 other) { return _mm512_sub_pd(a, other.a); }
	VD8 mul(VD8 other) { return _mm512_mul_pd(a, other.a); }
	VD8 div(VD8 other) { return _mm512_div_pd(a, other.a); }

	template <int i> VD8 rot() { return _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(a), _mm512_castpd_si512(a), i)); }

	double operator [] (size_t i) const { return at(i); }
	double& operator [] (size_t i) { return at(i); }

#ifdef _MSC_VER
	double at(size_t i) const { return a.m512d_f64[i]; }
	double& at(size_t i) { return a.m512d_f64[i]; }
#else
	double at(size_t i) const { return a[i]; }
	double& at(size_t i) { return a[i]; }
#endif

	friend std::ostream& operator << (std::ostream& os, const VD8& vec) {
		for (int i = 0; i < 8; i++) os << vec[i] << " ";
		return os;
	}

	double sum(int from, int to) const {
		double sum = 0.0;
		for (int i = from; i < to; i++) sum += at(i);
		return sum;
	}

	double sum() const {
		return at(0) + at(1) + at(2) + at(3) + at(4) + at(5) + at(6) + at(7);
	}

	VD8 clamp(VD8 lo, VD8 hi) const {
		return _mm512_min_pd(_mm512_max_pd(a, lo.a), hi.a);
	}

	void storeu(double* dest) const {
		_mm512_storeu_pd(dest, a);
	}

	void storeu(double* dest, __mmask8 mask) const {
		_mm512_mask_storeu_pd(dest, mask, a);
	}

	operator __m512d() { return a; }
};