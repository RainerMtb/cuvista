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

 //wrapper for __m512 (512 bits - 16 floats)
class V16f {
	__m512 a;

public:
	V16f() : a { _mm512_setzero_ps() } {}

	V16f(float a) : a { _mm512_set1_ps(a) } {}

	V16f(__m512 a) : a { a } {}

	V16f(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7,
		float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15) :
		a { _mm512_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) } {}

	V16f(float a, float b) : V16f(a, b, a, b, a, b, a, b, a, b, a, b, a, b, a, b) {}

	V16f(float a, float b, float c, float d) : V16f(a, b, c, d, a, b, c, d, a, b, c, d, a, b, c, d) {}

	V16f(const float* data) : a { _mm512_loadu_ps(data) } {}

	V16f(const float* data, __mmask16 mask) : a { _mm512_maskz_loadu_ps(mask, data) } {}

	V16f(const unsigned char* data) : a { _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm_loadu_epi8(data))) } {}

	V16f(const unsigned char* data, __mmask16 mask) : a { _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

	V16f operator + (V16f other) { return _mm512_add_ps(a, other.a); }
	V16f operator - (V16f other) { return _mm512_sub_ps(a, other.a); }
	V16f operator * (V16f other) { return _mm512_mul_ps(a, other.a); }
	V16f operator / (V16f other) { return _mm512_div_ps(a, other.a); }
	V16f operator += (V16f other) { a = _mm512_add_ps(a, other.a); return *this; }
	V16f operator -= (V16f other) { a = _mm512_sub_ps(a, other.a); return *this; }
	V16f operator *= (V16f other) { a = _mm512_mul_ps(a, other.a); return *this; }
	V16f operator /= (V16f other) { a = _mm512_div_ps(a, other.a); return *this; }
	V16f add(V16f other) const { return _mm512_add_ps(a, other.a); }
	V16f sub(V16f other) const { return _mm512_sub_ps(a, other.a); }
	V16f mul(V16f other) const { return _mm512_mul_ps(a, other.a); }
	V16f div(V16f other) const { return _mm512_div_ps(a, other.a); }

	template <int i> V16f rot() { return _mm512_castsi512_ps(_mm512_alignr_epi32(_mm512_castps_si512(a), _mm512_castps_si512(a), i)); }

	V16f broadcast(int i) const { return _mm512_permutexvar_ps(_mm512_set1_epi32(i), a); }

	float operator [] (size_t i) const { return at(i); }

#ifdef _MSC_VER
	float at(size_t i) const { return a.m512_f32[i]; }
#else
	float at(size_t i) const { return a[i]; }
#endif

	friend std::ostream& operator << (std::ostream& os, const V16f& vec) {
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

	V16f clamp(V16f lo, V16f hi) const {
		return _mm512_min_ps(_mm512_max_ps(a, lo.a), hi.a);
	}

	void storeu(float* dest) const {
		_mm512_storeu_ps(dest, a);
	}

	void storeu(float* dest, __mmask16 mask) const {
		_mm512_mask_storeu_ps(dest, mask, a);
	}

	operator __m512();
};


class V8d;

//wrapper for __m256 (256 bits - 8 floats)
class V8f {
	__m256 a;

public:
	V8f() : a { _mm256_setzero_ps() } {}

	V8f(float a) : a { _mm256_set1_ps(a) } {}

	V8f(__m256 a) : a { a } {}

	V8f(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) :
		a { _mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7) } {}

	V8f(float a, float b) : V8f(a, b, a, b, a, b, a, b) {}

	V8f(const float* data) : a { _mm256_loadu_ps(data) } {}

	V8f(const float* data, __mmask8 mask) : a { _mm256_maskz_loadu_ps(mask, data) } {}

	V8f(const unsigned char* data) : a { _mm256_cvtepu32_ps(_mm256_cvtepu8_epi32(_mm_maskz_loadu_epi8(0xFF, data))) } {}

	V8f(const unsigned char* data, __mmask8 mask) : a { _mm256_cvtepu32_ps(_mm256_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

	V8f operator + (V8f other) { return _mm256_add_ps(a, other.a); }
	V8f operator - (V8f other) { return _mm256_sub_ps(a, other.a); }
	V8f operator * (V8f other) { return _mm256_mul_ps(a, other.a); }
	V8f operator / (V8f other) { return _mm256_div_ps(a, other.a); }
	V8f operator += (V8f other) { a = _mm256_add_ps(a, other.a); return *this; }
	V8f operator -= (V8f other) { a = _mm256_sub_ps(a, other.a); return *this; }
	V8f operator *= (V8f other) { a = _mm256_mul_ps(a, other.a); return *this; }
	V8f operator /= (V8f other) { a = _mm256_div_ps(a, other.a); return *this; }
	V8f add(V8f other) const { return _mm256_add_ps(a, other.a); }
	V8f sub(V8f other) const { return _mm256_sub_ps(a, other.a); }
	V8f mul(V8f other) const { return _mm256_mul_ps(a, other.a); }
	V8f div(V8f other) const { return _mm256_div_ps(a, other.a); }

	template <int i> V8f rot() { return _mm256_castsi256_ps(_mm256_alignr_epi32(_mm256_castps_si256(a), _mm256_castps_si256(a), i)); }

	V8f broadcast(int i) const { return _mm256_permutexvar_ps(_mm256_set1_epi32(i), a); }

	float operator [] (size_t i) const { return at(i); }

#ifdef _MSC_VER
	float at(size_t i) const { return a.m256_f32[i]; }
#else
	float at(size_t i) const { return a[i]; }
#endif

	friend std::ostream& operator << (std::ostream& os, const V8f& vec) {
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

	V8f clamp(V8f lo, V8f hi) const {
		return _mm256_min_ps(_mm256_max_ps(a, lo.a), hi.a);
	}

	void storeu(float* dest) const {
		_mm256_storeu_ps(dest, a);
	}

	void storeu(float* dest, __mmask8 mask) const {
		_mm256_mask_storeu_ps(dest, mask, a);
	}

	operator __m256();
};


//wrapper for __m128 (128 bits - 4 floats)
class V4f {
	__m128 a;

public:
	V4f() : a { _mm_setzero_ps() } {}

	V4f(float a) : a { _mm_set_ps1(a) } {}

	V4f(__m128 a) : a { a } {}

	V4f(float v0, float v1, float v2, float v3) :
		a { _mm_setr_ps(v0, v1, v2, v3) } {}

	V4f(float a, float b) : V4f(a, b, a, b) {}

	V4f(const float* data) : a { _mm_loadu_ps(data) } {}

	V4f(const float* data, __mmask8 mask) : a { _mm_maskz_loadu_ps(mask, data) } {}

	V4f(const unsigned char* data) : a { _mm_cvtepu32_ps(_mm_cvtepu8_epi32(_mm_maskz_loadu_epi8(0xF, data))) } {}

	V4f(const unsigned char* data, __mmask8 mask) : a { _mm_cvtepu32_ps(_mm_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

	V4f operator + (V4f other) { return _mm_add_ps(a, other.a); }
	V4f operator - (V4f other) { return _mm_sub_ps(a, other.a); }
	V4f operator * (V4f other) { return _mm_mul_ps(a, other.a); }
	V4f operator / (V4f other) { return _mm_div_ps(a, other.a); }
	V4f operator += (V4f other) { a = _mm_add_ps(a, other.a); return *this; }
	V4f operator -= (V4f other) { a = _mm_sub_ps(a, other.a); return *this; }
	V4f operator *= (V4f other) { a = _mm_mul_ps(a, other.a); return *this; }
	V4f operator /= (V4f other) { a = _mm_div_ps(a, other.a); return *this; }
	V4f add(V4f other) const { return _mm_add_ps(a, other.a); }
	V4f sub(V4f other) const { return _mm_sub_ps(a, other.a); }
	V4f mul(V4f other) const { return _mm_mul_ps(a, other.a); }
	V4f div(V4f other) const { return _mm_div_ps(a, other.a); }

	template <int i> V4f rot() { return _mm_castsi128_ps(_mm_alignr_epi32(_mm_castps_si128(a), _mm_castps_si128(a), i)); }

	V4f broadcast(int i) const { return _mm_permutevar_ps(a, _mm_set1_epi32(i)); }

	float operator [] (size_t i) const { return at(i); }

#ifdef _MSC_VER
	float at(size_t i) const { return a.m128_f32[i]; }
#else
	float at(size_t i) const { return a[i]; }
#endif

	friend std::ostream& operator << (std::ostream& os, const V4f& vec) {
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

	V4f clamp(V4f lo, V4f hi) const {
		return _mm_min_ps(_mm_max_ps(a, lo.a), hi.a);
	}

	void storeu(float* dest) const {
		_mm_storeu_ps(dest, a);
	}

	void storeu(float* dest, __mmask8 mask) const {
		_mm_mask_storeu_ps(dest, mask, a);
	}

	operator __m128();
};


//wrapper for __m512d (512 bits - 8 doubles)
class V8d {
	__m512d a;

public:
	V8d() : a { _mm512_setzero_pd() } {}

	V8d(double a) : a { _mm512_set1_pd(a) } {}

	V8d(double a, double b) : V8d(a, b, a, b, a, b, a, b) {}

	V8d(__m512d a) : a { a } {}

	V8d(__m256 a) : a { _mm512_cvtps_pd(a) } {}

	V8d(V8f a) : a { _mm512_cvtps_pd(a) } {}

	V8d(double v0, double v1, double v2, double v3, double v4, double v5, double v6, double v7) :
		a { _mm512_setr_pd(v0, v1, v2, v3, v4, v5, v6, v7) } {}

	V8d(const double* data) : a { _mm512_loadu_pd(data) } {}

	V8d(const double* data, __mmask8 mask) : a { _mm512_maskz_loadu_pd(mask, data) } {}

	V8d(const unsigned char* data) : a { _mm512_cvtepu64_pd(_mm512_cvtepu8_epi64(_mm_maskz_loadu_epi8(0xFF, data))) } {}

	V8d(const unsigned char* data, __mmask8 mask) : a { _mm512_cvtepu64_pd(_mm512_cvtepu8_epi64(_mm_maskz_loadu_epi8(mask, data))) } {}

	V8d operator + (V8d other) { return _mm512_add_pd(a, other.a); }
	V8d operator - (V8d other) { return _mm512_sub_pd(a, other.a); }
	V8d operator * (V8d other) { return _mm512_mul_pd(a, other.a); }
	V8d operator / (V8d other) { return _mm512_div_pd(a, other.a); }
	V8d operator += (V8d other) { a = _mm512_add_pd(a, other.a); return *this; }
	V8d operator -= (V8d other) { a = _mm512_sub_pd(a, other.a); return *this; }
	V8d operator *= (V8d other) { a = _mm512_mul_pd(a, other.a); return *this; }
	V8d operator /= (V8d other) { a = _mm512_div_pd(a, other.a); return *this; }
	V8d add(V8d other) const { return _mm512_add_pd(a, other.a); }
	V8d sub(V8d other) const { return _mm512_sub_pd(a, other.a); }
	V8d mul(V8d other) const { return _mm512_mul_pd(a, other.a); }
	V8d div(V8d other) const { return _mm512_div_pd(a, other.a); }

	template <int i> V8d rot() { return _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(a), _mm512_castpd_si512(a), i)); }

	V8d broadcast(int i) const { return _mm512_permutexvar_pd(_mm512_set1_epi64(i), a); }

	double operator [] (size_t i) const { return at(i); }

#ifdef _MSC_VER
	double at(size_t i) const { return a.m512d_f64[i]; }
#else
	double at(size_t i) const { return a[i]; }
#endif

	friend std::ostream& operator << (std::ostream& os, const V8d& vec) {
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

	V8d clamp(V8d lo, V8d hi) const {
		return _mm512_min_pd(_mm512_max_pd(a, lo.a), hi.a);
	}

	void storeu(double* dest) const {
		_mm512_storeu_pd(dest, a);
	}

	void storeu(double* dest, __mmask8 mask) const {
		_mm512_mask_storeu_pd(dest, mask, a);
	}

	operator __m512d();
};
