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
#include <vector>
#include <cstdint>

struct Iotas {
	int32_t i32x8[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 };
	float fx8[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 };
	
	int32_t i32x16[32] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    float fx16[32] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

	int64_t i64x8[16] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
	double dx8[16] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
};

inline Iotas iotas;

//wrapper for __m512 (512 bits - 16 floats)
class V16f {
	__m512 a;

public:
	V16f();
	V16f(__m512 a);

	V16f(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7,
		float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15);
	V16f(float a, float b, float c, float d);
	V16f(float a, float b);
	V16f(float a);

	V16f(const float* data);
	V16f(const float* data, __mmask16 mask);
	V16f(const unsigned char* data);
	V16f(const unsigned char* data, __mmask16 mask);

	V16f operator + (V16f other) const;
	V16f operator - (V16f other) const;
	V16f operator * (V16f other) const;
	V16f operator / (V16f other) const;
	V16f operator += (V16f other);
	V16f operator -= (V16f other);
	V16f operator *= (V16f other);
	V16f operator /= (V16f other);
	V16f add(V16f other) const;
	V16f sub(V16f other) const;
	V16f mul(V16f other) const;
	V16f div(V16f other) const;

	template <int i> V16f rot() const {
		return _mm512_castsi512_ps(_mm512_alignr_epi32(_mm512_castps_si512(a), _mm512_castps_si512(a), i));
	}

	V16f rot(int i) const;

	V16f broadcast(int i) const;

	float at(size_t i) const;

	float operator [] (size_t i) const;

	friend std::ostream& operator << (std::ostream& os, const V16f& vec);

	float sum(int from, int to) const;

	float sum() const;

	V16f clamp(V16f lo, V16f hi) const;

	void storeu(float* dest) const;

	void storeu(float* dest, __mmask16 mask) const;

	std::vector<float> vector() const;

	operator __m512() const;
};


//wrapper for __m256 (256 bits - 8 floats)
class V8f {
	__m256 a;

public:
	V8f();
	V8f(__m256 a);

	V8f(float a, float b, float c, float d, float e, float f, float g, float h);
	V8f(float a, float b);
	V8f(float a);

	V8f(const float* data);
	V8f(const float* data, __mmask8 mask);
	V8f(const unsigned char* data);
	V8f(const unsigned char* data, __mmask8 mask);

	V8f operator + (V8f other) const;
	V8f operator - (V8f other) const;
	V8f operator * (V8f other) const;
	V8f operator / (V8f other) const;
	V8f operator += (V8f other);
	V8f operator -= (V8f other);
	V8f operator *= (V8f other);
	V8f operator /= (V8f other);
	V8f add(V8f other) const;
	V8f sub(V8f other) const;
	V8f mul(V8f other) const;
	V8f div(V8f other) const;

	template <int i> V8f rot() const {
		return _mm256_castsi256_ps(_mm256_alignr_epi32(_mm256_castps_si256(a), _mm256_castps_si256(a), i));
	}

	V8f rot(int i) const;

	V8f broadcast(int i) const;

	float at(size_t i) const;

	float operator [] (size_t i) const;

	friend std::ostream& operator << (std::ostream& os, const V8f& vec);

	float sum(int from, int to) const;

	float sum() const;

	V8f clamp(V8f lo, V8f hi) const;

	void storeu(float* dest) const;

	void storeu(float* dest, __mmask8 mask) const;

	std::vector<float> vector() const;

	operator __m256() const;
};


//wrapper for __m128 (128 bits - 4 floats)
class V4f {
	__m128 a;

public:
	V4f();
	V4f(__m128 a);

	V4f(float a, float b, float c, float d);
	V4f(float a, float b);
	V4f(float a);

	V4f(const float* data);
	V4f(const float* data, __mmask8 mask);
	V4f(const unsigned char* data);
	V4f(const unsigned char* data, __mmask8 mask);

	V4f operator + (V4f other) const;
	V4f operator - (V4f other) const;
	V4f operator * (V4f other) const;
	V4f operator / (V4f other) const;
	V4f operator += (V4f other);
	V4f operator -= (V4f other);
	V4f operator *= (V4f other);
	V4f operator /= (V4f other);
	V4f add(V4f other) const;
	V4f sub(V4f other) const;
	V4f mul(V4f other) const;
	V4f div(V4f other) const;

	template <int i> V4f rot() const {
		return _mm_castsi128_ps(_mm_alignr_epi32(_mm_castps_si128(a), _mm_castps_si128(a), i));
	}

	V4f rot(int i) const;

	V4f broadcast(int i) const;

	float at(size_t i) const;

	float operator [] (size_t i) const;
	
	friend std::ostream& operator << (std::ostream& os, const V4f& vec);

	float sum(int from, int to) const;

	float sum() const;

	V4f clamp(V4f lo, V4f hi) const;

	void storeu(float* dest) const;

	void storeu(float* dest, __mmask8 mask) const;

	std::vector<float> vector() const;

	operator __m128() const;
};


//wrapper for __m512d (512 bits - 8 doubles)
class V8d {
	__m512d a;

public:
	V8d();
	V8d(__m512d a);
	V8d(__m256 a);

	V8d(double a, double b, double c, double d, double e, double f, double g, double h);
	V8d(double a, double b);
	V8d(double a);

	V8d(const double* data);
	V8d(const double* data, __mmask8 mask);
	V8d(const unsigned char* data);
	V8d(const unsigned char* data, __mmask8 mask);

	V8d operator + (V8d other) const;
	V8d operator - (V8d other) const;
	V8d operator * (V8d other) const;
	V8d operator / (V8d other) const;
	V8d operator += (V8d other);
	V8d operator -= (V8d other);
	V8d operator *= (V8d other);
	V8d operator /= (V8d other);
	V8d add(V8d other) const;
	V8d sub(V8d other) const;
	V8d mul(V8d other) const;
	V8d div(V8d other) const;

	template <int i> V8d rot() const {
		return _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(a), _mm512_castpd_si512(a), i));
	}

	V8d rot(int i) const;

	V8d broadcast(int i) const;

	double at(size_t i) const;

	double operator [] (size_t i) const;

	friend std::ostream& operator << (std::ostream& os, const V8d& vec);

	double sum(int from, int to) const;

	double sum() const;

	V8d clamp(V8d lo, V8d hi) const;

	void storeu(double* dest) const;

	void storeu(double* dest, __mmask8 mask) const;

	std::vector<double> vector() const;

	operator __m512d() const;
};
