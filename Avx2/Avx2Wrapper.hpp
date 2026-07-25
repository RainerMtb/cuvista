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
#include <span>
#include "AffineSolverDirect.hpp"


//wrapper for __m256 (256 bits - 8 floats)
class alignas(32) V8f {
	__m256 a;

public:
	V8f();
	V8f(__m256 a);

	V8f(float a, float b, float c, float d, float e, float f, float g, float h);
	V8f(float a, float b);
	V8f(float a);

	V8f(const float* data);
	V8f(const unsigned char* data);

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

	float at(size_t i) const;

	float operator [] (size_t i) const;

	friend std::ostream& operator << (std::ostream& os, const V8f& vec);

	float sum(int from, int to) const;

	float sum() const;

	V8f clamp(V8f lo, V8f hi) const;

	void storeu(float* dest) const;

	std::vector<float> vector() const;

	operator __m256() const;
};


//wrapper for __m256d (256 bits - 4 doubles)
class alignas(32) V4d {
	__m256d a;

public:
	V4d();
	V4d(__m256d a);

	V4d(double a, double b, double c, double d);
	V4d(double a, double b);
	V4d(double a);

	V4d(const double* data);
	V4d(const unsigned char* data);

	V4d operator + (V4d other) const;
	V4d operator - (V4d other) const;
	V4d operator * (V4d other) const;
	V4d operator / (V4d other) const;
	V4d operator += (V4d other);
	V4d operator -= (V4d other);
	V4d operator *= (V4d other);
	V4d operator /= (V4d other);
	V4d add(V4d other) const;
	V4d sub(V4d other) const;
	V4d mul(V4d other) const;
	V4d div(V4d other) const;

	template <int i> V4d rot() const {
		return _mm256_castsi256_pd(_mm256_alignr_epi32(_mm256_castpd_si256(a), _mm256_castpd_si256(a), i));
	}

	double at(size_t i) const;

	double operator [] (size_t i) const;

	friend std::ostream& operator << (std::ostream& os, const V4d& vec);

	double sum(int from, int to) const;

	double sum() const;

	V4d clamp(V4d lo, V4d hi) const;

	void storeu(double* dest) const;

	std::vector<double> vector() const;

	operator __m256d() const;
};


//wrapper for __m128 (128 bits - 4 floats)
class alignas(32) V4f {
	__m128 a;

public:
	V4f();
	V4f(__m128 a);

	V4f(float a, float b, float c, float d);
	V4f(float a, float b);
	V4f(float a);

	V4f(const float* data);
	V4f(const unsigned char* data);

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

	float at(size_t i) const;

	float operator [] (size_t i) const;

	friend std::ostream& operator << (std::ostream& os, const V4f& vec);

	float sum(int from, int to) const;

	float sum() const;

	V4f clamp(V4f lo, V4f hi) const;

	void storeu(float* dest) const;

	std::vector<float> vector() const;

	operator __m128() const;
};


class AffineSolverAvx2 : public AffineSolverDirect {

public:
	AffineSolverAvx2(ThreadPoolBase& threadPool, size_t maxPoints);

	void computeSimilar(std::span<PointBase> points, Affine2D& dest) override;
};
