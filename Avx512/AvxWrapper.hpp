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

	static void yuvToRgbaPacked(V16f y, V16f u, V16f v, unsigned char* dest, V16f fu, V16f fv);
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

	static void inv(std::span<V8d> v);
	static void inv(std::span<V8d> v, std::span<size_t> piv);

	static double norm1(std::span<V8d> v);

	static void toConsole(std::span<V8d> v, int digits = 5);
	static void toConsole(V8d v, int digits = 5);
};


class AffineSolverAvx512 : public AffineSolverDirect {

public:
	AffineSolverAvx512(ThreadPoolBase& threadPool, size_t maxPoints);

	void computeSimilar(std::span<PointBase> points, Affine2D& dest) override;
};