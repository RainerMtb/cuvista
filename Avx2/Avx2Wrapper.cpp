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

#include "Avx2Wrapper.hpp"
#include "AvxUtil.hpp"

using namespace avx;

V8f::V8f() : a { _mm256_setzero_ps() } {}

V8f::V8f(float a) : a { _mm256_set1_ps(a) } {}

V8f::V8f(__m256 a) : a { a } {}

V8f::V8f(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) : a { _mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7) } {}

V8f::V8f(float a, float b) : V8f(a, b, a, b, a, b, a, b) {}

V8f::V8f(const float* data) : a { _mm256_loadu_ps(data) } {}

V8f::V8f(const unsigned char* data) : a { _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (data)))) } {}

V8f V8f::operator + (V8f other) const { return _mm256_add_ps(a, other.a); }
V8f V8f::operator - (V8f other) const { return _mm256_sub_ps(a, other.a); }
V8f V8f::operator * (V8f other) const { return _mm256_mul_ps(a, other.a); }
V8f V8f::operator / (V8f other) const { return _mm256_div_ps(a, other.a); }
V8f V8f::operator += (V8f other) { a = _mm256_add_ps(a, other.a); return *this; }
V8f V8f::operator -= (V8f other) { a = _mm256_sub_ps(a, other.a); return *this; }
V8f V8f::operator *= (V8f other) { a = _mm256_mul_ps(a, other.a); return *this; }
V8f V8f::operator /= (V8f other) { a = _mm256_div_ps(a, other.a); return *this; }
V8f V8f::add(V8f other) const { return _mm256_add_ps(a, other.a); }
V8f V8f::sub(V8f other) const { return _mm256_sub_ps(a, other.a); }
V8f V8f::mul(V8f other) const { return _mm256_mul_ps(a, other.a); }
V8f V8f::div(V8f other) const { return _mm256_div_ps(a, other.a); }

float V8f::operator [] (size_t i) const { return at(i); }

#ifdef _MSC_VER
float V8f::at(size_t i) const { return a.m256_f32[i]; }
#else
float V8f::at(size_t i) const { return a[i]; }
#endif

std::ostream& operator << (std::ostream& os, const V8f& vec) {
	for (int i = 0; i < 8; i++) os << vec[i] << " ";
	return os;
}

float V8f::sum(int from, int to) const {
	float sum = 0.0f;
	for (int i = from; i < to; i++) sum += at(i);
	return sum;
}

float V8f::sum() const {
	return at(0) + at(1) + at(2) + at(3) + at(4) + at(5) + at(6) + at(7);
}

V8f V8f::clamp(V8f lo, V8f hi) const {
	return _mm256_min_ps(_mm256_max_ps(a, lo.a), hi.a);
}

void V8f::storeu(float* dest) const {
	_mm256_storeu_ps(dest, a);
}

std::vector<float> V8f::vector() const {
	std::vector<float> v(8);
	_mm256_storeu_ps(v.data(), a);
	return v;
}

V8f::operator __m256() const { return a; }

//---------------------------------------------------


V4d::V4d() : a { _mm256_setzero_pd() } {}

V4d::V4d(double a) : a { _mm256_set1_pd(a) } {}

V4d::V4d(__m256d a) : a { a } {}

V4d::V4d(double a, double b, double c, double d) : a { _mm256_setr_pd(a, b, c, d) } {}

V4d::V4d(double a, double b) : V4d(a, b, a, b) {}

V4d::V4d(const double* data) : a { _mm256_loadu_pd(data) } {}

V4d::V4d(const unsigned char* data) : a { _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_loadu_si32(data))) } {}

V4d V4d::operator + (V4d other) const { return _mm256_add_pd(a, other.a); }
V4d V4d::operator - (V4d other) const { return _mm256_sub_pd(a, other.a); }
V4d V4d::operator * (V4d other) const { return _mm256_mul_pd(a, other.a); }
V4d V4d::operator / (V4d other) const { return _mm256_div_pd(a, other.a); }
V4d V4d::operator += (V4d other) { a = _mm256_add_pd(a, other.a); return *this; }
V4d V4d::operator -= (V4d other) { a = _mm256_sub_pd(a, other.a); return *this; }
V4d V4d::operator *= (V4d other) { a = _mm256_mul_pd(a, other.a); return *this; }
V4d V4d::operator /= (V4d other) { a = _mm256_div_pd(a, other.a); return *this; }
V4d V4d::add(V4d other) const { return _mm256_add_pd(a, other.a); }
V4d V4d::sub(V4d other) const { return _mm256_sub_pd(a, other.a); }
V4d V4d::mul(V4d other) const { return _mm256_mul_pd(a, other.a); }
V4d V4d::div(V4d other) const { return _mm256_div_pd(a, other.a); }

double V4d::operator [] (size_t i) const { return at(i); }

#ifdef _MSC_VER
double V4d::at(size_t i) const { return a.m256d_f64[i]; }
#else
double V4d::at(size_t i) const { return a[i]; }
#endif

std::ostream& operator << (std::ostream& os, const V4d& vec) {
	for (int i = 0; i < 4; i++) os << vec[i] << " ";
	return os;
}

double V4d::sum(int from, int to) const {
	double sum = 0.0;
	for (int i = from; i < to; i++) sum += at(i);
	return sum;
}

double V4d::sum() const {
	return at(0) + at(1) + at(2) + at(3);
}

V4d V4d::clamp(V4d lo, V4d hi) const {
	return _mm256_min_pd(_mm256_max_pd(a, lo.a), hi.a);
}

void V4d::storeu(double* dest) const {
	_mm256_storeu_pd(dest, a);
}

std::vector<double> V4d::vector() const {
	std::vector<double> v(4);
	_mm256_storeu_pd(v.data(), a);
	return v;
}

V4d::operator __m256d() const { return a; }

//compute similar transform using avx
AffineSolverAvx2::AffineSolverAvx2(ThreadPoolBase& threadPool, size_t maxPoints) :
	AffineSolverDirect(threadPool, maxPoints)
{}

//compute similar transform using avx
void AffineSolverAvx2::computeSimilar(std::span<PointBase> points, Affine2D& dest) {
	//util::ConsoleTimer timer("computeSimilar avx2");
	initParam(points);

	double* p = M.addr(4, 0);
	double* q = M.addr(5, 0);

	M[0][0] = b / n;
	M[0][1] = 0.0;
	M[0][2] = 0.0;
	M[0][3] = p[3] / n;

	M[1][0] = 0.0;
	M[1][1] = 1.0 + p[0] / n;
	M[1][2] = -p[3] / n;
	M[1][3] = 0.0;

	M[2][0] = -s0 / n;
	M[2][1] = s1 / n;
	M[2][2] = 1.0 + e - p[3] * f;
	M[2][3] = -p[3] * g;

	M[3][0] = -s1 / n;
	M[3][1] = -s0 / n;
	M[3][2] = 0.0;
	M[3][3] = 1.0 + e + p[3] * h;

	V4d pd_n0 = n;
	V4d pd_n1 = V4d(-n, n);
	V4d pd_e = V4d(e, 0);
	V4d pd_f = V4d(-f, f);
	V4d pd_g = g;
	V4d pd_ek = V4d(-k, e);
	V4d pd_h = h;
	V4d pd_j = V4d(j, -j);
	for (size_t idx = 4; idx < m; idx += 4) {
		V4d pd_a = p + idx;
		V4d pd_b = _mm256_shuffle_pd(pd_a, pd_a, 0b0101); //switch idx <-> idx+1
		(pd_a / pd_n0).storeu(M.addr(0, idx));
		(pd_b / pd_n1).storeu(M.addr(1, idx));
		(pd_e + pd_b * pd_f - pd_a * pd_g).storeu(M.addr(2, idx));
		(pd_ek + pd_a * pd_h + pd_b * pd_j).storeu(M.addr(3, idx));
	}

	//clear padding values to 0
	for (size_t i = 0; i < 6; i++) {
		double* ptr = M.addr(i, 0);
		std::fill(ptr + m, ptr + M.cols(), 0.0);
	}

	//back substitution step 1
	double* ptr5 = M.addr(5, 0);
	for (size_t k = 0; k < 4; k++) {
		double* ptrk = M.addr(k, 0);
		__m256d s = _mm256_set1_pd(0.0);
		__m256d x = _mm256_set1_pd(-M[k][k]);

		//inner product, first loop
		for (size_t i = k; i < m; i += 4) {
			__m256d a = _mm256_loadu_pd(ptrk + i);
			__m256d b = _mm256_loadu_pd(ptr5 + i);
			s = _mm256_fmadd_pd(a, b, s);
		}

		//horizontal sum
		__m256d b;
		b = _mm256_shuffle_pd(s, s, 0b0101);
		s = _mm256_add_pd(s, b);

		b = _mm256_permute2f128_pd(s, s, 0b0001);
		s = _mm256_add_pd(s, b);

		//second loop
		s = _mm256_div_pd(s, x);
		for (size_t i = k; i < m; i += 4) {
			__m256d a = _mm256_loadu_pd(ptrk + i);
			__m256d b = _mm256_loadu_pd(ptr5 + i);
			__m256d result = _mm256_fmadd_pd(s, a, b);
			_mm256_storeu_pd(ptr5 + i, result);
		}
	}

	//back substitution step 2, only need first four values
	for (int k = 3; k >= 0; k--) {
		q[k] /= -rd[k];
		for (int i = 0; i < k; i++) {
			q[i] -= q[k] * M[k][i];
		}
	}

	//readjust transform parameter values back to given points
	dest.setParam(q[0], q[1], -dx * q[0] - dy * q[1] + q[2] + dx, dx * q[1] - dy * q[0] + q[3] + dy);
}

//---------------------------------------------------


V4f::V4f() : a { _mm_setzero_ps() } {}

V4f::V4f(float a) : a { _mm_set_ps1(a) } {}

V4f::V4f(__m128 a) : a { a } {}

V4f::V4f(float v0, float v1, float v2, float v3) : a { _mm_setr_ps(v0, v1, v2, v3) } {}

V4f::V4f(float a, float b) : V4f(a, b, a, b) {}

V4f::V4f(const float* data) : a { _mm_loadu_ps(data) } {}

V4f::V4f(const unsigned char* data) : a { _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(data))) } {}

V4f V4f::operator + (V4f other) const { return _mm_add_ps(a, other.a); }
V4f V4f::operator - (V4f other) const { return _mm_sub_ps(a, other.a); }
V4f V4f::operator * (V4f other) const { return _mm_mul_ps(a, other.a); }
V4f V4f::operator / (V4f other) const { return _mm_div_ps(a, other.a); }
V4f V4f::operator += (V4f other) { a = _mm_add_ps(a, other.a); return *this; }
V4f V4f::operator -= (V4f other) { a = _mm_sub_ps(a, other.a); return *this; }
V4f V4f::operator *= (V4f other) { a = _mm_mul_ps(a, other.a); return *this; }
V4f V4f::operator /= (V4f other) { a = _mm_div_ps(a, other.a); return *this; }
V4f V4f::add(V4f other) const { return _mm_add_ps(a, other.a); }
V4f V4f::sub(V4f other) const { return _mm_sub_ps(a, other.a); }
V4f V4f::mul(V4f other) const { return _mm_mul_ps(a, other.a); }
V4f V4f::div(V4f other) const { return _mm_div_ps(a, other.a); }

float V4f::operator [] (size_t i) const { return at(i); }

#ifdef _MSC_VER
float V4f::at(size_t i) const { return a.m128_f32[i]; }
#else
float V4f::at(size_t i) const { return a[i]; }
#endif

std::ostream& operator << (std::ostream& os, const V4f& vec) {
	for (int i = 0; i < 4; i++) os << vec[i] << " ";
	return os;
}

float V4f::sum(int from, int to) const {
	float sum = 0.0f;
	for (int i = from; i < to; i++) sum += at(i);
	return sum;
}

float V4f::sum() const {
	return at(0) + at(1) + at(2) + at(3);
}

V4f V4f::clamp(V4f lo, V4f hi) const {
	return _mm_min_ps(_mm_max_ps(a, lo.a), hi.a);
}

void V4f::storeu(float* dest) const {
	_mm_storeu_ps(dest, a);
}

std::vector<float> V4f::vector() const {
	std::vector<float> v(4);
	_mm_storeu_ps(v.data(), a);
	return v;
}

V4f V4f::rot(int i) const {
	switch (i & 0b11) {
	case 1: return _mm_permute_ps(a, 0b00111001); break;
	case 2: return _mm_permute_ps(a, 0b01001110); break;
	case 3: return _mm_permute_ps(a, 0b10010011); break;
	}
	return a;
}

V4f::operator __m128() const { return a; }
