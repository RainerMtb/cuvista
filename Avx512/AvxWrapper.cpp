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

#include "AvxWrapper.hpp"
#include "AvxMat.hpp"
#include "AvxUtil.hpp"

using namespace avx;

V16f::V16f() : a { _mm512_setzero_ps() } {}

V16f::V16f(float a) : a { _mm512_set1_ps(a) } {}

V16f::V16f(__m512 a) : a { a } {}

V16f::V16f(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7,
	float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15) :
	a { _mm512_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) } {}

V16f::V16f(float a, float b) : V16f(a, b, a, b, a, b, a, b, a, b, a, b, a, b, a, b) {}

V16f::V16f(float a, float b, float c, float d) : V16f(a, b, c, d, a, b, c, d, a, b, c, d, a, b, c, d) {}

V16f::V16f(const float* data) : a { _mm512_loadu_ps(data) } {}

V16f::V16f(const float* data, __mmask16 mask) : a { _mm512_maskz_loadu_ps(mask, data) } {}

V16f::V16f(const unsigned char* data) : a { _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_epi8(data))) } {}

V16f::V16f(const unsigned char* data, __mmask16 mask) : a { _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

V16f V16f::operator + (V16f other) const { return _mm512_add_ps(a, other.a); }
V16f V16f::operator - (V16f other) const { return _mm512_sub_ps(a, other.a); }
V16f V16f::operator * (V16f other) const { return _mm512_mul_ps(a, other.a); }
V16f V16f::operator / (V16f other) const { return _mm512_div_ps(a, other.a); }
V16f V16f::operator += (V16f other) { a = _mm512_add_ps(a, other.a); return *this; }
V16f V16f::operator -= (V16f other) { a = _mm512_sub_ps(a, other.a); return *this; }
V16f V16f::operator *= (V16f other) { a = _mm512_mul_ps(a, other.a); return *this; }
V16f V16f::operator /= (V16f other) { a = _mm512_div_ps(a, other.a); return *this; }
V16f V16f::add(V16f other) const { return _mm512_add_ps(a, other.a); }
V16f V16f::sub(V16f other) const { return _mm512_sub_ps(a, other.a); }
V16f V16f::mul(V16f other) const { return _mm512_mul_ps(a, other.a); }
V16f V16f::div(V16f other) const { return _mm512_div_ps(a, other.a); }

V16f V16f::broadcast(int i) const { return _mm512_permutexvar_ps(_mm512_set1_epi32(i), a); }

float V16f::operator [] (size_t i) const { return at(i); }

float V16f::at(size_t i) const { 
	__mmask16 mask = 1 << i;
	float f;
	_mm512_mask_compressstoreu_ps(&f, mask, a);
	return f;
}

std::ostream& operator << (std::ostream& os, const V16f& vec) {
	for (int i = 0; i < 16; i++) os << vec[i] << " ";
	return os;
}

float V16f::sum(int from, int to) const {
	float sum = 0.0f;
	for (int i = from; i < to; i++) sum += at(i);
	return sum;
}

float V16f::sum() const {
	return at(0) + at(1) + at(2) + at(3) + at(4) + at(5) + at(6) + at(7) + at(8) + at(9) + at(10) + at(11) + at(12) + at(13) + at(14) + at(15);
}

V16f V16f::clamp(V16f lo, V16f hi) const {
	return _mm512_min_ps(_mm512_max_ps(a, lo.a), hi.a);
}

void V16f::storeu(float* dest) const {
	_mm512_storeu_ps(dest, a);
}

void V16f::storeu(float* dest, __mmask16 mask) const {
	_mm512_mask_storeu_ps(dest, mask, a);
}

std::vector<float> V16f::vector() const {
	std::vector<float> v(16);
	_mm512_storeu_ps(v.data(), a);
	return v;
}

V16f V16f::rot(int i) const {
	int offset = i & 0xF;
	__m512i idx = _mm512_loadu_epi32(iotas.i32x16 + offset);
	return _mm512_permutex2var_ps(a, idx, a);
}

V16f::operator __m512() const { return a; }

//convert individual vectors in float for Y U V to one vector holding uchar packed RGB
void V16f::yuvToRgbaPacked(V16f y, V16f u, V16f v, unsigned char* dest, V16f fu, V16f fv) {
	V16f ps255 = 255.0f;
	V16f ps0 = 0.0f;
	V16f rgba;

	//convert color
	rgba = (y - 16.0f) * 1.164f + (u - 128.0f) * fu + (v - 128.0f) * fv;
	rgba = _mm512_mask_max_ps(ps255, 0b0111'0111'0111'0111, rgba, ps0);

	//convert floats to uint8, saturate and store
	__m512i epi32 = _mm512_cvtps_epi32(rgba);
	_mm512_mask_cvtusepi32_storeu_epi8(dest, 0xFFFF, epi32);
}

//------------------------------------------------


V8d::V8d() : a { _mm512_setzero_pd() } {}

V8d::V8d(double a) : a { _mm512_set1_pd(a) } {}

V8d::V8d(double a, double b) : V8d(a, b, a, b, a, b, a, b) {}

V8d::V8d(__m512d a) : a { a } {}

V8d::V8d(__m256 a) : a { _mm512_cvtps_pd(a) } {}

V8d::V8d(double a, double b, double c, double d, double e, double f, double g, double h) : a { _mm512_setr_pd(a, b, c, d, e, f, g, h) } {}

V8d::V8d(const double* data) : a { _mm512_loadu_pd(data) } {}

V8d::V8d(const double* data, __mmask8 mask) : a { _mm512_maskz_loadu_pd(mask, data) } {}

V8d::V8d(const unsigned char* data) : a { _mm512_cvtepi64_pd(_mm512_cvtepu8_epi64(_mm_maskz_loadu_epi8(0xFF, data))) } {}

V8d::V8d(const unsigned char* data, __mmask8 mask) : a { _mm512_cvtepi64_pd(_mm512_cvtepu8_epi64(_mm_maskz_loadu_epi8(mask, data))) } {}

V8d V8d::operator + (V8d other) const { return _mm512_add_pd(a, other.a); }
V8d V8d::operator - (V8d other) const { return _mm512_sub_pd(a, other.a); }
V8d V8d::operator * (V8d other) const { return _mm512_mul_pd(a, other.a); }
V8d V8d::operator / (V8d other) const { return _mm512_div_pd(a, other.a); }
V8d V8d::operator += (V8d other) { a = _mm512_add_pd(a, other.a); return *this; }
V8d V8d::operator -= (V8d other) { a = _mm512_sub_pd(a, other.a); return *this; }
V8d V8d::operator *= (V8d other) { a = _mm512_mul_pd(a, other.a); return *this; }
V8d V8d::operator /= (V8d other) { a = _mm512_div_pd(a, other.a); return *this; }
V8d V8d::add(V8d other) const { return _mm512_add_pd(a, other.a); }
V8d V8d::sub(V8d other) const { return _mm512_sub_pd(a, other.a); }
V8d V8d::mul(V8d other) const { return _mm512_mul_pd(a, other.a); }
V8d V8d::div(V8d other) const { return _mm512_div_pd(a, other.a); }

V8d V8d::broadcast(int i) const { return _mm512_permutexvar_pd(_mm512_set1_epi64(i), a); }

double V8d::operator [] (size_t i) const { return at(i); }

double V8d::at(size_t i) const { 
	__mmask8 mask = 1 << i;
	double d;
	_mm512_mask_compressstoreu_pd(&d, mask, a);
	return d;
}

std::ostream& operator << (std::ostream& os, const V8d& vec) {
	for (int i = 0; i < 8; i++) os << vec[i] << " ";
	return os;
}

double V8d::sum(int from, int to) const {
	double sum = 0.0;
	for (int i = from; i < to; i++) sum += at(i);
	return sum;
}

double V8d::sum() const {
	return at(0) + at(1) + at(2) + at(3) + at(4) + at(5) + at(6) + at(7);
}

V8d V8d::clamp(V8d lo, V8d hi) const {
	return _mm512_min_pd(_mm512_max_pd(a, lo.a), hi.a);
}

void V8d::storeu(double* dest) const {
	_mm512_storeu_pd(dest, a);
}

void V8d::storeu(double* dest, __mmask8 mask) const {
	_mm512_mask_storeu_pd(dest, mask, a);
}

std::vector<double> V8d::vector() const {
	std::vector<double> v(8);
	_mm512_storeu_pd(v.data(), a);
	return v;
}

V8d V8d::rot(int i) const {
	int offset = i & 0x7;
	__m512i idx = _mm512_loadu_epi64(iotas.i64x8 + offset);
	return _mm512_permutex2var_pd(a, idx, a);
}

V8d::operator __m512d() const { return a; }

//compute similar transform using avx
AffineSolverAvx512::AffineSolverAvx512(ThreadPoolBase& threadPool, size_t maxPoints) :
	AffineSolverDirect(threadPool, maxPoints)
{}

//compute similar transform using avx
void AffineSolverAvx512::computeSimilar(std::span<PointBase> points, Affine2D& dest) {
	//util::ConsoleTimer timer("computeSimilar avx512");
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

	V8d pd_n0 = n;
	V8d pd_n1 = V8d(-n, n);
	V8d pd_e = V8d(e, 0);
	V8d pd_f = V8d(-f, f);
	V8d pd_g = g;
	V8d pd_ek = V8d(-k, e);
	V8d pd_h = h;
	V8d pd_j = V8d(j, -j);
	for (size_t idx = 4; idx < m; idx += 8) {
		V8d pd_a = p + idx;
		V8d pd_b = _mm512_permute_pd(pd_a, 0b0101'0101); //switch idx <-> idx+1
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
		__m512d s = _mm512_set1_pd(0.0);
		__m512d x = _mm512_set1_pd(-M[k][k]);

		//inner product, first loop
		for (size_t i = k; i < m; i += 8) {
			__m512d a = _mm512_loadu_pd(ptrk + i);
			__m512d b = _mm512_loadu_pd(ptr5 + i);
			s = _mm512_fmadd_pd(a, b, s);
		}

		//horizontal sum
		__m512d b;
		b = _mm512_shuffle_pd(s, s, 0b01010101);
		s = _mm512_add_pd(s, b);

		b = _mm512_shuffle_f64x2(s, s, 0b10110001);
		s = _mm512_add_pd(s, b);

		b = _mm512_shuffle_f64x2(s, s, 0b01001110);
		s = _mm512_add_pd(s, b);

		//second loop
		s = _mm512_div_pd(s, x);
		for (size_t i = k; i < m; i += 8) {
			__m512d a = _mm512_loadu_pd(ptrk + i);
			__m512d b = _mm512_loadu_pd(ptr5 + i);
			__m512d result = _mm512_fmadd_pd(s, a, b);
			_mm512_storeu_pd(ptr5 + i, result);
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

//invert matrix given in avx vectors
//matrix must be square
void V8d::inv(std::span<V8d> v) {
	size_t m = v.size();
	std::vector<size_t> piv(m);
	std::iota(piv.begin(), piv.end(), 0);
	inv(v, piv);
}

void V8d::inv(std::span<V8d> v, std::span<size_t> piv) {
	size_t m = v.size();
	V8d tmp;

	for (size_t j = 0; j < m; j++) {
		__mmask8 mask = 1 << j;

		for (size_t i = 0; i < j; i++) {
			__m512i idx = _mm512_set1_epi64(i);
			for (size_t k = i + 1; k < m; k++) {
				tmp = _mm512_permutexvar_pd(idx, v[k]);
				v[k] = _mm512_mask_sub_pd(v[k], mask, v[k], tmp * v[i]);
			}
		}

		//find pivot and exchange if necessary
		size_t p = j;
		tmp = _mm512_mask_abs_pd(tmp, mask, v[p]);
		for (size_t i = j + 1; i < m; i++) {
			V8d a = _mm512_mask_abs_pd(tmp, mask, v[i]);
			if (_mm512_cmp_pd_mask(a, tmp, _CMP_GT_OS)) {
				tmp = a;
				p = i;
			}
		}
		std::swap(v[p], v[j]);
		std::swap(piv[p], piv[j]);

		// Compute multipliers.
		for (size_t i = j + 1; i < m; i++) {
			v[i] = _mm512_mask_div_pd(v[i], mask, v[i], v[j]);
		}
	}

	// prepare temporary and destination vectors
	// v indentity matrix but rows reordered according to piv - will turn into result
	// x holds decomposed matrix
	std::vector<V8d> x(m);
	for (int i = 0; i < m; i++) {
		double p[8] = {};
		size_t idx = piv[i];
		p[idx] = 1.0;
		x[i] = v[i];
		v[i] = p;
	}

	//solve against identity
	for (size_t k = 0; k < m; k++) {
		__m512i tk = _mm512_set1_epi64(k);
		for (size_t i = k + 1; i < m; i++) {
			tmp = _mm512_permutexvar_pd(tk, x[i]); //broadcast x[i][k]
			v[i] -= v[k] * tmp;
		}
	}
	for (int64_t k = m - 1; k >= 0; k--) {
		__m512i tk = _mm512_set1_epi64(k);
		tmp = _mm512_permutexvar_pd(tk, x[k]); //broadcast x[k][k]
		v[k] /= tmp;
		for (int64_t i = 0; i < k; i++) {
			tmp = _mm512_permutexvar_pd(tk, x[i]); //broadcast x[i][k]
			v[i] -= v[k] * tmp;
		}
	}
}

//compute 1-norm of square matrix given in avx vectors
double V8d::norm1(std::span<V8d> v) {
	V8d sum;
	size_t m = v.size();
	for (size_t i = 0; i < m; i++) {
		sum += _mm512_abs_pd(v[i]);
	}
	__mmask8 mask = (1 << m) - 1;
	return _mm512_mask_reduce_max_pd(mask, sum);
}

//print matrix of avx vectors to console
void V8d::toConsole(std::span<V8d> v, int digits) {
	int siz = int(v.size());
	AvxMatd mat(siz, 8);
	for (int i = 0; i < siz; i++) v[i].storeu(mat.row(i));
	mat.toConsole(digits);
}

//print matrix of avx row vector to console
void V8d::toConsole(V8d v, int digits) {
	std::vector<V8d> vec = { v };
	toConsole(vec, digits);
}
