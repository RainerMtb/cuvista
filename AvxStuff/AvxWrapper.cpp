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

V16f::V16f(const unsigned char* data) : a { _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm_loadu_epi8(data))) } {}

V16f::V16f(const unsigned char* data, __mmask16 mask) : a { _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

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

#ifdef _MSC_VER
float V16f::at(size_t i) const { return a.m512_f32[i]; }
#else
float V16f::at(size_t i) const { return a[i]; }
#endif

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
	return sum(0, 16);
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
	__m512i idx = _mm512_loadu_epi32(Iota::i32);
	__m512i delta = _mm512_set1_epi32(i & 0xF);
	idx = _mm512_add_epi32(idx, delta);
	return _mm512_permutex2var_ps(a, idx, a);
}

V16f::operator __m512() { return a; }

//--------------------------------------------------------

V8f::V8f() : a { _mm256_setzero_ps() } {}

V8f::V8f(float a) : a { _mm256_set1_ps(a) } {}

V8f::V8f(__m256 a) : a { a } {}

V8f::V8f(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) :
	a { _mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7) } {}

V8f::V8f(float a, float b) : V8f(a, b, a, b, a, b, a, b) {}

V8f::V8f(const float* data) : a { _mm256_loadu_ps(data) } {}

V8f::V8f(const float* data, __mmask8 mask) : a { _mm256_maskz_loadu_ps(mask, data) } {}

V8f::V8f(const unsigned char* data) : a { _mm256_cvtepu32_ps(_mm256_cvtepu8_epi32(_mm_maskz_loadu_epi8(0xFF, data))) } {}

V8f::V8f(const unsigned char* data, __mmask8 mask) : a { _mm256_cvtepu32_ps(_mm256_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

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

V8f V8f::broadcast(int i) const { return _mm256_permutexvar_ps(_mm256_set1_epi32(i), a); }

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

void V8f::storeu(float* dest, __mmask8 mask) const {
	_mm256_mask_storeu_ps(dest, mask, a);
}

std::vector<float> V8f::vector() const {
	std::vector<float> v(8);
	_mm256_storeu_ps(v.data(), a);
	return v;
}

V8f V8f::rot(int i) const {
	__m256i idx = _mm256_loadu_epi32(Iota::i32);
	__m256i delta = _mm256_set1_epi32(i & 0b111);
	idx = _mm256_add_epi32(idx, delta);
	return _mm256_permutex2var_ps(a, idx, a);
}

V8f::operator __m256() { return a; }

//---------------------------------------------------

V4f::V4f() : a { _mm_setzero_ps() } {}

V4f::V4f(float a) : a { _mm_set_ps1(a) } {}

V4f::V4f(__m128 a) : a { a } {}

V4f::V4f(float v0, float v1, float v2, float v3) :
	a { _mm_setr_ps(v0, v1, v2, v3) } {}

V4f::V4f(float a, float b) : V4f(a, b, a, b) {}

V4f::V4f(const float* data) : a { _mm_loadu_ps(data) } {}

V4f::V4f(const float* data, __mmask8 mask) : a { _mm_maskz_loadu_ps(mask, data) } {}

V4f::V4f(const unsigned char* data) : a { _mm_cvtepu32_ps(_mm_cvtepu8_epi32(_mm_maskz_loadu_epi8(0xF, data))) } {}

V4f::V4f(const unsigned char* data, __mmask8 mask) : a { _mm_cvtepu32_ps(_mm_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

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

V4f V4f::broadcast(int i) const { return _mm_permutevar_ps(a, _mm_set1_epi32(i)); }

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

void V4f::storeu(float* dest, __mmask8 mask) const {
	_mm_mask_storeu_ps(dest, mask, a);
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

V4f::operator __m128() { return a; }

//------------------------------------------------

V8d::V8d() : a { _mm512_setzero_pd() } {}

V8d::V8d(double a) : a { _mm512_set1_pd(a) } {}

V8d::V8d(double a, double b) : V8d(a, b, a, b, a, b, a, b) {}

V8d::V8d(__m512d a) : a { a } {}

V8d::V8d(__m256 a) : a { _mm512_cvtps_pd(a) } {}

V8d::V8d(V8f a) : a { _mm512_cvtps_pd(a) } {}

V8d::V8d(double v0, double v1, double v2, double v3, double v4, double v5, double v6, double v7) :
	a { _mm512_setr_pd(v0, v1, v2, v3, v4, v5, v6, v7) } {}

V8d::V8d(const double* data) : a { _mm512_loadu_pd(data) } {}

V8d::V8d(const double* data, __mmask8 mask) : a { _mm512_maskz_loadu_pd(mask, data) } {}

V8d::V8d(const unsigned char* data) : a { _mm512_cvtepu64_pd(_mm512_cvtepu8_epi64(_mm_maskz_loadu_epi8(0xFF, data))) } {}

V8d::V8d(const unsigned char* data, __mmask8 mask) : a { _mm512_cvtepu64_pd(_mm512_cvtepu8_epi64(_mm_maskz_loadu_epi8(mask, data))) } {}

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

#ifdef _MSC_VER
double V8d::at(size_t i) const { return a.m512d_f64[i]; }
#else
double V8d::at(size_t i) const { return a[i]; }
#endif

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
	__m512i idx = _mm512_loadu_epi64(Iota::i64);
	__m512i delta = _mm512_set1_epi64(i & 0b111);
	idx = _mm512_add_epi64(idx, delta);
	return _mm512_permutex2var_pd(a, idx, a);
}

V8d::operator __m512d() { return a; }