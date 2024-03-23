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
class AvxMatFloat : public CoreMat<float> {
public:
	AvxMatFloat() : CoreMat<float>() {}
	AvxMatFloat(int h, int w) : CoreMat<float>(h, w) {}
	AvxMatFloat(int h, int w, float value) : CoreMat<float>(h, w, value) {}

	int w() const { return int(CoreMat::w); }
	int h() const { return int(CoreMat::h); }
	float* addr(size_t row, size_t col) override { return array + row * CoreMat::w + col; }
	const float* addr(size_t row, size_t col) const override { return array + row * CoreMat::w + col; }
	float* row(int r) { return addr(r, 0); }
	const float* row(int r) const { return addr(r, 0); }
	void fill(float value) { std::fill(array, array + numel(), value); }
	void saveAsBinary(const std::string& filename) { Matf::fromArray(h(), w(), array, false).saveAsBinary(filename); }
};


//wrapper for _mm512
class Vecf {
public:
	__m512 a;

	Vecf() : a { _mm512_setzero_ps() } {}

	Vecf(float a) : a { _mm512_set1_ps(a) } {}

	Vecf(__m512 a) : a { a } {}

	Vecf(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7,
		float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15) :
		a { _mm512_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) } {}

	Vecf(const float* data) : a { _mm512_loadu_ps(data) } {}

	Vecf(const float* data, __mmask16 mask) : a { _mm512_maskz_load_ps(mask, data) } {}

	Vecf(const unsigned char* data) : a { _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_epi8(data))) } {}

	Vecf(const unsigned char* data, __mmask16 mask) : a { _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, data))) } {}

	Vecf operator + (Vecf other) { return _mm512_add_ps(a, other.a); }
	Vecf operator - (Vecf other) { return _mm512_sub_ps(a, other.a); }
	Vecf operator * (Vecf other) { return _mm512_mul_ps(a, other.a); }
	Vecf operator / (Vecf other) { return _mm512_div_ps(a, other.a); }

	float operator [] (size_t i) const { return a.m512_f32[i]; }

	friend std::ostream& operator << (std::ostream& os, const Vecf& vec) {
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

	Vecf clamp(Vecf lo, Vecf hi) const {
		return _mm512_min_ps(_mm512_max_ps(a, lo.a), hi.a);
	}

	static Vecf clamp(Vecf value, Vecf lo, Vecf hi) {
		return value.clamp(lo, hi);
	}

	void storeu(float* dest) {
		_mm512_storeu_ps(dest, a);
	}

	void storeu(float* dest, __mmask16 mask) {
		_mm512_mask_storeu_ps(dest, mask, a);
	}
};
