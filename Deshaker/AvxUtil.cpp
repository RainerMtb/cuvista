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

#include <numeric>
#include <vector>
#include "AvxUtil.hpp"
#include "AvxMat.hpp"


 //transpose 4 vectors of 16 floats
void Avx::transpose16x4(std::span<VF16> data) {
	VF16 tmp[8];

	tmp[0] = _mm512_unpacklo_ps(data[0], data[1]);
	tmp[1] = _mm512_unpackhi_ps(data[0], data[1]);
	tmp[2] = _mm512_unpacklo_ps(data[2], data[3]);
	tmp[3] = _mm512_unpackhi_ps(data[2], data[3]);

	data[0] = _mm512_shuffle_ps(tmp[0], tmp[2], 0b0100'0100);
	data[1] = _mm512_shuffle_ps(tmp[0], tmp[2], 0b1110'1110);
	data[2] = _mm512_shuffle_ps(tmp[1], tmp[3], 0b0100'0100);
	data[3] = _mm512_shuffle_ps(tmp[1], tmp[3], 0b1110'1110);
}


//transpose 8 vectors of 16 floats
//result is returned again in 8 vectors of 16 floats
//each vector contains two 'blocks' of 8 floats representing data rows
void Avx::transpose16x8(std::span<VF16> data) {
	VF16 tmp[8];

	tmp[0] = _mm512_unpacklo_ps(data[0], data[1]);
	tmp[1] = _mm512_unpackhi_ps(data[0], data[1]);
	tmp[2] = _mm512_unpacklo_ps(data[2], data[3]);
	tmp[3] = _mm512_unpackhi_ps(data[2], data[3]);
	tmp[4] = _mm512_unpacklo_ps(data[4], data[5]);
	tmp[5] = _mm512_unpackhi_ps(data[4], data[5]);
	tmp[6] = _mm512_unpacklo_ps(data[6], data[7]);
	tmp[7] = _mm512_unpackhi_ps(data[6], data[7]);

	data[0] = _mm512_shuffle_ps(tmp[0], tmp[2], 0b0100'0100);
	data[1] = _mm512_shuffle_ps(tmp[0], tmp[2], 0b1110'1110);
	data[2] = _mm512_shuffle_ps(tmp[1], tmp[3], 0b0100'0100);
	data[3] = _mm512_shuffle_ps(tmp[1], tmp[3], 0b1110'1110);
	data[4] = _mm512_shuffle_ps(tmp[4], tmp[6], 0b0100'0100);
	data[5] = _mm512_shuffle_ps(tmp[4], tmp[6], 0b1110'1110);
	data[6] = _mm512_shuffle_ps(tmp[5], tmp[7], 0b0100'0100);
	data[7] = _mm512_shuffle_ps(tmp[5], tmp[7], 0b1110'1110);

	tmp[0] = _mm512_shuffle_f32x4(data[0], data[4], 0b1000'1000);
	tmp[1] = _mm512_shuffle_f32x4(data[1], data[5], 0b1000'1000);
	tmp[2] = _mm512_shuffle_f32x4(data[2], data[6], 0b1000'1000);
	tmp[3] = _mm512_shuffle_f32x4(data[3], data[7], 0b1000'1000);
	tmp[4] = _mm512_shuffle_f32x4(data[0], data[4], 0b1101'1101);
	tmp[5] = _mm512_shuffle_f32x4(data[1], data[5], 0b1101'1101);
	tmp[6] = _mm512_shuffle_f32x4(data[2], data[6], 0b1101'1101);
	tmp[7] = _mm512_shuffle_f32x4(data[3], data[7], 0b1101'1101);

	data[0] = _mm512_shuffle_f32x4(tmp[0], tmp[1], 0b1000'1000);
	data[1] = _mm512_shuffle_f32x4(tmp[2], tmp[3], 0b1000'1000);
	data[2] = _mm512_shuffle_f32x4(tmp[4], tmp[5], 0b1000'1000);
	data[3] = _mm512_shuffle_f32x4(tmp[6], tmp[7], 0b1000'1000);
	data[4] = _mm512_shuffle_f32x4(tmp[0], tmp[1], 0b1101'1101);
	data[5] = _mm512_shuffle_f32x4(tmp[2], tmp[3], 0b1101'1101);
	data[6] = _mm512_shuffle_f32x4(tmp[4], tmp[5], 0b1101'1101);
	data[7] = _mm512_shuffle_f32x4(tmp[6], tmp[7], 0b1101'1101);
}


//convert individual vectors in float for Y U V to one vector holding uchar packed RGB
__m128i Avx::yuvToRgbaPacked(VF4 y, VF4 u, VF4 v) {
	//distribute y, u, v values to 16 places
	__m512i index = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
	VF16 yy = _mm512_permutexvar_ps(index, _mm512_castps128_ps512(y));
	VF16 uu = _mm512_permutexvar_ps(index, _mm512_castps128_ps512(u));
	VF16 vv = _mm512_permutexvar_ps(index, _mm512_castps128_ps512(v));
	
	//factors for conversion yuv to rgb
	VF16 fu = { 0.0f, -0.337633f, 1.732446f, 0.0f };
	VF16 fv = { 1.370705f, -0.698001f, 0.0f, 0.0f };

	//convert
	VF16 ps255 = 255.0f;
	VF16 ps128 = 128.0f;
	VF16 rgba;
	rgba = yy + (uu - ps128) * fu + (vv - ps128) * fv;
	rgba = rgba.clamp(0.0f, 255.0f);
	rgba = _mm512_mask_blend_ps(0b1000'1000'1000'1000, rgba, ps255);

	//convert floats to uint8
	//default conversion in avx uses rint()
	return _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(rgba));
}


//invert matrix given in avx vectors
//matrix must be square
void Avx::inv(std::span<VD8> v) {
	size_t m = v.size();
	std::vector<size_t> piv(m);
	std::iota(piv.begin(), piv.end(), 0);
	VD8 tmp;

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
		tmp = _mm512_maskz_abs_pd(mask, v[p]);
		for (size_t i = j + 1; i < m; i++) {
			VD8 a = _mm512_maskz_abs_pd(mask, v[i]);
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
	std::vector<VD8> x(m);
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
double Avx::norm1(std::span<VD8> v) {
	VD8 sum;
	size_t m = v.size();
	for (size_t i = 0; i < m; i++) {
		sum += _mm512_abs_pd(v[i]);
	}
	__mmask8 mask = (1 << m) - 1;
	return _mm512_mask_reduce_max_pd(mask, sum);
}


//print matrix of avx vectors to console
void Avx::toConsole(std::span<VD8> v) {
	int siz = int(v.size());
	AvxMatd mat(siz, 8);
	for (int i = 0; i < siz; i++) v[i].storeu(mat.row(i));
	mat.toConsole();
}


//print matrix of avx vectors to console
void Avx::toConsole(std::span<VF16> v) {
	int siz = int(v.size());
	AvxMatf mat(siz, 16);
	for (int i = 0; i < siz; i++) v[i].storeu(mat.row(i));
	mat.toConsole();
}