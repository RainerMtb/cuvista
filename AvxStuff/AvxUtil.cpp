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
void avx::transpose16x4(std::span<VF16> data) {
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
void avx::transpose16x8(std::span<VF16> data) {
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
__m128i avx::yuvToRgbaPacked(VF4 y, VF4 u, VF4 v) {
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
void avx::inv(std::span<VD8> v) {
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
		tmp = _mm512_mask_abs_pd(tmp, mask, v[p]);
		for (size_t i = j + 1; i < m; i++) {
			VD8 a = _mm512_mask_abs_pd(tmp, mask, v[i]);
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
double avx::norm1(std::span<VD8> v) {
	VD8 sum;
	size_t m = v.size();
	for (size_t i = 0; i < m; i++) {
		sum += _mm512_abs_pd(v[i]);
	}
	__mmask8 mask = (1 << m) - 1;
	return _mm512_mask_reduce_max_pd(mask, sum);
}


//print matrix of avx vectors to console
void avx::toConsole(std::span<VD8> v, int digits) {
	int siz = int(v.size());
	AvxMatd mat(siz, 8);
	for (int i = 0; i < siz; i++) v[i].storeu(mat.row(i));
	mat.toConsole(digits);
}


void avx::toConsole(VD8 v, int digits) {
	std::vector<VD8> vec = { v };
	toConsole(vec, digits);
}


//compute similar transform using avx
void avx::computeSimilar(std::span<PointBase> points, Matd& M, Affine2D& affine) {
	assert(points.size() >= 2 && "similar transform needs at least two points");
	size_t m = points.size() * 2;
	Matd A = M.share(6, m + 8);

	int dy = points[0].y;        //represents a2, needs to be smallest value
	int dx = points[1].x;        //represents a3
	long long int nn = 0;        //int could overflow
	long long int s0 = 0;        //in docs s1
	long long int s1 = 0;        //in docs s2

	double* p = A.addr(4, 0);
	double* q = A.addr(5, 0);

	//accumulate s0, s1, nn and adjust coords
	for (size_t idx = 0, k = 0; idx < points.size(); idx++) {
		PointBase& pb = points[idx];
		int x = pb.x - dx;
		p[k] = x;
		s0 += x;
		nn += x * x;
		q[k] = pb.x + pb.u - dx;
		k++;

		int y = pb.y - dy;
		p[k] = y;
		s1 += y;
		nn += y * y;
		q[k] = pb.y + pb.v - dy;
		k++;
	}

	//compute parameters
	double sign0 = p[0] < 0 ? -1.0 : 1.0;
	double n = std::sqrt(nn) * sign0;
	double b = p[0] + n;
	double sign2 = sign0 * n * b < p[3] * s1 ? -1.0 : 1.0;
	double t = sign2 * std::sqrt(points.size() * nn - s0 * s0 - s1 * s1);
	double z = b * (n + t) - p[3] * s1;
	double e = n / t;
	double f = s1 / (b * t);

	double sn = s0 + n;
	double g = sn / (b * t);
	double h = (p[3] / b * (sn * sn + s1 * s1) - s1 * (n + t)) / (t * z);
	double j = sn * (t + n) / (t * z);
	double k = p[3] * n * sn / (t * z);

	double rd[] = { n, -n, t / n, t / n };

	//------------
	A[0][0] = b / n;
	A[0][1] = 0.0;
	A[0][2] = 0.0;
	A[0][3] = p[3] / n;

	A[1][0] = 0.0;
	A[1][1] = 1.0 + p[0] / n;
	A[1][2] = -p[3] / n;
	A[1][3] = 0.0;

	A[2][0] = -s0 / n;
	A[2][1] = s1 / n;
	A[2][2] = 1.0 + e - p[3] * f;
	A[2][3] = -p[3] * g;

	A[3][0] = -s1 / n;
	A[3][1] = -s0 / n;
	A[3][2] = 0.0;
	A[3][3] = 1.0 + e + p[3] * h;

	VD8 pd_n0 = n;
	VD8 pd_n1 = VD8(-n, n);
	VD8 pd_e = VD8(e, 0);
	VD8 pd_f = VD8(-f, f);
	VD8 pd_g = g;
	VD8 pd_ek = VD8(-k, e);
	VD8 pd_h = h;
	VD8 pd_j = VD8(j, -j);
	for (size_t idx = 4; idx < m; idx += 8) {
		VD8 pd_a = p + idx;
		VD8 pd_b = _mm512_permute_pd(pd_a, 0b0101'0101); //switch idx <-> idx+1
		(pd_a / pd_n0).storeu(A.addr(0, idx));
		(pd_b / pd_n1).storeu(A.addr(1, idx));
		(pd_e + pd_b * pd_f - pd_a * pd_g).storeu(A.addr(2, idx));
		(pd_ek + pd_a * pd_h + pd_b * pd_j).storeu(A.addr(3, idx));
	}

	//clear padding values to 0
	for (size_t i = 0; i < 6; i++) {
		for (size_t idx = m; idx < m + 8; idx++) {
			A.at(i, idx) = 0.0;
		}
	}

	//back substitution step 1
	for (size_t k = 0; k < 4; k++) {
		double s = 0.0;
		for (size_t i = k; i < m; i += 8) {
			VD8 a = A.addr(k, i);
			VD8 b = A.addr(5, i);
			s += _mm512_reduce_add_pd(a.mul(b));
		}
		s /= -A[k][k];
		VD8 pd_s = s;
		for (size_t i = k; i < m; i += 8) {
			VD8 a = A.addr(k, i);
			VD8 b = A.addr(5, i);
			(b + a * s).storeu(A.addr(5, i));
		}
	}

	//back substitution step 2, only need first four values
	for (int k = 3; k >= 0; k--) {
		q[k] /= -rd[k];
		for (int i = 0; i < k; i++) {
			q[i] -= q[k] * A[k][i];
		}
	}

	//readjust transform parameter values back to given points
	affine.setParam(q[0], q[1], -dx * q[0] - dy * q[1] + q[2] + dx, dx * q[1] - dy * q[0] + q[3] + dy);
}