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

#include "AvxUtil.hpp"


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
__m512i Avx::yuvToRgbPacked(VF16 y, VF16 u, VF16 v) {
	VF16 r = (y + ((v - 128.0f) * 1.370705f)).clamp(0.0f, 255.0f);
	VF16 g = (y - ((u - 128.0f) * 0.337633f) - ((v - 128.0f) * 0.698001f)).clamp(0.0f, 255.0f);
	VF16 b = (y + ((u - 128.0f) * 1.732446f)).clamp(0.0f, 255.0f);

	//convert floats to uint8 stored in 512 bits
	//default conversion in avx uses rint()
	__m512i ir = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(_mm512_cvtps_epi32(r)));
	__m512i ig = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(_mm512_cvtps_epi32(g)));
	__m512i ib = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(_mm512_cvtps_epi32(b)));

	//pack into the lower 3/4 of one 512 vector
	__m512i selectorRG = _mm512_setr_epi8(
		0, 64, 0, 1, 65, 0, 2, 66, 0, 3, 67, 0, 4, 68, 0, 5,
		69, 0, 6, 70, 0, 7, 71, 0, 8, 72, 0, 9, 73, 0, 10, 74,
		0, 11, 75, 0, 12, 76, 0, 13, 77, 0, 14, 78, 0, 15, 79, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	);
	__m512i selectorB = _mm512_setr_epi8(
		64, 65, 0, 67, 68, 1, 70, 71, 2, 73, 74, 3, 76, 77, 4, 79,
		80, 5, 82, 83, 6, 85, 86, 7, 88, 89, 8, 91, 92, 9, 94, 95,
		10, 97, 98, 11, 100, 101, 12, 103, 104, 13, 106, 107, 14, 109, 110, 15,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	);

	__m512i result;
	//combine red and green
	result = _mm512_permutex2var_epi8(ir, selectorRG, ig);
	//combine above with blue
	result = _mm512_permutex2var_epi8(ib, selectorB, result);
	return result;
}
