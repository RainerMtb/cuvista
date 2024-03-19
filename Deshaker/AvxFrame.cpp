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

#include <immintrin.h>
#include "AvxFrame.hpp"
#include "Util.hpp"

std::ostream& operator << (std::ostream& os, __m512 v) {
	for (int i = 0; i < 16; i++) os << v.m512_f32[i] << " ";
	return os;
}

std::ostream& operator << (std::ostream& os, __m256 v) {
	for (int i = 0; i < 8; i++) os << v.m256_f32[i] << " ";
	return os;
}

AvxFrame::AvxFrame(MainData& data, MovieReader& reader, MovieWriter& writer) :
	MovieFrame(data, reader, writer),
	mYUV(data.bufferCount),
	mPyr(data.pyramidCount, AvxMatFloat(data.pyramidRowCount, data.w, 0.0f)), 
	mFilterResult(data.h, alignValue(data.w, 16)),
	mFilterBuffer(data.w, alignValue(data.h, 16)) {}

std::string AvxFrame::getClassName() const {
	return "AVX 512: " + getCpuName();
}

std::string AvxFrame::getClassId() const {
	return "AVX 512";
}

void AvxFrame::inputData() {
	size_t idx = mBufferFrame.index % mYUV.size();
	mYUV[idx] = mBufferFrame;
}

float AvxFrame::sum(__m512 a, int from, int to) {
	float sum = 0.0f;
	for (int i = from; i < to; i++) sum += a.m512_f32[i];
	return sum;
}

float AvxFrame::sum(__m256 a, int from, int to) {
	float sum = 0.0f;
	for (int i = from; i < to; i++) sum += a.m256_f32[i];
	return sum;
}

void AvxFrame::filter(const float* srcptr, int h, int w, int stride, float* destptr, int destStride, __m512 ignore) {
	__m256 k = _mm256_setr_ps(0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0.0f, 0.0f, 0.0f);
	int ks = 5;
	int s = 2;
	for (int r = 0; r < h; r++) {
		const float* row = srcptr + r * stride;
		float* dest = destptr + r;

		__m256 input;
		input = _mm256_setr_ps(row[0], row[0], row[0], row[1], row[2], 0, 0, 0);
		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
		dest += destStride;

		input = _mm256_setr_ps(row[0], row[0], row[1], row[2], row[3], 0, 0, 0);
		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
		dest += destStride;

		for (int c = s; c < w - s; c++) {
			input = _mm256_loadu_ps(row + c - s);
			*dest = sum(_mm256_mul_ps(input, k), 0, ks);
			dest += destStride;
		}

		input = _mm256_setr_ps(row[w - 4], row[w - 3], row[w - 2], row[w - 1], row[w - 1], 0, 0, 0);
		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
		dest += destStride;

		input = _mm256_setr_ps(row[w - 3], row[w - 2], row[w - 1], row[w - 1], row[w - 1], 0, 0, 0);
		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
		dest += destStride;
	}
}


//__m512i offset = _mm512_set1_epi32(3);
//__m512i idxmin = _mm512_set1_epi32(0);
//__m512i idxmax = _mm512_set1_epi32(15);
//
//float* AvxFrame::filterTriple(__m512i index, __m512 input, __m512 k, float* dest, int destStride) {
//	__m512 a = _mm512_permutexvar_ps(index, input);
//	__m512 result = _mm512_mul_ps(a, k);
//	*dest = sum(result, 0, 5);
//	dest += destStride;
//	*dest = sum(result, 5, 10);
//	dest += destStride;
//	*dest = sum(result, 10, 15);
//	return dest + destStride;
//}
//
//float* AvxFrame::filterVector(__m512i index, __m512 input, __m512 k, float* dest, int destStride) {
//	float* ptr = dest;
//	for (int i = 0; i < 4; i++) {
//		__m512i vindex = _mm512_max_epi32(index, idxmin);
//		vindex = _mm512_min_epi32(vindex, idxmax);
//		ptr = filterTriple(vindex, input, k, ptr, destStride);
//		index = _mm512_add_epi32(index, offset);
//	}
//	return ptr;
//}
//
//void AvxFrame::filter(const float* srcptr, int h, int w, int stride, float* destptr, int destStride, __m512 k) {
//	for (int r = 0; r < h; r++) {
//		const float* row = srcptr + r * stride;
//		float* dest = destptr + r;
//		__m512 input;
//		__m512i index;
//
//		input = _mm512_loadu_ps(row);
//		index = _mm512_setr_epi32(-2, -1, 0, 1, 2, -1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0);
//		dest = filterVector(index, input, k, dest, destStride);
//
//		for (int c = 10; c < w - 16; c += 12) {
//			input = _mm512_loadu_ps(row + c);
//			index = _mm512_setr_epi32(0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 0);
//			dest = filterVector(index, input, k, dest, destStride);
//		}
//
//		input = _mm512_loadu_ps(row + w - 16);
//		index = _mm512_setr_epi32(2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 0);
//		dest = destptr + r + destStride * (w - 12);
//		filterVector(index, input, k, dest, destStride);
//	}
//}

void AvxFrame::downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride) {
	const int mask = 0b1011'0001;
	for (int r = 0; r < h - 1; r += 2) {
		for (int c = 0; c < w; c += 16) {
			const float* src = srcptr + r * stride + c;
			__m512 a = _mm512_loadu_ps(src);
			__m512 b = _mm512_loadu_ps(src + stride);
			__m512 f = _mm512_set1_ps(0.25f);

			a = _mm512_mul_ps(a, f);
			b = _mm512_mul_ps(b, f);
			__m512 result = _mm512_add_ps(a, b);
			__m512 ap = _mm512_permute_ps(a, mask);
			result = _mm512_add_ps(result, ap);
			__m512 bp = _mm512_permute_ps(b, mask);
			result = _mm512_add_ps(result, bp);
			__m512i index = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);
			result = _mm512_permutexvar_ps(index, result);

			float* dest = destptr + r / 2 * destStride + c / 2;
			_mm512_mask_storeu_ps(dest, 0xFF, result);
		}
	}
}

//void AvxFrame::downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride) {
//	const __m512i selector = _mm512_setr_epi32(0, 1, 0xF0, 0xF1, 2, 3, 0xF2, 0xF3, 4, 5, 0xF4, 0xF5, 6, 7, 0xF6, 0xF7);
//	for (int r = 0; r < h - 1; r += 2) {
//		for (int c = 0; c < w; c += 8) {
//			const float* src = srcptr + r * stride + c;
//			__m512 a = _mm512_maskz_loadu_ps(0xFF, src);
//			__m512 b = _mm512_maskz_loadu_ps(0xFF, src + stride);
//			__m512 input = _mm512_permutex2var_ps(a, selector, b);
//
//			__m512 f = _mm512_set1_ps(0.25f);
//			__m512 result = _mm512_mul_ps(input, f);
//
//			float* dest = destptr + r / 2 * destStride + c / 2;
//			dest[0] = result.m512_f32[0] + result.m512_f32[2] + result.m512_f32[1] + result.m512_f32[3];
//			dest[1] = result.m512_f32[4] + result.m512_f32[6] + result.m512_f32[5] + result.m512_f32[7];
//			dest[2] = result.m512_f32[8] + result.m512_f32[10] + result.m512_f32[9] + result.m512_f32[11];
//			dest[3] = result.m512_f32[12] + result.m512_f32[14] + result.m512_f32[13] + result.m512_f32[15];
//
//			//float b = _mm512_mask_reduce_add_ps(0b0000'0000'0000'1111, result); //will add in the order (0 + 2) + (1 + 3)
//		}
//	}
//}

void AvxFrame::createPyramid(int64_t frameIndex) {
	util::ConsoleTimer ic("avx pyramid");
	size_t pyrIdx = frameIndex % mPyr.size();
	AvxMatFloat& Y = mPyr[pyrIdx];

	//fill topmost level of pyramid
	size_t yuvIdx = frameIndex % mYUV.size();
	ImageYuv& yuv = mYUV[yuvIdx];
	constexpr float f = 1.0f / 255.0f;
	const __m512 fps = _mm512_set1_ps(f);

	for (int r = 0; r < mData.h; r++) {
		int c = 0;
		for (; c < mData.w / 16 * 16; c += 16) {
			__m128i epi8 = _mm_loadu_epi8(yuv.addr(0, r, c));
			__m512i epi32 = _mm512_cvtepu8_epi32(epi8);
			__m512 ps = _mm512_cvtepi32_ps(epi32);
			__m512 result = _mm512_mul_ps(ps, fps);
			_mm512_storeu_ps(Y.row(r) + c, result);
		}
		//image width may not align to 16, handle trailing pixels individually
		for (; c < mData.w; c++) {
			Y.at(r, c) = yuv.at(0, r, c) * f;
		}
	}

	//create pyramid levels below by downsampling level above
	int r = 0;
	int h = mData.h;
	int w = mData.w;
	for (size_t z = 1; z < mData.pyramidLevels; z++) {
		filter(Y.row(r), h, w, mData.w, mFilterBuffer.data(), mFilterBuffer.w(), filterKernels[0]);
		filter(mFilterBuffer.data(), w, h, mFilterBuffer.w(), mFilterResult.data(), mFilterResult.w(), filterKernels[0]);
		r += h;
		downsample(mFilterResult.data(), h, w, mFilterResult.w(), Y.row(r), Y.w());
		h /= 2;
		w /= 2;
		//if (z == 1) std::printf("avx %.14f\n", Y.at(r + 100, 100));
		//if (z == 1) mFilterBuffer.saveAsBinary("f:/filterAvx.dat");
	}
}

void AvxFrame::computeStart(int64_t frameIndex) {}

void AvxFrame::computeTerminate(int64_t frameIndex) {
	//TODO
}

void AvxFrame::outputData(const AffineTransform& trf, OutputContext outCtx) {
	//TODO
}

Matf AvxFrame::getTransformedOutput() const {
	//TODO
	return Matf();
}

void AvxFrame::getTransformedOutput(int64_t frameIndex, ImagePPM& image) {
	//TODO
}

Matf AvxFrame::getPyramid(size_t idx) const {
	return mPyr[idx];
}

void AvxFrame::getInputFrame(int64_t frameIndex, ImagePPM& image) {
	size_t idx = frameIndex % mYUV.size();
	mYUV[idx].toPPM(image, mPool);
}

ImageYuv AvxFrame::getInput(int64_t index) const {
	return mYUV[index % mYUV.size()];
}
